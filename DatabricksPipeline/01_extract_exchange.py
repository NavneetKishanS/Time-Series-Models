# Databricks notebook — Exchange Sequence Extraction
#
# Extracts exchange (between-patient transition) event sequences from
# hive_metastore.eventlog.common_eventlog for the target scanners.
#
# PatientId lives in examination_workflow.WorkflowValues["PatientId"] — it
# does NOT exist as a column in the eventlog.  We join via SerialNumber +
# time-range (pd.merge_asof) using WorkflowStartRefDateTime as patient
# arrival time.
#
# Output: exchange_sequences.pkl

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import re
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql import functions as F

os.makedirs(DBFS_OUTPUT_BASE, exist_ok=True)
print(f"Output directory: {DBFS_OUTPUT_BASE}")

# COMMAND ----------
# =============================================================================
# DIAGNOSTIC — confirm schemas and WorkflowValues keys (run once, then skip)
# =============================================================================

print("=" * 60)
print(f"EVENTLOG schema: {EVENTLOG_TABLE}")
print("=" * 60)
spark.table(EVENTLOG_TABLE).printSchema()

print("=" * 60)
print(f"EXAMINATION schema: {EXAMINATION_TABLE}")
print("=" * 60)
spark.table(EXAMINATION_TABLE).printSchema()

# Show sample WorkflowValues keys for the target scanners
print("=" * 60)
print("Sample WorkflowValues keys (target scanners):")
print("=" * 60)
sample_wf = (
    spark.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin(TARGET_SERIAL_NUMBERS))
    .filter(F.col("WorkflowValues").isNotNull())
    .select(F.map_keys("WorkflowValues").alias("keys"))
    .limit(10)
    .toPandas()
)
all_keys = set(k for row in sample_wf["keys"] for k in row)
print(f"Keys found in WorkflowValues: {sorted(all_keys)}")

# COMMAND ----------
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_temporal_features(df, dt_col='AdjustedEventDateTime'):
    """Add hour/day temporal features + cyclical encodings."""
    dt = df[dt_col]
    df = df.copy()
    df['hour_of_day'] = dt.dt.hour
    df['day_of_week']  = dt.dt.dayofweek       # Monday=0
    df['is_morning']   = (dt.dt.hour < 12).astype(int)
    df['hour_sin']     = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos']     = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin']      = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos']      = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['date']         = dt.dt.date
    return df


def build_conditioning(row):
    """Extract 12-key conditioning dict from a pandas Series."""
    def _f(key, default=0.0):
        v = row.get(key, default)
        try:
            v = float(v)
            return default if (v != v) else v
        except (TypeError, ValueError):
            return default
    return {
        'Age':               _f('Age'),
        'Weight':            _f('Weight'),
        'Height':            _f('Height'),
        'PTAB':              _f('PTAB'),
        'Direction_encoded': _f('Direction_encoded'),
        'hour_of_day':       _f('hour_of_day'),
        'day_of_week':       _f('day_of_week'),
        'is_morning':        _f('is_morning'),
        'hour_sin':          _f('hour_sin'),
        'hour_cos':          _f('hour_cos', 1.0),
        'dow_sin':           _f('dow_sin'),
        'dow_cos':           _f('dow_cos', 1.0),
    }


def add_ptab(df):
    """Extract PTAB from MRI_FRR_257 messages and forward-fill."""
    df = df.copy()
    df['PTAB'] = np.nan
    mask = df['MessageIdentification'] == 'MRI_FRR_257'
    def _parse(msg):
        if not isinstance(msg, str): return np.nan
        m = re.search(r'[-+]?\d+\.?\d*', msg)
        return float(m.group()) if m else np.nan
    df.loc[mask, 'PTAB'] = df.loc[mask, 'Message'].apply(_parse)
    df['PTAB'] = df['PTAB'].ffill().fillna(0.0)
    return df


def detect_segment_boundaries(df):
    """Detect row indices where PatientId or BodyGroup_to changes."""
    pids = df['PatientId'].fillna('__none__').values
    bgs  = df['BodyGroup_to'].fillna('UNKNOWN').values
    bounds = [0]
    for i in range(1, len(df)):
        if pids[i] != pids[i - 1] or bgs[i] != bgs[i - 1]:
            bounds.append(i)
    bounds.append(len(df))
    return bounds


def detect_day_boundaries(df):
    """Return list of (start_row, end_row) per calendar day."""
    dates = df['date'].values
    days  = []
    start = 0
    for i in range(1, len(df)):
        if dates[i] != dates[i - 1]:
            days.append((start, i))
            start = i
    days.append((start, len(df)))
    return days


def extract_exchange_sequences(df_scanner):
    """
    Extract exchange sequences from one scanner's sorted+merged DataFrame.

    PatientId and body-group data come from examination_workflow via
    merge_asof (backward) — so each event is assigned to the most recently
    started examination.  The exchange events that precede an examination
    therefore carry the PREVIOUS patient's PatientId; body_to is resolved
    by looking ahead to the next segment.
    """
    df = df_scanner.reset_index(drop=True)
    sequences = []

    df['sourceID_token'] = df['MessageIdentification'].map(
        lambda x: SOURCEID_VOCAB.get(x, SOURCEID_VOCAB['UNK'])
    )
    df = add_ptab(df)
    df = extract_temporal_features(df)

    # Cumulative timediff from first event of this scanner (seconds)
    t0 = df['AdjustedEventDateTime'].min()
    df['timediff'] = (df['AdjustedEventDateTime'] - t0).dt.total_seconds()

    day_bounds     = detect_day_boundaries(df)
    day_start_rows = {s for s, _ in day_bounds}
    exam_rows      = set(df.index[df['MessageIdentification'] == 'MRI_EXU_95'].tolist())
    bounds         = detect_segment_boundaries(df)

    first_of_day_segs = set()
    for day_start_row, _ in day_bounds:
        for seg_i in range(len(bounds) - 1):
            if bounds[seg_i] <= day_start_row < bounds[seg_i + 1]:
                first_of_day_segs.add(seg_i)
                break

    # -------------------------------------------------------------------------
    # 1. Between-patient (startup + between) exchange sequences
    # -------------------------------------------------------------------------
    for seg_i in range(len(bounds) - 1):
        seg_start = bounds[seg_i]
        seg_end   = bounds[seg_i + 1]
        segment   = df.iloc[seg_start:seg_end]

        if len(segment) == 0:
            continue

        # Exchange part: events BEFORE the first MRI_EXU_95 in this segment
        exam_in_seg = sorted(r for r in exam_rows if seg_start <= r < seg_end)
        if exam_in_seg:
            first_exam = exam_in_seg[0]
            exch_seg = (segment.loc[:first_exam - 1]
                        if first_exam > seg_start else pd.DataFrame())
        else:
            exch_seg = segment

        if len(exch_seg) < 2:
            continue

        sequence  = exch_seg['sourceID_token'].tolist()
        timediffs = exch_seg['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [max(0.0, d) for d in durations]

        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0
        if total_duration > MAX_EXCHANGE_DURATION:
            continue

        # body_from = current segment's body group (= previous patient via merge_asof)
        body_from_str = str(segment.iloc[0].get('BodyGroup_to', 'UNKNOWN')).upper()
        body_from = int(BODY_REGION_TO_ID.get(body_from_str, 10))

        # body_to = NEXT segment's body group (= incoming patient)
        if seg_i + 1 < len(bounds) - 1:
            next_seg_start = bounds[seg_i + 1]
            next_seg_end   = bounds[seg_i + 2]
            next_row       = df.iloc[next_seg_start]
            body_to_str    = str(next_row.get('BodyGroup_to', 'UNKNOWN')).upper()
            body_to        = int(BODY_REGION_TO_ID.get(body_to_str, 10))
        else:
            body_to = 10  # UNKNOWN (last segment)

        # Phase type
        if seg_i in first_of_day_segs or body_from == START_REGION_ID:
            phase_type = PHASE_TYPES['startup']
            body_from  = START_REGION_ID
        else:
            phase_type = PHASE_TYPES['between']

        # Conditioning: use the NEXT (incoming) patient's demographics to
        # match local preprocessing behaviour where possible.
        if seg_i + 1 < len(bounds) - 1:
            cond_row = df.iloc[bounds[seg_i + 1]]
        else:
            cond_row = segment.iloc[0]
        conditioning = build_conditioning(cond_row)

        start_dt = exch_seg.iloc[0]['AdjustedEventDateTime']
        if hasattr(start_dt, 'to_pydatetime'):
            start_dt = start_dt.to_pydatetime()

        sequences.append({
            'sequence':       sequence,
            'durations':      durations,
            'conditioning':   conditioning,
            'body_from':      body_from,
            'body_to':        body_to,
            'phase_type':     phase_type,
            'total_duration': total_duration,
            'start_datetime': start_dt,
        })

    # -------------------------------------------------------------------------
    # 2. Shutdown sequences (events after last examination of each day)
    # -------------------------------------------------------------------------
    for day_start, day_end in day_bounds:
        day_exam_rows = sorted(r for r in exam_rows if day_start <= r < day_end)
        if not day_exam_rows:
            continue

        last_exam_row = max(day_exam_rows)

        # Last segment boundary after the last exam within the day
        # [-1] is intentional: capture only the final post-exam segment
        post_exam_bounds = [b for b in bounds if last_exam_row < b <= day_end]
        if post_exam_bounds:
            shutdown_start = post_exam_bounds[-1]
        else:
            shutdown_start = last_exam_row + 1

        if shutdown_start >= day_end:
            continue

        shut_seg = df.iloc[shutdown_start:day_end].copy()
        shut_seg = shut_seg[shut_seg['MessageIdentification'] != 'MRI_EXU_95']

        if len(shut_seg) < 2:
            continue

        sequence  = shut_seg['sourceID_token'].tolist()
        timediffs = shut_seg['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [max(0.0, d) for d in durations]

        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0
        if total_duration > MAX_EXCHANGE_DURATION:
            continue

        body_from = int(BODY_REGION_TO_ID.get(
            str(shut_seg.iloc[0].get('BodyGroup_to', 'UNKNOWN')).upper(), 10
        ))

        start_dt = shut_seg.iloc[0]['AdjustedEventDateTime']
        if hasattr(start_dt, 'to_pydatetime'):
            start_dt = start_dt.to_pydatetime()

        sequences.append({
            'sequence':       sequence,
            'durations':      durations,
            'conditioning':   build_conditioning(shut_seg.iloc[0]),
            'body_from':      body_from,
            'body_to':        END_REGION_ID,
            'phase_type':     PHASE_TYPES['shutdown'],
            'total_duration': total_duration,
            'start_datetime': start_dt,
        })

    return sequences

# COMMAND ----------
# =============================================================================
# CELL: Load body group mapping from Excel
# =============================================================================

df_body_excel = pd.read_excel(BODY_GROUP_MAPPING_PATH)
print(f"Body mapping Excel columns: {df_body_excel.columns.tolist()}")
print(f"Rows: {len(df_body_excel)}")

# Columns confirmed as: ['BodyPart', 'BodyGroup', 'MRType']
if 'BodyPart' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _col_part, _col_group = 'BodyPart', 'BodyGroup'
elif 'BodyPartExamined' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _col_part, _col_group = 'BodyPartExamined', 'BodyGroup'
else:
    _col_part, _col_group = df_body_excel.columns[0], df_body_excel.columns[1]

body_part_to_group = {
    str(k).strip().upper(): str(v).strip().upper()
    for k, v in zip(df_body_excel[_col_part], df_body_excel[_col_group])
    if pd.notna(k) and pd.notna(v)
}
print(f"Loaded {len(body_part_to_group)} body-part → group mappings")
print(f"Sample mappings: {dict(list(body_part_to_group.items())[:5])}")

# COMMAND ----------
# =============================================================================
# CELL: Query eventlog from Spark
# No PatientId column exists in this table — it lives in examination_workflow.
# =============================================================================

df_eventlog_spark = (
    spark.table(EVENTLOG_TABLE)
    .filter(F.col("SerialNumber").isin(TARGET_SERIAL_NUMBERS))
    .filter(F.col("EventDateTime") >= DATE_START)
    .filter(F.col("EventDateTime") <= DATE_END)
    .filter(F.col("MessageIdentification").isin(REAL_EVENT_TYPES))
    .select(
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
        F.col("EventDateTime"),
        # Local time = EventDateTime + TimeZoneOffset seconds
        F.to_timestamp(
            F.unix_timestamp("EventDateTime") +
            F.coalesce(F.col("TimeZoneOffset"), F.lit(TIMEZONE_OFFSET_HOURS * 3600))
        ).alias("AdjustedEventDateTime"),
        F.col("MessageIdentification"),
        F.col("Message").cast("string").alias("Message"),
        F.col("Line").cast("long").alias("Line"),
    )
)

print(f"Eventlog row count: {df_eventlog_spark.count():,}")
df_eventlog_pd = df_eventlog_spark.toPandas()
print("Converted to pandas.")

# COMMAND ----------
# =============================================================================
# CELL: Query examination_workflow — one row per examination per scanner
#
# PatientId, Age, Weight, Height, Direction, BodyPart are all in WorkflowValues.
# WorkflowStartRefDateTime is the patient arrival / workflow start time.
# We group by (SerialNumber, WorkflowKey) to get one row per examination.
# =============================================================================

df_exams_spark = (
    spark.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin(TARGET_SERIAL_NUMBERS))
    .filter(F.col("WorkflowStartRefDateTime") >= DATE_START)
    .filter(F.col("WorkflowStartRefDateTime") <= DATE_END)
    .filter(F.col("WorkflowValues").isNotNull())
    .groupBy(
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
        "WorkflowKey",
        "WorkflowStartRefDateTime",
    )
    .agg(
        F.first(F.col("WorkflowValues")["PatientId"]).alias("PatientId"),
        F.first(F.col("WorkflowValues")["Age"]).alias("Age"),
        F.first(F.col("WorkflowValues")["Weight"]).alias("Weight"),
        F.first(F.col("WorkflowValues")["Height"]).alias("Height"),
        F.first(F.col("WorkflowValues")["Direction"]).alias("Direction"),
        # Body part: try common key names; adjust if your data uses a different key
        F.first(F.coalesce(
            F.col("WorkflowValues")["BodyPartExamined"],
            F.col("WorkflowValues")["BodyPart"],
            F.col("WorkflowValues")["RequestedBodyPart"],
        )).alias("BodyPartExamined"),
    )
    .orderBy("SerialNumber", "WorkflowStartRefDateTime")
)

df_exams_pd = df_exams_spark.toPandas()

# Parse numeric demographics
for col in ['Age', 'Weight', 'Height']:
    df_exams_pd[col] = pd.to_numeric(df_exams_pd[col], errors='coerce').fillna(0.0)

df_exams_pd['Direction_encoded'] = df_exams_pd['Direction'].apply(
    lambda x: 0 if str(x).strip().lower() == 'head first'
              else (1 if str(x).strip().lower() == 'feet first' else -1)
)

df_exams_pd['BodyGroup'] = df_exams_pd['BodyPartExamined'].apply(
    lambda x: body_part_to_group.get(str(x).strip().upper(), 'UNKNOWN')
              if pd.notna(x) else 'UNKNOWN'
)

print(f"Examination rows: {len(df_exams_pd):,}")
print(f"PatientId sample: {df_exams_pd['PatientId'].dropna().iloc[0] if len(df_exams_pd) > 0 else 'N/A'}")
print(f"BodyGroup distribution:\n{df_exams_pd['BodyGroup'].value_counts().head(10)}")
print(f"BodyPartExamined nulls: {df_exams_pd['BodyPartExamined'].isna().sum()} / {len(df_exams_pd)}")

# COMMAND ----------
# =============================================================================
# CELL: Per-scanner extraction loop
#
# For each scanner:
# 1. Sort events and examinations by time
# 2. pd.merge_asof (backward) assigns each event the data of the most recently
#    started examination — effectively labelling which patient each event
#    belongs to
# 3. Detect patient transitions and extract exchange sequences
# =============================================================================

exchange_sequences = []

for serial_number in TARGET_SERIAL_NUMBERS:
    print(f"\n--- Scanner {serial_number} ---")

    df_sc = df_eventlog_pd[df_eventlog_pd['SerialNumber'] == serial_number].copy()
    if len(df_sc) == 0:
        print("  No events found.")
        continue

    df_ex = df_exams_pd[df_exams_pd['SerialNumber'] == serial_number].copy()
    if len(df_ex) == 0:
        print("  No examinations found — skipping.")
        continue

    # Sort both by time
    df_sc = df_sc.sort_values(['AdjustedEventDateTime', 'Line']).reset_index(drop=True)
    df_ex = df_ex.sort_values('WorkflowStartRefDateTime').reset_index(drop=True)

    # merge_asof (backward): each event gets data from the most recent exam start
    df_merged = pd.merge_asof(
        df_sc,
        df_ex[['WorkflowStartRefDateTime', 'PatientId',
               'Age', 'Weight', 'Height', 'Direction_encoded', 'BodyGroup']],
        left_on='AdjustedEventDateTime',
        right_on='WorkflowStartRefDateTime',
        direction='backward',
    )

    df_merged['PatientId']  = df_merged['PatientId'].fillna('__no_patient__')
    df_merged['BodyGroup_to'] = df_merged['BodyGroup'].fillna('UNKNOWN').str.upper()
    df_merged['Age']               = df_merged['Age'].fillna(0.0)
    df_merged['Weight']            = df_merged['Weight'].fillna(0.0)
    df_merged['Height']            = df_merged['Height'].fillna(0.0)
    df_merged['Direction_encoded'] = df_merged['Direction_encoded'].fillna(0.0)

    seqs = extract_exchange_sequences(df_merged)
    exchange_sequences.extend(seqs)

    phase_counts = {}
    for s in seqs:
        phase_counts[s['phase_type']] = phase_counts.get(s['phase_type'], 0) + 1
    print(f"  {len(df_sc):,} events, {len(df_ex):,} exams → {len(seqs)} exchange sequences")
    print(f"  startup={phase_counts.get(0,0)}, between={phase_counts.get(1,0)}, "
          f"shutdown={phase_counts.get(2,0)}")

print(f"\nTotal exchange sequences: {len(exchange_sequences)}")

# COMMAND ----------
# =============================================================================
# CELL: Assertions
# =============================================================================

print("Running assertions...")

for i, seq in enumerate(exchange_sequences):
    assert len(seq['sequence']) == len(seq['durations']), \
        f"Seq {i}: length mismatch"
    assert seq['phase_type'] in {0, 1, 2}, \
        f"Seq {i}: invalid phase_type {seq['phase_type']}"
    assert all(d >= 0 for d in seq['durations']), \
        f"Seq {i}: negative duration"
    assert 0 <= seq['body_from'] <= END_REGION_ID, \
        f"Seq {i}: body_from {seq['body_from']} out of range"
    assert 0 <= seq['body_to'] <= END_REGION_ID, \
        f"Seq {i}: body_to {seq['body_to']} out of range"
    cond = seq['conditioning']
    for key in ['Age', 'Weight', 'Height', 'PTAB', 'Direction_encoded',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_morning']:
        assert key in cond, f"Seq {i}: missing conditioning key '{key}'"
    assert hasattr(seq['start_datetime'], 'date'), \
        f"Seq {i}: start_datetime not a datetime"

print(f"All assertions passed for {len(exchange_sequences)} sequences.")

# COMMAND ----------
# =============================================================================
# CELL: Save
# =============================================================================

with open(EXCHANGE_OUTPUT, 'wb') as f:
    pickle.dump(exchange_sequences, f)

print(f"Saved {len(exchange_sequences)} exchange sequences → {EXCHANGE_OUTPUT}")
