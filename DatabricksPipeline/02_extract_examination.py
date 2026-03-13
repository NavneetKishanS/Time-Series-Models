# Databricks notebook — Examination Sequence Extraction
#
# Extracts per-measurement examination event sequences.
# PatientId / demographics come from examination_workflow (not eventlog).
# Join strategy: pd.merge_asof(backward) on WorkflowStartRefDateTime.
#
# Output: examination_sequences.pkl

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
# HELPER FUNCTIONS
# =============================================================================

def extract_temporal_features(df, dt_col='AdjustedEventDateTime'):
    dt = df[dt_col]
    df = df.copy()
    df['hour_of_day'] = dt.dt.hour
    df['day_of_week']  = dt.dt.dayofweek
    df['is_morning']   = (dt.dt.hour < 12).astype(int)
    df['hour_sin']     = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos']     = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin']      = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos']      = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['date']         = dt.dt.date
    return df


def build_conditioning(row):
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


# -----------------------------------------------------------------------------
# Coil parsing
# -----------------------------------------------------------------------------

_COIL_ABBREV_MAP = {
    'HC1': 'HE1', 'HC2': 'HE2', 'HC3': 'HE3', 'HC4': 'HE4',
    'NC1': 'NE1', 'NC2': 'NE2',
    'BC': 'BC', 'SHL': 'SHL', 'FA': 'FA', 'TO': 'TO', 'FS': 'FS',
    '15K': '15K', 'SN': 'SN',
    'SP1': 'SP1', 'SP2': 'SP2', 'SP3': 'SP3', 'SP4': 'SP4',
    'SP5': 'SP5', 'SP6': 'SP6', 'SP7': 'SP7', 'SP8': 'SP8',
    'HW1': 'HW1', 'HW2': 'HW2', 'HW3': 'HW3',
    'HE1': 'HE1', 'HE2': 'HE2', 'HE3': 'HE3', 'HE4': 'HE4',
    'NE1': 'NE1', 'NE2': 'NE2',
    'BO1': 'BO1', 'BO2': 'BO2', 'BO3': 'BO3',
    'PA1': 'PA1', 'PA2': 'PA2', 'PA3': 'PA3',
    'PA4': 'PA4', 'PA5': 'PA5', 'PA6': 'PA6',
}


def _expand_coil_list(coil_str):
    active = []
    current_prefix = ''
    for part in coil_str.split(','):
        part = part.strip().upper()
        if not part:
            continue
        m_range = re.match(r'^([A-Z]+)(\d+)-(\d+)$', part)
        if m_range:
            pfx = m_range.group(1)
            current_prefix = pfx
            for n in range(int(m_range.group(2)), int(m_range.group(3)) + 1):
                active.append(f"{pfx}{n}")
            continue
        m_single = re.match(r'^([A-Z]+)(\d+)$', part)
        if m_single:
            current_prefix = m_single.group(1)
            active.append(f"{m_single.group(1)}{m_single.group(2)}")
            continue
        m_num = re.match(r'^(\d+)$', part)
        if m_num and current_prefix:
            active.append(f"{current_prefix}{m_num.group(1)}")
            continue
        active.append(part)
    return active


def parse_coil_message(msg):
    coil_config = {col: 0 for col in COIL_COLUMNS}
    if not isinstance(msg, str) or not msg.strip():
        return coil_config
    m = re.search(r'[Cc]onnected\s+coil\s+elements?:?\s*(.*?)(?:\s*\||\s*$)', msg)
    coil_str = m.group(1).strip() if m else msg
    for name in _expand_coil_list(coil_str):
        mapped = _COIL_ABBREV_MAP.get(name)
        if mapped and mapped in coil_config:
            coil_config[mapped] = 1
    return coil_config


def extract_examination_sequences(df_scanner):
    """
    Extract per-measurement examination sequences.
    PatientId and BodyGroup_to are pre-assigned via merge_asof.
    """
    df = df_scanner.reset_index(drop=True)
    sequences = []

    df['sourceID_token'] = df['MessageIdentification'].map(
        lambda x: SOURCEID_VOCAB.get(x, SOURCEID_VOCAB['UNK'])
    )
    df = add_ptab(df)
    df = extract_temporal_features(df)
    t0 = df['AdjustedEventDateTime'].min()
    df['timediff'] = (df['AdjustedEventDateTime'] - t0).dt.total_seconds()

    start_rows = df.index[df['MessageIdentification'] == 'MRI_MSR_100'].tolist()
    end_rows   = sorted(
        df.index[df['MessageIdentification'].isin(['MRI_MSR_104', 'MRI_MSR_34'])].tolist()
    )
    coil_rows  = df.index[df['MessageIdentification'] == 'MRI_CCS_11'].tolist()
    exam_rows  = df.index[df['MessageIdentification'] == 'MRI_EXU_95'].tolist()

    if not start_rows:
        return sequences

    for idx_in_list, start_row in enumerate(start_rows):
        # Find end boundary
        end_candidates = [r for r in end_rows if r > start_row]
        if end_candidates:
            end_row = end_candidates[0]
        elif idx_in_list + 1 < len(start_rows):
            end_row = start_rows[idx_in_list + 1] - 1
        else:
            end_row = len(df) - 1

        segment = df.iloc[start_row: end_row + 1]
        if len(segment) < 2:
            continue

        sequence  = segment['sourceID_token'].tolist()
        timediffs = segment['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [max(0.0, d) for d in durations]
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        # Body region: BodyGroup_to assigned via merge_asof
        body_region_str = str(segment.iloc[0].get('BodyGroup_to', 'UNKNOWN')).upper()
        # If merge_asof gave us the exchange segment's body region, try the
        # nearest MRI_EXU_95 before this start for a more accurate body region
        prior_exams = [r for r in exam_rows if r < start_row]
        if prior_exams:
            last_exam_row = max(prior_exams)
            bg = str(df.iloc[last_exam_row].get('BodyGroup_to', 'UNKNOWN')).upper()
            if bg != 'UNKNOWN':
                body_region_str = bg
        body_region = int(BODY_REGION_TO_ID.get(body_region_str, 10))

        # Coil config: most recent MRI_CCS_11 before this start
        prior_coils = [r for r in coil_rows if r < start_row]
        coil_config = (parse_coil_message(df.iloc[max(prior_coils)]['Message'])
                       if prior_coils else {col: 0 for col in COIL_COLUMNS})

        conditioning = build_conditioning(segment.iloc[0])

        start_dt = segment.iloc[0]['AdjustedEventDateTime']
        if hasattr(start_dt, 'to_pydatetime'):
            start_dt = start_dt.to_pydatetime()

        sequences.append({
            'sequence':       sequence,
            'durations':      durations,
            'conditioning':   conditioning,
            'body_region':    body_region,
            'coil_config':    coil_config,
            'total_duration': total_duration,
            'start_datetime': start_dt,
        })

    return sequences

# COMMAND ----------
# =============================================================================
# CELL: Load body group mapping
# =============================================================================

df_body_excel = pd.read_excel(BODY_GROUP_MAPPING_PATH)
print(f"Body mapping columns: {df_body_excel.columns.tolist()}, rows: {len(df_body_excel)}")

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

# COMMAND ----------
# =============================================================================
# CELL: Query eventlog
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
        F.to_timestamp(
            F.unix_timestamp("EventDateTime") +
            F.coalesce(F.col("TimeZoneOffset"), F.lit(TIMEZONE_OFFSET_HOURS * 3600))
        ).alias("AdjustedEventDateTime"),
        F.col("MessageIdentification"),
        F.col("Message").cast("string").alias("Message"),
        F.col("Line").cast("long").alias("Line"),
    )
)

print(f"Eventlog rows: {df_eventlog_spark.count():,}")
df_eventlog_pd = df_eventlog_spark.toPandas()
print("Converted to pandas.")

# COMMAND ----------
# =============================================================================
# CELL: Query examination_workflow
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
        F.first(F.coalesce(
            F.col("WorkflowValues")["BodyPartExamined"],
            F.col("WorkflowValues")["BodyPart"],
            F.col("WorkflowValues")["RequestedBodyPart"],
        )).alias("BodyPartExamined"),
    )
    .orderBy("SerialNumber", "WorkflowStartRefDateTime")
)

df_exams_pd = df_exams_spark.toPandas()

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
print(f"BodyGroup distribution:\n{df_exams_pd['BodyGroup'].value_counts().head(10)}")

# COMMAND ----------
# =============================================================================
# CELL: Per-scanner extraction loop
# =============================================================================

examination_sequences = []

for serial_number in TARGET_SERIAL_NUMBERS:
    print(f"\n--- Scanner {serial_number} ---")

    df_sc = df_eventlog_pd[df_eventlog_pd['SerialNumber'] == serial_number].copy()
    df_ex = df_exams_pd[df_exams_pd['SerialNumber'] == serial_number].copy()

    if len(df_sc) == 0 or len(df_ex) == 0:
        print("  No events or no examinations — skipping.")
        continue

    df_sc = df_sc.sort_values(['AdjustedEventDateTime', 'Line']).reset_index(drop=True)
    df_ex = df_ex.sort_values('WorkflowStartRefDateTime').reset_index(drop=True)

    # Assign patient data to each event via as-of merge
    df_merged = pd.merge_asof(
        df_sc,
        df_ex[['WorkflowStartRefDateTime', 'PatientId',
               'Age', 'Weight', 'Height', 'Direction_encoded', 'BodyGroup']],
        left_on='AdjustedEventDateTime',
        right_on='WorkflowStartRefDateTime',
        direction='backward',
    )

    df_merged['PatientId']         = df_merged['PatientId'].fillna('__no_patient__')
    df_merged['BodyGroup_to']      = df_merged['BodyGroup'].fillna('UNKNOWN').str.upper()
    df_merged['Age']               = df_merged['Age'].fillna(0.0)
    df_merged['Weight']            = df_merged['Weight'].fillna(0.0)
    df_merged['Height']            = df_merged['Height'].fillna(0.0)
    df_merged['Direction_encoded'] = df_merged['Direction_encoded'].fillna(0.0)

    seqs = extract_examination_sequences(df_merged)
    examination_sequences.extend(seqs)

    region_counts = {}
    for s in seqs:
        region_counts[s['body_region']] = region_counts.get(s['body_region'], 0) + 1
    region_summary = {ID_TO_BODY_REGION.get(k, str(k)): v
                      for k, v in sorted(region_counts.items())}
    print(f"  {len(df_sc):,} events, {len(df_ex):,} exams → {len(seqs)} exam sequences")
    print(f"  {region_summary}")

print(f"\nTotal examination sequences: {len(examination_sequences)}")

# COMMAND ----------
# =============================================================================
# CELL: Assertions
# =============================================================================

print("Running assertions...")
for i, seq in enumerate(examination_sequences):
    assert len(seq['sequence']) == len(seq['durations']), f"Seq {i}: length mismatch"
    assert 0 <= seq['body_region'] <= 10, f"Seq {i}: body_region out of range"
    assert set(seq['coil_config'].keys()) == set(COIL_COLUMNS), f"Seq {i}: coil_config wrong keys"
    assert all(d >= 0 for d in seq['durations']), f"Seq {i}: negative duration"
    for key in ['Age', 'Weight', 'Height', 'PTAB', 'Direction_encoded',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_morning']:
        assert key in seq['conditioning'], f"Seq {i}: missing '{key}'"
    assert hasattr(seq['start_datetime'], 'date'), f"Seq {i}: start_datetime not a datetime"

print(f"All assertions passed for {len(examination_sequences)} sequences.")

# COMMAND ----------
# =============================================================================
# CELL: Save
# =============================================================================

with open(EXAMINATION_OUTPUT, 'wb') as f:
    pickle.dump(examination_sequences, f)

print(f"Saved {len(examination_sequences)} examination sequences → {EXAMINATION_OUTPUT}")
