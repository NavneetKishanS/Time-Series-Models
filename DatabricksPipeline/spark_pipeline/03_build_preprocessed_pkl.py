# Databricks notebook source
# spark_pipeline — Build preprocessed_data.pkl (Spark-native)
#
# Same output as csv_pipeline/03_build_preprocessed_pkl.py:
# identical pkl structure (version, exchange, examination,
# daily_summaries, customer_schedules).
#
# Key differences vs. csv_pipeline/03:
#   Section 2b: eventlog stays as a Spark DataFrame (no toPandas on raw events)
#   Section 2c: examination_workflow toPandas is kept (it's small — unique
#               patients only) and then broadcast to all Spark workers
#   Section 2e: per-serial examination sequence extraction runs in parallel
#               via groupBy.applyInPandas instead of a sequential driver loop
#
# Output:  /dbfs/FileStore/spark_pipeline/preprocessed_data.pkl

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import re
import bisect
import json
import pickle
import time
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, StringType

PKL_OUTPUT = "/dbfs/FileStore/spark_pipeline/preprocessed_data.pkl"
os.makedirs("/dbfs/FileStore/spark_pipeline", exist_ok=True)

_TIMINGS = []
def _timeit(label, t0):
    dt = time.perf_counter() - t0
    _TIMINGS.append((label, dt))
    print(f"[timing] step03  {label:<25} {dt:9.1f}s")
    return time.perf_counter()

_BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                 'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
_BODY_REGION_TO_ID = {r: i for i, r in enumerate(_BODY_REGIONS)}
_START_REGION_ID   = 11
_END_REGION_ID     = 12

print(f"Serials:    {SERIAL_NUMBERS}")
print(f"Date range: {DATE_START} → {DATE_END}")
print(f"Output:     {PKL_OUTPUT}")

# COMMAND ----------

# =============================================================================
# HELPERS  (identical to csv_pipeline/03)
# =============================================================================

def _temporal_features(dt):
    h   = dt.hour
    dow = dt.dayofweek if hasattr(dt, 'dayofweek') else dt.weekday()
    return {
        'hour_of_day': int(h),
        'day_of_week': int(dow),
        'is_morning':  int(h < 12),
        'hour_sin':    float(np.sin(2 * np.pi * h / 24)),
        'hour_cos':    float(np.cos(2 * np.pi * h / 24)),
        'dow_sin':     float(np.sin(2 * np.pi * dow / 7)),
        'dow_cos':     float(np.cos(2 * np.pi * dow / 7)),
    }


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _conditioning(row, dt=None):
    if dt is None:
        dt = pd.to_datetime(row.get('datetime', pd.Timestamp.now()))
    temp = _temporal_features(dt)
    direction_raw = str(row.get('Direction', '') or '')
    dl = direction_raw.strip().lower()
    direction_enc = 0 if dl == 'head first' else (1 if dl == 'feet first' else -1)
    return {
        'Age':               _safe_float(row.get('Age', 0)),
        'Weight':            _safe_float(row.get('Weight', 0)),
        'Height':            _safe_float(row.get('Height', 0)),
        'PTAB':              _safe_float(row.get('PTAB', 0)),
        'Direction_encoded': float(direction_enc),
        **temp,
    }


def _seq_type_from_msg(msg):
    if not isinstance(msg, str):
        return SEQUENCE_TYPE_VOCAB['other']
    m = re.search(r"Sequence:\s*'([^']*)'", msg)
    return classify_sequence_type(m.group(1) if m else '')

# COMMAND ----------

# =============================================================================
# SECTION 1 — Exchange sequences from CSV  (unchanged from csv_pipeline/03)
# =============================================================================

print("\n" + "="*60)
print("SECTION 1: Exchange sequences from CSVs")
print("="*60)

_t = time.perf_counter()
exchange_sequences = []

for serial in SERIAL_NUMBERS:
    csv_path = f"{EXCHANGE_OUTPUT_DIR}/DATA_{serial}.csv"
    if not os.path.exists(csv_path):
        print(f"  {serial}: file not found — skipping")
        continue

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    name_col = 'token_name' if 'token_name' in df.columns else 'sourceID'
    df = df[df[name_col].notna() & (df[name_col].astype(str).str.strip() != '')].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        print(f"  {serial}: no event rows after filtering")
        continue

    if 'Direction' in df.columns:
        df['Direction_encoded'] = df['Direction'].apply(
            lambda x: 0 if str(x).strip().lower() == 'head first'
                      else (1 if str(x).strip().lower() == 'feet first' else -1)
        )
    else:
        df['Direction_encoded'] = 0

    def _to_region_id(series):
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            return numeric.astype(int)
        return series.apply(
            lambda x: _BODY_REGION_TO_ID.get(str(x).strip().upper(), 10) if pd.notna(x) else 10
        )
    df['body_from_id'] = _to_region_id(df['BodyGroup_from'])
    df['body_to_id']   = _to_region_id(df['BodyGroup_to'])

    if 'token_id' in df.columns:
        df['sourceID_token'] = pd.to_numeric(df['token_id'], errors='coerce').fillna(SOURCEID_VOCAB['UNK']).astype(int)
    else:
        df['sourceID_token'] = df['sourceID'].apply(
            lambda x: SOURCEID_VOCAB.get(str(x), SOURCEID_VOCAB['UNK'])
        )

    pid_col = 'PatientID_to' if 'PatientID_to' in df.columns else 'BodyGroup_to'
    df['block_id'] = (df[pid_col].astype(str) != df[pid_col].astype(str).shift()).cumsum()

    n_before = len(exchange_sequences)

    for _, block in df.groupby('block_id', sort=False):
        block = block[block['sourceID_token'].notna()].copy()
        if len(block) < 2:
            continue

        sequence  = block['sourceID_token'].astype(int).tolist()
        timediffs = pd.to_numeric(block['timediff'], errors='coerce').fillna(0).values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [min(MAX_PER_TOKEN_DURATION, max(0.0, d)) for d in durations]

        row = block.iloc[0]
        dt  = row['datetime']
        conditioning = _conditioning(row, dt)

        body_from      = int(row.get('body_from_id', 10))
        body_to        = int(row.get('body_to_id',   10))
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        if total_duration > MAX_EXCHANGE_DURATION:
            continue

        block_date   = dt.date() if pd.notna(dt) else None
        first_of_day = (block.iloc[0].name == df[df['datetime'].dt.date == block_date].index[0]
                        if block_date is not None else False)
        phase_type = PHASE_TYPES['startup'] if first_of_day else PHASE_TYPES['between']
        if first_of_day:
            body_from = _START_REGION_ID

        start_dt = dt.to_pydatetime() if pd.notna(dt) else datetime(2024, 1, 1)

        exchange_sequences.append({
            'sequence':       sequence,
            'durations':      durations,
            'conditioning':   conditioning,
            'body_from':      body_from,
            'body_to':        body_to,
            'phase_type':     phase_type,
            'total_duration': total_duration,
            'start_datetime': start_dt,
        })

    print(f"  {serial}: {len(exchange_sequences) - n_before} exchange sequences")

print(f"\nTotal exchange sequences: {len(exchange_sequences)}")
_t = _timeit('section1 exchange', _t)

# COMMAND ----------

# =============================================================================
# SECTION 2 — Examination sequences  (Spark-native parallel extraction)
# =============================================================================
# KEY CHANGE vs. csv_pipeline/03:
#   - eventlog stays as a Spark DataFrame (no toPandas on raw events)
#   - exam workflow is toPandas'd once (small: unique patients only) then
#     broadcast to all Spark workers
#   - the per-serial bisect extraction loop runs in parallel via applyInPandas
# =============================================================================

print("\n" + "="*60)
print("SECTION 2: Examination sequences from Spark")
print("="*60)

# ---- 2a. Body group mapping ----
df_body_excel = pd.read_excel(BODY_GROUP_MAPPING_PATH)
if 'BodyPart' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _cp, _cg = 'BodyPart', 'BodyGroup'
elif 'BodyPartExamined' in df_body_excel.columns:
    _cp, _cg = 'BodyPartExamined', 'BodyGroup'
else:
    _cp, _cg = df_body_excel.columns[0], df_body_excel.columns[1]

body_part_to_group = {
    str(k).strip().upper(): str(v).strip().upper()
    for k, v in zip(df_body_excel[_cp], df_body_excel[_cg])
    if pd.notna(k) and pd.notna(v)
}
print(f"Body mapping: {len(body_part_to_group)} entries")

# ---- 2b. Eventlog stays as Spark (no toPandas on raw events) ----
_t = time.perf_counter()

df_eventlog_spark = (
    spark.table(EVENTLOG_TABLE)
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
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
    .repartition(len(SERIAL_NUMBERS), F.col("SerialNumber"))
)
print(f"Eventlog rows (Spark): {df_eventlog_spark.count():,}")
_t = _timeit('eventlog Spark query', _t)

# ---- 2c. Exam workflow — small (unique patients), toPandas then broadcast ----
_t = time.perf_counter()

df_exams_spark = (
    spark.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
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
df_exams_pd = df_exams_spark.toPandas()   # small: unique patients only
_t = _timeit('exam workflow toPandas (small)', _t)

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

_unmapped = (
    df_exams_pd.loc[df_exams_pd['BodyGroup'] == 'UNKNOWN', 'BodyPartExamined']
    .dropna().astype(str).str.strip().str.upper()
)
_unmapped = _unmapped[_unmapped != ''].value_counts()
if len(_unmapped):
    print(f"  ⚠ {len(_unmapped)} distinct BodyPart values are UNMAPPED → UNKNOWN "
          f"({int(_unmapped.sum()):,} rows)")

# Broadcast exam data and all config constants to Spark workers
_exams_bc        = spark.sparkContext.broadcast(df_exams_pd.to_dict('records'))
_body_to_group_bc = spark.sparkContext.broadcast(body_part_to_group)
_sourceid_vocab_bc = spark.sparkContext.broadcast(SOURCEID_VOCAB)
_serial_numbers_bc = spark.sparkContext.broadcast(SERIAL_NUMBERS)
_coil_columns_bc   = spark.sparkContext.broadcast(COIL_COLUMNS)
_body_region_to_id_bc = spark.sparkContext.broadcast(_BODY_REGION_TO_ID)

# ---- 2d. Coil parsing helpers (same as csv_pipeline/03) ----

_COIL_ABBREV = {
    'HC1': 'HE1', 'HC2': 'HE2', 'HC3': 'HE3', 'HC4': 'HE4',
    'NC1': 'NE1', 'NC2': 'NE2',
    'BC': 'BC',   'SHL': 'SHL', 'FA': 'FA',  'TO': 'TO',
    'FS': 'FS',   '15K': '15K', 'SN': 'SN',
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

# ---- 2e. applyInPandas schema (one row = one JSON-encoded sequence dict) ----

_SEQ_SCHEMA = StructType([
    StructField('SerialNumber',   LongType(),   True),
    StructField('sequence_json',  StringType(), True),
])

# ---- 2f. Parallel per-serial examination sequence extraction ----

def extract_exam_sequences(pdf):
    """
    Process one SerialNumber group in parallel on a Spark worker.

    Performs merge_asof against broadcast exam data, adds PTAB, then uses
    bisect-based segment extraction (identical logic to csv_pipeline/03 Section 2e).
    Returns one row per valid examination sequence, with sequence data JSON-encoded
    (avoids needing a variable-length array type in the Spark schema).
    """
    import bisect, re, json
    import numpy as np
    import pandas as pd
    from datetime import datetime

    _empty = pd.DataFrame(columns=['SerialNumber', 'sequence_json'])

    if pdf.empty:
        return _empty

    serial = int(pdf['SerialNumber'].iloc[0])

    exams_all        = pd.DataFrame(_exams_bc.value)
    body_to_group    = _body_to_group_bc.value
    sourceid_vocab   = _sourceid_vocab_bc.value
    serial_numbers   = _serial_numbers_bc.value
    coil_columns     = _coil_columns_bc.value
    body_region_to_id = _body_region_to_id_bc.value

    serial_idx = serial_numbers.index(serial) if serial in serial_numbers else 0

    # --- Coil helpers ---
    _coil_abbrev = {
        'HC1': 'HE1', 'HC2': 'HE2', 'HC3': 'HE3', 'HC4': 'HE4',
        'NC1': 'NE1', 'NC2': 'NE2', 'BC': 'BC', 'SHL': 'SHL', 'FA': 'FA',
        'TO': 'TO', 'FS': 'FS', '15K': '15K', 'SN': 'SN',
        'SP1': 'SP1', 'SP2': 'SP2', 'SP3': 'SP3', 'SP4': 'SP4',
        'SP5': 'SP5', 'SP6': 'SP6', 'SP7': 'SP7', 'SP8': 'SP8',
        'HW1': 'HW1', 'HW2': 'HW2', 'HW3': 'HW3',
        'HE1': 'HE1', 'HE2': 'HE2', 'HE3': 'HE3', 'HE4': 'HE4',
        'NE1': 'NE1', 'NE2': 'NE2', 'BO1': 'BO1', 'BO2': 'BO2', 'BO3': 'BO3',
        'PA1': 'PA1', 'PA2': 'PA2', 'PA3': 'PA3',
        'PA4': 'PA4', 'PA5': 'PA5', 'PA6': 'PA6',
    }

    def _expand_coil_list_local(coil_str):
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

    def _parse_coil_message_local(msg, coil_columns):
        coil_config = {col: 0 for col in coil_columns}
        if not isinstance(msg, str) or not msg.strip():
            return coil_config
        m = re.search(r'[Cc]onnected\s+coil\s+elements?:?\s*(.*?)(?:\s*\||\s*$)', msg)
        coil_str = m.group(1).strip() if m else msg
        for name in _expand_coil_list_local(coil_str):
            mapped = _coil_abbrev.get(name)
            if mapped and mapped in coil_config:
                coil_config[mapped] = 1
        return coil_config

    def _add_ptab_local(df):
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

    def _temporal_features_local(dt):
        h   = dt.hour
        dow = dt.dayofweek if hasattr(dt, 'dayofweek') else dt.weekday()
        return {
            'hour_of_day': int(h),
            'day_of_week': int(dow),
            'is_morning':  int(h < 12),
            'hour_sin':    float(np.sin(2 * np.pi * h / 24)),
            'hour_cos':    float(np.cos(2 * np.pi * h / 24)),
            'dow_sin':     float(np.sin(2 * np.pi * dow / 7)),
            'dow_cos':     float(np.cos(2 * np.pi * dow / 7)),
        }

    def _safe_float_local(val, default=0.0):
        try:
            v = float(val)
            return default if (v != v or np.isinf(v)) else v
        except (TypeError, ValueError):
            return default

    def _seq_type_from_msg_local(msg, sequence_type_vocab, classify_fn):
        if not isinstance(msg, str):
            return sequence_type_vocab.get('other', 0)
        m = re.search(r"Sequence:\s*'([^']*)'", msg)
        return classify_fn(m.group(1) if m else '')

    # These keys mirror classify_sequence_type from config
    _SEQ_KEYS = ['localizer','scout','haste','space','tirm','vibe','dixon',
                 'medic','swi','tfl','flash','tse','gre']
    _SEQ_VOCAB = {
        'other':0,'scout':1,'localizer':2,'tse':3,'space':4,
        'haste':5,'gre':6,'flash':7,'epi':8,'tfl':9,'tirm':10,
        'vibe':11,'dixon':12,'swi':13,'medic':14,
    }

    def _classify_seq(raw):
        s = str(raw or '').lower()
        if not s: return _SEQ_VOCAB['other']
        for key in _SEQ_KEYS:
            if key in s: return _SEQ_VOCAB[key]
        if any(k in s for k in ('ep2d','epi','bold','diff','dwi')): return _SEQ_VOCAB['epi']
        return _SEQ_VOCAB['other']

    # Prepare serial-specific exam data
    df_sc = pdf.sort_values(['AdjustedEventDateTime', 'Line']).reset_index(drop=True)
    df_ex = exams_all[exams_all['SerialNumber'] == serial].copy()
    df_ex['WorkflowStartRefDateTime'] = pd.to_datetime(df_ex['WorkflowStartRefDateTime'])
    df_ex = df_ex.sort_values('WorkflowStartRefDateTime').reset_index(drop=True)

    if df_sc.empty or df_ex.empty:
        return _empty

    # merge_asof: assign exam workflow context to each event
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

    # Fallback: where merge_asof left BodyGroup_to UNKNOWN, extract body part
    # directly from MRI_EXU_95 messages (same regex as step 02 get_body_patient).
    # Covers the case where WorkflowValues['BodyPartExamined'] is absent or uses
    # naming conventions not in bodyupdated.xlsx.
    _mask_evu = df_merged['MessageIdentification'] == 'MRI_EXU_95'
    def _bg_from_evu95_local(msg):
        if not isinstance(msg, str):
            return None
        try:
            body = re.search(r'with body part < (.*) >', msg).group(1)
            if ' ' in body:
                try:
                    body = re.search(r'with body part < (.*) > <', msg).group(1)
                except AttributeError:
                    pass
            return body_to_group.get(str(body).strip().upper())
        except AttributeError:
            return None
    df_merged['_bg_msg'] = np.nan
    df_merged.loc[_mask_evu, '_bg_msg'] = df_merged.loc[_mask_evu, 'Message'].apply(_bg_from_evu95_local)
    df_merged['_bg_msg'] = df_merged['_bg_msg'].ffill()
    _needs_bg = df_merged['BodyGroup_to'] == 'UNKNOWN'
    df_merged.loc[_needs_bg, 'BodyGroup_to'] = df_merged.loc[_needs_bg, '_bg_msg'].fillna('UNKNOWN')
    df_merged.drop(columns=['_bg_msg'], inplace=True)

    df_merged['Age']               = df_merged['Age'].fillna(0.0)
    df_merged['Weight']            = df_merged['Weight'].fillna(0.0)
    df_merged['Height']            = df_merged['Height'].fillna(0.0)
    df_merged['Direction_encoded'] = df_merged['Direction_encoded'].fillna(0.0)

    df_merged = _add_ptab_local(df_merged)

    t0 = df_merged['AdjustedEventDateTime'].min()
    df_merged['timediff'] = (df_merged['AdjustedEventDateTime'] - t0).dt.total_seconds()

    df_merged['sourceID_token'] = df_merged['MessageIdentification'].map(
        lambda x: sourceid_vocab.get(x, sourceid_vocab.get('UNK', 17))
    )

    start_rows = df_merged.index[df_merged['MessageIdentification'] == 'MRI_MSR_100'].tolist()
    end_rows   = sorted(df_merged.index[
        df_merged['MessageIdentification'].isin(['MRI_MSR_104', 'MRI_MSR_34'])
    ].tolist())
    coil_rows  = df_merged.index[df_merged['MessageIdentification'] == 'MRI_CCS_11'].tolist()
    exam_rows  = df_merged.index[df_merged['MessageIdentification'] == 'MRI_EXU_95'].tolist()

    _MAX_EXAM    = 3000   # MAX_EXAMINATION_DURATION
    _MIN_EXAM    = 10     # MIN_EXAMINATION_DURATION
    _MAX_TOKEN   = 600    # MAX_PER_TOKEN_DURATION

    sequences = []

    for idx_in_list, start_row in enumerate(start_rows):
        _ec = bisect.bisect_right(end_rows, start_row)
        if _ec < len(end_rows):
            end_row = end_rows[_ec]
        elif idx_in_list + 1 < len(start_rows):
            end_row = start_rows[idx_in_list + 1] - 1
        else:
            end_row = len(df_merged) - 1

        segment = df_merged.iloc[start_row: end_row + 1]
        if len(segment) < 2:
            continue

        seq_tokens = segment['sourceID_token'].tolist()
        timediffs  = segment['timediff'].values.astype(float)
        durations  = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [min(_MAX_TOKEN, max(0.0, d)) for d in durations]
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        is_abort = (str(df_merged.iloc[end_row].get('MessageIdentification', '')) == 'MRI_MSR_34')

        if total_duration > _MAX_EXAM:
            continue
        if total_duration < _MIN_EXAM and not is_abort:
            continue

        seq_type = _classify_seq(segment.iloc[0].get('Message'))

        body_region_str = str(segment.iloc[0].get('BodyGroup_to', 'UNKNOWN')).upper()
        _pe = bisect.bisect_left(exam_rows, start_row) - 1
        if _pe >= 0:
            bg = str(df_merged.iloc[exam_rows[_pe]].get('BodyGroup_to', 'UNKNOWN')).upper()
            if bg != 'UNKNOWN':
                body_region_str = bg
        body_region = int(body_region_to_id.get(body_region_str, 10))

        _pc = bisect.bisect_left(coil_rows, start_row) - 1
        coil_config = (
            _parse_coil_message_local(df_merged.iloc[coil_rows[_pc]]['Message'], coil_columns)
            if _pc >= 0 else {col: 0 for col in coil_columns}
        )

        row  = segment.iloc[0]
        dt   = row['AdjustedEventDateTime']
        temp = _temporal_features_local(dt)
        conditioning = {
            'Age':               _safe_float_local(row.get('Age',    0)),
            'Weight':            _safe_float_local(row.get('Weight', 0)),
            'Height':            _safe_float_local(row.get('Height', 0)),
            'PTAB':              _safe_float_local(row.get('PTAB',   0)),
            'Direction_encoded': _safe_float_local(row.get('Direction_encoded', 0)),
            **temp,
        }

        start_dt = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt

        seq_dict = {
            'sequence':       seq_tokens,
            'durations':      [float(d) for d in durations],
            'conditioning':   conditioning,
            'body_region':    body_region,
            'sequence_type':  int(seq_type),
            'serial_idx':     int(serial_idx),
            'coil_config':    coil_config,
            'total_duration': total_duration,
            'start_datetime': start_dt.isoformat(),
        }
        sequences.append({'SerialNumber': serial, 'sequence_json': json.dumps(seq_dict)})

    if not sequences:
        return _empty
    return pd.DataFrame(sequences)


# ---- 2g. Run extraction in parallel ----
_t = time.perf_counter()

seq_spark = (
    df_eventlog_spark
    .groupBy('SerialNumber')
    .applyInPandas(extract_exam_sequences, schema=_SEQ_SCHEMA)
)

seq_pd = seq_spark.toPandas()   # collect small results to driver
_t = _timeit('exam extraction (parallel applyInPandas)', _t)

examination_sequences = []
for row in seq_pd.itertuples(index=False):
    d = json.loads(row.sequence_json)
    # Restore start_datetime from ISO string
    d['start_datetime'] = datetime.fromisoformat(d['start_datetime'])
    examination_sequences.append(d)

print(f"\nTotal examination sequences: {len(examination_sequences)}")

# Unmapped body region report
region_counts = {}
for s in examination_sequences:
    region_counts[s['body_region']] = region_counts.get(s['body_region'], 0) + 1
print(f"  Examination regions: {dict(sorted(region_counts.items()))}")

# Sequence-type distribution
st_counts = {}
for s in examination_sequences:
    name = ID_TO_SEQUENCE_TYPE.get(s.get('sequence_type', 0), 'other')
    st_counts[name] = st_counts.get(name, 0) + 1
print(f"  Exam seq types: {dict(sorted(st_counts.items(), key=lambda kv: -kv[1]))}")

# COMMAND ----------

# =============================================================================
# SECTION 3 — customer_schedules  (unchanged from csv_pipeline/03)
# =============================================================================

print("\n" + "="*60)
print("SECTION 3: customer_schedules from exam CSVs")
print("="*60)

_t = time.perf_counter()
customer_schedules = {}

for serial in SERIAL_NUMBERS:
    cid      = str(serial)
    csv_path = f"{EXAM_OUTPUT_DIR}/DATA_{serial}.csv"

    if not os.path.exists(csv_path):
        print(f"  {serial}: exam CSV not found — skipping")
        continue

    df_exam = pd.read_csv(csv_path)
    df_exam['startTime'] = pd.to_datetime(df_exam['startTime'], errors='coerce')
    df_exam = df_exam.dropna(subset=['startTime'])
    df_exam['date'] = df_exam['startTime'].dt.date

    customer_schedules[cid] = {}

    for date, day_df in df_exam.groupby('date'):
        date_str = str(date)
        day_df   = day_df.sort_values('startTime')
        patients = []
        seen     = set()

        for _, row in day_df.iterrows():
            pid = str(row.get('PatientID', '') or '')
            if not pid or pid in seen or pid in ('False', 'nan'):
                continue
            seen.add(pid)

            bp_str = str(row.get('BodyPart',  '') or '').strip().upper()
            bg_str = str(row.get('BodyGroup', '') or '').strip().upper()
            if not bg_str or bg_str in ('NAN', 'FALSE', 'UNKNOWN', ''):
                bg_str = body_part_to_group.get(bp_str, 'UNKNOWN')
            bg_id  = _BODY_REGION_TO_ID.get(bg_str, 10)

            direction_raw = str(row.get('Direction', '') or '').strip()
            hour = int(row['startTime'].hour)
            dow  = int(row['startTime'].dayofweek)

            patients.append({
                'patient_id':     pid,
                'body_region':    bg_str,
                'body_region_id': bg_id,
                'age':            _safe_float(row.get('Age',    0)),
                'weight':         _safe_float(row.get('Weight', 0)),
                'height':         _safe_float(row.get('Height', 0)),
                'direction':      direction_raw if direction_raw else 'Head First',
                'hour_of_day':    hour,
                'day_of_week':    dow,
            })

        if patients:
            customer_schedules[cid][date_str] = patients

    n_days = len(customer_schedules[cid])
    n_pats = sum(len(p) for p in customer_schedules[cid].values())
    print(f"  {cid}: {n_days} days, {n_pats} patients")

print(f"\nCustomer schedules: {len(customer_schedules)} scanners")
_t = _timeit('customer_schedules', _t)

# COMMAND ----------

# =============================================================================
# SECTION 4 — daily_summaries  (unchanged from csv_pipeline/03)
# =============================================================================

print("\n" + "="*60)
print("SECTION 4: daily_summaries from exchange CSVs")
print("="*60)

_t = time.perf_counter()
daily_summaries = []

for serial in SERIAL_NUMBERS:
    csv_path = f"{EXCHANGE_OUTPUT_DIR}/DATA_{serial}.csv"
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    name_col = 'token_name' if 'token_name' in df.columns else 'sourceID'
    df = df[df[name_col].notna() & (df[name_col].astype(str).str.strip() != '')].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['date']        = df['datetime'].dt.date
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['timediff']    = pd.to_numeric(df.get('timediff', 0), errors='coerce').fillna(0)

    for date, grp in df.groupby('date'):
        hourly_dist = grp.groupby('hour_of_day').size().reindex(range(24), fill_value=0).tolist()
        daily_summaries.append({
            'date':               date,
            'day_of_week':        int(grp['day_of_week'].iloc[0]),
            'is_weekend':         int(grp['day_of_week'].iloc[0] >= 5),
            'num_patients':       grp['PatientID_to'].nunique() if 'PatientID_to' in grp.columns else 0,
            'num_events':         len(grp),
            'start_hour':         int(grp['hour_of_day'].min()),
            'end_hour':           int(grp['hour_of_day'].max()),
            'total_duration_seconds':  float(grp['timediff'].sum()),
            'avg_duration_per_event':  float(grp['timediff'].mean()),
            'hourly_distribution': hourly_dist,
            'morning_event_ratio': float((grp['hour_of_day'] < 12).mean()),
            'body_regions':       (grp['BodyGroup_to_text'].value_counts().to_dict()
                                   if 'BodyGroup_to_text' in grp.columns
                                   else grp['BodyGroup_to'].value_counts().to_dict()
                                   if 'BodyGroup_to' in grp.columns else {}),
            'customer_id':        str(serial),
        })

daily_summaries.sort(key=lambda x: x['date'])
print(f"Daily summary entries: {len(daily_summaries)}")
_t = _timeit('daily_summaries', _t)

# COMMAND ----------

# =============================================================================
# SECTION 5 — Assemble and save preprocessed_data.pkl
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: Assemble and save")
print("="*60)

preprocessed_data = {
    'version':            4,
    'exchange':           exchange_sequences,
    'examination':        examination_sequences,
    'daily_summaries':    daily_summaries,
    'customer_schedules': customer_schedules,
}

_t = time.perf_counter()
with open(PKL_OUTPUT, 'wb') as f:
    pickle.dump(preprocessed_data, f)

size_mb = os.path.getsize(PKL_OUTPUT) / (1024 * 1024)
print(f"Saved → {PKL_OUTPUT}  ({size_mb:.1f} MB)")
_t = _timeit('pickle write', _t)

# COMMAND ----------

# =============================================================================
# SECTION 6 — Summary + download link
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Exchange sequences:    {len(exchange_sequences):,}")
print(f"  Examination sequences: {len(examination_sequences):,}")
print(f"  Daily summaries:       {len(daily_summaries):,}")
print(f"  Customers:             {len(customer_schedules)}")

print("\n" + "-"*60)
print("  TIMING BREAKDOWN")
print("-"*60)
for _label, _dt in _TIMINGS:
    print(f"[timing] step03  {_label:<25} {_dt:9.1f}s")
print(f"[timing] step03  {'TOTAL':<25} {sum(d for _, d in _TIMINGS):9.1f}s")

displayHTML('<a href="/files/spark_pipeline/preprocessed_data.pkl">Download preprocessed_data.pkl</a>')
