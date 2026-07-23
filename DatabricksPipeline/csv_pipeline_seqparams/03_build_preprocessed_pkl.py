# Databricks notebook source
# Databricks notebook — Build preprocessed_data.pkl (feature-rich examination pipeline)
#
# Forked from DatabricksPipeline/csv_pipeline/03_build_preprocessed_pkl.py,
# TRIMMED to the examination-sequence path only (train_examination_model()
# only ever reads preprocessed['examination'] — exchange/customer_schedules/
# daily_summaries are not needed here, so 01_exchange_preprocessing.py does
# not need forking at all for this pipeline).
#
# Adds MRI_SUT_1005 sequence-protocol parameters (TR, number of slices,
# trigger/gating mode) to each examination sequence's conditioning, with
# "scanning time" (SD58) explicitly excluded as a leakage risk — see
# csv_pipeline_seqparams/config.py's SUT_LEAKAGE_DENYLIST / assert_no_leakage.
#
# STATUS: SUT_SLOT_MAP / EXAMINATION_SEQPARAM_FEATURES are placeholders in
# config.py until DatabricksPipeline/sut_parameter_discovery.py confirms the
# real slot map. Running this notebook before that is safe (it will simply
# produce a pkl equivalent to the old 10-dim conditioning, since an empty
# EXAMINATION_SEQPARAM_FEATURES is a no-op, not a crash) but pointless for
# the actual feature-richness goal.
#
# Inputs:
#   hive_metastore.eventlog.common_eventlog          (Spark)
#   hive_metastore.examination.examination_workflow  (Spark)
#   /dbfs/FileStore/tables/bodyupdated.xlsx
#
# Output:
#   /dbfs/FileStore/csv_pipeline_seqparams/preprocessed_data.pkl
#   (examination sequences ONLY — no 'exchange'/'customer_schedules'/
#   'daily_summaries' keys, since nothing downstream reads them for this
#   pipeline)

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import re
import bisect
import pickle
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

os.makedirs("/dbfs/FileStore/csv_pipeline_seqparams", exist_ok=True)

# ---- Lightweight section timing (see csv_pipeline/03 for rationale) -------
_TIMINGS = []
def _timeit(label, t0):
    dt = time.perf_counter() - t0
    _TIMINGS.append((label, dt))
    print(f"[timing] step03-seqparams  {label:<20} {dt:9.1f}s")
    return time.perf_counter()

# AlternatingPipeline body region mapping (same values as parent config)
_BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                 'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
_BODY_REGION_TO_ID = {r: i for i, r in enumerate(_BODY_REGIONS)}

# Secondary normalizer: maps BodyGroup names that the Excel emits but that are
# NOT in the 11-region vocab to their canonical equivalents (byte-identical
# copy of csv_pipeline/03's own table — see that file for the full rationale).
_BODYGROUP_NORMALIZE = {
    'BRAIN': 'HEAD', 'SKULL': 'HEAD', 'FACE': 'HEAD', 'ORBIT': 'HEAD',
    'SELLA': 'HEAD', 'IAM': 'HEAD', 'EAR': 'HEAD', 'TEMPORAL': 'HEAD',
    'PITUITARY': 'HEAD',
    'THROAT': 'NECK', 'LARYNX': 'NECK', 'THYROID': 'NECK',
    'BREAST': 'CHEST', 'CARDIAC': 'CHEST', 'HEART': 'CHEST',
    'THORAX': 'CHEST', 'LUNG': 'CHEST', 'AORTA': 'CHEST',
    'MEDIASTINUM': 'CHEST', 'STERNUM': 'CHEST',
    'LIVER': 'ABDOMEN', 'KIDNEY': 'ABDOMEN', 'PANCREAS': 'ABDOMEN',
    'ADRENAL': 'ABDOMEN', 'SPLEEN': 'ABDOMEN', 'GI': 'ABDOMEN',
    'BOWEL': 'ABDOMEN', 'GALLBLADDER': 'ABDOMEN', 'ADRENAL GLAND': 'ABDOMEN',
    'PROSTATE': 'PELVIS', 'HIP': 'PELVIS', 'UTERUS': 'PELVIS',
    'OVARY': 'PELVIS', 'BLADDER': 'PELVIS', 'RECTUM': 'PELVIS',
    'SACRUM': 'PELVIS', 'SACRAL': 'PELVIS', 'COCCYX': 'PELVIS',
    'LSPINE': 'SPINE', 'CSPINE': 'SPINE', 'TSPINE': 'SPINE',
    'LUMBAR': 'SPINE', 'CERVICAL': 'SPINE', 'THORACIC': 'SPINE',
    'LUMBOSACRAL': 'SPINE', 'WHOLESPINE': 'SPINE', 'WHOLESPINE ': 'SPINE',
    'WHOLE SPINE': 'SPINE', 'L-SPINE': 'SPINE', 'C-SPINE': 'SPINE',
    'T-SPINE': 'SPINE', 'LS-SPINE': 'SPINE', 'SACROILIAC': 'SPINE',
    'SHOULDER': 'ARM', 'ELBOW': 'ARM', 'WRIST': 'ARM',
    'FOREARM': 'ARM', 'HUMERUS': 'ARM', 'UPPERARM': 'ARM',
    'UPPER ARM': 'ARM', 'UPPER_ARM': 'ARM',
    'KNEE': 'LEG', 'THIGH': 'LEG', 'ANKLE': 'LEG',
    'CALF': 'LEG', 'TIBIA': 'LEG', 'FIBULA': 'LEG', 'FEMUR': 'LEG',
    'LOWER LEG': 'LEG', 'LOWER_LEG': 'LEG',
    'FINGER': 'HAND', 'THUMB': 'HAND',
    'TOE': 'FOOT', 'HEEL': 'FOOT',
}


def _normalize_bodygroup(bg_str):
    s = str(bg_str).strip().upper()
    return _BODYGROUP_NORMALIZE.get(s, s)

print(f"Serials:    {SERIAL_NUMBERS}")
print(f"Date range: {DATE_START} → {DATE_END}")
print(f"Output:     {PKL_OUTPUT}")

# COMMAND ----------

# =============================================================================
# HELPERS
# =============================================================================

def _temporal_features(dt):
    h = dt.hour
    dow = dt.dayofweek if hasattr(dt, 'dayofweek') else dt.weekday()
    return {
        'hour_of_day': int(h),
        'day_of_week': int(dow),
        'is_morning': int(h < 12),
        'hour_sin': float(np.sin(2 * np.pi * h / 24)),
        'hour_cos': float(np.cos(2 * np.pi * h / 24)),
        'dow_sin': float(np.sin(2 * np.pi * dow / 7)),
        'dow_cos': float(np.cos(2 * np.pi * dow / 7)),
    }


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _normalize_body_label(val):
    if pd.isna(val):
        return ''
    text = str(val).strip().upper()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('BODY PART ', '')
    text = text.replace('BODYPART ', '')
    return text


def _canonical_body_label(val):
    label = _normalize_body_label(val)
    if not label:
        return ''
    alias_map = {
        'BREAST': 'BREAST', 'BREASTS': 'BREAST',
        'KNEE': 'KNEE', 'KNEES': 'KNEE',
        'SHOULDER': 'SHOULDER', 'SHOULDERS': 'SHOULDER',
        'HIP': 'HIP', 'HIPS': 'HIP',
        'HEART': 'HEART',
        'WHOLEBODY': 'WHOLEBODY', 'WHOLE BODY': 'WHOLEBODY',
        'TOTAL BODY': 'WHOLEBODY', 'WHOLE-BODY': 'WHOLEBODY',
        'WHOLE_BODY': 'WHOLEBODY',
    }
    return alias_map.get(label, label)


def _display_body_label(val):
    if pd.isna(val):
        return ''
    text = str(val).strip()
    if not text:
        return ''
    return text[:1].upper() + text[1:].lower()


_UNKNOWN_BODY_AUDIT_ROWS = []
_BODY_GROUP_SANITY = {}


def _record_unknown_body_audit(values, label):
    if not values:
        return
    for raw in values:
        raw_text = str(raw).strip()
        norm_text = _canonical_body_label(raw_text)
        if not norm_text:
            continue
        _UNKNOWN_BODY_AUDIT_ROWS.append({
            'source': label,
            'raw_body_label': raw_text,
            'normalized_body_label': norm_text,
            'display_body_label': _display_body_label(raw_text),
        })


def _print_unknown_body_audit():
    if not _UNKNOWN_BODY_AUDIT_ROWS:
        print("\nNo UNKNOWN body labels recorded.")
        return
    audit_df = pd.DataFrame(_UNKNOWN_BODY_AUDIT_ROWS)
    counts = (
        audit_df
        .groupby(['source', 'raw_body_label', 'normalized_body_label', 'display_body_label'], dropna=False)
        .size()
        .reset_index(name='row_count')
        .sort_values(['source', 'row_count', 'raw_body_label'], ascending=[True, False, True])
    )
    print("\n" + "=" * 60)
    print("UNKNOWN BODY LABEL AUDIT")
    print("=" * 60)
    for source, group in counts.groupby('source', sort=False):
        print(f"\n[{source}]")
        for _, row in group.iterrows():
            print(f"  {row['display_body_label']:<40} {int(row['row_count']):>6,}  ->  {row['normalized_body_label']}")


def _record_body_group_sanity(serial, raw_label, mapped_group, source):
    serial_key = str(serial)
    raw_text = str(raw_label).strip()
    group_text = str(mapped_group).strip().upper() if mapped_group is not None else 'UNKNOWN'
    if not raw_text:
        return
    serial_bucket = _BODY_GROUP_SANITY.setdefault(serial_key, {'known': {}, 'unknown': {}})
    target_bucket = serial_bucket['unknown'] if group_text == 'UNKNOWN' else serial_bucket['known']
    entry = target_bucket.setdefault(raw_text, {'group': group_text, 'count': 0, 'sources': set()})
    entry['count'] += 1
    entry['sources'].add(source)


def _print_body_group_sanity():
    if not _BODY_GROUP_SANITY:
        print("\nNo body-group sanity data recorded.")
        return
    print("\n" + "=" * 60)
    print("BODY GROUP SANITY SUMMARY")
    print("=" * 60)
    for serial in sorted(_BODY_GROUP_SANITY.keys()):
        print(f"\n[serial {serial}]")
        bucket = _BODY_GROUP_SANITY[serial]
        known_total = sum(item['count'] for item in bucket['known'].values())
        unknown_total = sum(item['count'] for item in bucket['unknown'].values())
        total = known_total + unknown_total
        unknown_pct = (unknown_total / total * 100.0) if total else 0.0
        print(f"  total_labels={total:,}  known={known_total:,}  unknown={unknown_total:,}  unknown_pct={unknown_pct:.1f}%")
        for status in ('known', 'unknown'):
            entries = bucket[status]
            if not entries:
                continue
            print(f"  {status.upper()}:")
            for raw_label, info in sorted(entries.items(), key=lambda kv: (-kv[1]['count'], kv[0])):
                sources = ",".join(sorted(info['sources']))
                print(f"    {raw_label:<35} {info['count']:>6,}  ->  {info['group']}  ({sources})")


def _seq_type_from_msg(msg):
    """Classify the MRI sequence family from an MRI_MSR_100 message."""
    if not isinstance(msg, str):
        return SEQUENCE_TYPE_VOCAB['other']
    m = re.search(r"Sequence:\s*'([^']*)'", msg)
    return classify_sequence_type(m.group(1) if m else '')

# COMMAND ----------

# =============================================================================
# COIL PARSING HELPERS (byte-identical copy of csv_pipeline/03)
# =============================================================================

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

def _parse_coil_message(msg):
    coil_config = {col: 0 for col in COIL_COLUMNS}
    if not isinstance(msg, str) or not msg.strip():
        return coil_config
    m = re.search(r'[Cc]onnected\s+coil\s+elements?:?\s*(.*?)(?:\s*\||\s*$)', msg)
    coil_str = m.group(1).strip() if m else msg
    for name in _expand_coil_list(coil_str):
        mapped = _COIL_ABBREV.get(name)
        if mapped and mapped in coil_config:
            coil_config[mapped] = 1
    return coil_config

def _add_ptab(df):
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

# COMMAND ----------

# =============================================================================
# SUT SEQUENCE-PARAMETER PARSING (NEW — the point of this fork)
# =============================================================================
# PLACEHOLDER parser pending DatabricksPipeline/sut_parameter_discovery.py.
# Assumes labeled "SD<n>: value" pairs (consistent with this scanner
# software's other labeled messages — Sequence:'...', Protocol:'...',
# CoilID '...') — if discovery finds a different format (e.g. an unlabeled
# positional list), update this function to match BEFORE trusting its
# output. Gracefully returns {} / partial dicts rather than raising, since
# SUT_SLOT_MAP is expected to be incomplete until Phase 1 confirms slots.
# =============================================================================

_SD_LABELED_RE = re.compile(r"SD(\d+)\s*[:=]\s*'?([^,'\n]+)'?")


def _parse_sut_message(msg):
    """Parse an MRI_SUT_1005 message into {stable_name: raw_value_str},
    for every slot present in both the message and SUT_SLOT_MAP."""
    if not isinstance(msg, str) or not SUT_SLOT_MAP:
        return {}
    raw_slots = {int(n): v.strip() for n, v in _SD_LABELED_RE.findall(msg)}
    return {
        name: raw_slots[slot]
        for slot, name in SUT_SLOT_MAP.items()
        if slot in raw_slots
    }


def _trigger_mode_id(sut_values):
    """Map the parsed trigger_mode string to its TRIGGER_MODE_VOCAB id,
    defaulting to 'none'/0 when absent or unrecognized."""
    raw = str(sut_values.get('trigger_mode', 'none')).strip().lower()
    return TRIGGER_MODE_VOCAB.get(raw, TRIGGER_MODE_VOCAB.get('unknown', 0))

# COMMAND ----------

# =============================================================================
# SECTION 2 — Examination sequences from Spark
#
# Forked from csv_pipeline/03_build_preprocessed_pkl.py's SECTION 2. Adds:
#   (a) MRI_SUT_1005 to the Spark query filter (REAL_EVENT_TYPES already
#       includes it via SOURCE_ID_FILTER_SEQPARAMS — see config.py's GAP
#       FOUND note: the old pipeline's filter does NOT include this)
#   (b) sut_rows + bisect join, same pattern as coil_rows / _pc below
#   (c) leakage gate #2 (assert_no_leakage on the runtime dict, right before
#       merging into conditioning)
# =============================================================================

print("\n" + "="*60)
print("SECTION 2: Examination sequences from Spark (+ SUT parameters)")
print("="*60)

_BODY_GROUP_SANITY.clear()
_UNKNOWN_BODY_AUDIT_ROWS.clear()

# ---- 2a. Load body group mapping ----
df_body_excel = pd.read_excel(BODY_GROUP_MAPPING_PATH)
if 'BodyPart' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _cp, _cg = 'BodyPart', 'BodyGroup'
elif 'BodyPartExamined' in df_body_excel.columns:
    _cp, _cg = 'BodyPartExamined', 'BodyGroup'
else:
    _cp, _cg = df_body_excel.columns[0], df_body_excel.columns[1]

body_part_to_group = {
    _canonical_body_label(k): str(v).strip().upper()
    for k, v in zip(df_body_excel[_cp], df_body_excel[_cg])
    if pd.notna(k) and pd.notna(v)
}
print(f"Body mapping: {len(body_part_to_group)} entries")

# ---- 2b. Query eventlog (filter WIDENED to include MRI_SUT_1005) ----
_t = time.perf_counter()
df_eventlog_spark = (
    spark.table(EVENTLOG_TABLE)
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
    .filter(F.col("EventDateTime") >= DATE_START)
    .filter(F.col("EventDateTime") <= DATE_END)
    .filter(F.col("MessageIdentification").isin(REAL_EVENT_TYPES))  # includes MRI_SUT_1005
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
_t = _timeit('eventlog.toPandas', _t)

# ---- 2c. Query examination_workflow ----
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
        F.first(F.col("WorkflowValues")["ProtocolName"]).alias("ProtocolName"),
    )
    .orderBy("SerialNumber", "WorkflowStartRefDateTime")
)
df_exams_pd = df_exams_spark.toPandas()
_t = _timeit('exam.toPandas', _t)

for col in ['Age', 'Weight', 'Height']:
    df_exams_pd[col] = pd.to_numeric(df_exams_pd[col], errors='coerce').fillna(0.0)

df_exams_pd['Direction_encoded'] = df_exams_pd['Direction'].apply(
    lambda x: 0 if str(x).strip().lower() == 'head first'
              else (1 if str(x).strip().lower() == 'feet first' else -1)
)
df_exams_pd['BodyGroup'] = df_exams_pd['BodyPartExamined'].apply(
    lambda x: body_part_to_group.get(_canonical_body_label(x), 'UNKNOWN')
              if pd.notna(x) else 'UNKNOWN'
)

def _body_group_from_protocol(protocol_name):
    if pd.isna(protocol_name) or not str(protocol_name).strip():
        return None
    for tok in re.split(r'[_\-\s]+', str(protocol_name).strip().upper()):
        label = _canonical_body_label(tok)
        if label and body_part_to_group.get(label, 'UNKNOWN') != 'UNKNOWN':
            return body_part_to_group[label]
    return None

_mask_unknown = df_exams_pd['BodyGroup'] == 'UNKNOWN'
if _mask_unknown.any():
    df_exams_pd.loc[_mask_unknown, 'BodyGroup'] = (
        df_exams_pd.loc[_mask_unknown, 'ProtocolName']
        .apply(lambda p: _body_group_from_protocol(p) or 'UNKNOWN')
    )
    _n_proto = int(_mask_unknown.sum() - (df_exams_pd['BodyGroup'] == 'UNKNOWN').sum())
    if _n_proto > 0:
        print(f"  ProtocolName fallback resolved {_n_proto:,} exam body groups")

df_exams_pd['BodyGroup'] = df_exams_pd['BodyGroup'].apply(_normalize_bodygroup)
print(f"Examination rows: {len(df_exams_pd):,}")

_unmapped = (
    df_exams_pd.loc[df_exams_pd['BodyGroup'] == 'UNKNOWN', 'BodyPartExamined']
    .dropna().astype(str).str.strip().str.upper()
)
_unmapped = _unmapped[_unmapped != ''].value_counts()
if len(_unmapped):
    print(f"  ⚠ {len(_unmapped)} distinct BodyPart values are UNMAPPED → UNKNOWN "
          f"({int(_unmapped.sum()):,} rows).")
    _record_unknown_body_audit(
        df_exams_pd.loc[df_exams_pd['BodyGroup'] == 'UNKNOWN', 'BodyPartExamined'].tolist(),
        'exam_workflow',
    )

# ---- 2e. Per-serial extraction loop ----

_t = time.perf_counter()
examination_sequences = []
_sut_parse_hits = 0
_sut_parse_total = 0

for serial in SERIAL_NUMBERS:
    print(f"\n  --- {serial} ---")
    serial_idx = SERIAL_NUMBERS.index(serial)
    df_sc = df_eventlog_pd[df_eventlog_pd['SerialNumber'] == int(serial)].copy()
    df_ex = df_exams_pd[df_exams_pd['SerialNumber'] == int(serial)].copy()

    if df_sc.empty or df_ex.empty:
        print(f"  No events or examinations — skipping.")
        continue

    df_sc = df_sc.sort_values(['AdjustedEventDateTime', 'Line']).reset_index(drop=True)
    df_ex = df_ex.sort_values('WorkflowStartRefDateTime').reset_index(drop=True)

    df_merged = pd.merge_asof(
        df_sc,
        df_ex[['WorkflowStartRefDateTime', 'PatientId',
               'Age', 'Weight', 'Height', 'Direction_encoded', 'BodyGroup']],
        left_on='AdjustedEventDateTime',
        right_on='WorkflowStartRefDateTime',
        direction='backward',
    )

    df_merged['PatientId'] = df_merged['PatientId'].fillna('__no_patient__')
    df_merged['BodyGroup_to'] = df_merged['BodyGroup'].fillna('UNKNOWN').str.upper()

    _mask_evu = df_merged['MessageIdentification'] == 'MRI_EXU_95'
    def _bg_from_evu95(msg):
        if not isinstance(msg, str):
            return None
        try:
            body = re.search(r'with body part < (.*) >', msg).group(1)
            if ' ' in body:
                try:
                    body = re.search(r'with body part < (.*) > <', msg).group(1)
                except AttributeError:
                    pass
            return body_part_to_group.get(_canonical_body_label(body))
        except AttributeError:
            return None
    df_merged['_bg_msg'] = None
    df_merged.loc[_mask_evu, '_bg_msg'] = df_merged.loc[_mask_evu, 'Message'].apply(_bg_from_evu95)
    df_merged['_bg_msg'] = df_merged['_bg_msg'].ffill()
    _needs_bg = df_merged['BodyGroup_to'] == 'UNKNOWN'
    df_merged.loc[_needs_bg, 'BodyGroup_to'] = (
        df_merged.loc[_needs_bg, '_bg_msg'].fillna('UNKNOWN').apply(_normalize_bodygroup)
    )
    df_merged.drop(columns=['_bg_msg'], inplace=True)

    _exam_final = pd.merge(
        df_merged.dropna(subset=['WorkflowStartRefDateTime'])
                 .drop_duplicates(subset=['WorkflowStartRefDateTime'])
                 [['WorkflowStartRefDateTime', 'BodyGroup_to']],
        df_ex[['WorkflowStartRefDateTime', 'BodyPartExamined']],
        on='WorkflowStartRefDateTime', how='left',
    )
    for _, _r in _exam_final.iterrows():
        _record_body_group_sanity(
            serial, _r.get('BodyPartExamined'),
            str(_r.get('BodyGroup_to', 'UNKNOWN')).upper(), 'exam_workflow',
        )

    df_merged['Age'] = df_merged['Age'].fillna(0.0)
    df_merged['Weight'] = df_merged['Weight'].fillna(0.0)
    df_merged['Height'] = df_merged['Height'].fillna(0.0)
    df_merged['Direction_encoded'] = df_merged['Direction_encoded'].fillna(0.0)

    df_merged = _add_ptab(df_merged)

    t0 = df_merged['AdjustedEventDateTime'].min()
    df_merged['timediff'] = (df_merged['AdjustedEventDateTime'] - t0).dt.total_seconds()

    df_merged['sourceID_token'] = df_merged['MessageIdentification'].map(
        lambda x: SOURCEID_VOCAB.get(x, SOURCEID_VOCAB['UNK'])
    )

    start_rows = df_merged.index[df_merged['MessageIdentification'] == 'MRI_MSR_100'].tolist()
    end_rows = sorted(df_merged.index[
        df_merged['MessageIdentification'].isin(['MRI_MSR_104', 'MRI_MSR_34'])
    ].tolist())
    coil_rows = df_merged.index[df_merged['MessageIdentification'] == 'MRI_CCS_11'].tolist()
    exam_rows = df_merged.index[df_merged['MessageIdentification'] == 'MRI_EXU_95'].tolist()
    # NEW — SUT sequence-protocol parameter rows. Join semantics (most-recent
    # -before, same as coil_rows) is a hypothesis — confirm against
    # sut_parameter_discovery.py STEP 2 (periodic vs. per-measurement).
    sut_rows = df_merged.index[df_merged['MessageIdentification'] == 'MRI_SUT_1005'].tolist()

    n_before = len(examination_sequences)

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

        sequence = segment['sourceID_token'].tolist()
        timediffs = segment['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0
        durations = [min(MAX_PER_TOKEN_DURATION, max(0.0, d)) for d in durations]
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        is_abort = (str(df_merged.iloc[end_row].get('MessageIdentification', '')) == 'MRI_MSR_34')

        if total_duration > MAX_EXAMINATION_DURATION:
            continue
        if total_duration < MIN_EXAMINATION_DURATION and not is_abort:
            continue

        seq_type = _seq_type_from_msg(segment.iloc[0].get('Message'))

        body_region_str = _normalize_bodygroup(str(segment.iloc[0].get('BodyGroup_to', 'UNKNOWN')).upper())
        _pe = bisect.bisect_left(exam_rows, start_row) - 1
        if _pe >= 0:
            bg = _normalize_bodygroup(str(df_merged.iloc[exam_rows[_pe]].get('BodyGroup_to', 'UNKNOWN')).upper())
            if bg != 'UNKNOWN':
                body_region_str = bg
        body_region = int(_BODY_REGION_TO_ID.get(body_region_str, 10))

        _pc = bisect.bisect_left(coil_rows, start_row) - 1
        coil_config = (
            _parse_coil_message(df_merged.iloc[coil_rows[_pc]]['Message'])
            if _pc >= 0 else {col: 0 for col in COIL_COLUMNS}
        )

        # NEW — SUT sequence-protocol parameters (most-recent-before start_row)
        _ps = bisect.bisect_left(sut_rows, start_row) - 1
        sut_values = (
            _parse_sut_message(df_merged.iloc[sut_rows[_ps]]['Message'])
            if _ps >= 0 else {}
        )
        _sut_parse_total += 1
        if sut_values:
            _sut_parse_hits += 1

        # Only the features in EXAMINATION_SEQPARAM_FEATURES ever reach the
        # model — never the raw unfiltered sut_values (which may include
        # scanning_time). Leakage gate #2: re-check the runtime dict keys,
        # not just the static config list, right before they get merged.
        numeric_seqparams = {
            name: _safe_float(sut_values.get(name))
            for name in EXAMINATION_SEQPARAM_FEATURES
        }
        assert_no_leakage(numeric_seqparams.keys())
        trigger_mode_id = _trigger_mode_id(sut_values)

        row = segment.iloc[0]
        dt = row['AdjustedEventDateTime']
        temp = _temporal_features(dt)
        conditioning = {
            'Age': _safe_float(row.get('Age', 0)),
            'Weight': _safe_float(row.get('Weight', 0)),
            'Height': _safe_float(row.get('Height', 0)),
            'PTAB': _safe_float(row.get('PTAB', 0)),
            'Direction_encoded': _safe_float(row.get('Direction_encoded', 0)),
            **temp,
        }
        conditioning.update(numeric_seqparams)

        start_dt = dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt

        examination_sequences.append({
            'sequence': sequence,
            'durations': durations,
            'conditioning': conditioning,
            'body_region': body_region,
            'sequence_type': int(seq_type),
            'serial_idx': int(serial_idx),
            'trigger_mode': int(trigger_mode_id),
            'coil_config': coil_config,
            # Audit-only — full unfiltered SUT parse, INCLUDING scanning_time.
            # Never merge this into 'conditioning'. Used only for Phase 1's
            # SD58-vs-real-duration validation cross-check and debugging.
            'sut_debug': sut_values,
            'total_duration': total_duration,
            'start_datetime': start_dt,
        })

    print(f"  → {len(examination_sequences) - n_before} examination sequences")

print(f"\nTotal examination sequences: {len(examination_sequences)}")
print(f"SUT parse hit rate: {_sut_parse_hits}/{_sut_parse_total} segments had a "
      f"preceding MRI_SUT_1005 event with at least one recognized slot "
      f"({100 * _sut_parse_hits / max(1, _sut_parse_total):.1f}%)")
_t = _timeit('exam extraction', _t)

_print_body_group_sanity()
_print_unknown_body_audit()

# COMMAND ----------

# =============================================================================
# SECTION 5 — Assemble and save preprocessed_data.pkl (examination only)
# =============================================================================

print("\n" + "="*60)
print("SECTION 5: Assemble and save")
print("="*60)

preprocessed_data = {
    'version': 'seqparams-1',   # distinct from csv_pipeline/'s integer versions —
                                # this pkl carries a DIFFERENT schema (examination
                                # only + trigger_mode + sut_debug), not a drop-in
                                # replacement for csv_pipeline/'s pkl.
    'examination': examination_sequences,
}

_t = time.perf_counter()
with open(PKL_OUTPUT, 'wb') as f:
    pickle.dump(preprocessed_data, f)

size_mb = os.path.getsize(PKL_OUTPUT) / (1024 * 1024)
print(f"Saved → {PKL_OUTPUT}  ({size_mb:.1f} MB)")
_t = _timeit('pickle write', _t)

# COMMAND ----------

# =============================================================================
# SECTION 6 — Summary
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Examination sequences: {len(examination_sequences):,}")

print("\n" + "-"*60)
print("  TIMING BREAKDOWN")
print("-"*60)
for _label, _dt in _TIMINGS:
    print(f"[timing] step03-seqparams  {_label:<20} {_dt:9.1f}s")
print(f"[timing] step03-seqparams  {'TOTAL':<20} {sum(d for _, d in _TIMINGS):9.1f}s")

if examination_sequences:
    region_counts = {}
    for s in examination_sequences:
        region_counts[s['body_region']] = region_counts.get(s['body_region'], 0) + 1
    print(f"  Examination regions:  {dict(sorted(region_counts.items()))}")

    st_counts = {}
    for s in examination_sequences:
        name = ID_TO_SEQUENCE_TYPE.get(s.get('sequence_type', 0), 'other')
        st_counts[name] = st_counts.get(name, 0) + 1
    print(f"  Examination seq types: {dict(sorted(st_counts.items(), key=lambda kv: -kv[1]))}")

    trig_counts = {}
    for s in examination_sequences:
        trig_counts[s.get('trigger_mode', 0)] = trig_counts.get(s.get('trigger_mode', 0), 0) + 1
    print(f"  Trigger mode distribution: {dict(sorted(trig_counts.items()))} "
          f"(all-zero => SUT parsing found nothing yet — expected until "
          f"sut_parameter_discovery.py confirms the real slot map)")

    if EXAMINATION_SEQPARAM_FEATURES:
        for name in EXAMINATION_SEQPARAM_FEATURES:
            vals = np.array([s['conditioning'].get(name, 0.0) for s in examination_sequences])
            print(f"  {name}: mean={vals.mean():.2f} std={vals.std():.2f} "
                  f"min={vals.min():.2f} max={vals.max():.2f} "
                  f"(all-zero/constant => parsing likely not finding this slot yet)")

    _msr34 = SOURCEID_VOCAB.get('MRI_MSR_34', 15)
    n_abort = sum(1 for s in examination_sequences if _msr34 in s['sequence'])
    print(f"  Examination aborts:   {n_abort} segments contain MRI_MSR_34 "
          f"({100*n_abort/len(examination_sequences):.1f}%)")

displayHTML('<a href="/files/csv_pipeline_seqparams/preprocessed_data.pkl">Download preprocessed_data.pkl</a>')

# COMMAND ----------

# =============================================================================
# NEXT STEP: run 04_train_models.py against this pkl, or 05_feature_analysis.py
# for the correlation matrix (does not need a trained model).
# =============================================================================
