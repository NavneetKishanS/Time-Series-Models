# Databricks notebook — Orchestration Preprocessing
#
# Builds customer_schedules and extracts orchestration training samples.
#
# Inputs:
#   - exchange_sequences.pkl   (from notebook 01)
#   - examination_sequences.pkl (from notebook 02)
#   - hive_metastore.eventlog.common_eventlog  (for patient ordering per day)
#   - hive_metastore.examination.examination_workflow (for demographics)
#
# Output: orchestration_data.pkl  →  dict with keys:
#   customer_schedules, daily_summaries, orch_samples, scanner_to_idx

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pyspark.sql import functions as F

os.makedirs(DBFS_OUTPUT_BASE, exist_ok=True)

# COMMAND ----------
# =============================================================================
# CELL 2: Load intermediate pickles from notebooks 01 and 02
# =============================================================================

with open(EXCHANGE_OUTPUT, 'rb') as f:
    exchange_sequences = pickle.load(f)

with open(EXAMINATION_OUTPUT, 'rb') as f:
    examination_sequences = pickle.load(f)

print(f"Loaded {len(exchange_sequences)} exchange sequences")
print(f"Loaded {len(examination_sequences)} examination sequences")

# COMMAND ----------
# =============================================================================
# CELL 3: Build customer_schedules via Spark
#
# Structure:
#   { str(serial_number): { 'YYYY-MM-DD': [ patient_dict, ... ] } }
#
# Patient dict keys (lowercase, matching AlternatingPipeline):
#   patient_id, body_region, body_region_id, age, weight, height,
#   direction, hour_of_day, day_of_week
# =============================================================================

# ---- 3a. One row per examination per scanner, from examination_workflow ----
# PatientId is in WorkflowValues — NOT a column in the eventlog.
# We use WorkflowStartRefDateTime as the patient's "first appearance" time.
# Deduplicate by (SerialNumber, WorkflowKey) to get one row per examination.
first_appear_spark = (
    spark.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin(TARGET_SERIAL_NUMBERS))
    .filter(F.col("WorkflowStartRefDateTime") >= DATE_START)
    .filter(F.col("WorkflowStartRefDateTime") <= DATE_END)
    .filter(F.col("WorkflowValues").isNotNull())
    .filter(F.col("WorkflowValues")["PatientId"].isNotNull())
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

df_first_pd = first_appear_spark.toPandas()

# Apply timezone offset to get local time for hour/day features
df_first_pd['FirstEventTime'] = pd.to_datetime(df_first_pd['WorkflowStartRefDateTime']) \
    + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)
df_first_pd['EventDate']   = df_first_pd['FirstEventTime'].dt.date
df_first_pd['hour_of_day'] = df_first_pd['FirstEventTime'].dt.hour
df_first_pd['day_of_week'] = df_first_pd['FirstEventTime'].dt.dayofweek  # Monday=0

for col in ['Age', 'Weight', 'Height']:
    df_first_pd[col] = pd.to_numeric(df_first_pd[col], errors='coerce').fillna(0.0)

df_first_pd['Direction'] = df_first_pd['Direction'].fillna('Head First')

print(f"Examinations: {len(df_first_pd):,} rows "
      f"across {df_first_pd['SerialNumber'].nunique()} scanners")

# ---- 3b. Body group mapping ----
df_body_excel = pd.read_excel(BODY_GROUP_MAPPING_PATH)
if 'BodyPart' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _cp, _cg = 'BodyPart', 'BodyGroup'
elif 'BodyPartExamined' in df_body_excel.columns and 'BodyGroup' in df_body_excel.columns:
    _cp, _cg = 'BodyPartExamined', 'BodyGroup'
else:
    _cp, _cg = df_body_excel.columns[0], df_body_excel.columns[1]

body_part_to_group = {
    str(k).strip().upper(): str(v).strip().upper()
    for k, v in zip(df_body_excel[_cp], df_body_excel[_cg])
    if pd.notna(k) and pd.notna(v)
}

# ---- 3c-cont. Map body part → group directly on df_first_pd ----
# Demographics are already in df_first_pd; no separate merge needed.
df_first_pd['BodyGroup'] = df_first_pd['BodyPartExamined'].apply(
    lambda x: body_part_to_group.get(str(x).strip().upper(), 'UNKNOWN')
              if pd.notna(x) else 'UNKNOWN'
)

# ---- 3d. Assemble customer_schedules ----
df_merged = df_first_pd.copy()
df_merged['BodyGroup'] = df_merged['BodyGroup'].fillna('UNKNOWN').str.upper()
df_merged['Age']       = df_merged['Age'].fillna(0.0)
df_merged['Weight']    = df_merged['Weight'].fillna(0.0)
df_merged['Height']    = df_merged['Height'].fillna(0.0)
df_merged['Direction'] = df_merged['Direction'].fillna('Head First')

customer_schedules = {}

for serial_number in TARGET_SERIAL_NUMBERS:
    cid = str(serial_number)
    customer_schedules[cid] = {}

    df_sc = df_merged[df_merged['SerialNumber'] == serial_number].copy()
    df_sc = df_sc.sort_values(['EventDate', 'FirstEventTime'])

    # Group by calendar date ('EventDate' is already a Python date object)
    for event_date, day_df in df_sc.groupby(df_sc['EventDate']):
        date_str = str(event_date)
        patients = []
        seen     = set()

        for _, row in day_df.iterrows():
            pid = str(row['PatientId'])
            if not pid or pid in seen:
                continue
            seen.add(pid)

            bg_str = str(row['BodyGroup']).upper()
            bg_id  = BODY_REGION_TO_ID.get(bg_str, 10)

            patients.append({
                'patient_id':    pid,
                'body_region':   bg_str,
                'body_region_id': bg_id,
                'age':           float(row['Age']),
                'weight':        float(row['Weight']),
                'height':        float(row['Height']),
                'direction':     str(row['Direction']),
                'hour_of_day':   int(row['hour_of_day']),
                'day_of_week':   int(row['day_of_week']),
            })

        if patients:
            customer_schedules[cid][date_str] = patients

    num_days     = len(customer_schedules[cid])
    total_pats   = sum(len(p) for p in customer_schedules[cid].values())
    print(f"  {cid}: {num_days} days, {total_pats} patients")

print(f"\nCustomer schedules built for {len(customer_schedules)} scanners")

# COMMAND ----------
# =============================================================================
# CELL 4: Build daily_summaries (one row per scanner/day)
# =============================================================================

# Daily summaries are derived from examination_workflow (PatientId is there,
# not in the eventlog).  num_patients = distinct PatientIds per scanner/day.
daily_summary_spark = (
    spark.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin(TARGET_SERIAL_NUMBERS))
    .filter(F.col("WorkflowStartRefDateTime") >= DATE_START)
    .filter(F.col("WorkflowStartRefDateTime") <= DATE_END)
    .filter(F.col("WorkflowValues").isNotNull())
    .filter(F.col("WorkflowValues")["PatientId"].isNotNull())
    .select(
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
        F.to_date(
            F.to_timestamp(
                F.unix_timestamp("WorkflowStartRefDateTime") +
                F.lit(TIMEZONE_OFFSET_HOURS * 3600)
            )
        ).alias("EventDate"),
        F.hour(
            F.to_timestamp(
                F.unix_timestamp("WorkflowStartRefDateTime") +
                F.lit(TIMEZONE_OFFSET_HOURS * 3600)
            )
        ).alias("HourOfDay"),
        F.col("WorkflowValues")["PatientId"].alias("PatientId"),
    )
    .groupBy("SerialNumber", "EventDate")
    .agg(
        F.countDistinct("PatientId").alias("num_patients"),
        F.count("*").alias("num_events"),
        F.min("HourOfDay").alias("start_hour"),
        F.max("HourOfDay").alias("end_hour"),
    )
    .orderBy("SerialNumber", "EventDate")
)

df_daily_pd = daily_summary_spark.toPandas()
df_daily_pd['EventDate'] = pd.to_datetime(df_daily_pd['EventDate'])
df_daily_pd['day_of_week'] = df_daily_pd['EventDate'].dt.dayofweek
df_daily_pd['is_weekend']  = (df_daily_pd['day_of_week'] >= 5).astype(int)
df_daily_pd['customer_id'] = df_daily_pd['SerialNumber'].astype(str)

daily_summaries = df_daily_pd.to_dict('records')
print(f"Daily summaries: {len(daily_summaries)} rows")

# COMMAND ----------
# =============================================================================
# CELL 5: Orchestration functions (verbatim copy from orchestration_preprocessing.py)
# =============================================================================

def _compute_scanner_stats(customer_schedules):
    """
    Compute per-scanner historical statistics for conditioning.

    Returns dict mapping customer_id -> {
        'avg_patients_per_day': float,
        'region_distribution': np.array of shape [11]
    }
    """
    stats = {}

    for customer_id, daily_schedules in customer_schedules.items():
        patient_counts = []
        region_counts  = np.zeros(NUM_BODY_REGIONS, dtype=np.float64)

        for date_str, patients in daily_schedules.items():
            patient_counts.append(len(patients))
            for patient in patients:
                region_id = patient.get('body_region_id', 10)
                if 0 <= region_id < NUM_BODY_REGIONS:
                    region_counts[region_id] += 1

        avg_patients  = np.mean(patient_counts) if patient_counts else 0.0
        total_regions = region_counts.sum()
        region_dist   = (region_counts / total_regions
                         if total_regions > 0 else np.zeros(NUM_BODY_REGIONS))

        stats[customer_id] = {
            'avg_patients_per_day': avg_patients,
            'region_distribution':  region_dist,
        }

    return stats


def _build_orchestration_conditioning(date_str, day_of_week, scanner_stats):
    """
    Build 17-dim conditioning vector for an orchestration sample.

    Features:
        [0] dow_sin
        [1] dow_cos
        [2] month_sin
        [3] month_cos
        [4] is_weekend
        [5] avg_patients_per_day
        [6:17] body_region_distribution (11 values)
    """
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            dt = datetime(2024, 1, 1)

    month = dt.month

    dow_sin   = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos   = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    avg_patients = scanner_stats['avg_patients_per_day']
    region_dist  = scanner_stats['region_distribution']

    conditioning = np.zeros(17, dtype=np.float32)
    conditioning[0] = dow_sin
    conditioning[1] = dow_cos
    conditioning[2] = month_sin
    conditioning[3] = month_cos
    conditioning[4] = is_weekend
    conditioning[5] = avg_patients
    conditioning[6:17] = region_dist

    return conditioning


def extract_orchestration_samples(preprocessed_data, break_threshold_hours=1):
    """
    Extract orchestration training samples from preprocessed customer schedules.

    Each sample represents one day at one scanner, encoded as a sequence of
    body region tokens with BREAK tokens inserted where gaps exceed threshold.

    Returns
    -------
    samples : list of dict
        Keys: tokens, conditioning, scanner_idx, start_datetime,
              customer_id, date_str, num_patients
    scanner_to_idx : dict
        Maps customer_id (str) → int index
    """
    customer_schedules = preprocessed_data.get('customer_schedules', {})
    scanner_to_idx     = {cid: idx for idx, cid
                          in enumerate(sorted(customer_schedules.keys()))}
    scanner_stats      = _compute_scanner_stats(customer_schedules)
    samples            = []

    for customer_id, daily_schedules in customer_schedules.items():
        scanner_idx = scanner_to_idx[customer_id]
        stats       = scanner_stats[customer_id]

        for date_str, patients in daily_schedules.items():
            if not patients:
                continue

            tokens = []
            for i, patient in enumerate(patients):
                if i > 0:
                    prev_hour = patients[i - 1].get('hour_of_day', 8)
                    curr_hour = patient.get('hour_of_day', 8)
                    if curr_hour - prev_hour >= break_threshold_hours:
                        tokens.append(BREAK_TOKEN_ID)

                body_region_id = patient.get('body_region_id', 10)
                tokens.append(body_region_id)

            day_of_week  = patients[0].get('day_of_week', 0)
            conditioning = _build_orchestration_conditioning(date_str, day_of_week, stats)

            try:
                start_datetime = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    start_datetime = datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    continue

            samples.append({
                'tokens':         tokens,
                'conditioning':   conditioning,
                'scanner_idx':    scanner_idx,
                'start_datetime': start_datetime,
                'customer_id':    customer_id,
                'date_str':       date_str,
                'num_patients':   len(patients),
            })

    return samples, scanner_to_idx

# COMMAND ----------
# =============================================================================
# CELL 6: Extract orchestration samples
# =============================================================================

stub_preprocessed = {'customer_schedules': customer_schedules}
orch_samples, scanner_to_idx = extract_orchestration_samples(stub_preprocessed)

print(f"Orchestration samples: {len(orch_samples)}")
print(f"Scanners: {len(scanner_to_idx)} → {scanner_to_idx}")

if orch_samples:
    seq_lens      = [len(s['tokens']) for s in orch_samples]
    patient_cnts  = [s['num_patients'] for s in orch_samples]
    break_cnts    = [s['tokens'].count(BREAK_TOKEN_ID) for s in orch_samples]
    print(f"Sequence length: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"avg={np.mean(seq_lens):.1f}")
    print(f"Patients/day: min={min(patient_cnts)}, max={max(patient_cnts)}, "
          f"avg={np.mean(patient_cnts):.1f}")
    print(f"BREAKs/day: avg={np.mean(break_cnts):.2f}")
    print(f"Sample conditioning shape: {orch_samples[0]['conditioning'].shape}")

# COMMAND ----------
# =============================================================================
# CELL 7: Validation
# =============================================================================

print("Validating orchestration samples...")

VALID_ORCH_TOKENS = set(range(NUM_BODY_REGIONS)) | {BREAK_TOKEN_ID}

for i, s in enumerate(orch_samples):
    assert s['conditioning'].shape == (17,), \
        f"Sample {i}: conditioning shape {s['conditioning'].shape} != (17,)"
    bad_tokens = [t for t in s['tokens'] if t not in VALID_ORCH_TOKENS]
    assert not bad_tokens, f"Sample {i}: invalid tokens {bad_tokens}"
    assert isinstance(s['customer_id'], str), \
        f"Sample {i}: customer_id must be str, got {type(s['customer_id'])}"
    assert hasattr(s['start_datetime'], 'date'), \
        f"Sample {i}: start_datetime is not a datetime ({type(s['start_datetime'])})"

print(f"All validations passed for {len(orch_samples)} samples.")

# Validate customer_schedules keys are strings
for cid in customer_schedules:
    assert isinstance(cid, str), f"customer_schedules key '{cid}' is not a string"
print("customer_schedules key types OK.")

# COMMAND ----------
# =============================================================================
# CELL 8: Save to DBFS
# =============================================================================

orch_data = {
    'customer_schedules': customer_schedules,
    'daily_summaries':    daily_summaries,
    'orch_samples':       orch_samples,
    'scanner_to_idx':     scanner_to_idx,
}

with open(ORCH_OUTPUT, 'wb') as f:
    pickle.dump(orch_data, f)

print(f"Saved orchestration data to {ORCH_OUTPUT}")
print(f"  customer_schedules: {len(customer_schedules)} scanners")
print(f"  daily_summaries: {len(daily_summaries)} entries")
print(f"  orch_samples: {len(orch_samples)}")
