# Databricks notebook source
# spark_pipeline — Exchange Preprocessing (Spark-native)
#
# Same business logic as csv_pipeline/01_exchange_preprocessing.py.
# Key difference: instead of pulling all 10 serials' raw eventlog to the
# driver with toPandas() and processing them sequentially, this version uses
# groupBy('SerialNumber').applyInPandas() so each serial is processed in
# parallel on a Spark worker.  The driver only receives the small processed
# result, not the raw event stream.
#
# Output:  /dbfs/FileStore/spark_pipeline/exchange/DATA_{serial}.csv
#          (identical schema to csv_pipeline/01 output)

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import re
import os
import time
import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType,
)

os.makedirs(EXCHANGE_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {EXCHANGE_OUTPUT_DIR}")

_TIMINGS = []
def _timeit(label, t0):
    dt = time.perf_counter() - t0
    _TIMINGS.append((label, dt))
    print(f"[timing] step01  {label:<25} {dt:9.1f}s")
    return time.perf_counter()

# COMMAND ----------
# =============================================================================
# PANDAS HELPER FUNCTIONS — identical to csv_pipeline/01
# These run inside applyInPandas on Spark workers (not on driver).
# =============================================================================

def _interpatient(df):
    df = df[~df['text'].str.contains('AdjustSeq|ServiceSeq', case=False)]
    mask = (df['datetime'].shift(1) == df['datetime']) & \
           (df['sourceID'].shift(1) == df['sourceID'])
    df = df[~mask]

    filtered_df = pd.DataFrame(columns=['datetime', 'sourceID', 'text',
                                        'timediff', 'BodyPart', 'PatientID'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    start_block     = False
    start_idx       = 0
    prev_patient_id = None
    patients        = 0
    date0           = df['datetime'].iloc[0].date()
    first_pat       = True
    first_ptab      = None

    indices_100 = df.index[df['sourceID'] == 'MRI_MSR_100'].tolist()

    for index, row in df.iterrows():
        code = row['sourceID']
        text = row['text']
        date = row['datetime'].date()
        end_idx = None

        if date0 != date:
            patients    = 0
            date0       = date
            start_block = False

        if code == 'MRI_FRR_257' and first_pat:
            first_ptab = row['text'].split()[-1]

        if code == 'MRI_MSR_104' and not start_block:
            start_idx   = index
            start_block = True
            first_pat   = False

        if code == 'MRI_MSR_100' and start_block:
            end_idx = index

        if code == 'MRI_EXU_95' and start_block:
            patient_id = re.search(r'Anonymised Patient ID < (.*) ><', text).group(1)

            if patient_id == prev_patient_id:
                start_block = False
                continue

            prev_patient_id = re.search(r'Anonymised Patient ID < (.*) >', text).group(1)
            body_part       = re.search(r'with body part < (.*) > <', text).group(1)
            df_copy         = df.copy()

            if not end_idx:
                candidates = [x for x in indices_100 if x > start_idx]
                end_idx = min(candidates) if candidates else index

            block = df_copy.loc[start_idx:end_idx]
            if block.empty:
                block = df_copy.loc[start_idx:index]

            time_diff_seconds = (block['datetime'] -
                                 block['datetime'].iloc[0]).dt.total_seconds()
            block = block.assign(timediff=time_diff_seconds)

            patients += 1
            if patients > 1:
                filtered_df = pd.concat([filtered_df, block], ignore_index=True)

            exu95_row = pd.DataFrame({
                'sourceID':  [''],
                'datetime':  [''],
                'text':      [''],
                'BodyPart':  [body_part],
                'PatientID': [prev_patient_id],
            })
            try:
                for c in filtered_df.columns:
                    if c not in exu95_row.columns:
                        exu95_row[c] = pd.NA
                exu95_row = exu95_row[filtered_df.columns]
                filtered_df = pd.concat([filtered_df, exu95_row], ignore_index=True)
            except Exception as e:
                print(f"Marker row append failed: {e}")

            start_block = False

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df, first_ptab


def _join_events(df):
    indices_264 = df.index[df['sourceID'] == 'MRI_FRR_264'].tolist()
    indices_265 = df.index[df['sourceID'] == 'MRI_FRR_265'].tolist()

    for idx_264 in indices_264:
        if idx_264 + 1 in df.index:
            df.at[idx_264, 'text'] = (str(df.at[idx_264, 'text']) + ' ' +
                                      str(df.at[idx_264 + 1, 'text']))

    df.drop(indices_265, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _to_bincolumns(row):
    text = row['text']
    if pd.notna(text):
        row['ZAxisInPossible']   = 1 if 'ZAxisInPossible: 1'  in text else (0 if 'ZAxisInPossible: 0'  in text else np.nan)
        row['ZAxisOutPossible']  = 1 if 'ZAxisOutPossible: 1' in text else (0 if 'ZAxisOutPossible: 0' in text else np.nan)
        row['YAxisDownPossible'] = 1 if 'YAxisDownPossible: 1' in text else (0 if 'YAxisDownPossible: 0' in text else np.nan)
        row['YAxisUpPossible']   = 1 if 'YAxisUpPossible: 1'  in text else (0 if 'YAxisUpPossible: 0'  in text else np.nan)
    return row


def _ptab(df, first_ptab):
    start = True
    df['PTAB'] = np.nan
    for idx, row in df.iterrows():
        if start and idx > 0:
            df.at[idx, 'PTAB'] = first_ptab
        if row['sourceID'] == 'MRI_FRR_257':
            df.at[idx, 'PTAB'] = row['text'].split()[-1]
            start = False
    return df


def _expand_coils(coils_str):
    m = re.search(r'Connected coil elements: ([^.)]*)', coils_str)
    coil_elements_str = m.group(1) if m else ''
    elements   = coil_elements_str.split(',')
    result     = []
    prev_prefix = ''
    for element in elements:
        element = element.strip()
        if not element:
            continue
        if not element.isnumeric():
            prefix_m = re.match(r'[A-Za-z]+', element)
            if prefix_m:
                prefix      = prefix_m.group()
                prev_prefix = prefix
                if '-' not in element:
                    result.append(element)
                else:
                    numbers = re.findall(r'\d+', element)
                    if len(numbers) == 2:
                        for i in range(int(numbers[0]), int(numbers[1]) + 1):
                            result.append(f"{prefix}{i}")
            else:
                if '-' in element:
                    parts = element.split('-')
                    try:
                        for i in range(int(parts[0]), int(parts[1]) + 1):
                            result.append(f"{prev_prefix}{i}")
                    except ValueError:
                        pass
                else:
                    result.append(element)
        else:
            result.append(f"{prev_prefix}{element}")
    return result


def _coils(df):
    df_copy     = df.copy()
    total_coils = []
    for idx, row in df.iterrows():
        if row['sourceID'] == 'MRI_CCS_11':
            active_coils = _expand_coils(str(row['text']))
            for coil in active_coils:
                if coil not in df_copy.columns:
                    df_copy[coil] = np.nan
                    total_coils.append(coil)
                df_copy.at[idx, coil] = 1

    indices = df_copy.index[df_copy['sourceID'] == 'MRI_CCS_11'].tolist()
    for i in range(len(indices) - 1):
        s, e = indices[i], indices[i + 1]
        df_copy.loc[s + 1:e - 1, total_coils] = df_copy.loc[s, total_coils].values
    if indices:
        last = indices[-1]
        df_copy.loc[last + 1:, total_coils] = df_copy.loc[last, total_coils].values
    df_copy[total_coils] = df_copy[total_coils].fillna(0)
    return df_copy

# COMMAND ----------
# =============================================================================
# BROADCAST SMALL LOOKUP DATA
# =============================================================================

# Body group mapping (small Excel file, broadcast to all workers)
_df_body = pd.read_excel(BODY_GROUP_MAPPING_PATH)
_df_body['BodyGroup'] = _df_body['BodyGroup'].str.upper()
_df_body['BodyPart']  = _df_body['BodyPart'].str.upper()
if 'MRType' in _df_body.columns:
    _df_body.drop(columns='MRType', inplace=True)
print(f"Body mapping rows: {len(_df_body)}")

df_body_bc = spark.sparkContext.broadcast(_df_body.to_dict('records'))

# Python config constants captured as plain dicts (safe to use inside UDFs)
_SOURCEID_VOCAB_BC    = spark.sparkContext.broadcast(SOURCEID_VOCAB)
_BODY_REGION_TO_ID_BC = spark.sparkContext.broadcast(BODY_REGION_TO_ID)
_BODY_REGIONS_BC      = spark.sparkContext.broadcast(BODY_REGIONS)
_SERIAL_NUMBERS_BC    = spark.sparkContext.broadcast(SERIAL_NUMBERS)

# COMMAND ----------
# =============================================================================
# SPARK QUERY — stays as Spark, no toPandas on raw events
# =============================================================================

_t0 = time.perf_counter()

eventlog_spark = (
    spark.read.table(EVENTLOG_TABLE)
    .select(
        F.date_format(F.col("EventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("EventDateTime"),
        "SerialNumber",
        "MessageIdentification",
        "TimeZoneOffset",
        "Message",
        "Line",
    )
    .filter(F.col("SerialNumber").isin([str(s) for s in SERIAL_NUMBERS] +
                                       [int(s) for s in SERIAL_NUMBERS]))
    .withColumn(
        "AdjustedEventDateTime",
        F.timestamp_seconds(
            F.unix_timestamp(F.col("EventDateTime")) + F.col("TimeZoneOffset")
        ),
    )
    .filter(F.to_date(F.col("AdjustedEventDateTime")).between(DATE_START, DATE_END))
    .filter(F.col("MessageIdentification").isin(EXCHANGE_SOURCE_TYPES))
    .select(
        F.date_format(F.col("AdjustedEventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("datetime"),
        F.col("MessageIdentification").alias("sourceID"),
        F.col("Message").cast("string").alias("text"),
        "Line",
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
    )
    .repartition(len(SERIAL_NUMBERS), F.col("SerialNumber"))   # one partition per serial
)

print(f"Eventlog row count: {eventlog_spark.count():,}")
_t0 = _timeit('eventlog Spark query', _t0)

# COMMAND ----------
# =============================================================================
# applyInPandas SCHEMA  — fixed output columns (coil columns dropped; they are
# not used downstream by step 03 or step 05 in the exchange path)
# =============================================================================

_EXCHANGE_SCHEMA = StructType([
    StructField('SN',                 StringType(), True),
    StructField('customer_idx',       LongType(),   True),
    StructField('sample_idx',         LongType(),   True),
    StructField('step',               LongType(),   True),
    StructField('token_id',           LongType(),   True),
    StructField('token_name',         StringType(), True),
    StructField('BodyGroup_from',     LongType(),   True),
    StructField('BodyGroup_to',       LongType(),   True),
    StructField('BodyGroup_from_text', StringType(), True),
    StructField('BodyGroup_to_text',  StringType(), True),
    StructField('PatientID_from',     StringType(), True),
    StructField('PatientID_to',       StringType(), True),
    StructField('predicted_mu',       DoubleType(), True),
    StructField('predicted_sigma',    DoubleType(), True),
    StructField('sampled_duration',   DoubleType(), True),
    StructField('total_time',         DoubleType(), True),
    StructField('timediff',           DoubleType(), True),
    StructField('datetime',           StringType(), True),
    StructField('PTAB',               StringType(), True),
])

# COMMAND ----------
# =============================================================================
# PER-SERIAL PROCESSING FUNCTION  (runs on Spark workers, in parallel)
# =============================================================================

def process_exchange_serial(pdf):
    """
    Process all events for a single SerialNumber group.

    Called once per SerialNumber by applyInPandas.  Runs on a Spark worker
    (not on the driver), so 10 serials are processed simultaneously.

    Returns rows with the fixed _EXCHANGE_SCHEMA (no Age/Weight/Height/Direction;
    those are joined afterwards as a Spark join against examination_workflow).
    """
    import re, numpy as np, pandas as pd

    if pdf.empty:
        return pd.DataFrame(columns=[f.name for f in _EXCHANGE_SCHEMA])

    serial_number   = int(pdf['SerialNumber'].iloc[0])
    serial_numbers  = _SERIAL_NUMBERS_BC.value
    sourceid_vocab  = _SOURCEID_VOCAB_BC.value
    body_region_to_id = _BODY_REGION_TO_ID_BC.value
    body_regions    = _BODY_REGIONS_BC.value
    df_body         = pd.DataFrame(df_body_bc.value)

    customer_idx = serial_numbers.index(serial_number) if serial_number in serial_numbers else 0

    _empty = pd.DataFrame(columns=[f.name for f in _EXCHANGE_SCHEMA])

    # Sort by datetime → Line (preserves event order within same timestamp)
    pdf['datetime'] = pd.to_datetime(pdf['datetime'])
    df_sorted = (
        pdf.groupby('datetime', group_keys=False)
           .apply(lambda g: g.sort_values('Line'))
           .reset_index(drop=True)
    )

    df_filter, first_ptab = _interpatient(df_sorted)
    if df_filter.empty:
        return _empty

    df_join = _join_events(df_filter)
    df_join.drop_duplicates(
        subset=['sourceID', 'datetime', 'PatientID'], keep='first', inplace=True
    )
    df_join.reset_index(drop=True, inplace=True)

    df_bool = df_join.apply(_to_bincolumns, axis=1)
    df_ptab = _ptab(df_bool, first_ptab)

    for c in ['ZAxisInPossible', 'ZAxisOutPossible', 'YAxisDownPossible', 'YAxisUpPossible', 'PTAB']:
        if c in df_ptab.columns:
            df_ptab[c] = df_ptab[c].ffill()

    df_coils = _coils(df_ptab)
    df_coils['SN'] = str(serial_number)

    # Body group merge
    exchange = df_coils.copy()
    if not exchange.empty and 'BodyPart' in exchange.columns:
        exchange['BodyPart'] = exchange['BodyPart'].str.upper()
        result = pd.merge(exchange, df_body, on='BodyPart', how='left')
    else:
        result = exchange.copy()

    body_parts_mask = result['BodyPart'].notnull()
    for col_name, src in [('BodyPart_from', 'BodyPart'),
                           ('BodyPart_to',   'BodyPart'),
                           ('BodyGroup_from', 'BodyGroup'),
                           ('BodyGroup_to',   'BodyGroup'),
                           ('PatientID_from', 'PatientID'),
                           ('PatientID_to',   'PatientID')]:
        if src in result.columns:
            s = result[src].where(body_parts_mask)
            result[col_name] = s.ffill() if col_name.endswith('_from') else s.bfill()

    drop_cols = [c for c in ['BodyPart', 'PatientID', 'BodyGroup'] if c in result.columns]
    df_filter_final = result[result['BodyPart'].isnull()].drop(columns=drop_cols)

    df_out = df_filter_final.copy()
    df_out['customer_idx'] = customer_idx

    df_out['sample_idx'] = (
        df_out['timediff'] < df_out['timediff'].shift(1).fillna(-1)
    ).cumsum()
    df_out['step'] = df_out.groupby('sample_idx').cumcount()

    df_out['token_id']   = df_out['sourceID'].map(lambda x: sourceid_vocab.get(str(x), 17))
    df_out['token_name'] = df_out['sourceID']

    df_out['BodyGroup_from_text'] = df_out['BodyGroup_from'].astype(str).str.upper()
    df_out['BodyGroup_to_text']   = df_out['BodyGroup_to'].astype(str).str.upper()
    df_out['BodyGroup_from'] = df_out['BodyGroup_from_text'].map(
        lambda x: body_region_to_id.get(x, len(body_regions) - 1))
    df_out['BodyGroup_to']   = df_out['BodyGroup_to_text'].map(
        lambda x: body_region_to_id.get(x, len(body_regions) - 1))

    df_out['total_time'] = df_out.groupby('sample_idx')['timediff'].transform('max')

    df_out['predicted_mu']     = float('nan')
    df_out['predicted_sigma']  = float('nan')
    df_out['sampled_duration'] = float('nan')

    # Return only the fixed output columns
    _out_cols = [
        'SN', 'customer_idx', 'sample_idx', 'step', 'token_id', 'token_name',
        'BodyGroup_from', 'BodyGroup_to', 'BodyGroup_from_text', 'BodyGroup_to_text',
        'PatientID_from', 'PatientID_to',
        'predicted_mu', 'predicted_sigma', 'sampled_duration', 'total_time',
        'timediff', 'datetime', 'PTAB',
    ]
    df_out = df_out[[c for c in _out_cols if c in df_out.columns]]

    # Ensure all schema columns exist (fill missing with None)
    for f in _EXCHANGE_SCHEMA:
        if f.name not in df_out.columns:
            df_out[f.name] = None

    # Cast LongType columns to avoid pandas int64 overflow errors
    for f in _EXCHANGE_SCHEMA:
        if str(f.dataType) == 'LongType()':
            df_out[f.name] = pd.to_numeric(df_out[f.name], errors='coerce').fillna(0).astype('int64')
        elif str(f.dataType) == 'DoubleType()':
            df_out[f.name] = pd.to_numeric(df_out[f.name], errors='coerce')

    return df_out[[f.name for f in _EXCHANGE_SCHEMA]]

# COMMAND ----------
# =============================================================================
# PARALLEL PROCESSING — each serial runs simultaneously on a Spark worker
# =============================================================================

_t = time.perf_counter()

exchange_spark = (
    eventlog_spark
    .groupBy('SerialNumber')
    .applyInPandas(process_exchange_serial, schema=_EXCHANGE_SCHEMA)
)

print(f"Exchange rows produced: {exchange_spark.count():,}")
_t = _timeit('applyInPandas (parallel, all serials)', _t)

# COMMAND ----------
# =============================================================================
# EXAMINATION WORKFLOW JOIN  — single Spark join, no per-serial toPandas
# =============================================================================

_t = time.perf_counter()

start_int = int(DATE_START.replace('-', ''))
end_int   = int(DATE_END.replace('-', ''))

exam_spark = (
    spark.read.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
    .filter(
        (F.col("Year").cast("int") * 10000 +
         F.col("Month").cast("int") * 100 +
         F.col("Day").cast("int")).between(start_int, end_int)
    )
    .withColumn("PatientId", col("WorkflowValues")["PatientId"])
    .dropDuplicates(["PatientId"])
    .select(
        "PatientId",
        col("WorkflowValues")["Position"].alias("Position"),
        col("WorkflowValues")["Weight"].alias("Weight"),
        col("WorkflowValues")["Age"].alias("Age"),
        col("WorkflowValues")["Height"].alias("Height"),
        col("WorkflowValues")["Direction"].alias("Direction"),
    )
)

# Broadcast hint: exam_spark is small (unique patients)
exchange_final = (
    exchange_spark
    .join(F.broadcast(exam_spark),
          exchange_spark['PatientID_to'] == exam_spark['PatientId'],
          how='left')
    .drop('PatientId')
)

_t = _timeit('exam workflow Spark join', _t)

# COMMAND ----------
# =============================================================================
# COLLECT RESULTS & WRITE CSVS
# Only the processed result comes to driver (not raw eventlog).
# =============================================================================

_t = time.perf_counter()

all_exchange_pd = exchange_final.toPandas()
_t = _timeit('result toPandas (small processed data)', _t)

print(f"Total exchange rows: {len(all_exchange_pd):,}")

for serial_number in SERIAL_NUMBERS:
    serial_df = all_exchange_pd[all_exchange_pd['SN'] == str(serial_number)].copy()
    if serial_df.empty:
        print(f"  {serial_number}: no rows")
        continue

    # Reorder to match csv_pipeline schema exactly
    _out_cols = [
        'SN', 'customer_idx', 'sample_idx', 'step', 'token_id', 'token_name',
        'BodyGroup_from', 'BodyGroup_to', 'BodyGroup_from_text', 'BodyGroup_to_text',
        'PatientID_from', 'PatientID_to',
        'predicted_mu', 'predicted_sigma', 'sampled_duration', 'total_time',
        'timediff', 'datetime',
        'Age', 'Weight', 'Height', 'Direction', 'PTAB',
    ]
    serial_df = serial_df[[c for c in _out_cols if c in serial_df.columns]]

    csv_path = f"{EXCHANGE_OUTPUT_DIR}/DATA_{serial_number}.csv"
    serial_df.to_csv(csv_path, index=False, header=True)
    print(f"  {serial_number}: {len(serial_df):,} rows → {csv_path}")

_t = _timeit('write CSVs', _t)

print("\nExchange preprocessing complete.")
print("\n" + "-"*60)
print("  TIMING BREAKDOWN")
print("-"*60)
for _label, _dt in _TIMINGS:
    print(f"[timing] step01  {_label:<25} {_dt:9.1f}s")
print(f"[timing] step01  {'TOTAL':<25} {sum(d for _, d in _TIMINGS):9.1f}s")
