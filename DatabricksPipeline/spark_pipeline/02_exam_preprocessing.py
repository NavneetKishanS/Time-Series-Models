# Databricks notebook source
# spark_pipeline — Exam Preprocessing (Spark-native)
#
# Same business logic as csv_pipeline/02_exam_preprocessing.py.
# Key difference: instead of 10 sequential per-serial Spark→toPandas calls,
# this version queries ALL serials in a single Spark scan, then uses
# groupBy('SerialNumber').applyInPandas() so each serial is processed in
# parallel on a Spark worker.  Examination workflow demographics are added
# via a single Spark join instead of 10 per-serial toPandas calls.
#
# Output:  /dbfs/FileStore/spark_pipeline/exam/DATA_{serial}.csv
#          (identical schema to csv_pipeline/02 output)

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import re
import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType, BooleanType,
)

os.makedirs(EXAM_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {EXAM_OUTPUT_DIR}")

_TIMINGS = []
def _timeit(label, t0):
    dt = time.perf_counter() - t0
    _TIMINGS.append((label, dt))
    print(f"[timing] step02  {label:<25} {dt:9.1f}s")
    return time.perf_counter()

# COMMAND ----------
# =============================================================================
# PANDAS HELPER FUNCTIONS — identical to csv_pipeline/02
# These run inside applyInPandas on Spark workers (not on driver).
# =============================================================================

def _ptab(df, first_ptab):
    start    = True
    df['PTAB'] = np.nan
    for idx, row in df.iterrows():
        if start and idx > 0:
            df.at[idx, 'PTAB'] = first_ptab
        if row['sourceID'] == 'MRI_FRR_257':
            df.at[idx, 'PTAB'] = row['text'].split()[-1]
            start = False
    return df


def _expand_coils_exam(coils_str, group):
    connected_coil = coils_str.split(',')
    sc   = []
    coil = ''
    for elem in connected_coil:
        if not elem:
            continue
        if elem.endswith('-') or elem.endswith('.'):
            elem = elem[:-1]
        if not elem.isnumeric():
            elem = elem.strip()
            coil = elem[:2]
            if '-' in elem:
                st    = elem[2:]
                st_sp = st.split('-')
                if len(st_sp) > 1:
                    try:
                        for i in range(int(st_sp[0]), int(st_sp[1]) + 1):
                            sc.append(f'#{group}_{coil}{i}')
                    except ValueError:
                        pass
            else:
                sc.append(f'#{group}_{elem}')
        else:
            sc.append(f'#{group}_{coil}{elem}')
    return sc


def _get_seq(text):
    sequence_type = re.search(r"Sequence: '(.*)'", text).group(1)
    protocol      = re.search(r"Protocol: '(.*)',", text).group(1)
    return sequence_type, protocol


def _get_body_patient(text):
    body = np.nan
    try:
        body = re.search(r'with body part < (.*) >', text).group(1)
        if ' ' in body:
            try:
                body = re.search(r'with body part < (.*) > <', text).group(1)
            except AttributeError:
                pass
    except AttributeError:
        pass
    try:
        patient_id   = re.search(r'Anonymised Patient ID < (.*) >', text).group(1)
        flag_patient = True
    except AttributeError:
        patient_id   = np.nan
        flag_patient = False
    return body, patient_id, flag_patient


def _data_transformation(df_):
    df_  = df_[~df_['text'].str.contains('AdjustSeq|ServiceSeq', case=False)]
    df_  = df_.reset_index(drop=True)

    df   = pd.DataFrame(index=range(df_.shape[0]), columns=['Sequence'])
    df['Sequence'] = 'False'

    finish_events         = ['MRI_MSR_104', 'MRI_MSR_18', 'MRI_FRR_18', 'MRI_MSR_21',
                             'MRI_MSR_34',  'MRI_MSR_26', 'MRI_MSR_22', 'MRI_MSR_40',
                             'MRI_MSR_25',  'MRI_MSR_24']
    finish_events_reduced = ['MRI_MSR_104', 'MRI_MSR_34', 'MRI_MSR_26',
                             'MRI_MSR_22',  'MRI_MSR_40', 'MRI_MSR_25', 'MRI_MSR_24']

    sc           = []
    coilList     = []
    start        = False
    flag_dot     = False
    startEXU     = False
    outEXU       = False
    conCoils     = ''
    body         = np.nan
    patient_id   = np.nan
    flag_patient = False
    datetime_start   = None
    start_index      = 0
    start_index_prev = 0

    for i, row in df_.iterrows():
        if row.sourceID == 'MRI_EXU_89':
            flag_dot = True
        if row.sourceID == 'MRI_CCS_1008':
            coil_ids = re.findall(r"CoilID\s+'(\w+)'", row.text)
            conCoils = '-'.join(coil_ids)
        if row.sourceID == 'MRI_MSR_100':
            start = True
            try:
                seq, protocol = _get_seq(row.text)
            except (AttributeError, TypeError):
                start = False
                continue
            datetime_start   = row.datetime
            start_index_prev = start_index
            start_index      = i
            startEXU         = False
        if start and row.sourceID == 'MRI_EXU_95':
            body, patient_id, flag_patient = _get_body_patient(row.text)
            startEXU = True
        if start and row.sourceID == 'MRI_MSR_101':
            s = row.text
            try:
                group         = re.search(r'#(\d):', s).group(1)
                selected_coil = re.search(r": '(.*)'\)", s).group(1)
                expanded      = _expand_coils_exam(selected_coil, group)
                sc = sc + expanded if sc else expanded
            except (AttributeError, TypeError):
                pass
        if row.sourceID in finish_events:
            if start and row.sourceID in finish_events_reduced:
                df_.at[i, 'startTime'] = datetime_start
                for elem in sc:
                    if elem not in df_.columns:
                        coilList.append(elem)
                    df_.at[i, elem] = True
                df.at[i, 'Sequence'] = seq
                df.at[i, 'Protocol'] = protocol
                finish_map = {
                    'MRI_MSR_104': 'Successful',
                    'MRI_MSR_34':  'Stopped by User',
                    'MRI_MSR_26':  'Scanning Error',
                    'MRI_MSR_22':  'Start MeasSys Failed',
                    'MRI_MSR_24':  'Stopped by Scanner',
                    'MRI_MSR_25':  'Stopped by ImageR',
                    'MRI_MSR_40':  'Stopped by CoilChange',
                }
                df.at[i, 'FinishEvent'] = finish_map.get(row.sourceID, 'Unknown')
                if not startEXU:
                    for offset in (-2, -1):
                        k = start_index + offset
                        if 0 <= k < len(df_) and df_['sourceID'].iloc[k] == 'MRI_EXU_95':
                            outEXU = True
                            break
                    if not outEXU:
                        try:
                            curr_t = pd.to_datetime(df_['datetime'].iloc[start_index])
                            prev_t = pd.to_datetime(df_['datetime'].iloc[start_index_prev])
                            if (curr_t - prev_t) <= pd.Timedelta(minutes=4) and pd.notna(body):
                                startEXU = True
                        except (ValueError, TypeError, KeyError):
                            pass
                if outEXU:
                    try:
                        body, patient_id, flag_patient = _get_body_patient(df_.iloc[k].text)
                        startEXU = True
                    except Exception:
                        pass
                    outEXU = False
                if startEXU:
                    df_.at[i, 'BodyPart'] = body
                    if flag_patient:
                        df_.at[i, 'PatientID'] = patient_id
                    else:
                        df_.at[i, 'MissingPatientID'] = True
                else:
                    df_.at[i, 'MissingBodyPart']  = True
                    df_.at[i, 'MissingPatientID'] = True
                df_.at[i, 'ConnectedCoils'] = conCoils if conCoils else np.nan
                if not conCoils:
                    df_.at[i, 'MissingConCoils'] = True
                df_.at[i, 'DOT'] = flag_dot
                df.fillna(False, inplace=True)
            start    = False
            startEXU = False
            sc       = []
        if row.sourceID == 'MRI_MPT_1005':
            flag_dot = False

    df_out = pd.concat([df_, df], axis=1)
    return df_out, coilList


def _measurement_time(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_sorted = df.sort_values(by='datetime')
    df_sorted = df_sorted.drop(df_sorted[df_sorted['Sequence'] == 'False'].index)
    df_sorted.rename(columns={'datetime': 'endTime'}, inplace=True)
    df_sorted['startTime'] = pd.to_datetime(df_sorted['startTime'])
    df_sorted['duration']  = (df_sorted['endTime'] -
                               df_sorted['startTime']).dt.total_seconds()
    return df_sorted


def _match_bp(df_info, df_bp):
    df_info = df_info.copy()
    df_bp   = df_bp.copy()
    df_info['bp_lower'] = df_info['BodyPart'].astype(str).str.lower()
    df_bp['bp_lower']   = df_bp['BodyPart'].astype(str).str.lower()
    df_bp.drop(columns='BodyPart', inplace=True)
    merged = pd.merge(df_info, df_bp, on='bp_lower', how='left')
    merged['BodyGroup'] = merged['BodyGroup'].fillna('Unknown')
    merged.drop(columns='bp_lower', inplace=True)
    return merged

# COMMAND ----------
# =============================================================================
# BROADCAST SMALL LOOKUP DATA
# =============================================================================

_df_bp = pd.read_excel(BODY_GROUP_MAPPING_PATH)
if 'MRType' in _df_bp.columns:
    _df_bp.drop(columns='MRType', inplace=True)
print(f"Body mapping rows: {len(_df_bp)}")

df_bp_bc = spark.sparkContext.broadcast(_df_bp.to_dict('records'))
_SERIAL_NUMBERS_BC = spark.sparkContext.broadcast(SERIAL_NUMBERS)

# COMMAND ----------
# =============================================================================
# SPARK QUERY — ALL serials in one scan (no per-serial loop)
# =============================================================================

_t = time.perf_counter()

eventlog_all = (
    spark.read.table(EVENTLOG_TABLE)
    .select(
        F.date_format(F.col("EventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("EventDateTime"),
        "SerialNumber",
        "MessageIdentification",
        "TimeZoneOffset",
        "Message",
        "Line",
    )
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
    .withColumn(
        "AdjustedEventDateTime",
        F.timestamp_seconds(
            F.unix_timestamp(F.col("EventDateTime")) + F.col("TimeZoneOffset")
        ),
    )
    .filter(F.to_date(F.col("AdjustedEventDateTime")).between(DATE_START, DATE_END))
    .filter(
        F.col("MessageIdentification").startswith("MRI_MSR") |
        F.col("MessageIdentification").isin(EXAM_EXTRA_SOURCE_TYPES)
    )
    .select(
        F.date_format(F.col("AdjustedEventDateTime"), "yyyy-MM-dd HH:mm:ss").alias("datetime"),
        F.col("MessageIdentification").alias("sourceID"),
        F.col("Message").cast("string").alias("text"),
        "Line",
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
    )
    .repartition(len(SERIAL_NUMBERS), F.col("SerialNumber"))
)

print(f"Eventlog row count: {eventlog_all.count():,}")
_t = _timeit('eventlog Spark query (all serials)', _t)

# COMMAND ----------
# =============================================================================
# applyInPandas OUTPUT SCHEMA
#
# Coil columns are included as a fixed superset (EXAM_COIL_COLS from config).
# Dynamic per-serial coil names produced by data_transformation are mapped to
# this fixed set; unknown coil names are discarded.
# =============================================================================

_BASE_FIELDS = [
    StructField('SN',              StringType(),  True),
    StructField('customer_idx',    LongType(),    True),
    StructField('sample_idx',      LongType(),    True),
    StructField('BodyPart',        StringType(),  True),
    StructField('BodyGroup',       StringType(),  True),
    StructField('PatientID',       StringType(),  True),
    StructField('Sequence',        StringType(),  True),
    StructField('Protocol',        StringType(),  True),
    StructField('ConnectedCoils',  StringType(),  True),
    StructField('DOT',             BooleanType(), True),
    StructField('FinishEvent',     StringType(),  True),
    StructField('startTime',       StringType(),  True),
    StructField('endTime',         StringType(),  True),
    StructField('duration',        LongType(),    True),
    StructField('timediff',        DoubleType(),  True),
    StructField('pauseTime',       DoubleType(),  True),
    StructField('StepCount',       LongType(),    True),
    StructField('PTAB',            DoubleType(),  True),
    StructField('MissingBodyPart', BooleanType(), True),
    StructField('MissingPatientID',BooleanType(), True),
    StructField('predicted_mu',    DoubleType(),  True),
    StructField('predicted_sigma', DoubleType(),  True),
    StructField('sampled_duration',DoubleType(),  True),
]
_COIL_FIELDS = [StructField(c, BooleanType(), True) for c in EXAM_COIL_COLS]

_EXAM_SCHEMA = StructType(_BASE_FIELDS + _COIL_FIELDS)

# COMMAND ----------
# =============================================================================
# PER-SERIAL PROCESSING FUNCTION  (runs on Spark workers, in parallel)
# =============================================================================

def process_exam_serial(pdf):
    """
    Process all events for a single SerialNumber group.

    Called once per SerialNumber by applyInPandas.  Runs in parallel on
    Spark workers.  Returns measurement-level rows (one per completed scan)
    with the fixed _EXAM_SCHEMA.
    """
    import re, numpy as np, pandas as pd

    _empty = pd.DataFrame(columns=[f.name for f in _EXAM_SCHEMA])

    if pdf.empty:
        return _empty

    serial_number   = int(pdf['SerialNumber'].iloc[0])
    serial_numbers  = _SERIAL_NUMBERS_BC.value
    df_bp           = pd.DataFrame(df_bp_bc.value)
    customer_idx    = serial_numbers.index(serial_number) if serial_number in serial_numbers else 0
    exam_coil_cols  = [f'#0_{c}' for c in [
        'BC', 'SP1','SP2','SP3','SP4','SP5','SP6','SP7','SP8', '15K',
        'HW1','HW2','HW3', 'HE1','HE2','HE3','HE4', 'NE1','NE2', 'SHL',
        'BO1','BO2','BO3', 'FA','TO','FS', 'PA1','PA2','PA3','PA4','PA5','PA6', 'SN',
    ]]

    pandas_df = pdf.sort_values(['datetime', 'Line']).reset_index(drop=True)

    # PTAB extraction
    pandas_df          = _ptab(pandas_df, 0)
    pandas_df['PTAB']  = pandas_df['PTAB'].ffill().bfill()

    df_ini, coil_list = _data_transformation(pandas_df)

    finish_keep = ['MRI_MSR_104', 'MRI_MSR_34', 'MRI_MSR_26',
                   'MRI_MSR_22',  'MRI_MSR_40', 'MRI_MSR_25', 'MRI_MSR_24']
    df_select = df_ini[
        df_ini['sourceID'].isin(finish_keep) & (df_ini['FinishEvent'] != False)
    ]
    df_pre = df_select.copy()
    df_pre.drop(columns=['sourceID', 'text'], axis=1, inplace=True, errors='ignore')

    df_sorted = _measurement_time(df_pre)
    df_sorted.fillna(False, inplace=True)
    df_sorted['SN'] = str(serial_number)
    df_sorted['FinishEvent'] = df_sorted['FinishEvent'].fillna(False)
    df_sorted = df_sorted[df_sorted['FinishEvent'] != False]

    if df_sorted.empty:
        return _empty

    df = df_sorted.copy()

    if 'PTAB' in df_sorted.columns:
        df['PTAB'] = df_sorted['PTAB'].values

    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime']   = pd.to_datetime(df['endTime'])

    s1  = df['PatientID'].replace({'False': np.nan})
    s1b = df['BodyPart'].replace({'False': np.nan})
    mask1 = (
        (df['PatientID'] == 'False') &
        (df['Sequence'].str.contains('Scout', case=False) |
         df['Protocol'].str.contains('localizer|Scout', case=False))
    )
    df.loc[mask1, 'PatientID'] = s1.bfill()
    df.loc[mask1, 'BodyPart']  = s1b.bfill()
    s3  = s1.ffill()
    s3b = s1b.ffill()
    mask2 = s3.eq(s1.bfill())
    df.loc[mask2, 'PatientID'] = s3
    df.loc[mask2, 'BodyPart']  = s3b
    df['PatientID'] = df['PatientID'].ffill()
    df['BodyPart']  = df['BodyPart'].ffill()
    mask3 = (df['PatientID'] == 'False') & \
            (df['ConnectedCoils'] == df['ConnectedCoils'].shift())
    df.loc[mask3, 'PatientID'] = df['PatientID'].shift()
    df.loc[mask3, 'BodyPart']  = df['BodyPart'].shift()
    df['PatientID'] = df['PatientID'].ffill()
    df['BodyPart']  = df['BodyPart'].ffill()
    mask4 = (
        (df['PatientID'] == 'False') &
        ((df['endTime'].shift(-1) - df['startTime']) <= pd.Timedelta(minutes=3))
    )
    df.loc[mask4, 'PatientID'] = df['PatientID'].shift(-1)
    df.loc[mask4, 'BodyPart']  = df['BodyPart'].shift(-1)

    mask5   = (df['duration'] == 0) & (df['FinishEvent'] == 'Successful')
    df      = df[~mask5]
    df      = df[df['duration'] <= 4000].copy()

    df = _match_bp(df, df_bp)

    df['pauseTime'] = (df['startTime'].shift(-1) - df['endTime']).dt.total_seconds()
    df['StepCount'] = df.groupby('PatientID').cumcount() + 1
    df.loc[df['PatientID'] != df['PatientID'].shift(-1), 'pauseTime'] = 0

    df = df.sort_values('startTime').reset_index(drop=True)
    df['timediff'] = df['startTime'].diff().dt.total_seconds().fillna(0)

    df['customer_idx'] = customer_idx
    df['sample_idx']   = (
        df['PatientID'] != df['PatientID'].shift(1, fill_value='__START__')
    ).cumsum() - 1

    df['predicted_mu']     = float('nan')
    df['predicted_sigma']  = float('nan')
    df['sampled_duration'] = float('nan')

    df['startTime'] = df['startTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['endTime']   = df['endTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['duration']  = df['duration'].astype('int64')

    # Ensure fixed coil columns (set to False if not present)
    for c in exam_coil_cols:
        if c not in df.columns:
            df[c] = False
        else:
            df[c] = df[c].fillna(False).astype(bool)

    # Ensure all schema columns present
    for f in _EXAM_SCHEMA:
        if f.name not in df.columns:
            df[f.name] = None

    # Type coercions for Spark schema
    for f in _EXAM_SCHEMA:
        if str(f.dataType) == 'LongType()':
            df[f.name] = pd.to_numeric(df[f.name], errors='coerce').fillna(0).astype('int64')
        elif str(f.dataType) == 'DoubleType()':
            df[f.name] = pd.to_numeric(df[f.name], errors='coerce')
        elif str(f.dataType) == 'BooleanType()':
            df[f.name] = df[f.name].fillna(False).astype(bool)
        elif str(f.dataType) == 'StringType()':
            df[f.name] = df[f.name].astype(str).where(df[f.name].notna(), None)

    return df[[f.name for f in _EXAM_SCHEMA]]

# COMMAND ----------
# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

_t = time.perf_counter()

exam_spark = (
    eventlog_all
    .groupBy('SerialNumber')
    .applyInPandas(process_exam_serial, schema=_EXAM_SCHEMA)
)

print(f"Exam rows produced: {exam_spark.count():,}")
_t = _timeit('applyInPandas (parallel, all serials)', _t)

# COMMAND ----------
# =============================================================================
# EXAMINATION WORKFLOW JOIN  — single Spark join for all demographics
# =============================================================================

_t = time.perf_counter()

start_int = int(DATE_START.replace('-', ''))
end_int   = int(DATE_END.replace('-', ''))

exam_workflow = (
    spark.read.table(EXAMINATION_TABLE)
    .filter(F.col("SerialNumber").isin([int(s) for s in SERIAL_NUMBERS]))
    .filter(
        (F.col("Year").cast("int") * 10000 +
         F.col("Month").cast("int") * 100 +
         F.col("Day").cast("int")).between(start_int, end_int)
    )
    .withColumn("PatientId", F.col("WorkflowValues")["PatientId"])
    .withColumn("Age",       F.col("WorkflowValues")["Age"])
    .withColumn("Weight",    F.col("WorkflowValues")["Weight"])
    .withColumn("Height",    F.col("WorkflowValues")["Height"])
    .withColumn("Direction", F.col("WorkflowValues")["Direction"])
    .select("PatientId", "Age", "Weight", "Height", "Direction")
    .dropDuplicates(["PatientId"])
)

exam_final = (
    exam_spark
    .join(F.broadcast(exam_workflow),
          exam_spark['PatientID'] == exam_workflow['PatientId'],
          how='left')
    .drop('PatientId')
)

_t = _timeit('exam workflow Spark join', _t)

# COMMAND ----------
# =============================================================================
# COLLECT & WRITE CSVS
# =============================================================================

_t = time.perf_counter()

all_exam_pd = exam_final.toPandas()
_t = _timeit('result toPandas (small processed data)', _t)

print(f"Total exam rows: {len(all_exam_pd):,}")

for serial_number in SERIAL_NUMBERS:
    serial_df = all_exam_pd[all_exam_pd['SN'] == str(serial_number)].copy()
    if serial_df.empty:
        print(f"  {serial_number}: no rows")
        continue

    csv_path = f"{EXAM_OUTPUT_DIR}/DATA_{serial_number}.csv"
    serial_df.to_csv(csv_path, index=False, header=True)
    print(f"  {serial_number}: {len(serial_df):,} rows → {csv_path}")

_t = _timeit('write CSVs', _t)

print("\nExam preprocessing complete.")
print("\n" + "-"*60)
print("  TIMING BREAKDOWN")
print("-"*60)
for _label, _dt in _TIMINGS:
    print(f"[timing] step02  {_label:<25} {_dt:9.1f}s")
print(f"[timing] step02  {'TOTAL':<25} {sum(d for _, d in _TIMINGS):9.1f}s")
