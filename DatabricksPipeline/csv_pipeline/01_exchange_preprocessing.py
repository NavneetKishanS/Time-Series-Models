# Databricks notebook — Exchange CSV Preprocessing
#
# Follows Patient_Exchange.ipynb workflow exactly.
# Loops over all 10 serial numbers and saves one CSV per serial to:
#   /dbfs/FileStore/csv_pipeline/exchange/DATA_{serial}.csv
#
# Runtime: << 1 hour for 10 serials over a 1-month window.

# COMMAND ----------
%pip install openpyxl

# COMMAND ----------
%run ./config

# COMMAND ----------

import re
import os
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

os.makedirs(EXCHANGE_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {EXCHANGE_OUTPUT_DIR}")

# COMMAND ----------
# =============================================================================
# HELPER FUNCTIONS  (faithful port of Patient_Exchange.ipynb)
# =============================================================================

def interpatient(df):
    """
    Extract blocks of events that occur between consecutive patient examinations.

    A block starts after MRI_MSR_104 (end of previous measurement) and ends at
    the next MRI_MSR_100 (start of next measurement).  The block is kept only
    when MRI_EXU_95 detects a *different* PatientID than the previous one.

    Returns:
        filtered_df  — rows that fall inside inter-patient exchange periods
        first_ptab   — PTAB value from the first MRI_FRR_257 of the first patient
    """
    df = df[~df['text'].str.contains('AdjustSeq|ServiceSeq', case=False)]
    # Drop consecutive duplicate (datetime, sourceID) pairs
    mask = (df['datetime'].shift(1) == df['datetime']) & \
           (df['sourceID'].shift(1) == df['sourceID'])
    df = df[~mask]

    filtered_df = pd.DataFrame(columns=['datetime', 'sourceID', 'text',
                                        'timediff', 'BodyPart', 'PatientID'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    start_block    = False
    start_idx      = 0
    prev_patient_id = None
    patients       = 0
    date0          = df['datetime'].iloc[0].date()
    first_pat      = True
    first_ptab     = None

    for index, row in df.iterrows():
        code = row['sourceID']
        text = row['text']
        date = row['datetime'].date()
        end_idx = None
        indices_100 = df.index[df['sourceID'] == 'MRI_MSR_100'].tolist()

        if date0 != date:           # New calendar day — reset per-day counters
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
                # Same patient → not an exchange block
                start_block = False
                continue

            prev_patient_id = re.search(r'Anonymised Patient ID < (.*) >', text).group(1)
            body_part       = re.search(r'with body part < (.*) > <', text).group(1)
            df_copy         = df.copy()

            # If MRI_EXU_95 arrived before MRI_MSR_100, find the nearest
            # MRI_MSR_100 after the start flag
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

            # Append a marker row with the incoming patient's info
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


def join_events(df):
    """
    Combine each MRI_FRR_264 message with the immediately following MRI_FRR_265
    message into a single row, then drop the MRI_FRR_265 rows.
    """
    indices_264 = df.index[df['sourceID'] == 'MRI_FRR_264'].tolist()
    indices_265 = df.index[df['sourceID'] == 'MRI_FRR_265'].tolist()

    for idx_264 in indices_264:
        if idx_264 + 1 in df.index:
            df.at[idx_264, 'text'] = (str(df.at[idx_264, 'text']) + ' ' +
                                      str(df.at[idx_264 + 1, 'text']))

    df.drop(indices_265, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Validate pairing
    invalid = [idx for idx in indices_264 if idx + 1 not in indices_265]
    if invalid:
        print(f"Warning: {len(invalid)} MRI_FRR_264 not followed by MRI_FRR_265 "
              f"at indices: {invalid}")
    else:
        print(f"All {len(indices_264)} MRI_FRR_264 correctly paired with MRI_FRR_265")
    return df


def to_bincolumns(row):
    """Parse MRI_FRR_264/265 combined text into binary axis-flag columns."""
    text = row['text']
    if pd.notna(text):
        z_in  = 1 if 'ZAxisInPossible: 1'  in text else (0 if 'ZAxisInPossible: 0'  in text else np.nan)
        z_out = 1 if 'ZAxisOutPossible: 1' in text else (0 if 'ZAxisOutPossible: 0' in text else np.nan)
        y_dn  = 1 if 'YAxisDownPossible: 1' in text else (0 if 'YAxisDownPossible: 0' in text else np.nan)
        y_up  = 1 if 'YAxisUpPossible: 1'  in text else (0 if 'YAxisUpPossible: 0'  in text else np.nan)
        row['ZAxisInPossible']   = z_in
        row['ZAxisOutPossible']  = z_out
        row['YAxisDownPossible'] = y_dn
        row['YAxisUpPossible']   = y_up
    return row


def ptab(df, first_ptab):
    """Extract PTAB position from MRI_FRR_257 events and propagate."""
    start = True
    df['PTAB'] = np.nan
    for idx, row in df.iterrows():
        if start and idx > 0:
            df.at[idx, 'PTAB'] = first_ptab
        if row['sourceID'] == 'MRI_FRR_257':
            df.at[idx, 'PTAB'] = row['text'].split()[-1]
            start = False
    return df


def expand_coils(coils_str):
    """
    Parse 'Connected coil elements: HC1-4,NC1,2,...' into a list of coil names.
    Returns e.g. ['HC1','HC2','HC3','HC4','NC1','NC2'].
    """
    m = re.search(r'Connected coil elements: ([^.)]*)', coils_str)
    coil_elements_str = m.group(1) if m else ''

    elements = coil_elements_str.split(',')
    result   = []
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


def coils(df):
    """
    Parse MRI_CCS_11 events and create binary coil columns.
    Forward-fills coil state between consecutive MRI_CCS_11 events.
    """
    df_copy     = df.copy()
    total_coils = []

    for idx, row in df.iterrows():
        if row['sourceID'] == 'MRI_CCS_11':
            active_coils = expand_coils(str(row['text']))
            for coil in active_coils:
                if coil not in df_copy.columns:
                    df_copy[coil] = np.nan
                    total_coils.append(coil)
                df_copy.at[idx, coil] = 1

    indices = df_copy.index[df_copy['sourceID'] == 'MRI_CCS_11'].tolist()

    # Forward-fill coil state between events
    for i in range(len(indices) - 1):
        s, e = indices[i], indices[i + 1]
        df_copy.loc[s + 1:e - 1, total_coils] = \
            df_copy.loc[s, total_coils].values

    if indices:
        last = indices[-1]
        df_copy.loc[last + 1:, total_coils] = \
            df_copy.loc[last, total_coils].values

    df_copy[total_coils] = df_copy[total_coils].fillna(0)
    return df_copy

# COMMAND ----------
# =============================================================================
# CELL: Load body group mapping from Excel
# =============================================================================

df_body = pd.read_excel(BODY_GROUP_MAPPING_PATH)
df_body['BodyGroup'] = df_body['BodyGroup'].str.upper()
df_body['BodyPart']  = df_body['BodyPart'].str.upper()
if 'MRType' in df_body.columns:
    df_body.drop(columns='MRType', inplace=True)

print(f"Body mapping rows: {len(df_body)}")
print(f"Sample:\n{df_body.head(3)}")

# COMMAND ----------
# =============================================================================
# CELL: Query eventlog from Spark (all 10 serials, 1-month window)
# =============================================================================

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
    .orderBy(F.col("datetime").asc())
)

print(f"Eventlog row count: {eventlog_spark.count():,}")
eventlog_pd = eventlog_spark.toPandas()
print("Converted to pandas.")

# COMMAND ----------
# =============================================================================
# CELL: Per-serial processing loop
# =============================================================================

for serial_number in SERIAL_NUMBERS:
    print(f"\n{'='*60}")
    print(f"Processing serial: {serial_number}")
    print(f"{'='*60}")

    # Filter to this serial (SerialNumber may be stored as int or string)
    df_sc = eventlog_pd[
        eventlog_pd['SerialNumber'].astype(str) == str(serial_number)
    ].copy()

    if df_sc.empty:
        print(f"  No events found — skipping.")
        continue

    print(f"  Events loaded: {len(df_sc):,}")

    # Sort by datetime then Line (secondary sort to preserve event order within
    # the same timestamp)
    df_sc['datetime'] = pd.to_datetime(df_sc['datetime'])
    df_sorted = (
        df_sc.groupby('datetime', group_keys=False)
             .apply(lambda g: g.sort_values('Line'))
             .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Step 1: Extract inter-patient exchange blocks
    # -------------------------------------------------------------------------
    df_filter, first_ptab = interpatient(df_sorted)
    if df_filter.empty:
        print("  No exchange blocks found — skipping.")
        continue

    print(f"  Exchange rows before dedup: {len(df_filter):,}")

    # -------------------------------------------------------------------------
    # Step 2: Join MRI_FRR_264/265 pairs
    # -------------------------------------------------------------------------
    df_join = join_events(df_filter)
    df_join.drop_duplicates(
        subset=['sourceID', 'datetime', 'PatientID'], keep='first', inplace=True
    )
    df_join.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------------------------
    # Step 3: Extract axis flags into binary columns
    # -------------------------------------------------------------------------
    df_bool = df_join.apply(to_bincolumns, axis=1)

    # -------------------------------------------------------------------------
    # Step 4: Add PTAB column
    # -------------------------------------------------------------------------
    df_ptab = ptab(df_bool, first_ptab)
    ff_cols = ['ZAxisInPossible', 'ZAxisOutPossible',
               'YAxisDownPossible', 'YAxisUpPossible', 'PTAB']
    for c in ff_cols:
        if c in df_ptab.columns:
            df_ptab[c] = df_ptab[c].ffill()

    # -------------------------------------------------------------------------
    # Step 5: Parse coil elements into binary columns
    # -------------------------------------------------------------------------
    df_coils = coils(df_ptab)
    df_coils['SN'] = str(serial_number)

    # -------------------------------------------------------------------------
    # Step 6: Merge body group mapping
    # -------------------------------------------------------------------------
    exchange = df_coils.copy()
    if not exchange.empty and 'BodyPart' in exchange.columns:
        # Normalise BodyPart case for the merge key
        exchange['BodyPart'] = exchange['BodyPart'].str.upper()
        result = pd.merge(exchange, df_body, on='BodyPart', how='left')
    else:
        print("  No BodyPart column — skipping body group merge.")
        result = exchange.copy()

    # Forward-fill / back-fill body and patient info
    body_parts_mask = result['BodyPart'].notnull()
    for col_name, src in [('BodyPart_from', 'BodyPart'),
                           ('BodyPart_to',   'BodyPart'),
                           ('BodyGroup_from', 'BodyGroup'),
                           ('BodyGroup_to',   'BodyGroup'),
                           ('PatientID_from', 'PatientID'),
                           ('PatientID_to',   'PatientID')]:
        if src in result.columns:
            s = result[src].where(body_parts_mask)
            if col_name.endswith('_from'):
                result[col_name] = s.ffill()
            else:
                result[col_name] = s.bfill()

    # Keep only rows that are actual event rows (BodyPart marker rows are dropped)
    drop_cols = [c for c in ['BodyPart', 'PatientID', 'BodyGroup']
                 if c in result.columns]
    df_filter_final = result[result['BodyPart'].isnull()].drop(columns=drop_cols)

    # -------------------------------------------------------------------------
    # Step 7: Merge examination_workflow for patient demographics
    # -------------------------------------------------------------------------
    sd = pd.to_datetime(DATE_START)
    ed = pd.to_datetime(DATE_END)
    start_int = int(DATE_START.replace('-', ''))
    end_int   = int(DATE_END.replace('-', ''))

    try:
        examination_df = spark.read.table(EXAMINATION_TABLE)
        exam_filtered = (
            examination_df
            .filter(F.col("SerialNumber").cast("long") == int(serial_number))
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
        exam_pd = exam_filtered.toPandas()
        print(f"  Examination records: {len(exam_pd):,}")

        df_merged = df_filter_final.merge(
            exam_pd[['PatientId', 'Position', 'Weight', 'Age', 'Height', 'Direction']],
            how='left',
            left_on='PatientID_to',
            right_on='PatientId',
        )
        if 'PatientId' in df_merged.columns:
            df_merged.drop(columns=['PatientId'], inplace=True)
    except Exception as e:
        print(f"  Warning: examination_workflow merge failed: {e}")
        df_merged = df_filter_final.copy()

    # -------------------------------------------------------------------------
    # Step 8: Save CSV
    # -------------------------------------------------------------------------
    csv_path = f"{EXCHANGE_OUTPUT_DIR}/DATA_{serial_number}.csv"
    df_merged.to_csv(csv_path, index=False, header=True)
    print(f"  Saved {len(df_merged):,} rows → {csv_path}")

print("\nExchange preprocessing complete.")
