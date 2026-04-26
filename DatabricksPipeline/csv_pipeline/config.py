# Databricks notebook source
"""
csv_pipeline/config.py — constants for the CSV-based pipeline.

Scoped to 10 serial numbers and a 1-month date range so the pipeline
runs quickly on Databricks.  Does NOT replace or modify the parent
DatabricksPipeline/config.py.
"""

# ============================================================================
# TARGET SCANNERS & DATE RANGE
# ============================================================================

SERIAL_NUMBERS = [183242, 176148, 176227, 175912, 175670,
                  183776, 182625, 176615, 176240, 175828]

DATE_START = "2024-01-01"
DATE_END   = "2024-01-31"

# ============================================================================
# TIMEZONE
# ============================================================================

TIMEZONE_OFFSET_HOURS = 1   # UTC+1 (CET); adjust per site

# ============================================================================
# DATABRICKS TABLE PATHS
# ============================================================================

EVENTLOG_TABLE    = "hive_metastore.eventlog.common_eventlog"
EXAMINATION_TABLE = "hive_metastore.examination.examination_workflow"

# ============================================================================
# FILE PATHS
# ============================================================================

BODY_GROUP_MAPPING_PATH = "/dbfs/FileStore/tables/bodyupdated.xlsx"

EXCHANGE_OUTPUT_DIR = "/dbfs/FileStore/csv_pipeline/exchange"
EXAM_OUTPUT_DIR     = "/dbfs/FileStore/csv_pipeline/exam"

# One CSV per serial:
#   EXCHANGE_OUTPUT_DIR/DATA_{serial}.csv
#   EXAM_OUTPUT_DIR/DATA_{serial}.csv

# ============================================================================
# SOURCE ID FILTER — 14 real event types (mirrors parent config REAL_EVENT_TYPES)
# ============================================================================

SOURCE_ID_FILTER = [
    "MRI_CCS_11",
    "MRI_EXU_95",
    "MRI_FRR_18",
    "MRI_FRR_257",
    "MRI_FRR_264",
    "MRI_FRR_2",
    "MRI_FRR_3",
    "MRI_FRR_34",
    "MRI_MPT_1005",
    "MRI_MSR_100",
    "MRI_MSR_104",
    "MRI_MSR_21",
    "MRI_MSR_34",
    "MRI_FRR_256",
]

# Exchange pipeline uses a broader filter (needs MRI_FRR_265 for join_events
# and additional MSR finish codes for interpatient block detection)
EXCHANGE_SOURCE_TYPES = [
    "MRI_FRR_2",    "MRI_FRR_265",  "MRI_FRR_264",
    "MRI_CCS_11",   "MRI_FRR_257",  "MRI_MPT_1005",  "MRI_FRR_256",
    "MRI_FRR_34",   "MRI_FRR_3",    "MRI_MSR_100",   "MRI_MSR_104",
    "MRI_EXU_95",   "MRI_MSR_18",   "MRI_FRR_18",    "MRI_MSR_21",
    "MRI_MSR_34",   "MRI_MSR_26",   "MRI_MSR_22",    "MRI_MSR_40",
    "MRI_MSR_25",   "MRI_MSR_24",
]

# Exam pipeline additional event types (beyond MRI_MSR_*)
EXAM_EXTRA_SOURCE_TYPES = [
    "MRI_FRR_18",   "MRI_EXU_95",   "MRI_EXU_89",
    "MRI_CCS_1008", "MRI_MPT_1005", "MRI_SUT_1005",
    "MRI_FRR_257",
]

# ============================================================================
# COIL ABBREVIATION MAP  (HC→HE, NC→NE, etc.)
# Used by the exam pipeline's expand_coils() for parent-pipeline compatibility
# ============================================================================

COIL_ABBREV_MAP = {
    'HC1': 'HE1', 'HC2': 'HE2', 'HC3': 'HE3', 'HC4': 'HE4',
    'NC1': 'NE1', 'NC2': 'NE2',
    'BC':  'BC',  'SHL': 'SHL', 'FA':  'FA',  'TO':  'TO',
    'FS':  'FS',  '15K': '15K', 'SN':  'SN',
    'SP1': 'SP1', 'SP2': 'SP2', 'SP3': 'SP3', 'SP4': 'SP4',
    'SP5': 'SP5', 'SP6': 'SP6', 'SP7': 'SP7', 'SP8': 'SP8',
    'HW1': 'HW1', 'HW2': 'HW2', 'HW3': 'HW3',
    'HE1': 'HE1', 'HE2': 'HE2', 'HE3': 'HE3', 'HE4': 'HE4',
    'NE1': 'NE1', 'NE2': 'NE2',
    'BO1': 'BO1', 'BO2': 'BO2', 'BO3': 'BO3',
    'PA1': 'PA1', 'PA2': 'PA2', 'PA3': 'PA3',
    'PA4': 'PA4', 'PA5': 'PA5', 'PA6': 'PA6',
}

# ============================================================================
# TOKEN VOCABULARY & BODY REGION MAP  (mirrors AlternatingPipeline/config.py)
# Used by preprocessing notebooks to add token_id and BodyGroup integer columns.
# ============================================================================

SOURCEID_VOCAB = {
    'PAD': 0, 'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3,
    'MRI_FRR_257': 4, 'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7,
    'MRI_FRR_34': 8, 'MRI_MPT_1005': 9, 'MRI_MSR_100': 10, 'START': 11,
    'MRI_MSR_104': 12, 'MRI_MSR_21': 13, 'END': 14, 'MRI_MSR_34': 15,
    'MRI_FRR_256': 16, 'UNK': 17,
}

BODY_REGIONS = [
    'HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
    'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN',
]
BODY_REGION_TO_ID = {r: i for i, r in enumerate(BODY_REGIONS)}

# ============================================================================
# CONSTANTS FORMERLY IN DatabricksPipeline/config.py (now removed)
# Required by 03_build_preprocessed_pkl.py
# ============================================================================

# Alias for SOURCE_ID_FILTER — used by step 03 Spark queries
REAL_EVENT_TYPES = SOURCE_ID_FILTER

# Maximum exchange block duration in seconds (2 hours); longer blocks are
# treated as overnight gaps and discarded (mirrors train_exchange.py)
MAX_EXCHANGE_DURATION = 7200

# Maximum total examination (MSR_100 → MSR_104) duration in seconds. Above
# this, the segment is almost certainly a missed change-point boundary
# (multi-day or cross-patient span) and would feed an outlier per-token
# duration into the Gaussian NLL training loss. Matches the 4000 s sanity
# filter that 05_generate_synthetic_data.py applies to its own output rows.
MAX_EXAMINATION_DURATION = 4000

# Minimum total examination duration in seconds. Below this, the segment is
# almost always a localizer / aborted measurement / calibration ping, not a
# real diagnostic exam. On the production pkl, 65.8% of segments fell under
# 10 s and dragged the mean total_duration to 43 s vs step 02's per-
# measurement CSV mean of 105 s. Filtering at 10 s realigns the training
# distribution with the real-data reference Qlik compares against.
MIN_EXAMINATION_DURATION = 10

# Cap on any single intra-event duration in seconds. Real per-token P90 is
# ~15 s and FINISH-token duration tracks total exam length (P90 ~141 s); a
# 600 s cap leaves headroom for legitimate long exams while clipping the
# segment-boundary artifacts (observed up to 316 712 s = 88 h on one token)
# that otherwise dominate the Gaussian NLL loss and inflate every
# synthesized duration.
MAX_PER_TOKEN_DURATION = 600

# Phase labels for exchange sequences
PHASE_TYPES = {
    'startup':  0,   # First exchange of the day
    'between':  1,   # Between-patient exchanges
    'shutdown': 2,   # Last exchange of the day
}

# Coil element columns used by examination sequences
COIL_COLUMNS = [
    'BC',
    'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8',
    '15K',
    'HW1', 'HW2', 'HW3',
    'HE1', 'HE2', 'HE3', 'HE4',
    'NE1', 'NE2',
    'SHL',
    'BO1', 'BO2', 'BO3',
    'FA', 'TO', 'FS',
    'PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6',
    'SN',
]
