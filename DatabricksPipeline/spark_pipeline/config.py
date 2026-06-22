# Databricks notebook source
"""
spark_pipeline/config.py — constants for the Spark-native pipeline.

Identical data subset and vocabulary as csv_pipeline/config.py; only the
output paths differ (spark_pipeline instead of csv_pipeline) so the two
pipelines can run side-by-side without clobbering each other.
"""

# ============================================================================
# TARGET SCANNERS & DATE RANGE  (same subset as csv_pipeline)
# ============================================================================

SERIAL_NUMBERS = [183242, 176148, 176227, 175912, 175670,
                  183776, 182625, 176615, 176240, 175828]

DATE_START = "2024-01-01"
DATE_END   = "2024-01-31"

# ============================================================================
# TIMEZONE
# ============================================================================

TIMEZONE_OFFSET_HOURS = 1   # UTC+1 (CET)

# ============================================================================
# DATABRICKS TABLE PATHS
# ============================================================================

EVENTLOG_TABLE    = "hive_metastore.eventlog.common_eventlog"
EXAMINATION_TABLE = "hive_metastore.examination.examination_workflow"

# ============================================================================
# FILE PATHS
# ============================================================================

BODY_GROUP_MAPPING_PATH = "/dbfs/FileStore/tables/bodyupdated.xlsx"

EXCHANGE_OUTPUT_DIR = "/dbfs/FileStore/spark_pipeline/exchange"
EXAM_OUTPUT_DIR     = "/dbfs/FileStore/spark_pipeline/exam"

# ============================================================================
# SOURCE ID FILTERS  (identical to csv_pipeline)
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

EXCHANGE_SOURCE_TYPES = [
    "MRI_FRR_2",    "MRI_FRR_265",  "MRI_FRR_264",
    "MRI_CCS_11",   "MRI_FRR_257",  "MRI_MPT_1005",  "MRI_FRR_256",
    "MRI_FRR_34",   "MRI_FRR_3",    "MRI_MSR_100",   "MRI_MSR_104",
    "MRI_EXU_95",   "MRI_MSR_18",   "MRI_FRR_18",    "MRI_MSR_21",
    "MRI_MSR_34",   "MRI_MSR_26",   "MRI_MSR_22",    "MRI_MSR_40",
    "MRI_MSR_25",   "MRI_MSR_24",
]

EXAM_EXTRA_SOURCE_TYPES = [
    "MRI_FRR_18",   "MRI_EXU_95",   "MRI_EXU_89",
    "MRI_CCS_1008", "MRI_MPT_1005", "MRI_SUT_1005",
    "MRI_FRR_257",
]

# ============================================================================
# COIL ABBREVIATION MAP
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
# TOKEN VOCABULARY & BODY REGION MAP
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
# SEQUENCE-TYPE VOCABULARY
# ============================================================================

SEQUENCE_TYPE_VOCAB = {
    'other': 0, 'scout': 1, 'localizer': 2, 'tse': 3, 'space': 4,
    'haste': 5, 'gre': 6, 'flash': 7, 'epi': 8, 'tfl': 9, 'tirm': 10,
    'vibe': 11, 'dixon': 12, 'swi': 13, 'medic': 14,
}
NUM_SEQUENCE_TYPES = len(SEQUENCE_TYPE_VOCAB)

_SEQUENCE_TYPE_KEYS = [
    'localizer', 'scout', 'haste', 'space', 'tirm', 'vibe', 'dixon',
    'medic', 'swi', 'tfl', 'flash', 'tse', 'gre',
]


def classify_sequence_type(raw):
    s = str(raw or '').lower()
    if not s:
        return SEQUENCE_TYPE_VOCAB['other']
    for key in _SEQUENCE_TYPE_KEYS:
        if key in s:
            return SEQUENCE_TYPE_VOCAB[key]
    if 'ep2d' in s or 'epi' in s or 'bold' in s or 'diff' in s or 'dwi' in s:
        return SEQUENCE_TYPE_VOCAB['epi']
    return SEQUENCE_TYPE_VOCAB['other']


ID_TO_SEQUENCE_TYPE = {v: k for k, v in SEQUENCE_TYPE_VOCAB.items()}

# ============================================================================
# CONSTANTS FROM AlternatingPipeline / csv_pipeline
# ============================================================================

REAL_EVENT_TYPES = SOURCE_ID_FILTER

MAX_EXCHANGE_DURATION    = 7200
MAX_EXAMINATION_DURATION = 3000
MIN_EXAMINATION_DURATION = 10
MAX_PER_TOKEN_DURATION   = 600

PHASE_TYPES = {
    'startup':  0,
    'between':  1,
    'shutdown': 2,
}

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

# Fixed exam coil columns (same prefix convention as step 02 / step 05)
EXAM_COIL_COLS = [f'#0_{c}' for c in COIL_COLUMNS]
