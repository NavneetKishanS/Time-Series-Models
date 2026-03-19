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
