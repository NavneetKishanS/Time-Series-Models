"""
DatabricksPipeline — shared constants.

Mirrors AlternatingPipeline/config.py exactly for preprocessed_data.pkl
format compatibility.  All notebooks %run this file.
"""

import os

# ============================================================================
# TARGET CUSTOMERS & DATE RANGE
# ============================================================================

TARGET_SERIAL_NUMBERS = [202594, 202551, 183242, 176148, 176227]
DATE_START = "2024-01-01"
DATE_END   = "2025-03-31"

# ============================================================================
# TIMEZONE
# ============================================================================

# UTC offset for the scanner site (e.g., +1 = CET).  Adjust per deployment.
TIMEZONE_OFFSET_HOURS = 1

# ============================================================================
# DATABRICKS TABLE / FILE PATHS
# ============================================================================

EVENTLOG_TABLE          = "hive_metastore.eventlog.common_eventlog"
EXAMINATION_TABLE       = "hive_metastore.examination.examination_workflow"
BODY_GROUP_MAPPING_PATH = "/dbfs/FileStore/tables/bodyupdated.xlsx"

DBFS_OUTPUT_BASE   = "/dbfs/FileStore/time_series_models/"
EXCHANGE_OUTPUT    = DBFS_OUTPUT_BASE + "exchange_sequences.pkl"
EXAMINATION_OUTPUT = DBFS_OUTPUT_BASE + "examination_sequences.pkl"
ORCH_OUTPUT        = DBFS_OUTPUT_BASE + "orchestration_data.pkl"
FINAL_OUTPUT       = DBFS_OUTPUT_BASE + "preprocessed_data.pkl"

# ============================================================================
# SOURCE ID VOCABULARY  (event token IDs — mirrors AlternatingPipeline exactly)
# ============================================================================

SOURCEID_VOCAB = {
    'PAD':         0,
    'MRI_CCS_11':  1,   # Coil change event
    'MRI_EXU_95':  2,   # Measurement start (examination marker)
    'MRI_FRR_18':  3,   # Scanner hardware
    'MRI_FRR_257': 4,   # Table movement
    'MRI_FRR_264': 5,   # Axis movement possible
    'MRI_FRR_2':   6,   # Door open warning
    'MRI_FRR_3':   7,   # Door closed
    'MRI_FRR_34':  8,   # Patient positioned
    'MRI_MPT_1005':9,   # Patient registered
    'MRI_MSR_100': 10,  # Start prepare
    'START':       11,  # Sequence start marker (synthetic)
    'MRI_MSR_104': 12,  # Measurement finished OK
    'MRI_MSR_21':  13,  # Measurement info
    'END':         14,  # Sequence end marker (synthetic)
    'MRI_MSR_34':  15,  # Measurement stopped by user
    'MRI_FRR_256': 16,  # PTAB position set
    'UNK':         17,  # Unknown token
}

# Real event types present in the eventlog (exclude synthetic tokens)
REAL_EVENT_TYPES = [k for k in SOURCEID_VOCAB if k not in ('PAD', 'START', 'END', 'UNK')]

# ============================================================================
# BODY REGIONS  (mirrors AlternatingPipeline/config.py exactly)
# ============================================================================

BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']

BODY_REGION_TO_ID = {r: i for i, r in enumerate(BODY_REGIONS)}
ID_TO_BODY_REGION = {i: r for i, r in enumerate(BODY_REGIONS)}

NUM_BODY_REGIONS   = 11
START_REGION_ID    = 11   # Special token for session start
END_REGION_ID      = 12   # Special token for session end
NUM_REGION_CLASSES = 13

# ============================================================================
# COIL COLUMNS  (33 entries — use all for coil_config dict)
# NUM_COIL_FEATURES=30 in AlternatingPipeline/config.py is inconsistent with
# the 33-entry list; we use all 33 to match what preprocessing.py produces.
# ============================================================================

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

# ============================================================================
# PHASE TYPES  (exchange model)
# ============================================================================

PHASE_TYPES = {
    'startup':  0,   # First exchange of the day (START → first body region)
    'between':  1,   # Between-patient exchanges
    'shutdown': 2,   # After last exam of the day (last body region → END)
}
NUM_PHASE_TYPES = len(PHASE_TYPES)

# ============================================================================
# FILTERING
# ============================================================================

MAX_EXCHANGE_DURATION = 7200   # seconds; filters overnight/multi-hour gaps

# ============================================================================
# ORCHESTRATION VOCABULARY  (mirrors AlternatingPipeline/config.py)
# ============================================================================

BREAK_TOKEN_ID    = 13
ORCH_PAD_TOKEN_ID = 14
ORCH_VOCAB_SIZE   = 15   # 11 body regions + START + END + BREAK + PAD

# ============================================================================
# CONDITIONING FEATURE NAMES  (for reference / documentation)
# ============================================================================
# build_conditioning_tensor() in training/utils.py expects this exact key order:
#   Age, Weight, Height, PTAB, Direction_encoded,
#   hour_sin, hour_cos, dow_sin, dow_cos, is_morning
CONDITIONING_KEYS = [
    'Age', 'Weight', 'Height', 'PTAB', 'Direction_encoded',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_morning',
]
