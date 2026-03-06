"""
Configuration for the Alternating Pipeline (Exchange <-> Examination)

This pipeline implements the sequential alternating approach:
  Exchange Model -> Examination Model -> Exchange Model -> ...

Unified Transformer architecture: both models share the same base config
but train on different data (exchange transitions vs examination sequences).
"""
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'PXChange_Refactored', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
CUSTOMER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'customers')

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_SAVE_DIR, CUSTOMER_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Sequence configuration
MAX_SEQ_LEN = 128  # Maximum sequence length for padding

# Body region vocabulary
BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
BODY_REGION_TO_ID = {region: i for i, region in enumerate(BODY_REGIONS)}
ID_TO_BODY_REGION = {i: region for i, region in enumerate(BODY_REGIONS)}

NUM_BODY_REGIONS = 11  # 0-10 for actual body regions
START_REGION_ID = 11   # Special token for session start
END_REGION_ID = 12     # Special token for session end
NUM_REGION_CLASSES = 13  # Total classes including START and END

# ============================================================================
# DURATION SCALING CONFIGURATION
# ============================================================================

# Duration multiplier - OBSOLETE with learned duration prediction
DURATION_MULTIPLIER = 1.0

# Base duration parameters - FALLBACK only
EXCHANGE_DURATION_SHAPE = 5.0
EXCHANGE_DURATION_SCALE = 10.0
EXAMINATION_DURATION_SHAPE = 2.0
EXAMINATION_DURATION_SCALE = 5.0

# ============================================================================
# BODY REGION FILTERING
# ============================================================================

# Body regions to exclude from generation (not present in training data)
EXCLUDED_BODY_REGIONS = ['CHEST']

# Convert to IDs for internal use
EXCLUDED_BODY_REGION_IDS = [BODY_REGION_TO_ID[r] for r in EXCLUDED_BODY_REGIONS if r in BODY_REGION_TO_ID]

# Valid body regions for generation
VALID_BODY_REGIONS = [r for r in BODY_REGIONS if r not in EXCLUDED_BODY_REGIONS]
VALID_BODY_REGION_IDS = [BODY_REGION_TO_ID[r] for r in VALID_BODY_REGIONS]

# ============================================================================
# CONDITIONING FEATURES
# ============================================================================

# Exchange Model conditioning (for body region transitions)
EXCHANGE_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'Direction_encoded'  # 0 = Head First, 1 = Feet First
]

# Temporal features (added during preprocessing)
TEMPORAL_FEATURES = [
    'hour_of_day',       # 0-23
    'day_of_week',       # 0-6 (Monday=0)
    'is_morning',        # 0 or 1 (before noon)
    'hour_sin',          # sin(2*pi*hour/24)
    'hour_cos',          # cos(2*pi*hour/24)
]

# Number of coil features (binary: 0 or 1)
NUM_COIL_FEATURES = 30  # Number of coil columns

# Total conditioning dimension for exchange model
EXCHANGE_TOTAL_CONDITIONING_DIM = (
    len(EXCHANGE_CONDITIONING_FEATURES) +
    len(TEMPORAL_FEATURES) +
    NUM_COIL_FEATURES
)

# Examination Model conditioning (for scan sequences within a body region)
EXAMINATION_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'Direction_encoded'
]

# Examination model also uses temporal features
EXAMINATION_TOTAL_CONDITIONING_DIM = (
    len(EXAMINATION_CONDITIONING_FEATURES) +
    len(TEMPORAL_FEATURES)
)

# ============================================================================
# SOURCE ID VOCABULARY (Event Tokens)
# ============================================================================

SOURCEID_VOCAB = {
    'PAD': 0,           # Padding token
    'MRI_CCS_11': 1,    # Coil change event
    'MRI_EXU_95': 2,    # Measurement start (examination marker)
    'MRI_FRR_18': 3,    # Scanner hardware
    'MRI_FRR_257': 4,   # Table movement
    'MRI_FRR_264': 5,   # Axis movement possible
    'MRI_FRR_2': 6,     # Door open warning
    'MRI_FRR_3': 7,     # Door closed
    'MRI_FRR_34': 8,    # Patient positioned
    'MRI_MPT_1005': 9,  # Patient registered
    'MRI_MSR_100': 10,  # Start prepare
    'START': 11,        # Sequence start marker
    'MRI_MSR_104': 12,  # Measurement finished OK
    'MRI_MSR_21': 13,   # Measurement info
    'END': 14,          # Sequence end marker
    'MRI_MSR_34': 15,   # Measurement stopped by user
    'MRI_FRR_256': 16,  # PTAB position set
    'UNK': 17           # Unknown token
}

ID_TO_SOURCEID = {v: k for k, v in SOURCEID_VOCAB.items()}
VOCAB_SIZE = len(SOURCEID_VOCAB)
START_TOKEN_ID = SOURCEID_VOCAB['START']
END_TOKEN_ID = SOURCEID_VOCAB['END']
PAD_TOKEN_ID = SOURCEID_VOCAB['PAD']

# ============================================================================
# COIL ELEMENT COLUMNS
# ============================================================================

COIL_COLUMNS = [
    'BC',   # Body coil
    'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8',  # Spine coils
    '15K',  # 15-channel knee coil
    'HW1', 'HW2', 'HW3',  # Hand/Wrist coils
    'HE1', 'HE2', 'HE3', 'HE4',  # Head coils
    'NE1', 'NE2',  # Neck coils
    'SHL',  # Shoulder coil
    'BO1', 'BO2', 'BO3',  # Body coils
    'FA', 'TO', 'FS',  # Foot/Ankle coils
    'PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6',  # Peripheral angiography coils
    'SN'   # Unknown
]

# ============================================================================
# PHASE TYPES (for exchange model)
# ============================================================================

PHASE_TYPES = {
    'startup': 0,    # First exchange of the day (START -> first body region)
    'between': 1,    # Between-patient exchanges
    'shutdown': 2,   # Last exchange of the day (last body region -> END)
}
NUM_PHASE_TYPES = len(PHASE_TYPES)

# ============================================================================
# UNIFIED SEQUENCE GENERATOR BASE CONFIG
# ============================================================================

SEQUENCE_GENERATOR_BASE_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'max_seq_len': MAX_SEQ_LEN,
    'num_duration_encoder_layers': 4,
    'num_body_regions': NUM_BODY_REGIONS,
    'num_region_classes': NUM_REGION_CLASSES,
    # Base conditioning: 5 patient + 5 temporal = 10
    'base_conditioning_dim': len(EXCHANGE_CONDITIONING_FEATURES) + 5,
}

# ============================================================================
# EXCHANGE MODEL CONFIGURATION (extends base)
# ============================================================================

EXCHANGE_MODEL_CONFIG = {
    **SEQUENCE_GENERATOR_BASE_CONFIG,
    'model_type': 'exchange',
    'has_phase_type': True,
    'body_region_mode': 'from_to',  # Uses body_from AND body_to
    'num_phase_types': NUM_PHASE_TYPES,
}

EXCHANGE_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'duration_loss_weight': 0.3,
    'augment_training': True,
    'duration_jitter_pct': 0.10,
}

# ============================================================================
# EXAMINATION MODEL CONFIGURATION (extends base)
# ============================================================================

EXAMINATION_MODEL_CONFIG = {
    **SEQUENCE_GENERATOR_BASE_CONFIG,
    'model_type': 'examination',
    'has_phase_type': False,
    'body_region_mode': 'single',  # Uses single body_region
}

EXAMINATION_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'duration_loss_weight': 0.3,
    'augment_training': True,
    'duration_jitter_pct': 0.10,
    'oversample_factor': 2,
}

# ============================================================================
# ORCHESTRATION MODEL VOCABULARY
# ============================================================================

BREAK_TOKEN_ID = 13
ORCH_PAD_TOKEN_ID = 14
ORCH_VOCAB_SIZE = 15           # 11 regions + START + END + BREAK + PAD
ORCH_MAX_SEQ_LEN = 40          # ~12 patients + breaks + START/END

# Orchestration conditioning (17-dim):
# dow_sin, dow_cos, month_sin, month_cos, is_weekend,
# avg_patients_per_day, body_region_distribution[11]
ORCH_BASE_CONDITIONING_DIM = 17
NUM_SCANNERS = 40
ORCH_SCANNER_EMB_DIM = 32

# ============================================================================
# ORCHESTRATION MODEL CONFIGURATION
# ============================================================================

ORCHESTRATION_MODEL_CONFIG = {
    'vocab_size': ORCH_VOCAB_SIZE,
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 4,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'max_seq_len': ORCH_MAX_SEQ_LEN,
    'base_conditioning_dim': ORCH_BASE_CONDITIONING_DIM,
    'num_scanners': NUM_SCANNERS,
    'scanner_emb_dim': ORCH_SCANNER_EMB_DIM,
    'pad_token_id': ORCH_PAD_TOKEN_ID,
    'start_token_id': START_REGION_ID,
    'end_token_id': END_REGION_ID,
    'break_token_id': BREAK_TOKEN_ID,
}

ORCHESTRATION_TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.0003,
    'warmup_steps': 2000,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 20,
}

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

GENERATION_CONFIG = {
    'temperature': 1.0,      # Sampling temperature
    'top_k': 10,             # Top-k sampling
    'top_p': 0.9,            # Nucleus sampling
    'max_length': MAX_SEQ_LEN
}

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

USE_GPU = True  # Set to False to force CPU usage

# ============================================================================
# TEMPORAL FORECASTING CONFIGURATION
# ============================================================================

CONTEXT_DAYS = 14
PREDICTION_HORIZON = 1
VALIDATION_DAYS = 2

# Conditioning dimension breakdown:
# - Patient demographics: 5 (Age, Weight, Height, PTAB, Direction_encoded)
# - Temporal features: 5 (hour_sin, hour_cos, dow_sin, dow_cos, is_morning)
# - Total base conditioning: 10
TEMPORAL_CONDITIONING_DIM = 5
