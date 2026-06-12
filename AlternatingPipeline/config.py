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
# SEQUENCE-TYPE VOCABULARY (examination scan-type conditioning)
# ============================================================================
# MRI pulse-sequence families parsed from the `Sequence` field of MRI_MSR_100
# messages (e.g. "%SiemensSeq%\\AALScout", "%SiemensSeq%\\tse"). The exam data
# carries this field but the examination model historically never saw it, so
# it predicted one body-region-mean duration for every scan. Feeding the
# sequence type lets the model learn that a scout (~20 s) and a TSE (~4 min)
# have very different durations.
#
# NOTE: this vocab MUST stay byte-identical to the copy in
# DatabricksPipeline/csv_pipeline/config.py — step 03 writes the IDs into the
# pkl and the training scripts read them back.
SEQUENCE_TYPE_VOCAB = {
    'other': 0, 'scout': 1, 'localizer': 2, 'tse': 3, 'space': 4,
    'haste': 5, 'gre': 6, 'flash': 7, 'epi': 8, 'tfl': 9, 'tirm': 10,
    'vibe': 11, 'dixon': 12, 'swi': 13, 'medic': 14,
}
NUM_SEQUENCE_TYPES = len(SEQUENCE_TYPE_VOCAB)
ID_TO_SEQUENCE_TYPE = {v: k for k, v in SEQUENCE_TYPE_VOCAB.items()}

# Substrings checked in order; first hit wins. 'epi' family also matches the
# diffusion/BOLD sequences that show up as ep2d_* / *bold* / *diff*.
_SEQUENCE_TYPE_KEYS = [
    'localizer', 'scout', 'haste', 'space', 'tirm', 'vibe', 'dixon',
    'medic', 'swi', 'tfl', 'flash', 'tse', 'gre',
]


def classify_sequence_type(raw):
    """Map a raw `Sequence` string to a SEQUENCE_TYPE_VOCAB id."""
    s = str(raw or '').lower()
    if not s:
        return SEQUENCE_TYPE_VOCAB['other']
    for key in _SEQUENCE_TYPE_KEYS:
        if key in s:
            return SEQUENCE_TYPE_VOCAB[key]
    if 'ep2d' in s or 'epi' in s or 'bold' in s or 'diff' in s or 'dwi' in s:
        return SEQUENCE_TYPE_VOCAB['epi']
    return SEQUENCE_TYPE_VOCAB['other']


# Number of distinct scanners in the csv_pipeline scope. Used as the size of
# the examination model's serial embedding so it can learn per-scanner
# duration offsets. serial_idx is the index of a serial in SERIAL_NUMBERS.
NUM_SERIALS = 10

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
    'd_model': 128,               # reduced from 256 — ~4x fewer attention params, much faster on CPU
    'nhead': 4,                   # reduced from 8 to match d_model
    'num_encoder_layers': 3,      # reduced from 6 — appropriate for ~3500 training sequences
    'num_decoder_layers': 3,      # reduced from 6
    'dim_feedforward': 512,       # reduced from 1024
    'dropout': 0.1,
    'max_seq_len': MAX_SEQ_LEN,
    'num_duration_encoder_layers': 2,  # reduced from 4
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
    # Log-space duration head. Real exchange per-token durations are heavily
    # right-skewed (median ~3 s, mean ~24 s); a Gaussian head fits the
    # inflated mean and over-predicts every exchange. 'log' makes the head
    # model log1p(duration), fitting the central tendency instead.
    'duration_mode': 'log',
    # Per-feature divisors bringing the raw conditioning vector
    # (Age, Weight, Height, PTAB, Direction, hour_sin, hour_cos,
    #  dow_sin, dow_cos, is_morning) to O(1). Raw Age≈50/Weight≈75 blow up
    # the pre-LayerNorm variance inside conditioning_projection, and the
    # LayerNorm then crushes the O(0.5) categorical embeddings (body region,
    # scan type, serial) to ~0.6% relative amplitude — measured: varying
    # sequence_type moved the encoded conditioning memory by max 0.006 while
    # activations are O(1), so conditioning was effectively erased.
    'conditioning_scale': [100.0, 100.0, 2.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0],
}

EXCHANGE_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 500,          # reduced from 4000 — ~4-5 epochs; 4000 caused LR to ramp through epoch 36
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'duration_loss_weight': 0.3,
    'duration_scale': 60.0,       # normalise durations — divide raw seconds by 60 before loss
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
    # Log-space duration head (like exchange). The duration target is now the
    # SPAN TOTAL placed on the finish token (see ExaminationDataset): totals
    # are right-skewed by scan type (scout ~19 s ... space ~235 s, swi tail
    # into the 10-min range), so the head models log1p(total/60) where a
    # Gaussian on the raw scale would be dominated by the tail.
    'duration_mode': 'log',
    # Scan-type + per-scanner conditioning. When enabled the examination
    # model embeds the MRI sequence type (scout/tse/...) and the scanner
    # serial alongside body_region, so it can produce duration variability
    # across scan types and customers instead of one flat body-region mean.
    'use_exam_conditioning': True,
    'num_sequence_types': NUM_SEQUENCE_TYPES,
    'num_serials': NUM_SERIALS,
    # Same O(1) rescale as the exchange model — see EXCHANGE_MODEL_CONFIG.
    # Without it the scan-type/serial embeddings are LayerNorm-crushed and
    # the duration head can only read sequence LENGTH, which the token
    # decoder (also conditioning-blind) generates flat — the root cause of
    # every flat-duration run through 2026-06-11.
    'conditioning_scale': [100.0, 100.0, 2.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0],
}

EXAMINATION_TRAINING_CONFIG = {
    'batch_size': 64,             # increased from 32 — halves steps/epoch on large dataset
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 500,          # reduced from 4000 — same reason as exchange model
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'duration_loss_weight': 0.3,
    'duration_scale': 60.0,       # normalise durations — divide raw seconds by 60 (1 min ref)
                                   # Lowered from 600 — real per-token durations average ~10 s, so
                                   # dividing by 600 compressed targets to ~0.02 and collapsed the
                                   # Gaussian duration head.  Scale 60 puts targets in the 0.15–0.3
                                   # range, matching the healthy exchange-model regime.
    'augment_training': True,
    'duration_jitter_pct': 0.10,
    # oversample_factor removed — 165K sequences is sufficient without duplication
    # Targeted oversampling of rare "Stopped by User" (MRI_MSR_34) abort
    # sequences. Replaces the removed inverse-frequency class weighting (which
    # collapsed the token decoder). Duplicates ONLY abort sequences x this
    # factor so the rare abort token is not crowded out of the softmax.
    # Abort sequences are ~3.9% of training; factor 4 -> ~13% share. Tune
    # against the MRI_MSR_34 rate in step-05 output: raise if aborts still
    # never appear, lower if they appear too often.
    'abort_oversample_factor': 4,
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
    'warmup_steps': 300,          # reduced from 2000 — orchestration dataset is smaller
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
