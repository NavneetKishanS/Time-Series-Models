"""
Configuration file for the Conditional Generation System
"""
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_SAVE_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Sequence configuration
MAX_SEQ_LEN = 128  # Maximum sequence length for padding

# Body region vocabulary
BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
NUM_BODY_REGIONS = 11  # 0-10 for actual body regions
START_REGION_ID = 11   # Special token for session start
END_REGION_ID = 12     # Special token for session end
NUM_REGION_CLASSES = 13  # Total classes including START and END

# Bucket configuration for pre-generation
BUCKET_SIZE = 1000  # Number of samples per body region

# Conditioning features (context) - original for sequence model
CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'BodyGroup_from',
    'BodyGroup_to',
    'PTAB',
    'entity_type'       # NEW: REAL_PATIENT=0 or PSEUDO_PATIENT_*=1/2/3
]

# Conditioning features for Exchange Model (patient-level, no body region)
EXCHANGE_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'entity_type'
]

# Conditioning features for Examination Model
EXAMINATION_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'entity_type'
]

# Symbolic sequence vocabulary (sourceID tokens)
SOURCEID_VOCAB = {
    'PAD': 0,           # Padding token
    'MRI_CCS_11': 1,    # Various scan types
    'MRI_EXU_95': 2,
    'MRI_FRR_18': 3,
    'MRI_FRR_257': 4,
    'MRI_FRR_264': 5,
    'MRI_FRR_2': 6,
    'MRI_FRR_3': 7,
    'MRI_FRR_34': 8,
    'MRI_MPT_1005': 9,
    'MRI_MSR_100': 10,
    'START': 11,        # Sequence start marker
    'MRI_MSR_104': 12,
    'MRI_MSR_21': 13,
    'END': 14,          # Sequence end marker
    'MRI_MSR_34': 15,
    'MRI_FRR_256': 16,
    'UNK': 17           # Unknown token
    # 'PAUSE': 18 removed - handled via pseudo-patient entities
}

# Pseudo-patient tokens (only used in pseudo-patient sequences)
PSEUDO_PATIENT_TOKENS = {
    'IDLE': 18  # Represents machine idle state (reuses ID 18)
}

VOCAB_SIZE = 19  # 18 sourceID tokens + IDLE token
START_TOKEN_ID = 11
END_TOKEN_ID = 14
PAD_TOKEN_ID = 0
IDLE_TOKEN_ID = 18  # Only appears in pseudo-patient sequences

# Special encoding for anatomical features when machine is idle
IDLE_STATE_ENCODING = -1  # Used for BodyGroup_from, BodyGroup_to when entity_type != REAL_PATIENT

# Additional sequence features (encoded alongside sourceID)
SEQUENCE_FEATURE_COLUMNS = [
    'Position_encoded',
    'Direction_encoded'
]

# ============================================================================
# MODEL ARCHITECTURE - EXCHANGE MODEL (Body Region Transitions)
# ============================================================================

EXCHANGE_MODEL_CONFIG = {
    'd_model': 128,                # Model dimension
    'hidden_dim': 256,             # Hidden layer dimension
    'num_layers': 3,               # Number of layers
    'dropout': 0.1,
    'conditioning_dim': len(EXCHANGE_CONDITIONING_FEATURES),
    'num_regions': NUM_REGION_CLASSES,  # Output classes (body regions + START + END)
}

# Training configuration for exchange model
EXCHANGE_TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2
}

# ============================================================================
# MODEL ARCHITECTURE - EXAMINATION MODEL (Body-Region-Specific Event Sequences)
# ============================================================================

EXAMINATION_MODEL_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'd_model': 256,                # Model dimension
    'nhead': 8,                    # Number of attention heads
    'num_encoder_layers': 6,       # Encoder depth
    'num_decoder_layers': 6,       # Decoder depth
    'dim_feedforward': 1024,       # FFN dimension
    'dropout': 0.1,
    'max_seq_len': MAX_SEQ_LEN,
    'conditioning_dim': len(EXAMINATION_CONDITIONING_FEATURES) + 1,  # +1 for body region
}

# Training configuration for examination model
EXAMINATION_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'validation_split': 0.2
}

# ============================================================================
# MODEL ARCHITECTURE - CONDITIONAL SEQUENCE GENERATOR (Original)
# ============================================================================

SEQUENCE_MODEL_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'd_model': 256,              # Model dimension
    'nhead': 8,                  # Number of attention heads
    'num_encoder_layers': 6,     # Encoder depth
    'num_decoder_layers': 6,     # Decoder depth
    'dim_feedforward': 1024,     # FFN dimension
    'dropout': 0.1,
    'max_seq_len': MAX_SEQ_LEN,
    'conditioning_dim': len(CONDITIONING_FEATURES)
}

# Training configuration for sequence model
SEQUENCE_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'grad_clip_value': 1.0, # Renamed from 'gradient_clip'
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'validation_split': 0.2
}

# Sampling configuration for sequence generation
SEQUENCE_SAMPLING_CONFIG = {
    'temperature': 1.0,      # Sampling temperature (higher = more random)
    'top_k': 10,             # Top-k sampling
    'top_p': 0.9,            # Nucleus sampling
    'max_length': MAX_SEQ_LEN
}

# ============================================================================
# MODEL ARCHITECTURE - CONDITIONAL COUNTS GENERATOR
# ============================================================================

COUNTS_MODEL_CONFIG = {
    'd_model': 256,              # Model dimension
    'nhead': 8,                  # Number of attention heads
    'num_encoder_layers': 6,     # Number of encoder layers
    'num_cross_attention_layers': 4,  # Cross-attention layers
    'dim_feedforward': 1024,     # FFN dimension
    'dropout': 0.1,
    'max_seq_len': MAX_SEQ_LEN,
    'conditioning_dim': len(CONDITIONING_FEATURES),
    'sequence_feature_dim': VOCAB_SIZE + len(SEQUENCE_FEATURE_COLUMNS),
    'output_heads': 2            # μ and σ predictions
}

# Training configuration for counts model
COUNTS_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'grad_clip_value': 1.0, # Renamed from 'gradient_clip'
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'min_sigma': 0.1            # Minimum standard deviation
}

# Sampling configuration for count generation
COUNTS_SAMPLING_CONFIG = {
    'distribution': 'gamma',     # 'gamma', 'normal', or 'poisson'
    'num_samples': 1,            # Number of samples to draw per sequence
    'clip_negative': True        # Clip negative values to 0
}

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

# For sequence model: cross-entropy with label smoothing
SEQUENCE_LOSS = 'cross_entropy'

# For counts model: negative log-likelihood of Gamma distribution
COUNTS_LOSS = 'gamma_nll'  # Options: 'gamma_nll', 'gaussian_nll', 'mse'

# ============================================================================
# EVALUATION METRICS
# ============================================================================

METRICS = {
    'sequence': ['accuracy', 'perplexity', 'bleu'],
    'counts': ['mae', 'rmse', 'mape', 'nll']
}

# ============================================================================
# RANDOM SEEDS
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

USE_GPU = True  # Set to False to force CPU usage
