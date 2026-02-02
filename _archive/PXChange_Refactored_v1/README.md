# Conditional Generation System for MRI Scan Sequences

A two-stage conditional generation system for modeling MRI scan workflows using Transformer-based architectures. This system separates **structural patterns** (symbolic sequences) from **quantitative behavior** (numerical counts with uncertainty).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Theoretical Foundation](#theoretical-foundation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Overview

This system models MRI scan sequences as a combination of:

1. **Symbolic sequences**: The order and type of scan procedures (sourceID tokens)
2. **Numerical counts**: The duration of each step with uncertainty quantification

### Key Features

- ✅ **Two-stage generation**: Separates structure from quantity
- ✅ **Uncertainty modeling**: Outputs μ and σ for each prediction
- ✅ **Autoregressive sequence generation**: Coherent symbolic sequences
- ✅ **Parallel count prediction**: Efficient numerical predictions
- ✅ **Conditioning on context**: Patient demographics and scan parameters
- ✅ **Flexible sampling**: Temperature, top-k, and nucleus sampling
- ✅ **Gamma distribution**: Realistic positive count sampling

---

## Architecture

### High-Level Pipeline

```
Conditioning (Patient/Scan Context)
         ↓
┌────────────────────────────────────┐
│ Conditional Sequence Generator      │
│ (Transformer Encoder-Decoder)       │
│ • Auto-regressive token generation  │
│ • Categorical distribution          │
└────────────────────────────────────┘
         ↓
   Symbolic Sequence
         ↓
┌────────────────────────────────────┐
│ Conditional Counts Generator        │
│ (Transformer Encoder + Cross-Attn) │
│ • Parallel count prediction         │
│ • Outputs μ and σ per position      │
└────────────────────────────────────┘
         ↓
   Sample from Gamma(μ, σ)
         ↓
   Realistic Count Sequences
```

### Model Components

#### 1. Conditional Sequence Generator

**Purpose**: Generate plausible symbolic sequences (scan procedure orders)

**Architecture**:
- **Input**: Patient/scan conditioning features (Age, Weight, Height, BodyGroup, etc.)
- **Encoder**: Processes conditioning into memory representation
- **Decoder**: Auto-regressive generation with causal masking
- **Output**: Token probabilities at each step

**Key Properties**:
- Auto-regressive: Each token depends on previous tokens
- Categorical distribution: Samples from vocabulary
- Temperature/top-k/nucleus sampling for diversity

**Training**:
- Loss: Cross-entropy with label smoothing
- Metrics: Accuracy, Perplexity
- Teacher forcing during training

#### 2. Conditional Counts Generator

**Purpose**: Predict numerical durations with uncertainty

**Architecture**:
- **Conditioning Encoder**: Processes patient/scan context
- **Sequence Encoder**: Processes symbolic sequence + features
- **Cross-Attention**: Sequence attends to conditioning
- **Dual Output Heads**: Predicts μ (mean) and σ (uncertainty)

**Key Properties**:
- Non-autoregressive: All positions predicted in parallel
- Parametric output: Gamma distribution parameters
- Uncertainty quantification: Explicit σ for each prediction

**Training**:
- Loss: Negative log-likelihood of Gamma distribution
- Metrics: MAE, RMSE, MAPE
- Gamma parameterization: α = (μ/σ)², β = μ/σ²

---

## Theoretical Foundation

### Probabilistic Formulation

Given:
- Conditioning sequence: **c** = [c₁, ..., cₗ] (patient/scan context)
- Symbolic sequence: **x** = [x₁, ..., xₜ] (scan procedure tokens)
- Numeric sequence: **y** = [y₁, ..., yₜ] (step durations)

#### Sequence Model (Symbolic)

```
P(x | c) = ∏ₜ P(xₜ | x₍<t₎, c)
```

- Each xₜ is **categorical** over vocabulary
- **Auto-regressive**: xₜ depends on all previous tokens
- Sampling: Sequential, token-by-token

#### Counts Model (Numeric)

```
P(y | c, x) = ∏ₜ P(yₜ | c, x)
```

- Each yₜ follows **Gamma(α, β)** distribution
- **Conditionally independent**: yₜ ⊥ yₜ' | c, x
- Parameters: α = (μ/σ)², β = μ/σ²
- Sampling: Parallel, all positions simultaneously

### Why This Approach?

| Aspect | Benefit |
|--------|---------|
| **Separation of concerns** | Structure vs. quantity modeled independently |
| **Uncertainty quantification** | σ captures prediction confidence |
| **Interpretability** | Symbolic sequences are human-readable |
| **Efficiency** | Parallel count prediction (no sequential dependency) |
| **Flexibility** | Can sample multiple plausible outcomes |

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn tqdm
```

### Setup

1. Clone the repository
2. Navigate to `PXChange_Refactored/`
3. Ensure preprocessed data exists in `../PXChange/data/{dataset_id}/preprocessed_{dataset_id}.csv`

---

## Usage

### 1. Training Both Models

Train both the sequence generator and counts generator:

```bash
python main_pipeline.py train
```

**Options**:
```bash
python main_pipeline.py train \
    --dataset-ids 176401 176886 176887 \
    --batch-size 32 \
    --val-split 0.2 \
    --skip-sequence   # Skip sequence model training
    --skip-counts     # Skip counts model training
```

**Output**:
- `saved_models/sequence_model_best.pt`
- `saved_models/counts_model_best.pt`
- `saved_models/conditioning_scaler.pkl`
- Training curve plots

### 2. Generating Sequences

Generate new sequences using trained models:

```bash
python main_pipeline.py generate
```

**Options**:
```bash
python main_pipeline.py generate \
    --num-conditioning 10 \      # Number of conditioning examples
    --num-samples 5 \             # Samples per conditioning
    --output-file results.csv \  # Output filename
    --use-gpu                     # Use GPU if available
```

**Output**:
- `outputs/generated_sequences.csv` with columns:
  - `conditioning_idx`: Which conditioning example
  - `sample_idx`: Which sample for that conditioning
  - `step`: Step number in sequence
  - `token_id`, `token_name`: Symbolic token
  - `predicted_mu`, `predicted_sigma`: Predicted parameters
  - `sampled_duration`: Sampled duration
  - `total_time`: Total sequence time

### 3. Evaluating Results

Compare generated sequences to true data:

```bash
python main_pipeline.py evaluate \
    --generated-file generated_sequences.csv \
    --dataset-ids 176401 176886
```

**Output**:
- Distribution comparisons
- Token frequency analysis
- Statistical summaries

---

## Project Structure

```
PXChange_Refactored/
├── config.py                       # Configuration (all hyperparameters)
├── main_pipeline.py                # Main entry point
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py              # DataLoader and Dataset classes
│   └── sequence_encoder.py         # Encoding/decoding utilities
│
├── models/
│   ├── __init__.py
│   ├── layers.py                   # Shared components (positional encoding, etc.)
│   ├── conditional_sequence_generator.py    # Sequence model
│   └── conditional_counts_generator.py      # Counts model
│
├── training/
│   ├── __init__.py
│   ├── train_sequence_model.py     # Sequence model training
│   └── train_counts_model.py       # Counts model training
│
├── generation/
│   ├── __init__.py
│   └── generate_pipeline.py        # Complete generation pipeline
│
├── outputs/                        # Generated results
├── saved_models/                   # Trained model checkpoints
├── visualizations/                 # Training curves and plots
│
└── README.md                       # This file
```

---

## Model Details

### Conditional Sequence Generator

**Input**:
- Conditioning: `[batch_size, 6]` - (Age, Weight, Height, BodyGroup_from, BodyGroup_to, PTAB)
- Target tokens: `[batch_size, seq_len]` - sourceID sequence

**Architecture**:
- d_model: 256
- Attention heads: 8
- Encoder layers: 6
- Decoder layers: 6
- FFN dimension: 1024
- Dropout: 0.1

**Output**:
- Logits: `[batch_size, seq_len, vocab_size]`

**Vocabulary**: 18 tokens (START, END, PAD, 15 scan types)

### Conditional Counts Generator

**Input**:
- Conditioning: `[batch_size, 6]`
- Sequence tokens: `[batch_size, seq_len]`
- Sequence features: `[batch_size, seq_len, 2]` - (Position, Direction)

**Architecture**:
- d_model: 256
- Attention heads: 8
- Encoder layers: 6
- Cross-attention layers: 4
- FFN dimension: 1024
- Dropout: 0.1

**Output**:
- μ: `[batch_size, seq_len]` - Mean duration
- σ: `[batch_size, seq_len]` - Standard deviation (min: 0.1)

**Sampling**:
```python
alpha = (mu / sigma) ** 2
beta = mu / sigma ** 2
duration ~ Gamma(alpha, beta)
```

---

## Configuration

Edit `config.py` to customize:

### Model Architecture

```python
SEQUENCE_MODEL_CONFIG = {
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1
}

COUNTS_MODEL_CONFIG = {
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_cross_attention_layers': 4,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'min_sigma': 0.1
}
```

### Training

```python
SEQUENCE_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'early_stopping_patience': 15
}

COUNTS_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'early_stopping_patience': 15
}
```

### Sampling

```python
SEQUENCE_SAMPLING_CONFIG = {
    'temperature': 1.0,      # Higher = more random
    'top_k': 10,             # Top-k sampling
    'top_p': 0.9,            # Nucleus sampling
    'max_length': 128
}
```

---

## Examples

### Example 1: Train and Generate

```bash
# Train models
python main_pipeline.py train --dataset-ids 176401 176886

# Generate 100 sequences (10 conditioning × 10 samples each)
python main_pipeline.py generate --num-conditioning 10 --num-samples 10

# Evaluate
python main_pipeline.py evaluate
```

### Example 2: Custom Conditioning

Create a CSV file `custom_conditioning.csv`:

```csv
Age,Weight,Height,BodyGroup_from,BodyGroup_to,PTAB
45,70,1.75,1,1,-500000
60,80,1.80,3,3,-600000
```

Generate:

```bash
python main_pipeline.py generate \
    --conditioning-file custom_conditioning.csv \
    --num-samples 20 \
    --output-file custom_results.csv
```

### Example 3: Python API

```python
import torch
from models import ConditionalSequenceGenerator, ConditionalCountsGenerator
from generation import generate_sequences_and_counts
import numpy as np

# Load trained models
seq_model = ConditionalSequenceGenerator()
seq_model.load_state_dict(torch.load('saved_models/sequence_model_best.pt')['model_state_dict'])

counts_model = ConditionalCountsGenerator()
counts_model.load_state_dict(torch.load('saved_models/counts_model_best.pt')['model_state_dict'])

# Define conditioning
conditioning = np.array([[45, 70, 1.75, 1, 1, -500000]])  # Age, Weight, Height, etc.

# Generate
results = generate_sequences_and_counts(
    seq_model,
    counts_model,
    conditioning,
    num_samples=5
)

print(results)
```

---

## Citation

If you use this system, please cite:

```
@software{conditional_mri_generation,
  title={Conditional Generation System for MRI Scan Sequences},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Time-Series-Models}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
