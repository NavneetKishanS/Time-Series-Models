# Quick Start Guide

Get up and running with the Conditional Generation System in 5 minutes.

## Prerequisites

Ensure you have:
1. Python 3.8+
2. PyTorch installed
3. Preprocessed data in `../PXChange/data/`

## Step 1: Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Step 2: Verify Data

Check that preprocessed data exists:

```bash
python -c "from preprocessing import load_preprocessed_data; df = load_preprocessed_data(); print(f'Loaded {len(df)} rows')"
```

Expected output:
```
✓ Loaded dataset 176401: 1234 sequences
✓ Loaded dataset 176886: 567 sequences
...
✓ Total loaded: XXXX sequences from Y datasets
```

## Step 3: Train Models (20-60 minutes)

```bash
python main_pipeline.py train
```

This will:
- ✅ Load and prepare data
- ✅ Train Conditional Sequence Generator
- ✅ Train Conditional Counts Generator
- ✅ Save models to `saved_models/`
- ✅ Generate training curve plots

**Faster testing** (reduced model size):

Edit `config.py` and set:
```python
SEQUENCE_MODEL_CONFIG = {
    'd_model': 128,  # Reduced from 256
    'num_encoder_layers': 3,  # Reduced from 6
    'num_decoder_layers': 3,
    ...
}

SEQUENCE_TRAINING_CONFIG = {
    'epochs': 20,  # Reduced from 100
    ...
}
```

## Step 4: Generate Sequences (1-5 minutes)

```bash
python main_pipeline.py generate --num-conditioning 5 --num-samples 10
```

This will:
- ✅ Load trained models
- ✅ Generate 50 sequences (5 × 10)
- ✅ Save to `outputs/generated_sequences.csv`
- ✅ Print example sequences

## Step 5: Evaluate Results

```bash
python main_pipeline.py evaluate
```

This will:
- ✅ Compare generated vs. true sequences
- ✅ Show distribution statistics
- ✅ Display token frequencies

## Example Output

### Generation Examples

```
Example 1 (Conditioning=0, Sample=0):
  Length: 25 steps
  Sequence: START -> MRI_CCS_11 -> MRI_FRR_18 -> ... -> END
  Durations (first 10): [0.0, 12.3, 8.1, 15.6, ...]
  Total time: 342.5s

Example 2 (Conditioning=0, Sample=1):
  Length: 28 steps
  Sequence: START -> MRI_CCS_11 -> MRI_FRR_3 -> ... -> END
  Durations (first 10): [0.0, 10.8, 9.4, 14.2, ...]
  Total time: 389.2s
```

### Evaluation Statistics

```
Sequence Length:
  True - Mean: 26.3, Std: 8.5, Range: [10, 85]
  Generated - Mean: 25.7, Std: 7.2, Range: [12, 72]

Total Time (seconds):
  True - Mean: 456.2, Std: 182.3, Range: [120.0, 1800.0]
  Generated - Mean: 438.9, Std: 176.8, Range: [115.3, 1750.2]
```

## Next Steps

### Customize Conditioning

Create `my_conditioning.csv`:
```csv
Age,Weight,Height,BodyGroup_from,BodyGroup_to,PTAB
35,65,1.70,1,1,-500000
50,75,1.80,3,8,-600000
```

Generate:
```bash
python main_pipeline.py generate \
    --conditioning-file my_conditioning.csv \
    --num-samples 20 \
    --output-file my_results.csv
```

### Adjust Sampling Parameters

Edit `config.py`:
```python
SEQUENCE_SAMPLING_CONFIG = {
    'temperature': 1.2,  # More diversity (default: 1.0)
    'top_k': 5,          # More focused (default: 10)
    'top_p': 0.95,       # Include more tail (default: 0.9)
}
```

Then regenerate:
```bash
python main_pipeline.py generate
```

### Use Python API

```python
import torch
import numpy as np
from models import ConditionalSequenceGenerator, ConditionalCountsGenerator
from generation import generate_sequences_and_counts

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_checkpoint = torch.load('saved_models/sequence_model_best.pt', map_location=device)
seq_model = ConditionalSequenceGenerator(seq_checkpoint['config'])
seq_model.load_state_dict(seq_checkpoint['model_state_dict'])
seq_model.to(device)
seq_model.eval()

counts_checkpoint = torch.load('saved_models/counts_model_best.pt', map_location=device)
counts_model = ConditionalCountsGenerator(counts_checkpoint['config'])
counts_model.load_state_dict(counts_checkpoint['model_state_dict'])
counts_model.to(device)
counts_model.eval()

# Generate
conditioning = np.array([[45, 70, 1.75, 1, 1, -500000]])
results = generate_sequences_and_counts(
    seq_model, counts_model, conditioning, num_samples=5, device=device
)

print(results.head(20))
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in `config.py`:
```python
SEQUENCE_TRAINING_CONFIG = {
    'batch_size': 16,  # Reduced from 32
    ...
}
```

### Issue: Models not found

**Solution**: Run training first:
```bash
python main_pipeline.py train
```

### Issue: Data not found

**Solution**: Check data path in `config.py`:
```python
DATA_DIR = 'C:/path/to/your/data'
```

### Issue: Training too slow

**Solution**:
1. Use GPU: Ensure CUDA is available
2. Reduce model size (see Step 3 above)
3. Train on subset:
```bash
python main_pipeline.py train --dataset-ids 176401
```

## Common Commands Cheatsheet

```bash
# Train everything
python main_pipeline.py train

# Train only sequence model
python main_pipeline.py train --skip-counts

# Train only counts model
python main_pipeline.py train --skip-sequence

# Generate with defaults
python main_pipeline.py generate

# Generate many samples
python main_pipeline.py generate --num-conditioning 20 --num-samples 50

# Evaluate
python main_pipeline.py evaluate

# Evaluate custom file
python main_pipeline.py evaluate --generated-file my_results.csv
```

## Help

```bash
python main_pipeline.py --help
python main_pipeline.py train --help
python main_pipeline.py generate --help
python main_pipeline.py evaluate --help
```

---

**That's it!** You now have a working conditional generation system for MRI scan sequences.

For more details, see [README.md](README.md).
