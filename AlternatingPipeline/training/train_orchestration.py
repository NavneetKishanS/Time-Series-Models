"""
Training script for the Orchestration Model.

Trains the model to generate day-level body region sequences conditioned on
scanner identity, day-of-week, month, and historical body region distributions.
"""
import math
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ORCHESTRATION_MODEL_CONFIG, ORCHESTRATION_TRAINING_CONFIG,
    MODEL_SAVE_DIR, RANDOM_SEED, USE_GPU,
    ORCH_MAX_SEQ_LEN, ORCH_PAD_TOKEN_ID,
    START_REGION_ID, END_REGION_ID,
)
from models.orchestration_model import create_orchestration_model
from data.preprocessing import load_preprocessed_data
from data.orchestration_preprocessing import extract_orchestration_samples
from training.utils import temporal_split, make_pad_collate


class OrchestrationDataset(Dataset):
    """Dataset for orchestration (day-level body region sequence) training."""

    def __init__(self, samples, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = ORCH_MAX_SEQ_LEN

        self.max_seq_len = max_seq_len
        self.data = []

        for sample in samples:
            tokens = sample['tokens']
            conditioning = sample['conditioning']
            scanner_idx = sample['scanner_idx']

            # Input: [START, region1, region2, ..., regionN]
            # Target: [region1, region2, ..., regionN, END]
            input_seq = [START_REGION_ID] + tokens[:max_seq_len - 1]
            target_seq = tokens[:max_seq_len - 1] + [END_REGION_ID]

            # Pad to max_seq_len
            pad_len = max_seq_len - len(input_seq)
            input_seq = input_seq + [ORCH_PAD_TOKEN_ID] * pad_len
            target_seq = target_seq + [ORCH_PAD_TOKEN_ID] * pad_len

            self.data.append({
                'conditioning': torch.tensor(conditioning, dtype=torch.float32),
                'scanner_idx': scanner_idx,
                'input_seq': torch.tensor(input_seq, dtype=torch.long),
                'target_seq': torch.tensor(target_seq, dtype=torch.long),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['conditioning'],
            torch.tensor(item['scanner_idx'], dtype=torch.long),
            item['input_seq'],
            item['target_seq'],
        )


def compute_region_class_weights(samples, vocab_size, pad_token_id, smoothing=0.5):
    """Inverse-frequency class weights for the orchestration token loss.

    *** CURRENTLY UNUSED — DO NOT RE-ENABLE AS-IS. ***
    This is the same phantom-weight bug that collapsed the examination model
    (Stage 1): regions with ZERO target counts (NECK/HAND/FOOT/START in real
    schedules) get the capped inverse-frequency weight (~3.44 after mean
    normalisation) while every REAL region is crushed to ~0.006-0.025. Combined
    with label smoothing (0.1), the weighted cross-entropy is MINIMISED by
    putting ~91% of the softmax mass on the never-seen classes — measured
    analytically on real schedules: optimum = NECK/START/FOOT/HAND at 22.5%
    each, vs HEAD 1.7%. That is exactly the NECK+FOOT≈91% collapse observed in
    synthetic output, and why fixing the serve-time conditioning (7a58aca)
    changed nothing. Without weights the optimum equals the real region
    distribution (HEAD 28%, SPINE 20%, UNKNOWN 13%, ...), which is the goal.
    The "long tail" this weighting was meant to surface is better served by
    the per-scanner region_distribution conditioning, which is fed correctly
    at generation since 7a58aca.
    """
    counts = np.zeros(vocab_size, dtype=np.float64)
    for sample in samples:
        for tok in sample['tokens']:
            if 0 <= tok < vocab_size:
                counts[tok] += 1
        counts[END_REGION_ID] += 1  # every sample's target ends with END
    counts = counts + counts.sum() * 1e-6
    freq = counts / counts.sum()
    weights = (1.0 / freq) ** smoothing
    weights[pad_token_id] = 0.0
    nonzero = weights[weights > 0]
    weights = weights / nonzero.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_orchestration_model(data_path=None, config=None, training_config=None,
                               save_dir=None, verbose=True):
    """
    Train the Orchestration Model.

    Args:
        data_path: Path to preprocessed data pickle file
        config: Model config dict
        training_config: Training config dict
        save_dir: Directory to save model
        verbose: Print progress

    Returns:
        Trained model, training history
    """
    if config is None:
        config = ORCHESTRATION_MODEL_CONFIG
    if training_config is None:
        training_config = ORCHESTRATION_TRAINING_CONFIG
    if save_dir is None:
        save_dir = os.path.join(MODEL_SAVE_DIR, 'orchestration')

    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print("Loading data...")

    if data_path is None:
        preprocessed = load_preprocessed_data()
    else:
        with open(data_path, 'rb') as f:
            preprocessed = pickle.load(f)

    samples, scanner_to_idx = extract_orchestration_samples(preprocessed)
    if verbose:
        print(f"Extracted {len(samples)} orchestration samples from "
              f"{len(scanner_to_idx)} scanners")

    # Save scanner mapping for inference
    with open(os.path.join(save_dir, 'scanner_to_idx.pkl'), 'wb') as f:
        pickle.dump(scanner_to_idx, f)

    # Temporal split
    train_samples, val_samples = temporal_split(samples, val_days=2)

    if verbose:
        print(f"Temporal split: Train={len(train_samples)}, Val={len(val_samples)}")
        if train_samples and val_samples:
            train_dates = [s['start_datetime'] for s in train_samples]
            val_dates = [s['start_datetime'] for s in val_samples]
            print(f"  Train date range: {min(train_dates)} to {max(train_dates)}")
            print(f"  Val date range: {min(val_dates)} to {max(val_dates)}")

    # Create datasets
    train_dataset = OrchestrationDataset(train_samples)
    val_dataset = OrchestrationDataset(val_samples)

    if verbose:
        print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

    # Trim each batch to its longest real sequence — the tuple layout is
    # (conditioning, scanner_idx, input_seq, target_seq); positions 2/3 are the
    # per-token fields, measured off the PAD-terminated input_seq at position 2.
    collate = make_pad_collate(seq_indices=(2, 3), length_index=2,
                               pad_token_id=ORCH_PAD_TOKEN_ID)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    # Create model
    model = create_orchestration_model(config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # NOTE: inverse-frequency region class weighting was REMOVED here — the
    # phantom weights on zero-count regions (NECK/HAND/FOOT/START) made the
    # weighted loss optimal at ~91% mass on regions that never occur in real
    # schedules (see compute_region_class_weights docstring). The desired
    # per-scanner region mix is carried by the region_distribution
    # conditioning instead.
    if verbose:
        from collections import Counter
        _tok_counts = Counter(t for s in train_samples for t in s['tokens'])
        _region_tot = sum(c for t, c in _tok_counts.items() if 0 <= t < 11) or 1
        from config import BODY_REGIONS as _BR
        _top = sorted(((t, c) for t, c in _tok_counts.items() if 0 <= t < 11),
                      key=lambda x: -x[1])[:5]
        print("Train region distribution (top 5): " +
              ", ".join(f"{_BR[t]} {100*c/_region_tot:.0f}%" for t, c in _top))

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=1e-4,
    )

    total_steps = training_config['epochs'] * len(train_loader)

    def lr_lambda(step):
        warmup_steps = training_config['warmup_steps']
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [], 'val_perplexity': [],
    }
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(training_config['epochs']):
        model.train()
        train_loss = 0.0

        for batch_data in tqdm(train_loader, disable=not verbose, desc=f"Epoch {epoch+1}"):
            conditioning, scanner_ids, input_seq, target_seq = batch_data

            conditioning = conditioning.to(device)
            scanner_ids = scanner_ids.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            logits = model(conditioning, scanner_ids, input_seq)

            loss = model.compute_loss(
                logits, target_seq,
                label_smoothing=training_config['label_smoothing'],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config['gradient_clip']
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                conditioning, scanner_ids, input_seq, target_seq = batch_data

                conditioning = conditioning.to(device)
                scanner_ids = scanner_ids.to(device)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)

                logits = model(conditioning, scanner_ids, input_seq)

                loss = model.compute_loss(logits, target_seq)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_perplexity = np.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, 'orchestration_model_best.pt'),
            )
        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, 'orchestration_model_final.pt'),
    )

    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    if verbose:
        print(f"\nTraining complete. Models saved to {save_dir}")

    return model, history


if __name__ == "__main__":
    print("Training Orchestration Model...")
    print("=" * 60)

    model, history = train_orchestration_model(verbose=True)

    print("\nFinal Results:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation perplexity: {min(history['val_perplexity']):.2f}")
