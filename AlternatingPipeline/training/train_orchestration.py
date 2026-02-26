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
from training.utils import temporal_split


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = create_orchestration_model(config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
