"""
Training script for the Exchange Model.

Trains the model to predict body region transitions given patient conditioning.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXCHANGE_MODEL_CONFIG, EXCHANGE_TRAINING_CONFIG,
    MODEL_SAVE_DIR, RANDOM_SEED, USE_GPU
)
from models.exchange_model import create_exchange_model
from data.preprocessing import load_preprocessed_data


def temporal_split(sequences, val_days=2):
    """
    Split sequences temporally to prevent data leakage.

    Args:
        sequences: List of sequence dicts with 'start_datetime' key
        val_days: Number of days to hold out for validation

    Returns:
        train_sequences, val_sequences
    """
    sorted_seqs = sorted(sequences, key=lambda s: s['start_datetime'])

    if len(sorted_seqs) == 0:
        return [], []

    last_date = sorted_seqs[-1]['start_datetime']
    if hasattr(last_date, 'date'):
        last_date = last_date.date() if hasattr(last_date, 'date') else last_date

    if hasattr(last_date, '__sub__'):
        cutoff = last_date - timedelta(days=val_days)
    else:
        from datetime import datetime
        last_dt = datetime.combine(last_date, datetime.min.time())
        cutoff_dt = last_dt - timedelta(days=val_days)
        cutoff = cutoff_dt.date()

    train_sequences = []
    val_sequences = []

    for seq in sorted_seqs:
        seq_date = seq['start_datetime']
        if hasattr(seq_date, 'date'):
            seq_date = seq_date.date()

        if seq_date < cutoff:
            train_sequences.append(seq)
        else:
            val_sequences.append(seq)

    return train_sequences, val_sequences


def safe_float(val, default=0.0):
    """Safely convert a value to float, handling errors and NaN."""
    if val is None:
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


class ExchangeDataset(Dataset):
    """Dataset for exchange (body region transition) training."""

    def __init__(self, exchange_sequences):
        """
        Args:
            exchange_sequences: List of exchange sequence dicts from preprocessing
        """
        self.data = []

        for seq in exchange_sequences:
            # Extract conditioning features with safe conversion
            # NOW INCLUDES TEMPORAL FEATURES (10 dims instead of 5)
            cond = seq['conditioning']
            conditioning = torch.tensor([
                # Patient demographics (5 features)
                safe_float(cond.get('Age', 0)),
                safe_float(cond.get('Weight', 0)),
                safe_float(cond.get('Height', 0)),
                safe_float(cond.get('PTAB', 0)),
                safe_float(cond.get('Direction_encoded', 0)),
                # Temporal features (5 features) - NEW!
                safe_float(cond.get('hour_sin', 0.0)),
                safe_float(cond.get('hour_cos', 1.0)),
                safe_float(cond.get('dow_sin', 0.0)),
                safe_float(cond.get('dow_cos', 1.0)),
                safe_float(cond.get('is_morning', 0)),
            ], dtype=torch.float32)

            # Body region transition
            body_from = seq['body_from']
            body_to = seq['body_to']

            self.data.append({
                'conditioning': conditioning,
                'current_region': body_from,
                'target_region': body_to
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['conditioning'],
            torch.tensor(item['current_region'], dtype=torch.long),
            torch.tensor(item['target_region'], dtype=torch.long)
        )


def train_exchange_model(data_path=None, config=None, training_config=None,
                         save_dir=None, verbose=True):
    """
    Train the Exchange Model.

    Args:
        data_path: Path to preprocessed data pickle file
        config: Model config dict (default: EXCHANGE_MODEL_CONFIG)
        training_config: Training config dict (default: EXCHANGE_TRAINING_CONFIG)
        save_dir: Directory to save model (default: MODEL_SAVE_DIR)
        verbose: Print progress

    Returns:
        Trained model, training history
    """
    if config is None:
        config = EXCHANGE_MODEL_CONFIG
    if training_config is None:
        training_config = EXCHANGE_TRAINING_CONFIG
    if save_dir is None:
        save_dir = os.path.join(MODEL_SAVE_DIR, 'exchange')

    os.makedirs(save_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Device
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

    exchange_sequences = preprocessed['exchange']
    if verbose:
        print(f"Loaded {len(exchange_sequences)} exchange sequences")

    # TEMPORAL SPLIT (NEW: prevents data leakage from future patterns)
    train_sequences, val_sequences = temporal_split(exchange_sequences, val_days=2)

    if verbose:
        print(f"Temporal split: Train={len(train_sequences)}, Val={len(val_sequences)}")
        if train_sequences and val_sequences:
            train_dates = [s['start_datetime'] for s in train_sequences]
            val_dates = [s['start_datetime'] for s in val_sequences]
            print(f"  Train date range: {min(train_dates)} to {max(train_dates)}")
            print(f"  Val date range: {min(val_dates)} to {max(val_dates)}")

    # Create datasets
    train_dataset = ExchangeDataset(train_sequences)
    val_dataset = ExchangeDataset(val_sequences)

    if verbose:
        print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = create_exchange_model(config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(training_config['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for conditioning, current_region, target_region in tqdm(train_loader, disable=not verbose,
                                                                  desc=f"Epoch {epoch+1}"):
            conditioning = conditioning.to(device)
            current_region = current_region.to(device)
            target_region = target_region.to(device)

            optimizer.zero_grad()
            logits = model(conditioning, current_region)
            loss = criterion(logits, target_region)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for conditioning, current_region, target_region in val_loader:
                conditioning = conditioning.to(device)
                current_region = current_region.to(device)
                target_region = target_region.to(device)

                logits = model(conditioning, current_region)
                loss = criterion(logits, target_region)
                val_loss += loss.item()

                # Accuracy
                predictions = logits.argmax(dim=-1)
                correct += (predictions == target_region).sum().item()
                total += target_region.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'exchange_model_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'exchange_model_final.pt'))

    # Save history
    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    if verbose:
        print(f"\nTraining complete. Models saved to {save_dir}")

    return model, history


if __name__ == "__main__":
    print("Training Exchange Model...")
    print("=" * 60)

    model, history = train_exchange_model(verbose=True)

    print("\nFinal Results:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
