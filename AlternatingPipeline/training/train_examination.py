"""
Training script for the Examination Model (Unified Transformer).

Trains the model to generate MRI event sequences for specific body regions.
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
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG,
    MODEL_SAVE_DIR, RANDOM_SEED, USE_GPU, MAX_SEQ_LEN,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID, ID_TO_BODY_REGION
)
from models.examination_model import create_examination_model
from data.preprocessing import load_preprocessed_data
from data.archive_duration_priors import load_examination_priors
from training.utils import temporal_split, build_conditioning_tensor


class ExaminationDataset(Dataset):
    """Dataset for examination (scan sequence) training."""

    def __init__(self, examination_sequences, max_seq_len=None, augment=False):
        if max_seq_len is None:
            max_seq_len = MAX_SEQ_LEN

        self.max_seq_len = max_seq_len
        self.augment = augment
        self.data = []

        for seq in examination_sequences:
            conditioning = build_conditioning_tensor(seq['conditioning'])

            body_region = seq['body_region']
            tokens = seq['sequence']
            durations = seq.get('durations', [0.0] * len(tokens))

            # Input: [START, tok1, tok2, ..., tokN]
            # Target: [tok1, tok2, ..., tokN, END]
            input_seq = [START_TOKEN_ID] + tokens[:max_seq_len - 1]
            target_seq = tokens[:max_seq_len - 1] + [END_TOKEN_ID]
            target_durations = durations[:max_seq_len - 1] + [0.0]

            # Pad
            pad_len = max_seq_len - len(input_seq)
            input_seq = input_seq + [PAD_TOKEN_ID] * pad_len
            target_seq = target_seq + [PAD_TOKEN_ID] * pad_len
            target_durations = target_durations + [0.0] * pad_len

            self.data.append({
                'conditioning': conditioning,
                'body_region': body_region,
                'input_seq': torch.tensor(input_seq, dtype=torch.long),
                'target_seq': torch.tensor(target_seq, dtype=torch.long),
                'target_durations': target_durations,  # kept as list for augmentation
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        durations = list(item['target_durations'])
        if self.augment:
            noise = np.random.normal(0, 0.10, len(durations))
            durations = [max(0.0, d * (1 + n)) for d, n in zip(durations, noise)]
        return (
            item['conditioning'],
            torch.tensor(item['body_region'], dtype=torch.long),
            item['input_seq'],
            item['target_seq'],
            torch.tensor(durations, dtype=torch.float32),
        )


def train_examination_model(data_path=None, config=None, training_config=None,
                            save_dir=None, verbose=True):
    """
    Train the Examination Model.

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
        config = EXAMINATION_MODEL_CONFIG
    if training_config is None:
        training_config = EXAMINATION_TRAINING_CONFIG
    if save_dir is None:
        save_dir = os.path.join(MODEL_SAVE_DIR, 'examination')

    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Load archive duration priors
    archive_priors = load_examination_priors()
    if verbose:
        if archive_priors:
            print(f"Loaded archive priors for {len(archive_priors)} body groups: {list(archive_priors.keys())}")
        else:
            print("No archive priors found — prior regularization disabled.")

    # Load data
    if verbose:
        print("Loading data...")

    if data_path is None:
        preprocessed = load_preprocessed_data()
    else:
        with open(data_path, 'rb') as f:
            preprocessed = pickle.load(f)

    examination_sequences = preprocessed['examination']
    if verbose:
        print(f"Loaded {len(examination_sequences)} examination sequences")

    # Temporal split
    train_sequences, val_sequences = temporal_split(examination_sequences, val_days=2)

    if verbose:
        print(f"Temporal split: Train={len(train_sequences)}, Val={len(val_sequences)}")
        if train_sequences and val_sequences:
            train_dates = [s['start_datetime'] for s in train_sequences]
            val_dates = [s['start_datetime'] for s in val_sequences]
            print(f"  Train date range: {min(train_dates)} to {max(train_dates)}")
            print(f"  Val date range: {min(val_dates)} to {max(val_dates)}")

    # Create datasets
    augment = training_config.get('augment_training', False)
    train_dataset = ExaminationDataset(train_sequences, augment=augment)
    val_dataset = ExaminationDataset(val_sequences, augment=False)

    if verbose:
        print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")

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
    model = create_examination_model(config)
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
    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': [], 'train_duration_loss': [], 'train_prior_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(training_config['epochs']):
        model.train()
        train_loss = 0.0
        train_dur_loss = 0.0
        train_prior_loss = 0.0

        for conditioning, body_region, input_seq, target_seq, target_durations in tqdm(
            train_loader, disable=not verbose, desc=f"Epoch {epoch+1}"
        ):
            conditioning = conditioning.to(device)
            body_region = body_region.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            target_durations = target_durations.to(device)

            optimizer.zero_grad()

            logits, duration_mu, duration_sigma = model(
                conditioning, {'body_region': body_region}, input_seq
            )

            token_loss = model.compute_loss(
                logits, target_seq,
                label_smoothing=training_config['label_smoothing']
            )

            pad_mask = (target_seq == PAD_TOKEN_ID)
            duration_loss = model.compute_duration_loss(
                duration_mu, duration_sigma, target_durations, ignore_mask=pad_mask
            )

            duration_weight = training_config.get('duration_loss_weight', 0.5)
            loss = token_loss + duration_weight * duration_loss

            # Prior regularization: penalize mean prediction deviating from archive prior
            batch_prior_loss = 0.0
            if archive_priors:
                prior_weight = training_config.get('prior_loss_weight', 0.1)
                # Use the mode body region in this batch for the prior lookup
                mode_region_id = int(body_region.mode().values.item())
                body_region_name = ID_TO_BODY_REGION.get(mode_region_id, '').capitalize()
                if body_region_name in archive_priors:
                    prior_mean = archive_priors[body_region_name]['mean']
                    prior_std = archive_priors[body_region_name]['std']
                    prior_loss = ((duration_mu.mean() - prior_mean) / prior_std).pow(2)
                    loss = loss + prior_weight * prior_loss
                    batch_prior_loss = prior_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()
            train_dur_loss += duration_loss.item()
            train_prior_loss += batch_prior_loss

        train_loss /= len(train_loader)
        train_dur_loss /= len(train_loader)
        train_prior_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for conditioning, body_region, input_seq, target_seq, target_durations in val_loader:
                conditioning = conditioning.to(device)
                body_region = body_region.to(device)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                target_durations = target_durations.to(device)

                logits, duration_mu, duration_sigma = model(
                    conditioning, {'body_region': body_region}, input_seq
                )
                token_loss = model.compute_loss(logits, target_seq)
                pad_mask = (target_seq == PAD_TOKEN_ID)
                dur_loss = model.compute_duration_loss(
                    duration_mu, duration_sigma, target_durations, ignore_mask=pad_mask
                )
                duration_weight = training_config.get('duration_loss_weight', 0.5)
                loss = token_loss + duration_weight * dur_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_perplexity = np.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)
        history['train_duration_loss'].append(train_dur_loss)
        history['train_prior_loss'].append(train_prior_loss)

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}, "
                  f"dur_loss={train_dur_loss:.4f}, prior_loss={train_prior_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_final.pt'))

    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    if verbose:
        print(f"\nTraining complete. Models saved to {save_dir}")

    return model, history


if __name__ == "__main__":
    print("Training Examination Model...")
    print("=" * 60)

    model, history = train_examination_model(verbose=True)

    print("\nFinal Results:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation perplexity: {min(history['val_perplexity']):.2f}")
