"""
Training script for Examination Model (Body-Region-Specific Sequence Generation)

Trains a sequence generator to produce MRI event sequences for specific body regions,
conditioned on patient features.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG,
    MODEL_SAVE_DIR, START_TOKEN_ID, PAD_TOKEN_ID, BODY_REGIONS
)
from models.examination_model import create_examination_model
from preprocessing.examination_dataset import (
    load_examination_data, create_examination_dataloaders
)
from preprocessing.sequence_encoder import create_decoder_input


def get_warmup_scheduler(optimizer, warmup_steps):
    """Learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def calculate_perplexity(logits, targets, ignore_index=PAD_TOKEN_ID):
    """Calculate perplexity metric."""
    with torch.no_grad():
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = nn.functional.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )
        perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_accuracy(logits, targets, ignore_index=PAD_TOKEN_ID):
    """Calculate token prediction accuracy."""
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        mask = targets != ignore_index
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        body_region = batch['body_region'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)

        # Create decoder input (shift right with START token)
        decoder_input = create_decoder_input(sequence_tokens, start_token_id=START_TOKEN_ID)

        # Forward pass
        logits = model(conditioning, body_region, decoder_input)

        # Compute loss
        loss = model.compute_loss(
            logits,
            sequence_tokens,
            ignore_index=PAD_TOKEN_ID,
            label_smoothing=config.get('label_smoothing', 0.1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            perplexity = calculate_perplexity(logits, sequence_tokens)
            accuracy = calculate_accuracy(logits, sequence_tokens)

        total_loss += loss.item()
        total_perplexity += perplexity
        total_accuracy += accuracy
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{perplexity:.2f}',
            'acc': f'{accuracy:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_perplexity, avg_accuracy


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        conditioning = batch['conditioning'].to(device)
        body_region = batch['body_region'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)

        decoder_input = create_decoder_input(sequence_tokens, start_token_id=START_TOKEN_ID)
        logits = model(conditioning, body_region, decoder_input)

        loss = model.compute_loss(
            logits,
            sequence_tokens,
            ignore_index=PAD_TOKEN_ID,
            label_smoothing=0.0
        )

        perplexity = calculate_perplexity(logits, sequence_tokens)
        accuracy = calculate_accuracy(logits, sequence_tokens)

        total_loss += loss.item()
        total_perplexity += perplexity
        total_accuracy += accuracy
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_perplexity, avg_accuracy


def train_examination_model(dataset_id=None, epochs=None, save_name=None,
                            filter_body_region=None):
    """
    Train the Examination Model.

    Args:
        dataset_id: Optional specific dataset ID to train on
        epochs: Override number of epochs
        save_name: Override save name for the model
        filter_body_region: Optional body region ID to filter training data

    Returns:
        model: Trained model
        history: Training history dict
    """
    config = EXAMINATION_TRAINING_CONFIG.copy()
    if epochs is not None:
        config['epochs'] = epochs

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    if dataset_id:
        df = load_examination_data(dataset_ids=[dataset_id], use_combined=False)
    else:
        df = load_examination_data()

    train_loader, val_loader, scaler = create_examination_dataloaders(
        df,
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        filter_body_region=filter_body_region
    )

    if len(train_loader.dataset) == 0:
        print("No training data available!")
        return None, None

    # Create model
    print("\nCreating model...")
    model = create_examination_model()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_warmup_scheduler(optimizer, config.get('warmup_steps', 4000))

    # Training history
    history = {
        'train_loss': [],
        'train_ppl': [],
        'train_acc': [],
        'val_loss': [],
        'val_ppl': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_loss, train_ppl, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_ppl, val_acc = validate_epoch(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['val_acc'].append(val_acc)

        print(f"Train - Loss: {train_loss:.4f}, PPL: {train_ppl:.2f}, Acc: {train_acc:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            if save_name is None:
                if filter_body_region is not None:
                    region_name = BODY_REGIONS[filter_body_region]
                    save_name = f"examination_model_{region_name}"
                elif dataset_id:
                    save_name = f"examination_model_{dataset_id}"
                else:
                    save_name = "examination_model"

            save_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_best.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'epoch': epoch,
                'config': EXAMINATION_MODEL_CONFIG,
                'history': history,
                'filter_body_region': filter_body_region
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save final model
    if save_name is None:
        save_name = "examination_model"
    final_save_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'epoch': epoch,
        'config': EXAMINATION_MODEL_CONFIG,
        'history': history,
        'filter_body_region': filter_body_region
    }, final_save_path)
    print(f"\nSaved final model to {final_save_path}")

    # Plot training curves
    plot_training_curves(history, save_name)

    return model, history


def plot_training_curves(history, save_name):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Examination Model - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Perplexity
    axes[1].plot(history['train_ppl'], label='Train')
    axes[1].plot(history['val_ppl'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Examination Model - Perplexity')
    axes[1].legend()
    axes[1].grid(True)

    # Accuracy
    axes[2].plot(history['train_acc'], label='Train')
    axes[2].plot(history['val_acc'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Examination Model - Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_curves.png")
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Examination Model")
    parser.add_argument("--dataset-id", type=str, default=None,
                        help="Specific dataset ID to train on")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Name for saved model files")
    parser.add_argument("--body-region", type=int, default=None,
                        help="Filter to specific body region ID")

    args = parser.parse_args()

    model, history = train_examination_model(
        dataset_id=args.dataset_id,
        epochs=args.epochs,
        save_name=args.save_name,
        filter_body_region=args.body_region
    )
