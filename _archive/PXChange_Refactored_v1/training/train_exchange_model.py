"""
Training script for Exchange Model (Body Region Transitions)

Trains a classifier to predict: current_region -> next_region
given patient conditioning features.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXCHANGE_MODEL_CONFIG, EXCHANGE_TRAINING_CONFIG,
    MODEL_SAVE_DIR, BODY_REGIONS, START_REGION_ID, END_REGION_ID
)
from models.exchange_model import create_exchange_model
from preprocessing.exchange_dataset import load_exchange_data, create_exchange_dataloaders


def calculate_accuracy(logits, targets):
    """Calculate classification accuracy."""
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        accuracy = correct.mean()
    return accuracy.item()


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        current_region = batch['current_region'].to(device)
        next_region = batch['next_region'].to(device)

        # Forward pass
        logits = model(conditioning, current_region)

        # Compute loss
        loss = criterion(logits, next_region)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        optimizer.step()

        # Metrics
        accuracy = calculate_accuracy(logits, next_region)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Validation"):
        # Move data to device
        conditioning = batch['conditioning'].to(device)
        current_region = batch['current_region'].to(device)
        next_region = batch['next_region'].to(device)

        # Forward pass
        logits = model(conditioning, current_region)

        # Compute loss
        loss = criterion(logits, next_region)

        # Metrics
        accuracy = calculate_accuracy(logits, next_region)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def train_exchange_model(dataset_id=None, epochs=None, save_name=None):
    """
    Train the Exchange Model.

    Args:
        dataset_id: Optional specific dataset ID to train on
        epochs: Override number of epochs
        save_name: Override save name for the model

    Returns:
        model: Trained model
        history: Training history dict
    """
    config = EXCHANGE_TRAINING_CONFIG.copy()
    if epochs is not None:
        config['epochs'] = epochs

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    if dataset_id:
        df = load_exchange_data(dataset_ids=[dataset_id], use_combined=False)
    else:
        df = load_exchange_data()

    train_loader, val_loader, scaler = create_exchange_dataloaders(
        df,
        batch_size=config['batch_size'],
        validation_split=config['validation_split']
    )

    # Create model
    print("\nCreating model...")
    model = create_exchange_model()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0)
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, config)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            if save_name is None:
                save_name = f"exchange_model_{dataset_id}" if dataset_id else "exchange_model"
            save_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_best.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'epoch': epoch,
                'config': EXCHANGE_MODEL_CONFIG,
                'history': history
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save final model
    if save_name is None:
        save_name = f"exchange_model_{dataset_id}" if dataset_id else "exchange_model"
    final_save_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'epoch': epoch,
        'config': EXCHANGE_MODEL_CONFIG,
        'history': history
    }, final_save_path)
    print(f"\nSaved final model to {final_save_path}")

    # Plot training curves
    plot_training_curves(history, save_name)

    return model, history


def plot_training_curves(history, save_name):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Exchange Model - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Exchange Model - Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}_curves.png")
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")
    plt.close()


def evaluate_model(model, dataloader, device):
    """Evaluate model and print detailed statistics."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            conditioning = batch['conditioning'].to(device)
            current_region = batch['current_region'].to(device)
            next_region = batch['next_region'].to(device)

            logits = model(conditioning, current_region)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(next_region.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Overall accuracy
    accuracy = (all_predictions == all_targets).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    region_names = BODY_REGIONS + ['START', 'END']

    for class_id in range(len(region_names)):
        mask = all_targets == class_id
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == class_id).mean()
            print(f"  {region_names[class_id]:10s}: {class_acc:.3f} ({mask.sum()} samples)")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exchange Model")
    parser.add_argument("--dataset-id", type=str, default=None,
                        help="Specific dataset ID to train on")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Name for saved model files")

    args = parser.parse_args()

    model, history = train_exchange_model(
        dataset_id=args.dataset_id,
        epochs=args.epochs,
        save_name=args.save_name
    )
