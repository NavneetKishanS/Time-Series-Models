"""
Training script for the Conditional Counts Generator model.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COUNTS_MODEL_CONFIG, COUNTS_TRAINING_CONFIG,
    CONDITIONING_FEATURES, MODEL_SAVE_DIR
)
from models.conditional_counts_generator import ConditionalCountsGenerator
from preprocessing.data_loader import load_preprocessed_data, create_dataloaders


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot and save training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Counts Model Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (NLL)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

def gamma_nll_loss(mu, sigma, y_true, min_sigma=0.01):
    """
    Negative Log-Likelihood loss for a Gamma distribution.
    Assumes mu and sigma are parameters of the distribution.
    
    Args:
        mu (torch.Tensor): Mean of the distribution.
        sigma (torch.Tensor): Standard deviation.
        y_true (torch.Tensor): True values.
        min_sigma (float): Minimum value for sigma to avoid division by zero.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Clamp sigma to avoid numerical issues
    sigma = torch.clamp(sigma, min=min_sigma)
    
    # Convert (mu, sigma) to (shape, rate) of Gamma distribution
    sigma_sq = sigma.pow(2)
    shape = mu.pow(2) / sigma_sq
    rate = mu / sigma_sq
    
    # Ensure shape and rate are positive
    shape = torch.clamp(shape, min=1e-6)
    rate = torch.clamp(rate, min=1e-6)

    # Create Gamma distribution
    gamma_dist = torch.distributions.Gamma(shape, rate)
    
    # Calculate negative log-likelihood
    # Add small epsilon to y_true to avoid log(0) for zero durations
    nll = -gamma_dist.log_prob(y_true + 1e-8)
    
    return nll.mean()


def train_epoch(model, dataloader, optimizer, device, scaler=None, grad_clip_value=1.0):
    """
    Run one training epoch.
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training epoch", leave=False):
        # Unpack batch and move to device
        conditioning = batch['conditioning'].to(device)
        seq_tokens = batch['sequence_tokens'].to(device)
        seq_features = batch['sequence_features'].to(device)
        seq_counts = batch['step_durations'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(conditioning, seq_tokens, seq_features, mask)

        # Calculate loss, applying mask
        loss = gamma_nll_loss(mu[mask], sigma[mask], seq_counts[mask])
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device, scaler=None):
    """
    Run one validation epoch.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation epoch", leave=False):
            # Unpack batch and move to device
            conditioning = batch['conditioning'].to(device)
            seq_tokens = batch['sequence_tokens'].to(device)
            seq_features = batch['sequence_features'].to(device)
            seq_counts = batch['step_durations'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            mu, sigma = model(conditioning, seq_tokens, seq_features, mask)

            # Calculate loss
            loss = gamma_nll_loss(mu[mask], sigma[mask], seq_counts[mask])
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_counts_model(data_path, validation_split=0.15, save_model=True, plot_curves=True):
    """
    Main training loop for the counts model.
    """
    # Config
    config = COUNTS_TRAINING_CONFIG
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading preprocessed data and creating dataloaders...")
    df = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df, batch_size=config['batch_size'], validation_split=validation_split)

    print(f"\nTraining data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")

    # Initialize model
    model_config = COUNTS_MODEL_CONFIG
    model = ConditionalCountsGenerator(model_config).to(device)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.2
    )

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, config['grad_clip_value'])
        train_losses.append(train_loss)

        # Validation
        val_loss = validate_epoch(model, val_loader, device, scaler)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Learning rate update
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                model_path = os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pth')
                torch.save(model.state_dict(), model_path)
                print(f"  -> Best model saved to {model_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Plot training curves
    if plot_curves:
        save_path = os.path.join(MODEL_SAVE_DIR, 'counts_training_curves.png')
        plot_training_curves(train_losses, val_losses, save_path=save_path)
        
    # --- Final evaluation and prediction example ---
    print("\n--- Generating example prediction ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pth')))
    model.eval()

    with torch.no_grad():
        # Get one sample batch from validation set
        sample_batch = next(iter(val_loader))

        sample_cond = sample_batch['conditioning'][0:1].to(device)
        sample_tok = sample_batch['sequence_tokens'][0:1].to(device)
        sample_feat = sample_batch['sequence_features'][0:1].to(device)
        sample_counts = sample_batch['step_durations'][0:1].to(device)
        sample_mask = sample_batch['mask'][0:1].to(device)

        # Predict mu and sigma
        mu, sigma = model(sample_cond, sample_tok, sample_feat, sample_mask)
        
        # Get valid steps
        valid_mask = sample_mask.squeeze(0)
        true_counts = sample_counts.squeeze(0)[valid_mask].cpu().numpy()
        pred_mu = mu.squeeze(0)[valid_mask].cpu().numpy()
        pred_sigma = sigma.squeeze(0)[valid_mask].cpu().numpy()
        
        print("\nExample Prediction:")
        print(f"  Conditioning: {sample_cond.squeeze(0).cpu().numpy()}")
        print("  Step | True Duration | Pred Mean | Pred StdDev")
        print("-" * 50)
        for i in range(len(true_counts)):
            print(f"  {i:<4} | {true_counts[i]:<13.2f} | {pred_mu[i]:<9.2f} | {pred_sigma[i]:<9.2f}")


    return best_val_loss


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_file_path = sys.argv[1]
        if not os.path.exists(data_file_path):
            print(f"Error: Data file not found at {data_file_path}")
            sys.exit(1)
    else:
        # Default path
        data_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'preprocessed', 'all_preprocessed.csv'
        )

    print(f"Using data file: {data_file_path}")
    train_counts_model(data_path=data_file_path)