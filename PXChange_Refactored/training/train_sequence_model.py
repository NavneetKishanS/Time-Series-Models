"""
Training script for the Conditional Sequence Generator model.
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
    SEQUENCE_MODEL_CONFIG, SEQUENCE_TRAINING_CONFIG, SEQUENCE_SAMPLING_CONFIG, # Added SEQUENCE_SAMPLING_CONFIG
    CONDITIONING_FEATURES, PAD_TOKEN_ID,
    MODEL_SAVE_DIR, START_TOKEN_ID, END_TOKEN_ID
)
from models.conditional_sequence_generator import ConditionalSequenceGenerator
from preprocessing.data_loader import load_preprocessed_data, create_dataloaders
from preprocessing.sequence_encoder import sequence_to_text, decode_sequences


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot and save training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Sequence Model Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, grad_clip_value=1.0):
    """
    Run one training epoch.
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training epoch", leave=False):
        # Unpack batch and move to device
        conditioning = batch['conditioning'].to(device)
        sequence_tokens = batch['sequence_tokens'].to(device)
        seq_length = batch['seq_length'].to(device)

        # Create sequences_in (input to decoder, shifted right with START_TOKEN_ID)
        # sequences_in will have START_TOKEN_ID at position 0, and then the actual sequence tokens.
        # The target sequences_out will be the original sequence_tokens.
        sequences_in = torch.full_like(sequence_tokens, PAD_TOKEN_ID, device=device)
        sequences_in[:, 0] = START_TOKEN_ID # Prepend START token to all sequences

        # Shift original sequence tokens to the right for sequences_in
        for i in range(sequence_tokens.size(0)):
            current_len = seq_length[i].item() # Get the actual length for this sequence
            # Only shift if sequence has elements beyond the START_TOKEN_ID
            if current_len > 0:
                sequences_in[i, 1:current_len] = sequence_tokens[i, :current_len-1]
        
        sequences_out = sequence_tokens # The target for the loss calculation

        optimizer.zero_grad()

        # Forward pass
        logits = model(conditioning, sequences_in)

        # Reshape for loss calculation
        # Logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
        # Target: [batch, seq_len] -> [batch * seq_len]
        logits_flat = logits.view(-1, logits.shape[-1])
        sequences_out_flat = sequences_out.reshape(-1)

        # Calculate loss
        loss = criterion(logits_flat, sequences_out_flat)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        optimizer.step()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, scaler=None):
    """
    Run one validation epoch.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation epoch", leave=False):
            # Unpack batch and move to device
            conditioning = batch['conditioning'].to(device)
            sequence_tokens = batch['sequence_tokens'].to(device)
            seq_length = batch['seq_length'].to(device)

            # Create sequences_in (input to decoder, shifted right with START_TOKEN_ID)
            # sequences_in will have START_TOKEN_ID at position 0, and then the actual sequence tokens.
            # The target sequences_out will be the original sequence_tokens.
            sequences_in = torch.full_like(sequence_tokens, PAD_TOKEN_ID, device=device)
            sequences_in[:, 0] = START_TOKEN_ID # Prepend START token to all sequences

            # Shift original sequence tokens to the right for sequences_in
            for i in range(sequence_tokens.size(0)):
                current_len = seq_length[i].item() # Get the actual length for this sequence
                # Only shift if sequence has elements beyond the START_TOKEN_ID
                if current_len > 0:
                    sequences_in[i, 1:current_len] = sequence_tokens[i, :current_len-1]
            
            sequences_out = sequence_tokens # The target for the loss calculation
            
            # Forward pass
            logits = model(conditioning, sequences_in)

            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.shape[-1])
            sequences_out_flat = sequences_out.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_flat, sequences_out_flat)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_sequence_model(data_path, validation_split=0.15, save_model=True, plot_curves=True):
    """
    Main training loop for the sequence model.
    """
    # Config
    config = SEQUENCE_TRAINING_CONFIG
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading preprocessed data and creating dataloaders...")
    df = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df, batch_size=config['batch_size'], validation_split=validation_split)

    print(f"\nTraining data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    # Initialize model
    model_config = SEQUENCE_MODEL_CONFIG
    model = ConditionalSequenceGenerator(model_config).to(device)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, config['grad_clip_value'])
        train_losses.append(train_loss)

        # Validation
        val_loss = validate_epoch(model, val_loader, criterion, device, scaler)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Learning rate update
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pth')
                torch.save(model.state_dict(), model_path)
                print(f"  -> Best model saved to {model_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Plot training curves
    if plot_curves:
        save_path = os.path.join(MODEL_SAVE_DIR, 'sequence_training_curves.png')
        plot_training_curves(train_losses, val_losses, save_path=save_path)
        
    # --- Final evaluation and generation example ---
    print("\n--- Generating example sequence ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pth')))
    model.eval()

    with torch.no_grad():
        # Get one sample batch from validation set
        # Using next(iter(val_loader)) for simplicity to get one batch
        sample_batch = next(iter(val_loader))

        sample_cond = sample_batch['conditioning'][0:1].to(device)
        true_seq_out = sample_batch['sequence_tokens'][0:1].to(device) # Target is sequence_tokens

        # Generate sequence
        generated_seq = model.generate(
            sample_cond,
            max_length=model.max_seq_len,
            temperature=SEQUENCE_SAMPLING_CONFIG['temperature'],
            top_k=SEQUENCE_SAMPLING_CONFIG['top_k']
        )
        
        # Decode and print
        true_sequence_text = sequence_to_text(true_seq_out.cpu().numpy()[0])
        generated_sequence_text = sequence_to_text(generated_seq.cpu().numpy()[0])
        
        print(f"\nConditioning data (scaled):\n{pd.Series(sample_cond[0].cpu().numpy(), index=CONDITIONING_FEATURES)}")
        print(f"\nTrue sequence:\n  -> {' '.join(true_sequence_text)}")
        print(f"\nGenerated sequence:\n  -> {' '.join(generated_sequence_text)}")
        
        # Another example (more tokens)
        print("\n--- Another example ---")
        sample_cond_2 = sample_batch['conditioning'][1:2].to(device) # Get second sample
        true_seq_out_2 = sample_batch['sequence_tokens'][1:2].to(device)
        generated_seq_2 = model.generate(
            sample_cond_2,
            max_length=model.max_seq_len,
            temperature=SEQUENCE_SAMPLING_CONFIG['temperature'],
            top_k=SEQUENCE_SAMPLING_CONFIG['top_k']
        )
        true_text_2 = sequence_to_text(true_seq_out_2.cpu().numpy()[0])
        gen_text_2 = sequence_to_text(generated_seq_2.cpu().numpy()[0])
        print(f"\nTrue sequence 2:\n  -> {' '.join(true_text_2)}")
        print(f"\nGenerated sequence 2:\n  -> {' '.join(gen_text_2)}")

    return best_val_loss


if __name__ == '__main__':
    # Get data path from command line arguments
    if len(sys.argv) > 1:
        data_file_path = sys.argv[1]
        if not os.path.exists(data_file_path):
            print(f"Error: Data file not found at {data_file_path}")
            sys.exit(1)
    else:
        # Default path if not provided
        data_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'preprocessed', 'all_preprocessed.csv'
        )

    print(f"Using data file: {data_file_path}")
    train_sequence_model(data_path=data_file_path)