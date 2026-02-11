"""
Shared layers and components for transformer models.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention is All You Need".
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DurationHead(nn.Module):
    """
    Output head that predicts duration parameters (mu, sigma) using Gaussian distribution.
    Used for per-token duration prediction in examination model.
    """

    def __init__(self, d_model, min_sigma=0.1):
        super(DurationHead, self).__init__()
        self.min_sigma = min_sigma

        self.mu_head = nn.Linear(d_model, 1)
        self.sigma_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            mu: [batch_size, seq_len] - mean parameters
            sigma: [batch_size, seq_len] - standard deviation parameters
        """
        mu = torch.nn.functional.softplus(self.mu_head(x)).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(x)).squeeze(-1) + self.min_sigma

        return mu, sigma


def create_attention_mask(seq_len, causal=False, device='cpu'):
    """
    Create attention mask for transformer.

    Args:
        seq_len: Sequence length
        causal: If True, creates causal (look-ahead) mask
        device: torch device

    Returns:
        mask: Attention mask [seq_len, seq_len]
    """
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    else:
        mask = torch.zeros(seq_len, seq_len, device=device).bool()

    return mask


def create_key_padding_mask(sequence_tokens, pad_token_id=0):
    """
    Create key padding mask for transformer.

    Args:
        sequence_tokens: [batch_size, seq_len]
        pad_token_id: ID of padding token

    Returns:
        mask: [batch_size, seq_len] - True for padding positions
    """
    return sequence_tokens == pad_token_id
