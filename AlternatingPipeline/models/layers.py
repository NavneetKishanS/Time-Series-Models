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


class SinglePassDurationHead(nn.Module):
    """
    Duration head for single-pass estimation over a complete sequence.

    Takes hidden states from a bidirectional encoder and predicts
    (mu, sigma) per token position using a shared MLP trunk with
    separate output heads.
    """

    def __init__(self, d_model, hidden_dim=128, dropout=0.1, min_sigma=0.1):
        super(SinglePassDurationHead, self).__init__()
        self.min_sigma = min_sigma

        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            mu: [batch_size, seq_len] - mean duration parameters
            sigma: [batch_size, seq_len] - std deviation parameters
        """
        h = self.shared_mlp(x)
        mu = torch.nn.functional.softplus(self.mu_head(h)).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(h)).squeeze(-1) + self.min_sigma
        return mu, sigma


class MixtureDurationHead(nn.Module):
    """Predict a conditional mixture of duration distributions per position."""

    def __init__(self, d_model, num_components=3, hidden_dim=128, dropout=0.1,
                 min_sigma=0.05, initial_means=None, initial_sigmas=None):
        super(MixtureDurationHead, self).__init__()
        if num_components < 2:
            raise ValueError("num_components must be at least 2 for a mixture head")

        self.num_components = int(num_components)
        self.min_sigma = float(min_sigma)
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mixture_logits_head = nn.Linear(hidden_dim, self.num_components)
        self.mu_head = nn.Linear(hidden_dim, self.num_components)
        self.sigma_head = nn.Linear(hidden_dim, self.num_components)

        self._initialise_component_biases(initial_means, initial_sigmas)

    @staticmethod
    def _inverse_softplus(values):
        values = torch.as_tensor(values, dtype=torch.float32).clamp_min(1e-4)
        return torch.log(torch.expm1(values))

    def _initialise_component_biases(self, initial_means, initial_sigmas):
        if initial_means is None:
            initial_means = torch.linspace(0.25, 1.75, self.num_components)
        if initial_sigmas is None:
            initial_sigmas = torch.linspace(0.20, 0.50, self.num_components)
        if len(initial_means) != self.num_components:
            raise ValueError("initial_means must match num_components")
        if len(initial_sigmas) != self.num_components:
            raise ValueError("initial_sigmas must match num_components")

        means = torch.as_tensor(initial_means, dtype=torch.float32)
        sigmas = torch.as_tensor(initial_sigmas, dtype=torch.float32)
        if (means <= 0).any():
            raise ValueError("initial_means must be positive")
        if (sigmas <= self.min_sigma).any():
            raise ValueError("initial_sigmas must be greater than min_sigma")

        with torch.no_grad():
            self.mixture_logits_head.bias.zero_()
            self.mu_head.bias.copy_(self._inverse_softplus(means))
            self.sigma_head.bias.copy_(
                self._inverse_softplus(sigmas - self.min_sigma)
            )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            mixture_logits: [batch_size, seq_len, num_components]
            mu: [batch_size, seq_len, num_components]
            sigma: [batch_size, seq_len, num_components]
        """
        h = self.shared_mlp(x)
        mixture_logits = self.mixture_logits_head(h)
        mu = torch.nn.functional.softplus(self.mu_head(h))
        sigma = (
            torch.nn.functional.softplus(self.sigma_head(h)) + self.min_sigma
        )
        return mixture_logits, mu, sigma


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
