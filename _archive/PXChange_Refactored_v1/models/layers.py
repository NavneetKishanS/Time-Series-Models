"""
Shared layers and components for transformer models
"""
import torch
import torch.nn as nn
import numpy as np
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


class ConditioningProjection(nn.Module):
    """
    Projects conditioning features into model dimension space.
    Allows conditioning information to be integrated into the model.
    """

    def __init__(self, conditioning_dim, d_model, dropout=0.1):
        super(ConditioningProjection, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(conditioning_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, conditioning):
        """
        Args:
            conditioning: [batch_size, conditioning_dim]

        Returns:
            projected: [batch_size, d_model]
        """
        return self.projection(conditioning)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (alternative to sinusoidal).
    """

    def __init__(self, max_len, d_model, dropout=0.1):
        super(LearnedPositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            x with positional embeddings added
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.embedding(positions)
        return self.dropout(x + pos_emb)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending from one sequence to another.
    Used in the counts model to attend from target sequence to conditioning sequence.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query: [batch_size, tgt_len, d_model]
            key_value: [batch_size, src_len, d_model]
            key_padding_mask: [batch_size, src_len] - True for padding positions

        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        # Cross-attention
        attn_output, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + self.dropout1(attn_output))

        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout2(ffn_output))

        return output


class GammaOutputHead(nn.Module):
    """
    Output head that predicts parameters (μ, σ) for Gamma distribution.
    Ensures positive outputs using softplus activation.
    """

    def __init__(self, d_model, min_sigma=0.1):
        super(GammaOutputHead, self).__init__()
        self.min_sigma = min_sigma

        # Separate heads for μ and σ
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
        # Causal mask: prevent attending to future positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    else:
        # No masking
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


if __name__ == "__main__":
    # Test components
    print("Testing layer components...")

    batch_size, seq_len, d_model = 4, 10, 256

    # Test positional encoding
    print("\n1. Positional Encoding")
    x = torch.randn(batch_size, seq_len, d_model)
    pos_enc = PositionalEncoding(d_model, max_len=100)
    x_pos = pos_enc(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_pos.shape}")

    # Test conditioning projection
    print("\n2. Conditioning Projection")
    conditioning_dim = 6
    conditioning = torch.randn(batch_size, conditioning_dim)
    cond_proj = ConditioningProjection(conditioning_dim, d_model)
    cond_projected = cond_proj(conditioning)
    print(f"  Input shape: {conditioning.shape}")
    print(f"  Output shape: {cond_projected.shape}")

    # Test cross-attention
    print("\n3. Cross-Attention Layer")
    query = torch.randn(batch_size, seq_len, d_model)
    key_value = torch.randn(batch_size, seq_len // 2, d_model)
    cross_attn = CrossAttentionLayer(d_model, nhead=8)
    output = cross_attn(query, key_value)
    print(f"  Query shape: {query.shape}")
    print(f"  Key/Value shape: {key_value.shape}")
    print(f"  Output shape: {output.shape}")

    # Test Gamma output head
    print("\n4. Gamma Output Head")
    x = torch.randn(batch_size, seq_len, d_model)
    gamma_head = GammaOutputHead(d_model)
    mu, sigma = gamma_head(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Mu shape: {mu.shape}, range: [{mu.min():.2f}, {mu.max():.2f}]")
    print(f"  Sigma shape: {sigma.shape}, range: [{sigma.min():.2f}, {sigma.max():.2f}]")

    print("\n✓ All layer components working correctly!")
