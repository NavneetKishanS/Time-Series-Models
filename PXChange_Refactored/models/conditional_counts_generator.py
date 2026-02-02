"""
Conditional Counts Generator: Transformer Encoder with Cross-Attention
Predicts numerical counts (step durations) conditioned on symbolic sequences and context.
Non-autoregressive: all positions predicted in parallel.
Outputs: μ (mean) and σ (uncertainty) for each position.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COUNTS_MODEL_CONFIG, PAD_TOKEN_ID, VOCAB_SIZE
from models.layers import (
    PositionalEncoding, ConditioningProjection,
    CrossAttentionLayer, GammaOutputHead,
    create_key_padding_mask
)


class ConditionalCountsGenerator(nn.Module):
    """
    Transformer encoder with cross-attention for count prediction.

    Architecture:
        1. Conditioning encoder: processes patient/scan context
        2. Sequence encoder: processes symbolic sequence (sourceID + features)
        3. Cross-attention: attends from sequence to conditioning
        4. Dual output heads: predicts μ and σ for each position
    """

    def __init__(self, config=None):
        super(ConditionalCountsGenerator, self).__init__()

        if config is None:
            config = COUNTS_MODEL_CONFIG

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_cross_attention_layers = config['num_cross_attention_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.max_seq_len = config['max_seq_len']
        self.conditioning_dim = config['conditioning_dim']
        self.sequence_feature_dim = config['sequence_feature_dim']
        self.min_sigma = config.get('min_sigma', 0.1)

        # Conditioning projection
        self.conditioning_projection = ConditioningProjection(
            self.conditioning_dim, self.d_model, dropout=self.dropout
        )

        # Token embedding for symbolic sequence
        self.token_embedding = nn.Embedding(VOCAB_SIZE, self.d_model // 2, padding_idx=PAD_TOKEN_ID)

        # Projection for additional sequence features
        self.feature_projection = nn.Linear(2, self.d_model // 2)  # Position and Direction encoded

        # Combine token embeddings and features
        self.sequence_projection = nn.Linear(self.d_model, self.d_model)

        # Positional encoding
        self.pos_encoder_cond = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)
        self.pos_encoder_seq = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)

        # Conditioning encoder
        cond_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.conditioning_encoder = nn.TransformerEncoder(
            cond_encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # Sequence encoder
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            seq_encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # Cross-attention layers (sequence attends to conditioning)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            )
            for _ in range(self.num_cross_attention_layers)
        ])

        # Output head for Gamma distribution parameters
        self.output_head = GammaOutputHead(self.d_model, min_sigma=self.min_sigma)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_conditioning(self, conditioning):
        """
        Encode conditioning features.

        Args:
            conditioning: [batch_size, conditioning_dim]

        Returns:
            cond_encoded: [batch_size, 1, d_model]
        """
        # Project conditioning
        cond_proj = self.conditioning_projection(conditioning)  # [batch_size, d_model]

        # Add sequence dimension
        cond_seq = cond_proj.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add positional encoding
        cond_pos = self.pos_encoder_cond(cond_seq)

        # Encode
        cond_encoded = self.conditioning_encoder(cond_pos)

        return cond_encoded

    def encode_sequence(self, sequence_tokens, sequence_features, mask=None):
        """
        Encode symbolic sequence with additional features.

        Args:
            sequence_tokens: [batch_size, seq_len] - sourceID tokens
            sequence_features: [batch_size, seq_len, feature_dim] - Position, Direction
            mask: [batch_size, seq_len] - Boolean mask (True = valid)

        Returns:
            seq_encoded: [batch_size, seq_len, d_model]
        """
        # Embed tokens
        token_emb = self.token_embedding(sequence_tokens)  # [batch_size, seq_len, d_model/2]

        # Project features
        feature_emb = self.feature_projection(sequence_features)  # [batch_size, seq_len, d_model/2]

        # Concatenate
        combined = torch.cat([token_emb, feature_emb], dim=-1)  # [batch_size, seq_len, d_model]

        # Project to d_model
        seq_proj = self.sequence_projection(combined)

        # Add positional encoding
        seq_pos = self.pos_encoder_seq(seq_proj)

        # Create padding mask
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = padding
        else:
            key_padding_mask = create_key_padding_mask(sequence_tokens, pad_token_id=PAD_TOKEN_ID)

        # Encode
        seq_encoded = self.sequence_encoder(seq_pos, src_key_padding_mask=key_padding_mask)

        return seq_encoded

    def forward(self, conditioning, sequence_tokens, sequence_features, mask=None):
        """
        Forward pass to predict count parameters.

        Args:
            conditioning: [batch_size, conditioning_dim]
            sequence_tokens: [batch_size, seq_len]
            sequence_features: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] - Boolean mask (True = valid)

        Returns:
            mu: [batch_size, seq_len] - mean parameters
            sigma: [batch_size, seq_len] - std deviation parameters
        """
        # Encode conditioning
        cond_encoded = self.encode_conditioning(conditioning)  # [batch_size, 1, d_model]

        # Encode sequence
        seq_encoded = self.encode_sequence(sequence_tokens, sequence_features, mask)  # [batch_size, seq_len, d_model]

        # Apply cross-attention layers
        cross_output = seq_encoded
        for cross_attn_layer in self.cross_attention_layers:
            cross_output = cross_attn_layer(
                query=cross_output,
                key_value=cond_encoded
            )

        # Predict μ and σ
        mu, sigma = self.output_head(cross_output)  # Each [batch_size, seq_len]

        return mu, sigma

    def sample_counts(self, mu, sigma, num_samples=1):
        """
        Sample from the predicted Gamma distribution.

        Args:
            mu (torch.Tensor): Mean of the distribution [batch_size, seq_len]
            sigma (torch.Tensor): Standard deviation of the distribution [batch_size, seq_len]
            num_samples (int): Number of samples to draw for each position.

        Returns:
            torch.Tensor: Sampled counts [batch_size, seq_len, num_samples]
        """
        # Convert (mu, sigma) to (shape, rate) of Gamma distribution
        # shape = (mu / sigma)^2, rate = mu / sigma^2
        sigma_sq = sigma.pow(2)
        shape = mu.pow(2) / (sigma_sq + 1e-8)
        rate = mu / (sigma_sq + 1e-8)

        # Ensure shape and rate are positive
        shape = torch.clamp(shape, min=1e-6)
        rate = torch.clamp(rate, min=1e-6)

        # Create Gamma distribution
        gamma_dist = torch.distributions.Gamma(shape, rate)

        # Sample from the distribution
        samples = gamma_dist.sample((num_samples,)).permute(1, 2, 0)

        return samples