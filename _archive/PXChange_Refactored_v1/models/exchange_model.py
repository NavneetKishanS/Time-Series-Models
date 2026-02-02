"""
Exchange Model for body region transition prediction.

Predicts: Given conditioning + current body region, what is the next body region?

This is a probabilistic classifier that outputs a distribution over possible
next body regions. During inference, we sample from this distribution to
generate stochastic body region sequences.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

pxchange_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, pxchange_dir)

import config as px_config


class ExchangeModel(nn.Module):
    """
    Neural network for predicting body region transitions.

    Architecture:
    - Embedding for current body region
    - MLP for conditioning features
    - Combined through hidden layers
    - Output: softmax over next body regions

    Input:
        conditioning: [batch, conditioning_dim] - patient features
        current_region: [batch] - current body region ID

    Output:
        logits: [batch, num_regions] - logits for next body region
    """

    def __init__(
        self,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        conditioning_dim=5,
        num_regions=13  # 11 body regions + START + END
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions
        self.conditioning_dim = conditioning_dim

        # Embedding for body region
        self.region_embedding = nn.Embedding(num_regions, d_model)

        # Project conditioning to d_model
        self.conditioning_proj = nn.Linear(conditioning_dim, d_model)

        # Combined input dimension (region embedding + conditioning projection)
        combined_dim = d_model * 2

        # Build hidden layers
        layers = []
        input_dim = combined_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_regions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, conditioning, current_region):
        """
        Forward pass.

        Args:
            conditioning: [batch, conditioning_dim] - patient features
            current_region: [batch] - current body region ID

        Returns:
            logits: [batch, num_regions] - logits for next body region
        """
        # Embed current region
        region_emb = self.region_embedding(current_region)  # [batch, d_model]

        # Project conditioning
        cond_proj = self.conditioning_proj(conditioning)  # [batch, d_model]

        # Combine
        combined = torch.cat([region_emb, cond_proj], dim=-1)  # [batch, d_model*2]

        # Pass through hidden layers
        hidden = self.hidden_layers(combined)  # [batch, hidden_dim]

        # Output logits
        logits = self.output_proj(hidden)  # [batch, num_regions]

        return logits

    def predict(self, conditioning, current_region, temperature=1.0, top_k=None):
        """
        Predict next body region by sampling from the distribution.

        Args:
            conditioning: [batch, conditioning_dim] - patient features
            current_region: [batch] - current body region ID
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k options

        Returns:
            next_region: [batch] - predicted next body region ID
            probs: [batch, num_regions] - probability distribution
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(conditioning, current_region)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            next_region = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_region, probs

    def predict_greedy(self, conditioning, current_region):
        """
        Predict next body region using greedy decoding (argmax).

        Args:
            conditioning: [batch, conditioning_dim] - patient features
            current_region: [batch] - current body region ID

        Returns:
            next_region: [batch] - predicted next body region ID
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(conditioning, current_region)
            next_region = logits.argmax(dim=-1)
        return next_region

    def generate_sequence(self, conditioning, max_steps=10, temperature=1.0,
                          top_k=None, start_region_id=11, end_region_id=12):
        """
        Generate a complete body region sequence from START to END.

        Args:
            conditioning: [1, conditioning_dim] or [conditioning_dim] - patient features
            max_steps: Maximum number of body regions to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            start_region_id: ID for START token (default 11)
            end_region_id: ID for END token (default 12)

        Returns:
            regions: List of body region IDs (excluding START, including END)
        """
        self.eval()

        # Ensure conditioning is 2D
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        device = next(self.parameters()).device
        conditioning = conditioning.to(device)

        regions = []
        current_region = torch.tensor([start_region_id], device=device)

        for _ in range(max_steps):
            next_region, _ = self.predict(conditioning, current_region,
                                          temperature=temperature, top_k=top_k)

            region_id = next_region.item()
            regions.append(region_id)

            if region_id == end_region_id:
                break

            current_region = next_region

        return regions


def create_exchange_model(config=None):
    """
    Create an Exchange Model instance from config.

    Args:
        config: Optional config dict. If None, uses px_config.EXCHANGE_MODEL_CONFIG

    Returns:
        ExchangeModel instance
    """
    if config is None:
        config = px_config.EXCHANGE_MODEL_CONFIG

    model = ExchangeModel(
        d_model=config.get('d_model', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1),
        conditioning_dim=config.get('conditioning_dim', 5),
        num_regions=config.get('num_regions', 13)
    )

    return model


if __name__ == "__main__":
    # Test model
    print("Testing Exchange Model...")
    print("=" * 60)

    model = create_exchange_model()
    print(f"Model created: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    conditioning = torch.randn(batch_size, 5)
    current_region = torch.randint(0, 13, (batch_size,))

    logits = model(conditioning, current_region)
    print(f"\nForward pass:")
    print(f"  Input conditioning: {conditioning.shape}")
    print(f"  Input current_region: {current_region.shape}")
    print(f"  Output logits: {logits.shape}")

    # Test prediction
    next_region, probs = model.predict(conditioning, current_region)
    print(f"\nPrediction:")
    print(f"  Next regions: {next_region}")
    print(f"  Probabilities shape: {probs.shape}")

    # Test sequence generation
    print(f"\nSequence generation:")
    for i in range(3):
        cond = torch.randn(5)
        seq = model.generate_sequence(cond, temperature=1.0)
        body_regions = px_config.BODY_REGIONS + ['START', 'END']
        seq_names = [body_regions[r] if r < len(body_regions) else f"ID_{r}" for r in seq]
        print(f"  Sample {i+1}: {' -> '.join(seq_names)}")
