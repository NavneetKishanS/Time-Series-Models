"""
Examination Model: Conditional Sequence Generator with Body Region Conditioning

Generates MRI event sequences (sourceID tokens) for a specific body region,
conditioned on patient context features.

This is an adaptation of ConditionalSequenceGenerator that explicitly incorporates
body region as a conditioning input.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXAMINATION_MODEL_CONFIG, START_TOKEN_ID, END_TOKEN_ID,
    PAD_TOKEN_ID, NUM_BODY_REGIONS, BODY_REGIONS
)
from models.layers import (
    PositionalEncoding, ConditioningProjection,
    create_attention_mask, create_key_padding_mask
)


class ExaminationModel(nn.Module):
    """
    Transformer encoder-decoder for body-region-specific sequence generation.

    Generates MRI event sequences conditioned on:
    - Patient features (age, weight, height, PTAB, entity_type)
    - Body region being examined (HEAD, CHEST, SPINE, etc.)

    Architecture:
        1. Conditioning encoder: processes patient features
        2. Body region embedding: converts region ID to embedding
        3. Combined conditioning: patient features + body region embedding
        4. Transformer decoder: generates sequence auto-regressively
        5. Output projection: predicts next token probabilities
    """

    def __init__(self, config=None):
        super(ExaminationModel, self).__init__()

        if config is None:
            config = EXAMINATION_MODEL_CONFIG

        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.max_seq_len = config['max_seq_len']
        # conditioning_dim includes the +1 for body region in config, but we handle it separately
        self.base_conditioning_dim = config['conditioning_dim'] - 1  # Subtract 1 for body region
        self.num_body_regions = NUM_BODY_REGIONS

        # Body region embedding
        self.body_region_embedding = nn.Embedding(self.num_body_regions, self.d_model // 4)

        # Conditioning projection (patient features + body region embedding)
        combined_input_dim = self.base_conditioning_dim + self.d_model // 4
        self.conditioning_projection = nn.Sequential(
            nn.Linear(combined_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model)
        )

        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=PAD_TOKEN_ID)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)
        self.pos_decoder = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=self.dropout)

        # Transformer encoder (processes conditioning as a sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # Transformer decoder (generates sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_conditioning(self, conditioning, body_region):
        """
        Encode conditioning features and body region into memory for decoder.

        Args:
            conditioning: [batch_size, base_conditioning_dim] - patient features
            body_region: [batch_size] - body region IDs

        Returns:
            memory: [batch_size, 1, d_model] - encoded conditioning
        """
        # Embed body region
        region_emb = self.body_region_embedding(body_region)  # [batch_size, d_model//4]

        # Concatenate conditioning with body region embedding
        combined = torch.cat([conditioning, region_emb], dim=-1)  # [batch_size, combined_dim]

        # Project to d_model
        cond_proj = self.conditioning_projection(combined)  # [batch_size, d_model]

        # Add sequence dimension and positional encoding
        cond_seq = cond_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        cond_encoded = self.pos_encoder(cond_seq)

        # Pass through encoder
        memory = self.transformer_encoder(cond_encoded)  # [batch_size, 1, d_model]

        return memory

    def forward(self, conditioning, body_region, target_tokens):
        """
        Forward pass for training (teacher forcing).

        Args:
            conditioning: [batch_size, base_conditioning_dim] - patient features
            body_region: [batch_size] - body region IDs
            target_tokens: [batch_size, seq_len] - target sequence with START prepended

        Returns:
            logits: [batch_size, seq_len, vocab_size] - token logits
        """
        batch_size, seq_len = target_tokens.shape

        # Encode conditioning with body region
        memory = self.encode_conditioning(conditioning, body_region)

        # Embed target tokens
        tgt_emb = self.token_embedding(target_tokens)  # [batch_size, seq_len, d_model]
        tgt_emb = tgt_emb * (self.d_model ** 0.5)  # Scale embeddings
        tgt_emb = self.pos_decoder(tgt_emb)

        # Create causal mask for decoder
        tgt_mask = create_attention_mask(seq_len, causal=True, device=target_tokens.device)

        # Create padding mask
        tgt_key_padding_mask = create_key_padding_mask(target_tokens, pad_token_id=PAD_TOKEN_ID)

        # Decoder
        decoder_output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    @torch.no_grad()
    def generate(self, conditioning, body_region, max_length=None,
                 temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate sequences auto-regressively (inference mode).

        Args:
            conditioning: [batch_size, base_conditioning_dim] or [base_conditioning_dim]
            body_region: [batch_size] or scalar - body region ID
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold

        Returns:
            generated_sequences: [batch_size, seq_len] - generated token IDs
        """
        self.eval()

        if max_length is None:
            max_length = self.max_seq_len

        # Handle single sample input
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)
        if isinstance(body_region, int) or body_region.dim() == 0:
            body_region = torch.tensor([body_region] if isinstance(body_region, int)
                                       else [body_region.item()],
                                       device=conditioning.device)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        # Encode conditioning with body region
        memory = self.encode_conditioning(conditioning, body_region)

        # Initialize with START token
        generated = torch.full((batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Embed current sequence
            tgt_emb = self.token_embedding(generated)
            tgt_emb = tgt_emb * (self.d_model ** 0.5)
            tgt_emb = self.pos_decoder(tgt_emb)

            # Create causal mask
            seq_len = generated.shape[1]
            tgt_mask = create_attention_mask(seq_len, causal=True, device=device)

            # Decoder
            decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            # Get logits for next token
            logits = self.output_projection(decoder_output)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have generated END token
            if (next_token == END_TOKEN_ID).all():
                break

        return generated

    def compute_loss(self, logits, targets, ignore_index=PAD_TOKEN_ID, label_smoothing=0.0):
        """
        Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            ignore_index: Index to ignore in loss (padding)
            label_smoothing: Label smoothing factor

        Returns:
            loss: Scalar loss value
        """
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        return loss


def create_examination_model(config=None):
    """
    Create an Examination Model instance from config.

    Args:
        config: Optional config dict. If None, uses EXAMINATION_MODEL_CONFIG

    Returns:
        ExaminationModel instance
    """
    if config is None:
        config = EXAMINATION_MODEL_CONFIG

    return ExaminationModel(config)


if __name__ == "__main__":
    # Test the model
    print("Testing Examination Model...")
    print("=" * 60)

    # Initialize model
    model = create_examination_model()
    print(f"Model created")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    batch_size = 4
    seq_len = 20
    base_conditioning_dim = 5  # Age, Weight, Height, PTAB, entity_type

    conditioning = torch.randn(batch_size, base_conditioning_dim)
    body_region = torch.randint(0, NUM_BODY_REGIONS, (batch_size,))
    target_tokens = torch.randint(1, 17, (batch_size, seq_len))

    # Forward pass
    print(f"\nTesting forward pass...")
    logits = model(conditioning, body_region, target_tokens)
    print(f"  Conditioning shape: {conditioning.shape}")
    print(f"  Body region shape: {body_region.shape}")
    print(f"  Target tokens shape: {target_tokens.shape}")
    print(f"  Output logits shape: {logits.shape}")

    # Test loss
    loss = model.compute_loss(logits, target_tokens, label_smoothing=0.1)
    print(f"\nLoss computed: {loss.item():.4f}")

    # Test generation
    print(f"\nTesting generation for each body region...")
    single_cond = torch.randn(base_conditioning_dim)

    for region_id in [0, 2, 5]:  # HEAD, CHEST, SPINE
        generated = model.generate(single_cond, region_id, max_length=30,
                                   temperature=1.0, top_k=10)
        print(f"  {BODY_REGIONS[region_id]}: generated {generated.shape[1]} tokens")
        print(f"    Tokens: {generated[0, :10].tolist()}...")

    print("\nAll tests passed!")
