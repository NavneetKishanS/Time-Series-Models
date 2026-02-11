"""
Examination Model: Conditional Sequence Generator with Body Region Conditioning

Generates MRI event sequences (sourceID tokens) for a specific body region,
conditioned on patient context features.

This model handles the scan sequence generation during examinations:
- Scan protocols (LOCALIZER, T2_TSE, T1_VIBE, etc.)
- Per-scan durations
- Coil configuration usage
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXAMINATION_MODEL_CONFIG, START_TOKEN_ID, END_TOKEN_ID,
    PAD_TOKEN_ID, NUM_BODY_REGIONS, BODY_REGIONS
)
from models.layers import (
    PositionalEncoding, DurationHead, create_attention_mask, create_key_padding_mask
)


class ExaminationModel(nn.Module):
    """
    Transformer encoder-decoder for body-region-specific sequence generation.

    Generates MRI event sequences conditioned on:
    - Patient features (age, weight, height, PTAB, direction)
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
        self.base_conditioning_dim = config['conditioning_dim'] - 1  # Subtract 1 for body region
        self.num_body_regions = config.get('num_body_regions', NUM_BODY_REGIONS)

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

        # Duration prediction head (predicts mu, sigma for each token's duration)
        self.duration_head = DurationHead(self.d_model, min_sigma=0.1)

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
        region_emb = self.body_region_embedding(body_region)  # [batch_size, d_model//4]
        combined = torch.cat([conditioning, region_emb], dim=-1)  # [batch_size, combined_dim]
        cond_proj = self.conditioning_projection(combined)  # [batch_size, d_model]
        cond_seq = cond_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        cond_encoded = self.pos_encoder(cond_seq)
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
            duration_mu: [batch_size, seq_len] - predicted duration means
            duration_sigma: [batch_size, seq_len] - predicted duration stds
        """
        batch_size, seq_len = target_tokens.shape

        memory = self.encode_conditioning(conditioning, body_region)

        tgt_emb = self.token_embedding(target_tokens)  # [batch_size, seq_len, d_model]
        tgt_emb = tgt_emb * (self.d_model ** 0.5)  # Scale embeddings
        tgt_emb = self.pos_decoder(tgt_emb)

        tgt_mask = create_attention_mask(seq_len, causal=True, device=target_tokens.device)
        tgt_key_padding_mask = create_key_padding_mask(target_tokens, pad_token_id=PAD_TOKEN_ID)

        decoder_output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        logits = self.output_projection(decoder_output)
        duration_mu, duration_sigma = self.duration_head(decoder_output)

        return logits, duration_mu, duration_sigma

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
            generated_durations: [batch_size, seq_len] - predicted durations per token
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

        memory = self.encode_conditioning(conditioning, body_region)

        # Initialize with START token
        generated = torch.full((batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=device)
        generated_durations = [torch.zeros(batch_size, device=device)]  # 0 duration for START

        for _ in range(max_length - 1):
            tgt_emb = self.token_embedding(generated)
            tgt_emb = tgt_emb * (self.d_model ** 0.5)
            tgt_emb = self.pos_decoder(tgt_emb)

            seq_len = generated.shape[1]
            tgt_mask = create_attention_mask(seq_len, causal=True, device=device)

            decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            # Token prediction
            logits = self.output_projection(decoder_output)
            next_token_logits = logits[:, -1, :] / temperature

            # Duration prediction for the last position
            dur_mu, dur_sigma = self.duration_head(decoder_output)
            last_mu = dur_mu[:, -1]
            last_sigma = dur_sigma[:, -1]
            sampled_duration = torch.normal(last_mu, last_sigma).clamp(min=0.0)

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

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            generated_durations.append(sampled_duration)

            if (next_token == END_TOKEN_ID).all():
                break

        return generated, torch.stack(generated_durations, dim=1)

    def compute_loss(self, logits, targets, ignore_index=None, label_smoothing=0.0):
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
        if ignore_index is None:
            ignore_index = PAD_TOKEN_ID

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        return loss

    def compute_duration_loss(self, mu, sigma, target_durations, ignore_mask=None):
        """
        Compute Gaussian NLL loss for duration prediction.

        Args:
            mu: [batch_size, seq_len] predicted duration means
            sigma: [batch_size, seq_len] predicted duration stds
            target_durations: [batch_size, seq_len] actual durations in seconds
            ignore_mask: [batch_size, seq_len] True for positions to ignore (padding)

        Returns:
            loss: Scalar loss value
        """
        # Gaussian NLL: 0.5 * (log(sigma^2) + (target - mu)^2 / sigma^2)
        variance = sigma ** 2 + 1e-8
        nll = 0.5 * (torch.log(variance) + (target_durations - mu) ** 2 / variance)

        if ignore_mask is not None:
            nll = nll.masked_fill(ignore_mask, 0.0)
            num_valid = (~ignore_mask).sum().clamp(min=1)
            return nll.sum() / num_valid

        return nll.mean()


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
    print("Testing Examination Model...")
    print("=" * 60)

    model = create_examination_model()
    print(f"Model created")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    batch_size = 4
    seq_len = 20
    base_conditioning_dim = 5  # Age, Weight, Height, PTAB, Direction

    conditioning = torch.randn(batch_size, base_conditioning_dim)
    body_region = torch.randint(0, NUM_BODY_REGIONS, (batch_size,))
    target_tokens = torch.randint(1, 17, (batch_size, seq_len))

    # Forward pass
    print(f"\nTesting forward pass...")
    logits, dur_mu, dur_sigma = model(conditioning, body_region, target_tokens)
    print(f"  Conditioning shape: {conditioning.shape}")
    print(f"  Body region shape: {body_region.shape}")
    print(f"  Target tokens shape: {target_tokens.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Duration mu shape: {dur_mu.shape}")
    print(f"  Duration sigma shape: {dur_sigma.shape}")

    # Test loss
    loss = model.compute_loss(logits, target_tokens, label_smoothing=0.1)
    print(f"\nToken loss: {loss.item():.4f}")

    # Test duration loss
    dummy_durations = torch.rand(batch_size, seq_len) * 300  # Random durations 0-300s
    dur_loss = model.compute_duration_loss(dur_mu, dur_sigma, dummy_durations)
    print(f"Duration loss: {dur_loss.item():.4f}")

    # Test generation
    print(f"\nTesting generation for each body region...")
    single_cond = torch.randn(base_conditioning_dim)

    for region_id in [0, 2, 5]:  # HEAD, CHEST, SPINE
        generated, durations = model.generate(single_cond, region_id, max_length=30,
                                              temperature=1.0, top_k=10)
        print(f"  {BODY_REGIONS[region_id]}: generated {generated.shape[1]} tokens")
        print(f"    Tokens: {generated[0, :10].tolist()}...")
        print(f"    Durations: {durations[0, :10].tolist()}")

    print("\nAll tests passed!")
