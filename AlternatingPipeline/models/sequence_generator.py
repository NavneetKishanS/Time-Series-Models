"""
Unified Sequence Generator Model.

A single Transformer architecture used by both exchange and examination models.
The exchange model uses body_from + body_to + phase_type conditioning.
The examination model uses a single body_region conditioning.

Architecture:
    Conditioning Encoder -> memory
    Token Decoder (autoregressive) -> logits
    Duration Encoder (bidirectional, single-pass) -> (mu, sigma) per token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXCHANGE_MODEL_CONFIG, EXAMINATION_MODEL_CONFIG,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID,
    GENERATION_CONFIG
)
from models.layers import (
    PositionalEncoding, SinglePassDurationHead,
    create_attention_mask, create_key_padding_mask
)


class SequenceGeneratorModel(nn.Module):
    """
    Unified Transformer for MRI event sequence generation.

    Used by both exchange and examination models with different configs.
    """

    def __init__(self, config):
        super(SequenceGeneratorModel, self).__init__()

        self.config = config
        self.model_type = config['model_type']
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.has_phase_type = config.get('has_phase_type', False)
        self.body_region_mode = config.get('body_region_mode', 'single')

        nhead = config['nhead']
        num_encoder_layers = config['num_encoder_layers']
        num_decoder_layers = config['num_decoder_layers']
        num_duration_encoder_layers = config.get('num_duration_encoder_layers', 4)
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        base_conditioning_dim = config['base_conditioning_dim']
        num_body_regions = config.get('num_body_regions', 11)
        num_region_classes = config.get('num_region_classes', 13)

        # =====================================================================
        # Conditioning Encoder
        # =====================================================================
        region_emb_dim = self.d_model // 4  # 64 for d_model=256

        # Examination scan-type + per-scanner conditioning (off by default;
        # enabled via config for the examination model only).
        self.use_exam_conditioning = config.get('use_exam_conditioning', False)

        if self.body_region_mode == 'from_to':
            # Exchange: embed body_from AND body_to, concat both
            self.body_from_embedding = nn.Embedding(num_region_classes, region_emb_dim)
            self.body_to_embedding = nn.Embedding(num_region_classes, region_emb_dim)
            cond_input_dim = base_conditioning_dim + 2 * region_emb_dim
        else:
            # Examination: embed single body_region
            self.body_region_embedding = nn.Embedding(num_body_regions, region_emb_dim)
            cond_input_dim = base_conditioning_dim + region_emb_dim

            if self.use_exam_conditioning:
                # Sequence type (scout/tse/...) and scanner serial. Small
                # embeddings — these are coarse categoricals, not the main
                # signal — concatenated onto the conditioning vector.
                exam_emb_dim = self.d_model // 8  # 16 for d_model=128
                self.sequence_type_embedding = nn.Embedding(
                    config['num_sequence_types'], exam_emb_dim
                )
                self.serial_embedding = nn.Embedding(
                    config['num_serials'], exam_emb_dim
                )
                cond_input_dim += 2 * exam_emb_dim

        if self.has_phase_type:
            phase_emb_dim = self.d_model // 8  # 32 for d_model=256
            self.phase_type_embedding = nn.Embedding(
                config.get('num_phase_types', 3), phase_emb_dim
            )
            cond_input_dim += phase_emb_dim

        self.conditioning_projection = nn.Sequential(
            nn.Linear(cond_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.conditioning_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # =====================================================================
        # Token Decoder (autoregressive)
        # =====================================================================
        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_TOKEN_ID
        )
        self.pos_decoder = PositionalEncoding(self.d_model, max_len=self.max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.token_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        # =====================================================================
        # Duration Encoder (bidirectional, single-pass)
        # =====================================================================
        self.duration_token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.duration_pos_encoding = PositionalEncoding(self.d_model, max_len=self.max_seq_len + 1, dropout=dropout)

        duration_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.duration_encoder = nn.TransformerEncoder(
            duration_encoder_layer, num_layers=num_duration_encoder_layers
        )

        self.duration_head = SinglePassDurationHead(self.d_model, hidden_dim=128, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_conditioning(self, conditioning, body_region_info, phase_type=None):
        """
        Encode conditioning features into memory for the decoder.

        Args:
            conditioning: [batch, base_conditioning_dim] - patient + temporal features
            body_region_info: dict with keys depending on body_region_mode:
                'from_to': {'body_from': [batch], 'body_to': [batch]}
                'single': {'body_region': [batch]}
            phase_type: [batch] - phase type IDs (exchange only), or None

        Returns:
            memory: [batch, 1, d_model]
        """
        parts = [conditioning]

        if self.body_region_mode == 'from_to':
            from_emb = self.body_from_embedding(body_region_info['body_from'])
            to_emb = self.body_to_embedding(body_region_info['body_to'])
            parts.extend([from_emb, to_emb])
        else:
            region_emb = self.body_region_embedding(body_region_info['body_region'])
            parts.append(region_emb)

            if self.use_exam_conditioning:
                region_t = body_region_info['body_region']
                # sequence_type / serial_idx default to 0 ('other' / first
                # scanner) when a caller does not supply them.
                seq_t = body_region_info.get('sequence_type')
                if seq_t is None:
                    seq_t = torch.zeros_like(region_t)
                ser_t = body_region_info.get('serial_idx')
                if ser_t is None:
                    ser_t = torch.zeros_like(region_t)
                parts.append(self.sequence_type_embedding(seq_t))
                parts.append(self.serial_embedding(ser_t))

        if self.has_phase_type and phase_type is not None:
            phase_emb = self.phase_type_embedding(phase_type)
            parts.append(phase_emb)

        combined = torch.cat(parts, dim=-1)
        cond_proj = self.conditioning_projection(combined)  # [batch, d_model]
        cond_seq = cond_proj.unsqueeze(1)  # [batch, 1, d_model]
        cond_encoded = self.pos_encoder(cond_seq)
        memory = self.conditioning_encoder(cond_encoded)  # [batch, 1, d_model]
        return memory

    def _get_conditioning_token(self, conditioning, body_region_info, phase_type=None):
        """
        Get a single conditioning token embedding for the duration encoder.
        Reuses the conditioning encoder to produce [batch, 1, d_model].
        """
        return self._encode_conditioning(conditioning, body_region_info, phase_type)

    def estimate_durations(self, token_ids, conditioning, body_region_info, phase_type=None):
        """
        Estimate durations for a complete token sequence (single-pass, bidirectional).

        Args:
            token_ids: [batch, seq_len] - token IDs (the complete generated/target sequence)
            conditioning: [batch, base_conditioning_dim]
            body_region_info: dict
            phase_type: [batch] or None

        Returns:
            mu: [batch, seq_len]
            sigma: [batch, seq_len]
        """
        batch_size, seq_len = token_ids.shape

        # Get conditioning token
        cond_token = self._get_conditioning_token(conditioning, body_region_info, phase_type)
        # cond_token: [batch, 1, d_model]

        # Embed tokens
        tok_emb = self.duration_token_embedding(token_ids)  # [batch, seq_len, d_model]

        # Prepend conditioning token
        combined = torch.cat([cond_token, tok_emb], dim=1)  # [batch, 1+seq_len, d_model]
        combined = self.duration_pos_encoding(combined)

        # Bidirectional encoder (NO causal mask)
        encoded = self.duration_encoder(combined)  # [batch, 1+seq_len, d_model]

        # Extract token positions (skip conditioning token at position 0)
        token_hidden = encoded[:, 1:, :]  # [batch, seq_len, d_model]

        mu, sigma = self.duration_head(token_hidden)
        return mu, sigma

    def forward(self, conditioning, body_region_info, target_tokens, phase_type=None):
        """
        Training forward pass with teacher forcing.

        Args:
            conditioning: [batch, base_conditioning_dim]
            body_region_info: dict
            target_tokens: [batch, seq_len] - input sequence (START + tokens)
            phase_type: [batch] or None

        Returns:
            logits: [batch, seq_len, vocab_size]
            duration_mu: [batch, seq_len]
            duration_sigma: [batch, seq_len]
        """
        batch_size, seq_len = target_tokens.shape

        # Encode conditioning -> memory
        memory = self._encode_conditioning(conditioning, body_region_info, phase_type)

        # Decode with teacher forcing
        tgt_emb = self.token_embedding(target_tokens)
        tgt_emb = tgt_emb * (self.d_model ** 0.5)
        tgt_emb = self.pos_decoder(tgt_emb)

        tgt_mask = create_attention_mask(seq_len, causal=True, device=target_tokens.device)
        tgt_key_padding_mask = create_key_padding_mask(target_tokens, pad_token_id=PAD_TOKEN_ID)

        decoder_output = self.token_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_projection(decoder_output)

        # Duration estimation via bidirectional encoder on target tokens
        duration_mu, duration_sigma = self.estimate_durations(
            target_tokens, conditioning, body_region_info, phase_type
        )

        return logits, duration_mu, duration_sigma

    @torch.no_grad()
    def generate(self, conditioning, body_region_info, phase_type=None,
                 max_length=None, temperature=1.0, top_k=0, top_p=0.9,
                 return_stats=False):
        """
        Two-phase generation:
        1. Autoregressive token generation
        2. Single-pass duration estimation over the complete sequence

        Args:
            conditioning: [batch, base_conditioning_dim] or [base_conditioning_dim]
            body_region_info: dict
            phase_type: [batch] or scalar or None
            max_length: max sequence length
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold
            return_stats: if True, also return (duration_mu, duration_sigma)

        Returns:
            generated_tokens: [batch, seq_len]
            generated_durations: [batch, seq_len]
            duration_mu: [batch, seq_len]  — only if return_stats=True
            duration_sigma: [batch, seq_len] — only if return_stats=True
        """
        self.eval()

        if max_length is None:
            max_length = self.max_seq_len

        # Handle single-sample input
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        # Ensure body_region_info tensors are batched
        body_region_info = self._ensure_batched(body_region_info, conditioning.device)

        if phase_type is not None:
            if isinstance(phase_type, int):
                phase_type = torch.tensor([phase_type], device=conditioning.device)
            elif phase_type.dim() == 0:
                phase_type = phase_type.unsqueeze(0)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        # Encode conditioning
        memory = self._encode_conditioning(conditioning, body_region_info, phase_type)

        # Phase 1: Autoregressive token generation
        generated = torch.full((batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            tgt_emb = self.token_embedding(generated)
            tgt_emb = tgt_emb * (self.d_model ** 0.5)
            tgt_emb = self.pos_decoder(tgt_emb)

            seq_len = generated.shape[1]
            tgt_mask = create_attention_mask(seq_len, causal=True, device=device)

            decoder_output = self.token_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            next_token_logits = self.output_projection(decoder_output[:, -1, :]) / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == END_TOKEN_ID).all():
                break

        # Phase 2: Single-pass duration estimation
        duration_mu, duration_sigma = self.estimate_durations(
            generated, conditioning, body_region_info, phase_type
        )

        # Sample durations from Normal(mu, sigma), clamp to non-negative
        durations = torch.normal(duration_mu, duration_sigma).clamp(min=0.0)

        if return_stats:
            return generated, durations, duration_mu, duration_sigma
        return generated, durations

    def _ensure_batched(self, body_region_info, device):
        """Ensure body_region_info values are batched tensors on the right device."""
        result = {}
        for key, val in body_region_info.items():
            if isinstance(val, int):
                val = torch.tensor([val], device=device)
            elif isinstance(val, torch.Tensor):
                val = val.to(device)
                if val.dim() == 0:
                    val = val.unsqueeze(0)
            result[key] = val
        return result

    def compute_loss(self, logits, targets, ignore_index=None, label_smoothing=0.0,
                     class_weights=None):
        """Compute cross-entropy loss with optional label smoothing.

        class_weights: optional [vocab_size] tensor up-weighting rare tokens
        (e.g. the MRI_MSR_34 abort token) so they are not crowded out of the
        softmax by frequent workflow events.
        """
        if ignore_index is None:
            ignore_index = PAD_TOKEN_ID

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        weight = None
        if class_weights is not None:
            weight = class_weights.to(logits_flat.device, dtype=logits_flat.dtype)

        return F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            weight=weight,
        )

    def compute_duration_loss(self, mu, sigma, target_durations, ignore_mask=None):
        """Compute Gaussian NLL loss for duration prediction."""
        variance = sigma ** 2 + 1e-8
        nll = 0.5 * (torch.log(variance) + (target_durations - mu) ** 2 / variance)

        if ignore_mask is not None:
            nll = nll.masked_fill(ignore_mask, 0.0)
            num_valid = (~ignore_mask).sum().clamp(min=1)
            return nll.sum() / num_valid

        return nll.mean()


def create_sequence_generator(config):
    """Create a SequenceGeneratorModel from a config dict."""
    return SequenceGeneratorModel(config)


if __name__ == "__main__":
    print("Testing Unified Sequence Generator Model...")
    print("=" * 60)

    # Test Exchange config
    print("\n--- Exchange Model ---")
    exchange_model = create_sequence_generator(EXCHANGE_MODEL_CONFIG)
    num_params = sum(p.numel() for p in exchange_model.parameters())
    print(f"Parameters: {num_params:,}")

    batch = 4
    seq_len = 20
    cond = torch.randn(batch, EXCHANGE_MODEL_CONFIG['base_conditioning_dim'])
    body_from = torch.randint(0, 13, (batch,))
    body_to = torch.randint(0, 11, (batch,))
    phase = torch.randint(0, 3, (batch,))
    input_seq = torch.randint(1, 17, (batch, seq_len))

    logits, dur_mu, dur_sigma = exchange_model(
        cond, {'body_from': body_from, 'body_to': body_to}, input_seq, phase_type=phase
    )
    print(f"Forward: logits={logits.shape}, dur_mu={dur_mu.shape}, dur_sigma={dur_sigma.shape}")

    gen_tokens, gen_durs = exchange_model.generate(
        cond[:1], {'body_from': body_from[:1], 'body_to': body_to[:1]},
        phase_type=phase[:1], max_length=30
    )
    print(f"Generate: tokens={gen_tokens.shape}, durations={gen_durs.shape}")

    # Test Examination config
    print("\n--- Examination Model ---")
    exam_model = create_sequence_generator(EXAMINATION_MODEL_CONFIG)
    num_params = sum(p.numel() for p in exam_model.parameters())
    print(f"Parameters: {num_params:,}")

    body_region = torch.randint(0, 11, (batch,))
    logits, dur_mu, dur_sigma = exam_model(
        cond, {'body_region': body_region}, input_seq
    )
    print(f"Forward: logits={logits.shape}, dur_mu={dur_mu.shape}, dur_sigma={dur_sigma.shape}")

    gen_tokens, gen_durs = exam_model.generate(
        cond[:1], {'body_region': body_region[:1]}, max_length=30
    )
    print(f"Generate: tokens={gen_tokens.shape}, durations={gen_durs.shape}")

    print("\nAll tests passed!")
