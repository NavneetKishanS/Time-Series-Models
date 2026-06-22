"""
Orchestration Model: Autoregressive Transformer for day-level patient scheduling.

Decides WHAT body regions to scan and in WHAT ORDER for a given day,
including BREAK tokens for pauses between patients.

Architecture (no duration encoder):
    Conditioning Encoder (scanner + day features) -> memory
    Token Decoder (autoregressive) -> next body region logits

Vocabulary (15 tokens):
    0-10: Body regions (HEAD, NECK, CHEST, ABDOMEN, PELVIS, SPINE, ARM, LEG, HAND, FOOT, UNKNOWN)
    11: START
    12: END
    13: BREAK (pseudo-patient for pauses)
    14: PAD (orchestration-specific padding)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ORCHESTRATION_MODEL_CONFIG
from models.layers import PositionalEncoding, create_attention_mask, create_key_padding_mask


class OrchestrationModel(nn.Module):
    """
    Autoregressive Transformer for day-level body region sequence generation.

    Simpler than SequenceGeneratorModel: no duration encoder,
    smaller dimensionality, different vocabulary.
    """

    def __init__(self, config=None):
        super(OrchestrationModel, self).__init__()

        if config is None:
            config = ORCHESTRATION_MODEL_CONFIG

        self.config = config
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.pad_token_id = config['pad_token_id']
        self.start_token_id = config['start_token_id']
        self.end_token_id = config['end_token_id']
        self.break_token_id = config['break_token_id']

        nhead = config['nhead']
        num_encoder_layers = config['num_encoder_layers']
        num_decoder_layers = config['num_decoder_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        base_conditioning_dim = config['base_conditioning_dim']
        num_scanners = config['num_scanners']
        scanner_emb_dim = config['scanner_emb_dim']

        # =====================================================================
        # Conditioning Encoder
        # =====================================================================
        self.scanner_embedding = nn.Embedding(num_scanners, scanner_emb_dim)

        cond_input_dim = base_conditioning_dim + scanner_emb_dim  # 17 + 32 = 49

        self.conditioning_projection = nn.Sequential(
            nn.Linear(cond_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.cond_pos_encoder = PositionalEncoding(
            self.d_model, max_len=self.max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.conditioning_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,  # nested tensor path is 5-10x slower on CPU
        )

        # =====================================================================
        # Token Decoder (autoregressive)
        # =====================================================================
        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=self.pad_token_id
        )
        self.pos_decoder = PositionalEncoding(
            self.d_model, max_len=self.max_seq_len, dropout=dropout
        )

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

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_conditioning(self, conditioning, scanner_ids):
        """
        Encode day-level conditioning into memory for the decoder.

        Args:
            conditioning: [batch, 17] - day features
            scanner_ids: [batch] - scanner indices

        Returns:
            memory: [batch, 1, d_model]
        """
        scanner_emb = self.scanner_embedding(scanner_ids)  # [batch, 32]
        combined = torch.cat([conditioning, scanner_emb], dim=-1)  # [batch, 49]

        cond_proj = self.conditioning_projection(combined)  # [batch, d_model]
        cond_seq = cond_proj.unsqueeze(1)  # [batch, 1, d_model]
        cond_encoded = self.cond_pos_encoder(cond_seq)
        memory = self.conditioning_encoder(cond_encoded)  # [batch, 1, d_model]

        return memory

    def forward(self, conditioning, scanner_ids, input_tokens):
        """
        Training forward pass with teacher forcing.

        Args:
            conditioning: [batch, 17] - day-level features
            scanner_ids: [batch] - scanner indices
            input_tokens: [batch, seq_len] - input sequence (START + body regions)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_tokens.shape

        # Encode conditioning -> memory
        memory = self._encode_conditioning(conditioning, scanner_ids)

        # Decode with teacher forcing
        tgt_emb = self.token_embedding(input_tokens)
        tgt_emb = tgt_emb * (self.d_model ** 0.5)
        tgt_emb = self.pos_decoder(tgt_emb)

        tgt_mask = create_attention_mask(seq_len, causal=True, device=input_tokens.device)
        tgt_key_padding_mask = create_key_padding_mask(
            input_tokens, pad_token_id=self.pad_token_id
        )

        decoder_output = self.token_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_projection(decoder_output)
        return logits

    @torch.no_grad()
    def generate(self, conditioning, scanner_ids, max_length=None,
                 temperature=1.0, top_k=0, top_p=0.9, allowed_tokens=None):
        """
        Autoregressive generation of body region sequence.

        Args:
            conditioning: [batch, 17] or [17]
            scanner_ids: [batch] or scalar
            max_length: max tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold
            allowed_tokens: optional iterable of token IDs the sampler may
                emit (e.g. this scanner's real region support + END/BREAK).
                Tokens outside the set get -inf logits. Guards against the
                model emitting regions that never occur on this scanner.

        Returns:
            generated_tokens: [batch, seq_len] - token IDs including START/END
        """
        self.eval()

        if max_length is None:
            max_length = self.max_seq_len

        # Handle single-sample input
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        if isinstance(scanner_ids, int):
            scanner_ids = torch.tensor([scanner_ids], device=conditioning.device)
        elif isinstance(scanner_ids, torch.Tensor) and scanner_ids.dim() == 0:
            scanner_ids = scanner_ids.unsqueeze(0)

        scanner_ids = scanner_ids.to(conditioning.device)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        allowed_mask = None
        if allowed_tokens is not None:
            allowed_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            allowed_mask[list(allowed_tokens)] = True
            allowed_mask[self.end_token_id] = True  # must always be able to stop

        # Encode conditioning
        memory = self._encode_conditioning(conditioning, scanner_ids)

        # Autoregressive generation
        generated = torch.full(
            (batch_size, 1), self.start_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_length - 1):
            tgt_emb = self.token_embedding(generated)
            tgt_emb = tgt_emb * (self.d_model ** 0.5)
            tgt_emb = self.pos_decoder(tgt_emb)

            seq_len = generated.shape[1]
            tgt_mask = create_attention_mask(seq_len, causal=True, device=device)

            decoder_output = self.token_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            next_token_logits = self.output_projection(decoder_output[:, -1, :]) / temperature

            # Mask out PAD token - should never be generated
            next_token_logits[:, self.pad_token_id] = -float('Inf')

            # Restrict to the caller's allowed token set (scanner region support)
            if allowed_mask is not None:
                next_token_logits[:, ~allowed_mask] = -float('Inf')

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k_val
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
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

            if (next_token == self.end_token_id).all():
                break

        return generated

    def compute_loss(self, logits, targets, label_smoothing=0.0, class_weights=None):
        """Compute cross-entropy loss ignoring PAD tokens.

        class_weights: optional [vocab_size] tensor up-weighting rare body
        regions so the model does not collapse to the few common ones and
        actually emits the long tail (e.g. Heart/Knee).
        """
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        weight = None
        if class_weights is not None:
            weight = class_weights.to(logits_flat.device, dtype=logits_flat.dtype)

        return F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=self.pad_token_id,
            label_smoothing=label_smoothing,
            weight=weight,
        )


def create_orchestration_model(config=None):
    """Create an OrchestrationModel instance."""
    if config is None:
        config = ORCHESTRATION_MODEL_CONFIG
    return OrchestrationModel(config)


if __name__ == "__main__":
    print("Testing Orchestration Model...")
    print("=" * 60)

    model = create_orchestration_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    batch = 4
    seq_len = 15
    cond = torch.randn(batch, ORCHESTRATION_MODEL_CONFIG['base_conditioning_dim'])
    scanner_ids = torch.randint(0, ORCHESTRATION_MODEL_CONFIG['num_scanners'], (batch,))
    input_seq = torch.randint(0, 13, (batch, seq_len))  # 0-12 valid tokens

    # Test forward pass
    logits = model(cond, scanner_ids, input_seq)
    print(f"Forward: logits={logits.shape}")

    # Test loss computation
    target_seq = torch.randint(0, 13, (batch, seq_len))
    loss = model.compute_loss(logits, target_seq, label_smoothing=0.1)
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    gen_tokens = model.generate(
        cond[:1], scanner_ids[:1], max_length=20, temperature=1.0, top_k=10
    )
    print(f"Generate: tokens={gen_tokens.shape}")
    print(f"Generated sequence: {gen_tokens[0].tolist()}")

    # Test single-sample generation
    gen_tokens_single = model.generate(
        cond[0], scanner_ids[0].item(), max_length=20
    )
    print(f"Single generate: tokens={gen_tokens_single.shape}")

    print("\nAll tests passed!")
