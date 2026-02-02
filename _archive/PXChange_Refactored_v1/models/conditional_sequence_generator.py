"""
Conditional Sequence Generator: Transformer Encoder-Decoder
Generates symbolic sequences (sourceID tokens) conditioned on patient/scan context.
Auto-regressive: each token depends on previous tokens and conditioning.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_MODEL_CONFIG, START_TOKEN_ID, END_TOKEN_ID,
    PAD_TOKEN_ID, SEQUENCE_SAMPLING_CONFIG
)
from models.layers import (
    PositionalEncoding, ConditioningProjection,
    create_attention_mask, create_key_padding_mask
)


class ConditionalSequenceGenerator(nn.Module):
    """
    Transformer encoder-decoder for conditional sequence generation.

    Architecture:
        1. Conditioning encoder: processes patient/scan context
        2. Token embedding + positional encoding
        3. Transformer encoder: encodes conditioning
        4. Transformer decoder: generates sequence auto-regressively
        5. Output projection: predicts next token probabilities
    """

    def __init__(self, config=None):
        super(ConditionalSequenceGenerator, self).__init__()

        if config is None:
            config = SEQUENCE_MODEL_CONFIG

        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.max_seq_len = config['max_seq_len']
        self.conditioning_dim = config['conditioning_dim']

        # Conditioning projection
        self.conditioning_projection = ConditioningProjection(
            self.conditioning_dim, self.d_model, dropout=self.dropout
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

    def encode_conditioning(self, conditioning):
        """
        Encode conditioning features into memory for decoder.

        Args:
            conditioning: [batch_size, conditioning_dim]

        Returns:
            memory: [batch_size, 1, d_model] - encoded conditioning
        """
        # Project conditioning to d_model
        cond_proj = self.conditioning_projection(conditioning)  # [batch_size, d_model]

        # Add sequence dimension and positional encoding
        cond_seq = cond_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        cond_encoded = self.pos_encoder(cond_seq)

        # Pass through encoder
        memory = self.transformer_encoder(cond_encoded)  # [batch_size, 1, d_model]

        return memory

    def forward(self, conditioning, target_tokens):
        """
        Forward pass for training (teacher forcing).

        Args:
            conditioning: [batch_size, conditioning_dim]
            target_tokens: [batch_size, seq_len] - target sequence with START prepended

        Returns:
            logits: [batch_size, seq_len, vocab_size] - token logits
        """
        batch_size, seq_len = target_tokens.shape

        # Encode conditioning
        memory = self.encode_conditioning(conditioning)  # [batch_size, 1, d_model]

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
        )  # [batch_size, seq_len, d_model]

        # Project to vocabulary
        logits = self.output_projection(decoder_output)  # [batch_size, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, conditioning, max_length=None, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate sequences auto-regressively (inference mode).

        Args:
            conditioning: [batch_size, conditioning_dim]
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

        batch_size = conditioning.shape[0]
        device = conditioning.device

        # Encode conditioning
        memory = self.encode_conditioning(conditioning)

        # Initialize with START token
        generated = torch.full((batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(conditioning, generated)  # [batch_size, curr_len, vocab_size]

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have generated END token
            if (next_token == END_TOKEN_ID).all():
                break

        return generated