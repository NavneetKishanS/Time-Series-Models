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
    SOURCEID_VOCAB, GENERATION_CONFIG
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
        # 'gaussian' (default) or 'log' — see compute_duration_loss / generate.
        self.duration_mode = config.get('duration_mode', 'gaussian')

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
        # Sequence-protocol trigger/gating mode (ECG, peripheral, none, ...),
        # off by default. Independent of use_exam_conditioning — same opt-in
        # precedent, examination-model only.
        self.use_sut_conditioning = config.get('use_sut_conditioning', False)

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
                # Per-position bias injected into the duration encoder for every
                # token. The conditioning token carries sequence_type once, but
                # one conditioning-token → attention weight per position means
                # the gradient from the single supervised finish token barely
                # reaches it. Adding this bias to every position gives the
                # duration encoder a direct per-type signal at the only place
                # the NLL gradient flows. Zero-initialized so old checkpoints
                # loaded with strict=False behave as if the bias does not exist.
                self.duration_seq_type_bias = nn.Embedding(
                    config['num_sequence_types'], self.d_model
                )

                # PROTOCOL — the operator-selected protocol name, the strongest
                # duration signal available and the one this model has never
                # been shown. Measured held out on the real exam CSVs (group
                # means fitted on 80%, scored on 20%):
                #
                #     sequence_type (12 values)    R2 31.2%   MAE 53.5s
                #     Sequence raw string (44)     R2 42.2%   MAE 46.6s
                #     protocol identity (~1,590)   R2 82.0%   MAE 17.2s
                #     (this model, for reference)             MAE 50.3s
                #
                # A per-protocol lookup beats the trained model threefold. The
                # variance was never missing — the 76%-unexplained figure came
                # from decomposing by the coarse sequence_type bucket, which is
                # all the model could see. Within a protocol, duration is close
                # to deterministic (residual sd 25.7s, median CV 0.134).
                #
                # It gets BOTH paths deliberately: the conditioning embedding
                # for the token decoder, and a per-position bias for the
                # duration encoder. sequence_type is the only channel this
                # model has ever responded to and the per-position bias is why;
                # everything riding the conditioning token alone measures
                # ~0.000s under perturbation. Optional, so exchange and any
                # config predating this are untouched.
                # Deliberately ADDED to the projected conditioning rather than
                # concatenated onto its input: concatenation widens
                # conditioning_projection, and PyTorch refuses a shape-mismatched
                # checkpoint no matter what `strict` says, so every existing
                # examination checkpoint would stop loading the moment this
                # landed — steps 05/06/07 would all break for the length of a
                # retrain. protocol_cond_proj is zero-initialised, so this is a
                # strict no-op until trained while gradients still flow
                # (d(Wx)/dW = x), the same contract duration_cond_bias uses.
                num_protocols = config.get('num_protocols')
                if num_protocols:
                    self.protocol_embedding = nn.Embedding(
                        num_protocols, exam_emb_dim
                    )
                    self.protocol_cond_proj = nn.Linear(exam_emb_dim, self.d_model)
                    self.duration_protocol_bias = nn.Embedding(
                        num_protocols, self.d_model
                    )

            if self.use_sut_conditioning:
                # Trigger/gating mode (ECG, peripheral, none, ...) — small
                # categorical embedding, same sizing convention as the exam
                # conditioning embeddings above.
                trigger_emb_dim = self.d_model // 8
                self.trigger_mode_embedding = nn.Embedding(
                    config['num_trigger_modes'], trigger_emb_dim
                )
                cond_input_dim += trigger_emb_dim

        if self.has_phase_type:
            phase_emb_dim = self.d_model // 8  # 32 for d_model=256
            self.phase_type_embedding = nn.Embedding(
                config.get('num_phase_types', 3), phase_emb_dim
            )
            cond_input_dim += phase_emb_dim

        # Per-feature divisors bringing raw conditioning features to O(1)
        # BEFORE they are concatenated with the categorical embeddings. Raw
        # Age≈50/Weight≈75 dominate the pre-LayerNorm variance of
        # conditioning_projection, and the LayerNorm then attenuates the
        # O(0.5) embedding contributions (body region, scan type, serial) to
        # ~0.6% relative amplitude — conditioning was effectively erased.
        #
        # ALWAYS registered as a persistent buffer (default = identity ones
        # when a config omits 'conditioning_scale'). The trained values then
        # travel INSIDE the checkpoint, so a serve-side model built from a
        # stale config still receives them via load_state_dict — the buffer
        # slot always exists. Registering it only conditionally caused step 05
        # to fail with "Unexpected key conditioning_scale" whenever its config
        # was a step behind the trained checkpoint.
        cond_scale = config.get('conditioning_scale')
        if cond_scale is None:
            cond_scale = [1.0] * base_conditioning_dim
        self.register_buffer(
            'conditioning_scale',
            torch.tensor(cond_scale, dtype=torch.float32),
        )

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
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,  # nested tensor path is 5-10x slower on CPU
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
            duration_encoder_layer, num_layers=num_duration_encoder_layers,
            enable_nested_tensor=False,  # nested tensor path is 5-10x slower on CPU
        )

        self.duration_head = SinglePassDurationHead(self.d_model, hidden_dim=128, dropout=dropout)

        # Per-position projection of the conditioning token into the duration
        # encoder. The conditioning token is prepended at position 0 and the
        # duration head reads only positions 1..N, so conditioning can reach it
        # ONLY through attention back to that one position — and measurement
        # showed it does not. On the 2026-07-24 examination checkpoint
        # (06_compare_models.py Criterion 1b), perturbing each channel moved the
        # duration head by:
        #
        #     sequence_type (scout vs space)    249.475s   <- per-position bias
        #     body_region   (ABDOMEN vs SPINE)    0.000s   <- cond token only
        #     serial_idx    (0 vs 1)              0.001s   <- cond token only
        #     TR+num_slices (1x vs 3x)            0.002s   <- cond token only
        #
        # sequence_type is the only channel with a per-position path
        # (duration_seq_type_bias) and the only one the head responds to.
        # body_region has a ~2.3x real duration spread (ABDOMEN ~63s vs SPINE
        # ~144s), so a 0.000s response is a model defect, not a property of the
        # data — the head had learned to ignore position 0 entirely once the
        # easier per-position route existed for the dominant feature.
        #
        # ZERO-INITIALISED, so this is a strict no-op until trained: every
        # existing checkpoint loads (the key is simply missing) and predicts
        # exactly as it did before. Gradients still flow — d(Wx)/dW = x — so it
        # learns from the first step, the same way duration_seq_type_bias did.
        # Created unconditionally: the exchange model has the identical dead-
        # conditioning shape (Stage 3b's "no variability across customers"), and
        # zero-init means adding it there costs nothing until that model is
        # retrained too.
        self.duration_cond_bias = nn.Linear(self.d_model, self.d_model)

        self._init_weights()

    # Parameters that must start at exactly zero so the paths they feed are
    # no-ops for checkpoints trained before they existed.
    _ZERO_INIT_PARAMS = (
        'duration_seq_type_bias', 'duration_cond_bias',
        'duration_protocol_bias', 'protocol_cond_proj',
    )

    def _init_weights(self):
        for name, p in self.named_parameters():
            if any(zero_name in name for zero_name in self._ZERO_INIT_PARAMS):
                # Covers bias vectors too: nn.Linear's default init is uniform,
                # NOT zero, so a dim>1 check alone would leave the bias live and
                # silently shift every pre-existing checkpoint's predictions.
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _force_token_subset(logits, row_idx, token_ids):
        """Restrict one batch row to a non-empty set of token IDs."""
        token_ids = list(token_ids)
        if not token_ids:
            raise ValueError("token_ids must contain at least one token")

        keep = logits[row_idx, token_ids].clone()
        logits[row_idx].fill_(-float('Inf'))
        # Model logits should already be finite. Falling back to zero keeps the
        # constraint deterministic even if an upstream numerical issue occurs.
        logits[row_idx, token_ids] = torch.where(
            torch.isfinite(keep), keep, torch.zeros_like(keep)
        )

    def _apply_generation_constraints(self, logits, generated, remaining_steps):
        """Apply model-specific grammar before top-k/top-p sampling."""
        logits = logits.clone()

        # Control/placeholder tokens are training scaffolding, never real events.
        blocked = {PAD_TOKEN_ID, START_TOKEN_ID}
        unk_token_id = SOURCEID_VOCAB.get('UNK')
        if unk_token_id is not None:
            blocked.add(unk_token_id)
        logits[:, list(blocked)] = -float('Inf')

        exam_start_id = SOURCEID_VOCAB.get('MRI_MSR_100')
        exam_finish_ids = tuple(
            token_id for token_id in (
                SOURCEID_VOCAB.get('MRI_MSR_104'),
                SOURCEID_VOCAB.get('MRI_MSR_34'),
            )
            if token_id is not None
        )

        for row_idx in range(generated.shape[0]):
            row = generated[row_idx]

            # Batched callers can finish at different steps. Keep completed
            # rows terminal while unfinished rows continue decoding.
            if (row == END_TOKEN_ID).any():
                self._force_token_subset(logits, row_idx, [END_TOKEN_ID])
                continue

            content = row[1:]  # remove the initial START marker

            if self.model_type == 'examination':
                if exam_start_id is None or not exam_finish_ids:
                    raise RuntimeError(
                        "Examination grammar tokens are missing from SOURCEID_VOCAB"
                    )

                if content.numel() == 0:
                    # Every rendered measurement must open with MRI_MSR_100.
                    self._force_token_subset(logits, row_idx, [exam_start_id])
                    continue

                last_token = int(content[-1].item())
                if last_token in exam_finish_ids:
                    # A successful/aborted finish is immediately followed by END.
                    self._force_token_subset(logits, row_idx, [END_TOKEN_ID])
                    continue

                # A measurement cannot restart or terminate before a finish event.
                logits[row_idx, exam_start_id] = -float('Inf')
                logits[row_idx, END_TOKEN_ID] = -float('Inf')

                # Reserve the final slot for END. If the model has not closed the
                # measurement in time, select its preferred valid finish token.
                if remaining_steps <= 2:
                    self._force_token_subset(logits, row_idx, exam_finish_ids)
            else:
                # Exchange blocks must contain at least one real event.
                if content.numel() == 0:
                    logits[row_idx, END_TOKEN_ID] = -float('Inf')

                # Always return a closed sequence when max_length is reached.
                if remaining_steps == 1:
                    self._force_token_subset(logits, row_idx, [END_TOKEN_ID])

        if not torch.isfinite(logits).any(dim=-1).all():
            raise RuntimeError(
                "Generation constraints removed every candidate token for at least one sample"
            )
        return logits

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
        # Always a registered buffer (identity ones when unconfigured), so the
        # division is a no-op for models without a configured scale.
        conditioning = conditioning / self.conditioning_scale
        parts = [conditioning]

        if self.body_region_mode == 'from_to':
            from_emb = self.body_from_embedding(body_region_info['body_from'])
            to_emb = self.body_to_embedding(body_region_info['body_to'])
            parts.extend([from_emb, to_emb])
        else:
            region_emb = self.body_region_embedding(body_region_info['body_region'])
            parts.append(region_emb)
            region_t = body_region_info['body_region']

            if self.use_exam_conditioning:
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

            if self.use_sut_conditioning:
                # trigger_mode defaults to 0 ('none'/first class) when a
                # caller does not supply it — same pattern as sequence_type.
                trig_t = body_region_info.get('trigger_mode')
                if trig_t is None:
                    trig_t = torch.zeros_like(region_t)
                parts.append(self.trigger_mode_embedding(trig_t))

        if self.has_phase_type and phase_type is not None:
            phase_emb = self.phase_type_embedding(phase_type)
            parts.append(phase_emb)

        combined = torch.cat(parts, dim=-1)
        cond_proj = self.conditioning_projection(combined)  # [batch, d_model]

        # Protocol joins here rather than in `parts` so the projection's input
        # width — and every existing checkpoint — stays valid. Absent key ->
        # RARE_PROTOCOL_ID (0), the reserved bucket for protocols below the
        # frequency floor and anything unseen at serve time.
        if hasattr(self, 'protocol_embedding'):
            proto_t = body_region_info.get('protocol')
            if proto_t is None:
                proto_t = torch.zeros(
                    cond_proj.shape[0], dtype=torch.long, device=cond_proj.device
                )
            cond_proj = cond_proj + self.protocol_cond_proj(
                self.protocol_embedding(proto_t)
            )

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

        # Inject per-sequence-type bias at every token position when available.
        # This gives the duration encoder a direct per-type signal at each
        # position rather than relying solely on the single conditioning token,
        # whose gradient from the finish-token NLL is diluted across all heads.
        if self.use_exam_conditioning and hasattr(self, 'duration_seq_type_bias'):
            seq_t = body_region_info.get('sequence_type')
            if seq_t is not None:
                st_bias = self.duration_seq_type_bias(seq_t)  # [batch, d_model]
                tok_emb = tok_emb + st_bias.unsqueeze(1)      # broadcast over seq dim

        # Same mechanism for the protocol, which carries far more of the
        # duration signal than the scan type does (82.0% of held-out variance
        # vs 31.2%). Zero-initialised, so this is a strict no-op for any
        # checkpoint trained before it existed.
        if hasattr(self, 'duration_protocol_bias'):
            proto_t = body_region_info.get('protocol')
            if proto_t is None:
                proto_t = torch.zeros(
                    token_ids.shape[0], dtype=torch.long, device=token_ids.device
                )
            tok_emb = tok_emb + self.duration_protocol_bias(proto_t).unsqueeze(1)

        # Inject the FULL conditioning at every token position, for the same
        # reason and by the same mechanism. Position 0 below still carries the
        # conditioning token; this adds a direct route that does not depend on
        # the encoder choosing to attend back to it. Zero-initialised, so this
        # contributes exactly nothing until trained — see __init__ for the
        # measurements that motivated it.
        if hasattr(self, 'duration_cond_bias'):
            tok_emb = tok_emb + self.duration_cond_bias(cond_token)  # broadcast over seq dim

        # Prepend conditioning token
        combined = torch.cat([cond_token, tok_emb], dim=1)  # [batch, 1+seq_len, d_model]
        combined = self.duration_pos_encoding(combined)

        # Key-padding mask so PAD positions cannot influence the encoding.
        # Training feeds max_seq_len padded sequences while generation feeds
        # short UNPADDED ones; without this mask the encoder learns to lean on
        # the ~100 PAD embeddings as context, and at generation (no pads) its
        # output collapses to a flat prior — measured: a model probed with its
        # own padded training input predicts mu correctly per scan type, the
        # identical unpadded sequence returns mu≈0 everywhere. The mask makes
        # both input styles equivalent. Position 0 (conditioning token) is
        # never masked.
        # END is masked for the same train/serve parity reason: training input
        # is [START + tokens] and never contains END (it exists only in the
        # shifted targets), but generate() passes the full generated sequence
        # which ends with END — an embedding this encoder never trained on.
        pad_mask = (token_ids == PAD_TOKEN_ID) | (token_ids == END_TOKEN_ID)  # [batch, seq_len]
        full_pad_mask = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=token_ids.device),
             pad_mask], dim=1
        )

        # Bidirectional encoder (NO causal mask)
        encoded = self.duration_encoder(
            combined, src_key_padding_mask=full_pad_mask
        )  # [batch, 1+seq_len, d_model]

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
                 return_stats=False, enforce_constraints=True):
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
            enforce_constraints: enforce valid control-token and examination
                state transitions. Disable only for decoder diagnostics.

        Returns:
            generated_tokens: [batch, seq_len]
            generated_durations: [batch, seq_len]
            duration_mu: [batch, seq_len]  — only if return_stats=True
            duration_sigma: [batch, seq_len] — only if return_stats=True
        """
        self.eval()

        if max_length is None:
            max_length = self.max_seq_len
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        min_length = 4 if self.model_type == 'examination' else 3
        if enforce_constraints and max_length < min_length:
            raise ValueError(
                f"max_length must be at least {min_length} for constrained "
                f"{self.model_type} generation"
            )

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

        for step_idx in range(max_length - 1):
            tgt_emb = self.token_embedding(generated)
            tgt_emb = tgt_emb * (self.d_model ** 0.5)
            tgt_emb = self.pos_decoder(tgt_emb)

            seq_len = generated.shape[1]
            tgt_mask = create_attention_mask(seq_len, causal=True, device=device)

            decoder_output = self.token_decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            next_token_logits = self.output_projection(decoder_output[:, -1, :]) / temperature

            if enforce_constraints:
                remaining_steps = (max_length - 1) - step_idx
                next_token_logits = self._apply_generation_constraints(
                    next_token_logits, generated, remaining_steps
                )

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
            if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
                raise RuntimeError("Invalid sampling probabilities after token filtering")
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == END_TOKEN_ID).all():
                break

        # Phase 2: Single-pass duration estimation
        duration_mu, duration_sigma = self.estimate_durations(
            generated, conditioning, body_region_info, phase_type
        )

        # Sample durations. In 'log' mode the head models log1p(duration),
        # so sample in log space and invert with expm1; otherwise sample the
        # Normal directly. Both clamp to non-negative.
        if self.duration_mode == 'log':
            durations = torch.expm1(
                torch.normal(duration_mu, duration_sigma)
            ).clamp(min=0.0)
        else:
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
        """Compute Gaussian NLL loss for duration prediction.

        In 'log' mode the target is transformed with log1p, so the head
        models log-durations. This fits the central tendency of the heavily
        right-skewed exchange duration distribution instead of the inflated
        mean. log1p is zero-safe: the 0 s first/END tokens map to 0.
        """
        if self.duration_mode == 'log':
            target_durations = torch.log1p(target_durations.clamp(min=0.0))
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
