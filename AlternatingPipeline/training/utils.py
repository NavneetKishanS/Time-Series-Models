"""
Shared training utilities for exchange and examination models.
"""
import functools

import numpy as np
import torch
from datetime import timedelta


def dynamic_pad_collate(batch, seq_indices, length_index, pad_token_id):
    """Collate that trims a batch down to its longest real (non-pad) sequence.

    The datasets pre-pad every sequence to the global MAX_SEQ_LEN (128 for the
    exchange/examination models). The transformer's self-attention is O(L^2),
    so on a CPU cluster (no GPU available — see step 04) padding a ~15-token
    scan out to 128 burns ~70x the attention FLOPs it needs. This collate
    stacks the batch normally, then slices every per-token field
    (those at `seq_indices`) down to the longest *non-pad* sequence IN THE
    BATCH.

    The model derives its causal mask, key-padding mask and positional encoding
    from the input length at runtime, so a batch trimmed to length L is
    mathematically identical to the same batch padded to 128 — the trailing
    columns are pure PAD that the masks already zero out. This is a free
    speedup, not an approximation.

    Args:
        batch: list of sample tuples from a Dataset.
        seq_indices: iterable of tuple positions holding per-token tensors
            shaped [max_seq_len] (input_seq / target_seq / durations).
        length_index: position of the token tensor used to measure real length
            (an input/target sequence padded with `pad_token_id`).
        pad_token_id: id used for padding in the `length_index` field.
    """
    fields = list(zip(*batch))  # transpose: one tuple per column
    length_field = torch.stack(fields[length_index])  # [B, max_seq_len]
    keep_len = int((length_field != pad_token_id).sum(dim=1).max().item())
    keep_len = max(1, keep_len)

    seq_indices = set(seq_indices)
    out = []
    for i, col in enumerate(fields):
        stacked = torch.stack(col)
        if i in seq_indices:
            stacked = stacked[:, :keep_len]
        out.append(stacked)
    return tuple(out)


def make_pad_collate(seq_indices, length_index, pad_token_id):
    """Build a picklable dynamic-padding collate_fn (see dynamic_pad_collate)."""
    return functools.partial(
        dynamic_pad_collate,
        seq_indices=tuple(seq_indices),
        length_index=length_index,
        pad_token_id=pad_token_id,
    )


def temporal_split(sequences, val_days=2):
    """
    Split sequences temporally to prevent data leakage.

    Train on earlier data, validate on later data.

    Args:
        sequences: List of sequence dicts with 'start_datetime' key
        val_days: Number of days to hold out for validation

    Returns:
        train_sequences, val_sequences
    """
    sorted_seqs = sorted(sequences, key=lambda s: s['start_datetime'])

    if len(sorted_seqs) == 0:
        return [], []

    last_date = sorted_seqs[-1]['start_datetime']
    if hasattr(last_date, 'date'):
        last_date = last_date.date()

    if hasattr(last_date, '__sub__'):
        cutoff = last_date - timedelta(days=val_days)
    else:
        from datetime import datetime
        last_dt = datetime.combine(last_date, datetime.min.time())
        cutoff_dt = last_dt - timedelta(days=val_days)
        cutoff = cutoff_dt.date()

    train_sequences = []
    val_sequences = []

    for seq in sorted_seqs:
        seq_date = seq['start_datetime']
        if hasattr(seq_date, 'date'):
            seq_date = seq_date.date()

        if seq_date < cutoff:
            train_sequences.append(seq)
        else:
            val_sequences.append(seq)

    return train_sequences, val_sequences


def safe_float(val, default=0.0):
    """Safely convert a value to float, handling errors and NaN."""
    if val is None:
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def build_conditioning_tensor(conditioning_dict):
    """
    Convert a conditioning dict to a 10-dim tensor.

    Order: Age, Weight, Height, PTAB, Direction_encoded,
           hour_sin, hour_cos, dow_sin, dow_cos, is_morning

    Args:
        conditioning_dict: Dict with conditioning feature values

    Returns:
        torch.Tensor of shape [10]
    """
    return torch.tensor([
        safe_float(conditioning_dict.get('Age', 0)),
        safe_float(conditioning_dict.get('Weight', 0)),
        safe_float(conditioning_dict.get('Height', 0)),
        safe_float(conditioning_dict.get('PTAB', 0)),
        safe_float(conditioning_dict.get('Direction_encoded', 0)),
        safe_float(conditioning_dict.get('hour_sin', 0.0)),
        safe_float(conditioning_dict.get('hour_cos', 1.0)),
        safe_float(conditioning_dict.get('dow_sin', 0.0)),
        safe_float(conditioning_dict.get('dow_cos', 1.0)),
        safe_float(conditioning_dict.get('is_morning', 0)),
    ], dtype=torch.float32)
