"""
Shared training utilities for exchange and examination models.
"""
import numpy as np
import torch
from datetime import timedelta


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
