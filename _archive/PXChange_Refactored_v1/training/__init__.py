"""
Training modules for conditional generation models
"""

from .train_sequence_model import train_sequence_model
from .train_counts_model import train_counts_model

__all__ = [
    'train_sequence_model',
    'train_counts_model'
]
