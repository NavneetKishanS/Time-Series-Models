"""
Models for the Alternating Pipeline.

Unified Transformer architecture:
- SequenceGeneratorModel: Core model used by both exchange and examination
- ExchangeModel: SequenceGeneratorModel with exchange config (from_to regions + phase_type)
- ExaminationModel: SequenceGeneratorModel with examination config (single region)
- OrchestrationModel: Autoregressive Transformer for day-level patient scheduling
"""
from .layers import PositionalEncoding, SinglePassDurationHead, create_attention_mask, create_key_padding_mask
from .sequence_generator import SequenceGeneratorModel, create_sequence_generator
from .exchange_model import ExchangeModel, create_exchange_model
from .examination_model import ExaminationModel, create_examination_model
from .orchestration_model import OrchestrationModel, create_orchestration_model

__all__ = [
    'PositionalEncoding',
    'SinglePassDurationHead',
    'create_attention_mask',
    'create_key_padding_mask',
    'SequenceGeneratorModel',
    'create_sequence_generator',
    'ExchangeModel',
    'create_exchange_model',
    'ExaminationModel',
    'create_examination_model',
    'OrchestrationModel',
    'create_orchestration_model',
]
