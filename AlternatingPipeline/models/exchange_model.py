"""
Exchange Model: Wrapper around the unified SequenceGeneratorModel.

Generates event sequences for body region transitions (exchange phases),
conditioned on body_from, body_to, phase_type, and patient features.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXCHANGE_MODEL_CONFIG
from models.sequence_generator import SequenceGeneratorModel, create_sequence_generator


# The exchange model IS a SequenceGeneratorModel with exchange config
ExchangeModel = SequenceGeneratorModel


def create_exchange_model(config=None):
    """
    Create an Exchange Model instance.

    Args:
        config: Optional config dict. If None, uses EXCHANGE_MODEL_CONFIG

    Returns:
        SequenceGeneratorModel configured for exchange
    """
    if config is None:
        config = EXCHANGE_MODEL_CONFIG
    return create_sequence_generator(config)


if __name__ == "__main__":
    from config import BODY_REGIONS

    print("Testing Exchange Model (Unified Transformer)...")
    print("=" * 60)

    model = create_exchange_model()
    print(f"Model type: {model.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 20
    cond_dim = EXCHANGE_MODEL_CONFIG['base_conditioning_dim']
    conditioning = torch.randn(batch_size, cond_dim)
    body_from = torch.randint(0, 13, (batch_size,))
    body_to = torch.randint(0, 11, (batch_size,))
    phase_type = torch.randint(0, 3, (batch_size,))
    input_seq = torch.randint(1, 17, (batch_size, seq_len))

    logits, dur_mu, dur_sigma = model(
        conditioning,
        {'body_from': body_from, 'body_to': body_to},
        input_seq,
        phase_type=phase_type
    )
    print(f"\nForward pass:")
    print(f"  Logits: {logits.shape}")
    print(f"  Duration mu: {dur_mu.shape}")
    print(f"  Duration sigma: {dur_sigma.shape}")

    # Test generation
    print(f"\nGeneration:")
    for pt in range(3):
        gen_tokens, gen_durs = model.generate(
            conditioning[:1],
            {'body_from': body_from[:1], 'body_to': body_to[:1]},
            phase_type=pt,
            max_length=30,
            temperature=1.0, top_k=10
        )
        print(f"  Phase {pt}: {gen_tokens.shape[1]} tokens, "
              f"total duration={gen_durs[0].sum().item():.1f}s")

    print("\nAll tests passed!")
