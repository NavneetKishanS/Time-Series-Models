"""
Examination Model: Wrapper around the unified SequenceGeneratorModel.

Generates MRI event sequences for a specific body region,
conditioned on patient context features.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXAMINATION_MODEL_CONFIG, NUM_BODY_REGIONS, BODY_REGIONS
from models.sequence_generator import SequenceGeneratorModel, create_sequence_generator


# The examination model IS a SequenceGeneratorModel with examination config
ExaminationModel = SequenceGeneratorModel


def create_examination_model(config=None):
    """
    Create an Examination Model instance.

    Args:
        config: Optional config dict. If None, uses EXAMINATION_MODEL_CONFIG

    Returns:
        SequenceGeneratorModel configured for examination
    """
    if config is None:
        config = EXAMINATION_MODEL_CONFIG
    return create_sequence_generator(config)


if __name__ == "__main__":
    print("Testing Examination Model (Unified Transformer)...")
    print("=" * 60)

    model = create_examination_model()
    print(f"Model type: {model.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 20
    cond_dim = EXAMINATION_MODEL_CONFIG['base_conditioning_dim']
    conditioning = torch.randn(batch_size, cond_dim)
    body_region = torch.randint(0, NUM_BODY_REGIONS, (batch_size,))
    input_seq = torch.randint(1, 17, (batch_size, seq_len))

    logits, dur_mu, dur_sigma = model(
        conditioning, {'body_region': body_region}, input_seq
    )
    print(f"\nForward pass:")
    print(f"  Logits: {logits.shape}")
    print(f"  Duration mu: {dur_mu.shape}")
    print(f"  Duration sigma: {dur_sigma.shape}")

    # Test loss
    target_seq = torch.randint(1, 17, (batch_size, seq_len))
    loss = model.compute_loss(logits, target_seq, label_smoothing=0.1)
    print(f"\nToken loss: {loss.item():.4f}")

    dummy_durations = torch.rand(batch_size, seq_len) * 300
    dur_loss = model.compute_duration_loss(dur_mu, dur_sigma, dummy_durations)
    print(f"Duration loss: {dur_loss.item():.4f}")

    # Test generation
    print(f"\nGeneration per body region:")
    single_cond = torch.randn(cond_dim)
    for region_id in [0, 5, 9]:  # HEAD, SPINE, FOOT
        gen_tokens, gen_durs = model.generate(
            single_cond, {'body_region': region_id},
            max_length=30, temperature=1.0, top_k=10
        )
        print(f"  {BODY_REGIONS[region_id]}: {gen_tokens.shape[1]} tokens, "
              f"total duration={gen_durs[0].sum().item():.1f}s")

    print("\nAll tests passed!")
