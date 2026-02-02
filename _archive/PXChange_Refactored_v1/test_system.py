"""
System verification script - tests all components without training
"""
import os
import sys
import torch
import numpy as np

def test_config():
    """Test configuration imports."""
    print("Testing configuration...")
    try:
        from config import (
            SEQUENCE_MODEL_CONFIG, COUNTS_MODEL_CONFIG,
            CONDITIONING_FEATURES, VOCAB_SIZE, MAX_SEQ_LEN
        )
        print(f"  [OK] Config loaded")
        print(f"    - Vocab size: {VOCAB_SIZE}")
        print(f"    - Max sequence length: {MAX_SEQ_LEN}")
        print(f"    - Conditioning features: {len(CONDITIONING_FEATURES)}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        return False


def test_preprocessing():
    """Test data preprocessing."""
    print("\nTesting preprocessing...")
    try:
        from preprocessing import load_preprocessed_data, create_dataloaders
        from preprocessing.sequence_encoder import encode_sequences, decode_sequences

        # Test encoding/decoding
        test_tokens = ['START', 'MRI_CCS_11', 'END']
        encoded = encode_sequences(test_tokens)
        decoded = decode_sequences(encoded, remove_special_tokens=False)
        assert decoded == test_tokens, "Encoding/decoding mismatch"
        print(f"  [OK] Sequence encoding/decoding works")

        # Try to load data
        try:
            df = load_preprocessed_data()
            print(f"  [OK] Data loading works ({len(df)} rows)")

            # Try to create dataloaders
            train_loader, val_loader, scaler = create_dataloaders(df, batch_size=4)
            print(f"  [OK] DataLoader creation works")
            print(f"    - Training batches: {len(train_loader)}")
            print(f"    - Validation batches: {len(val_loader)}")

            # Test batch
            batch = next(iter(train_loader))
            print(f"  [OK] Batch loading works")
            for key in batch.keys():
                print(f"    - {key}: {batch[key].shape}")

        except Exception as e:
            print(f"  [WARN] Data loading skipped (expected if no preprocessed data): {e}")

        return True
    except Exception as e:
        print(f"  [FAIL] Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model architectures."""
    print("\nTesting models...")
    try:
        from models import ConditionalSequenceGenerator, ConditionalCountsGenerator
        from models.layers import PositionalEncoding, ConditioningProjection

        # Test sequence model
        print("  Testing Conditional Sequence Generator...")
        seq_config = {
            'vocab_size': 18,
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'max_seq_len': 32,
            'conditioning_dim': 6
        }

        seq_model = ConditionalSequenceGenerator(seq_config)
        conditioning = torch.randn(2, 6)
        target_tokens = torch.randint(1, 17, (2, 10))
        logits = seq_model(conditioning, target_tokens)
        assert logits.shape == (2, 10, 18), f"Unexpected shape: {logits.shape}"
        print(f"    [OK] Forward pass works: {logits.shape}")

        # Test generation
        generated = seq_model.generate(conditioning, max_length=20)
        assert generated.shape[0] == 2, f"Unexpected batch size: {generated.shape}"
        print(f"    [OK] Generation works: {generated.shape}")

        # Test counts model
        print("  Testing Conditional Counts Generator...")
        counts_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_cross_attention_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'max_seq_len': 32,
            'conditioning_dim': 6,
            'sequence_feature_dim': 18,
            'min_sigma': 0.1
        }

        counts_model = ConditionalCountsGenerator(counts_config)
        sequence_tokens = torch.randint(1, 17, (2, 10))
        sequence_features = torch.randn(2, 10, 2)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mu, sigma = counts_model(conditioning, sequence_tokens, sequence_features, mask)
        assert mu.shape == (2, 10), f"Unexpected mu shape: {mu.shape}"
        assert sigma.shape == (2, 10), f"Unexpected sigma shape: {sigma.shape}"
        print(f"    [OK] Forward pass works: mu={mu.shape}, sigma={sigma.shape}")

        # Test sampling
        samples = counts_model.sample_counts(mu, sigma, num_samples=3)
        assert samples.shape == (2, 10, 3), f"Unexpected samples shape: {samples.shape}"
        print(f"    [OK] Sampling works: {samples.shape}")

        return True
    except Exception as e:
        print(f"  [FAIL] Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_pipeline():
    """Test generation pipeline."""
    print("\nTesting generation pipeline...")
    try:
        from models import ConditionalSequenceGenerator, ConditionalCountsGenerator
        from generation import generate_sequences_and_counts

        # Create small test models
        seq_config = {
            'vocab_size': 18,
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 128,
            'dropout': 0.0,
            'max_seq_len': 64,  # Increased for generation test
            'conditioning_dim': 6
        }

        counts_config = {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 1,
            'num_cross_attention_layers': 1,
            'dim_feedforward': 128,
            'dropout': 0.0,
            'max_seq_len': 64,  # Increased for generation test
            'conditioning_dim': 6,
            'sequence_feature_dim': 18,
            'min_sigma': 0.1
        }

        seq_model = ConditionalSequenceGenerator(seq_config)
        counts_model = ConditionalCountsGenerator(counts_config)

        # Test with dummy conditioning
        conditioning_data = np.random.randn(2, 6)

        results = generate_sequences_and_counts(
            seq_model,
            counts_model,
            conditioning_data,
            num_samples=3,
            device='cpu',
            verbose=False
        )

        assert len(results) > 0, "No results generated"
        assert 'token_name' in results.columns, "Missing token_name column"
        assert 'sampled_duration' in results.columns, "Missing sampled_duration column"
        assert 'total_time' in results.columns, "Missing total_time column"

        print(f"  [OK] Generation pipeline works")
        print(f"    - Generated {len(results)} steps")
        print(f"    - {results['sample_idx'].nunique()} unique sequences")

        return True
    except Exception as e:
        print(f"  [FAIL] Generation pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that all required directories exist or can be created."""
    print("\nTesting directory structure...")
    try:
        from config import OUTPUT_DIR, MODEL_SAVE_DIR, VISUALIZATION_DIR

        dirs = {
            'Output': OUTPUT_DIR,
            'Models': MODEL_SAVE_DIR,
            'Visualizations': VISUALIZATION_DIR
        }

        for name, path in dirs.items():
            if os.path.exists(path):
                print(f"  [OK] {name} directory exists: {path}")
            else:
                os.makedirs(path, exist_ok=True)
                print(f"  [OK] {name} directory created: {path}")

        return True
    except Exception as e:
        print(f"  [FAIL] Directory structure error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("SYSTEM VERIFICATION")
    print("="*70)

    results = {
        'Configuration': test_config(),
        'Preprocessing': test_preprocessing(),
        'Models': test_models(),
        'Generation Pipeline': test_generation_pipeline(),
        'Directory Structure': test_directory_structure()
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status:8} - {name}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("[OK] ALL TESTS PASSED - System is ready!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Train models: python main_pipeline.py train")
        print("  2. Generate sequences: python main_pipeline.py generate")
        print("  3. See QUICKSTART.md for more details")
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED - Please fix errors above")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
