"Main Pipeline: Complete training and generation workflow"
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEQUENCE_MODEL_CONFIG, SEQUENCE_TRAINING_CONFIG,
    COUNTS_MODEL_CONFIG, COUNTS_TRAINING_CONFIG,
    MODEL_SAVE_DIR, OUTPUT_DIR, RANDOM_SEED, DATA_DIR
)
from preprocessing.preprocess_raw_data import preprocess_all_datasets
from models.conditional_sequence_generator import ConditionalSequenceGenerator
from models.conditional_counts_generator import ConditionalCountsGenerator
from training.train_sequence_model import train_sequence_model
from training.train_counts_model import train_counts_model
from generation.generate_pipeline import generate_sequences_and_counts, save_generated_results, print_generation_examples, generate_sequence_pool
from resample_sequences import resample_sequences
from config import CONDITIONING_FEATURES


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def preprocess_pipeline(args):
    """
    Preprocessing pipeline to convert raw CSV files to training format.
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING PIPELINE")
    print(f"{'='*70}\n")

    # Check if raw CSV files exist
    import glob
    raw_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not raw_files:
        print(f"Error: No CSV files found in {DATA_DIR}")
        print(f"Please place your raw MRI scan CSV files in the data directory.")
        return

    print(f"Found {len(raw_files)} raw CSV files")

    # Run preprocessing
    preprocess_all_datasets()

    print(f"\n{'='*70}")
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Preprocessed data saved to: {os.path.join(DATA_DIR, 'preprocessed')}")
    print(f"\nYou can now train the models with: python main_pipeline.py train")


def train_pipeline(args):
    """
    Complete training pipeline for both models.
    """
    print(f"\n{'='*70}")
    print(f"CONDITIONAL GENERATION TRAINING PIPELINE")
    print(f"{'='*70}\n")

    # Set seeds
    set_random_seeds()

    data_path = os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Run 'preprocess' first.")

    # Train sequence model
    if 'sequence' in args.models or 'all' in args.models:
        print(f"\n{'='*70}")
        print("Step 1: Training Conditional Sequence Generator...")
        print(f"{'='*70}\n")
        train_sequence_model(data_path, validation_split=args.val_split)
    else:
        print("\n[SKIP] Skipping sequence model training")

    # Train counts model
    if 'counts' in args.models or 'all' in args.models:
        print(f"\n{'='*70}")
        print("Step 2: Training Conditional Counts Generator...")
        print(f"{'='*70}\n")
        train_counts_model(data_path, validation_split=args.val_split)
    else:
        print("\n[SKIP] Skipping counts model training")

    print(f"\n{'='*70}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Models saved to: {MODEL_SAVE_DIR}")


def generate_pipeline(args):
    """
    Generation pipeline using trained models.
    """
    print(f"\n{'='*70}")
    print(f"CONDITIONAL GENERATION PIPELINE")
    print(f"{'='*70}\n")

    # Set seeds
    set_random_seeds()

    # Check if models exist
    seq_model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pth')
    counts_model_path = os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pth')
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'conditioning_scaler.pkl')

    if not os.path.exists(seq_model_path):
        raise FileNotFoundError(f"Sequence model not found at {seq_model_path}. Train models first!")
    if not os.path.exists(counts_model_path):
        raise FileNotFoundError(f"Counts model not found at {counts_model_path}. Train models first!")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}\n")

    # Load models
    print("Step 1: Loading trained models...")

    # Load sequence model
    sequence_model = ConditionalSequenceGenerator(SEQUENCE_MODEL_CONFIG)
    sequence_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    sequence_model.to(device)
    sequence_model.eval()
    print("[OK] Sequence model loaded")

    # Load counts model
    counts_model = ConditionalCountsGenerator(COUNTS_MODEL_CONFIG)
    counts_model.load_state_dict(torch.load(counts_model_path, map_location=device))
    counts_model.to(device)
    counts_model.eval()
    print("[OK] Counts model loaded")

    # Load scaler
    import pickle
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            conditioning_scaler = pickle.load(f)
        print("[OK] Conditioning scaler loaded")
    else:
        conditioning_scaler = None
        print("[WARN] Scaler not found. Assuming conditioning data is not scaled.")

    # Load conditioning data
    print("\nStep 2: Preparing conditioning data...")
    if args.conditioning_file:
        conditioning_df = pd.read_csv(args.conditioning_file)
        if 'dataset_id' not in conditioning_df.columns:
            raise ValueError("Conditioning file must contain 'dataset_id' column for per-customer generation")
        print(f"[OK] Loaded conditioning from {args.conditioning_file}")
        print(f"    Customers found: {conditioning_df['dataset_id'].nunique()}")
    else:
        # Use preprocessed data - get one representative row per customer
        preprocessed_path = os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv')
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_path}. Run 'preprocess' first.")
            
        df = pd.read_csv(preprocessed_path)
        # Get one row per customer (dataset_id)
        conditioning_df = df.groupby('dataset_id').first().reset_index()
        print(f"[OK] Loaded conditioning data for {len(conditioning_df)} customers")

    # Calculate number of sequences per customer from original data (to match input volume)
    print("\nStep 2b: Calculating sequence counts per customer...")
    if args.match_input_volume:
        full_df_path = os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv')
        df_full = pd.read_csv(full_df_path)
        sequences_per_customer = df_full.groupby('dataset_id')['SeqOrder'].nunique().to_dict()
        # Convert all keys to strings for consistent lookup
        sequences_per_customer = {str(k): v for k, v in sequences_per_customer.items()}
        print(f"[OK] Will generate matching number of sequences per customer (avg: {np.mean(list(sequences_per_customer.values())):.1f})")
        print(f"    Example: Customer 175832 will generate {sequences_per_customer.get('175832', 'N/A')} sequences")
        print(f"    Total sequences to generate: {sum(sequences_per_customer.values())}")
    else:
        sequences_per_customer = None
        print(f"[OK] Will generate {args.num_samples} sequences per customer")

    # Generate sequences and counts (per customer)
    print("\nStep 3: Generating sequences and counts (per customer)...")
    results_df = generate_sequences_and_counts(
        sequence_model,
        counts_model,
        conditioning_df,
        conditioning_scaler=conditioning_scaler,
        num_samples_per_customer=args.num_samples,
        sequences_per_customer=sequences_per_customer,
        remove_repetitions=not args.keep_repetitions,
        device=device,
        verbose=True
    )

    # Save results
    print("\nStep 4: Saving results...")
    output_file = save_generated_results(results_df, filename=args.output_file)

    # Print examples
    if not results_df.empty:
        print_generation_examples(results_df, num_examples=min(5, len(conditioning_df)))

    print(f"\n{'='*70}")
    print("GENERATION PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Results saved to: {output_file}")


def generate_calibrated_pipeline(args):
    """
    Generates a calibrated set of sequences by creating a pool and resampling.
    """
    print(f"\n{'='*70}")
    print(f"CALIBRATED GENERATION PIPELINE")
    print(f"{'='*70}\n")

    # Set seeds
    set_random_seeds()

    # Check if models exist
    seq_model_path = os.path.join(MODEL_SAVE_DIR, 'sequence_model_best.pth')
    counts_model_path = os.path.join(MODEL_SAVE_DIR, 'counts_model_best.pth')

    if not os.path.exists(seq_model_path) or not os.path.exists(counts_model_path):
        raise FileNotFoundError("Models not found. Train models first with 'train' command.")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}\n")

    # --- Step 1: Generate a large pool of sequences ---
    print("--- Step 1: Generating sequence pool ---")
    
    # Load models
    sequence_model = ConditionalSequenceGenerator(SEQUENCE_MODEL_CONFIG)
    sequence_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    sequence_model.to(device)
    
    counts_model = ConditionalCountsGenerator(COUNTS_MODEL_CONFIG)
    counts_model.load_state_dict(torch.load(counts_model_path, map_location=device))
    counts_model.to(device)

    pool_df = generate_sequence_pool(
        sequence_model,
        counts_model,
        num_samples_per_category=args.num_samples_per_category,
        remove_repetitions=not args.keep_repetitions,
        device=device,
        verbose=True
    )

    # Save the pool
    pool_output_path = os.path.join(OUTPUT_DIR, args.pool_file)
    save_generated_results(pool_df, filename=args.pool_file)

    # --- Step 2: Resample the pool to match true distribution ---
    print("\n--- Step 2: Resampling sequence pool to match true distribution ---")
    
    true_data_path = os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv')
    calibrated_output_path = os.path.join(OUTPUT_DIR, args.output_file)

    resample_sequences(
        generated_pool_path=pool_output_path,
        true_data_path=true_data_path,
        output_path=calibrated_output_path,
        feature_column='BodyGroup_from'
    )
    
    print(f"\n{'='*70}")
    print("CALIBRATED GENERATION PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    print(f"[OK] Calibrated results saved to: {calibrated_output_path}")


def evaluate_pipeline(args):
    """
    Evaluation pipeline: compare generated vs true sequences.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION PIPELINE")
    print(f"{'='*70}\n")

    # Load generated results
    generated_file = os.path.join(OUTPUT_DIR, args.generated_file)
    if not os.path.exists(generated_file):
        raise FileNotFoundError(f"Generated results not found at {generated_file}")

    generated_df = pd.read_csv(generated_file)
    print(f"[OK] Loaded generated results from {generated_file}")
    
    if generated_df.empty:
        print("[WARN] Generated data is empty. Cannot perform evaluation.")
        return
        
    print(f"  Total customers: {generated_df['SN'].nunique()}")
    print(f"  Total sequences: {len(generated_df.groupby(['SN', 'sample_idx']))}")
    print(f"  Average length: {generated_df.groupby(['SN', 'sample_idx'])['step'].max().mean():.1f}")
    print(f"  Average total time: {generated_df.groupby(['SN', 'sample_idx'])['total_time'].first().mean():.1f}s")

    # Load true data for comparison
    print("\n[OK] Loading true data for comparison...")
    true_data_path = os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv')
    if not os.path.exists(true_data_path):
        raise FileNotFoundError(f"True data not found at {true_data_path}. Run 'preprocess' first.")
    
    df = pd.read_csv(true_data_path)

    # Compute statistics
    true_lengths = df.groupby('SeqOrder').size()
    true_total_times = df.groupby('SeqOrder')['step_duration'].sum()

    generated_lengths = generated_df.groupby(['SN', 'sample_idx']).size()
    generated_total_times = generated_df.groupby(['SN', 'sample_idx'])['total_time'].first()

    print(f"\n{'='*70}")
    print("COMPARISON STATISTICS")
    print(f"{'='*70}\n")

    print("Sequence Length:")
    print(f"  True - Mean: {true_lengths.mean():.1f}, Std: {true_lengths.std():.1f}, Range: [{true_lengths.min()}, {true_lengths.max()}]")
    print(f"  Generated - Mean: {generated_lengths.mean():.1f}, Std: {generated_lengths.std():.1f}, Range: [{generated_lengths.min()}, {generated_lengths.max()}]")

    print("\nTotal Time (seconds):")
    print(f"  True - Mean: {true_total_times.mean():.1f}, Std: {true_total_times.std():.1f}, Range: [{true_total_times.min():.1f}, {true_total_times.max():.1f}]")
    print(f"  Generated - Mean: {generated_total_times.mean():.1f}, Std: {generated_total_times.std():.1f}, Range: [{generated_total_times.min():.1f}, {generated_total_times.max():.1f}]")

    # Token distribution
    print("\nToken Distribution (top 10):")
    true_tokens = df['sourceID'].value_counts(normalize=True).head(10)
    generated_tokens = generated_df['token_id'].value_counts(normalize=True).head(10)

    print("\n  True:")
    for token_id, ratio in true_tokens.items():
        print(f"    Token {token_id}: {ratio*100:.2f}%")

    print("\n  Generated:")
    for token_id, ratio in generated_tokens.items():
        print(f"    Token {token_id}: {ratio*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Conditional Generation Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command', required=True)

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw CSV files')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--models', nargs='+', default=['all'], choices=['sequence', 'counts', 'all'], help='Models to train')
    train_parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sequences and counts per customer')
    generate_parser.add_argument('--conditioning-file', type=str, default=None, help='CSV file with conditioning data (must have dataset_id column)')
    generate_parser.add_argument('--num-samples', type=int, default=15, help='Number of sequences to generate per customer (ignored if --match-input-volume)')
    generate_parser.add_argument('--match-input-volume', action='store_true', help='Generate same number of sequences as in original data per customer')
    generate_parser.add_argument('--keep-repetitions', action='store_true', help='Keep consecutive token repetitions (do not filter)')
    generate_parser.add_argument('--output-file', type=str, default='generated_sequences.csv', help='Output filename')
    generate_parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage even if GPU is available')

    # Calibrated Generate command
    calibrated_generate_parser = subparsers.add_parser('generate-calibrated', help='Generate a calibrated set of sequences via pooling and resampling')
    calibrated_generate_parser.add_argument('--num-samples-per-category', type=int, default=500, help='Number of sequences to generate for each body region in the pool.')
    calibrated_generate_parser.add_argument('--keep-repetitions', action='store_true', help='Keep consecutive token repetitions (do not filter).')
    calibrated_generate_parser.add_argument('--pool-file', type=str, default='generated_sequences_pool.csv', help='Filename for the intermediate sequence pool.')
    calibrated_generate_parser.add_argument('--output-file', type=str, default='generated_sequences_calibrated.csv', help='Final output filename for calibrated sequences.')
    calibrated_generate_parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage even if GPU is available.')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated sequences')
    eval_parser.add_argument('--generated-file', type=str, default='generated_sequences_calibrated.csv', help='Generated results file to evaluate.')

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_pipeline(args)
    elif args.command == 'train':
        train_pipeline(args)
    elif args.command == 'generate':
        generate_pipeline(args)
    elif args.command == 'generate-calibrated':
        generate_calibrated_pipeline(args)
    elif args.command == 'evaluate':
        evaluate_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
