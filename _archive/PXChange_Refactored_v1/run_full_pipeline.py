"""
Complete Pipeline Runner - Retrains models and generates sequences matching input volume

This script runs the complete pipeline:
1. Preprocesses raw CSV files (if needed)
2. Trains both sequence and counts models
3. Generates sequences matching the input data volume for each customer

Usage:
    # Run everything (preprocess + train + generate)
    python run_full_pipeline.py --all

    # Just train models (skip preprocessing)
    python run_full_pipeline.py --train

    # Just generate (skip preprocessing and training)
    python run_full_pipeline.py --generate

    # Train and generate (skip preprocessing)
    python run_full_pipeline.py --train --generate
"""
import subprocess
import sys
import os
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False

    print(f"\n[OK] {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete PXChange pipeline")
    parser.add_argument('--all', action='store_true',
                       help='Run all steps (preprocess, train, generate)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing step')
    parser.add_argument('--train', action='store_true',
                       help='Run training step')
    parser.add_argument('--generate', action='store_true',
                       help='Run generation step')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for generation (default: CPU)')
    parser.add_argument('--match-input-volume', action='store_true', default=True,
                       help='Generate same number of sequences as input (default: True)')
    parser.add_argument('--keep-repetitions', action='store_true',
                       help='Keep consecutive token repetitions in generated data')
    parser.add_argument('--output-file', type=str, default='generated_sequences.csv',
                       help='Output filename for generated sequences')

    args = parser.parse_args()

    # If no flags specified, show help
    if not any([args.all, args.preprocess, args.train, args.generate]):
        parser.print_help()
        print("\nExample: python run_full_pipeline.py --all")
        sys.exit(0)

    # Determine which steps to run
    run_preprocess = args.all or args.preprocess
    run_train = args.all or args.train
    run_generate = args.all or args.generate

    print("="*80)
    print("PXCHANGE COMPLETE PIPELINE")
    print("="*80)
    print(f"\nSteps to run:")
    print(f"  Preprocess: {run_preprocess}")
    print(f"  Train: {run_train}")
    print(f"  Generate: {run_generate}")
    print()

    # Step 1: Preprocessing
    if run_preprocess:
        success = run_command(
            "python main_pipeline.py preprocess",
            "Step 1: Preprocessing raw CSV files"
        )
        if not success:
            print("\n[ERROR] Preprocessing failed. Aborting pipeline.")
            sys.exit(1)

    # Step 2: Training
    if run_train:
        # Check if preprocessing was done
        preprocessed_dir = os.path.join("data", "preprocessed")
        if not os.path.exists(preprocessed_dir):
            print("\n[ERROR] Preprocessed data not found. Run preprocessing first!")
            print("        Use: python run_full_pipeline.py --preprocess")
            sys.exit(1)

        train_cmd = f"python main_pipeline.py train --batch-size {args.batch_size}"

        success = run_command(
            train_cmd,
            "Step 2: Training sequence and counts models"
        )
        if not success:
            print("\n[ERROR] Training failed. Aborting pipeline.")
            sys.exit(1)

    # Step 3: Generation
    if run_generate:
        # Check if models exist
        model_dir = "saved_models"
        seq_model = os.path.join(model_dir, "sequence_model_best.pt")
        counts_model = os.path.join(model_dir, "counts_model_best.pt")

        if not os.path.exists(seq_model) or not os.path.exists(counts_model):
            print("\n[ERROR] Trained models not found. Run training first!")
            print("        Use: python run_full_pipeline.py --train")
            sys.exit(1)

        # Build generation command
        gen_cmd = "python main_pipeline.py generate"

        if args.match_input_volume:
            gen_cmd += " --match-input-volume"

        if args.keep_repetitions:
            gen_cmd += " --keep-repetitions"

        if args.use_gpu:
            gen_cmd += " --use-gpu"

        gen_cmd += f" --output-file {args.output_file}"

        success = run_command(
            gen_cmd,
            "Step 3: Generating sequences (matching input volume)"
        )
        if not success:
            print("\n[ERROR] Generation failed.")
            sys.exit(1)

        # Print summary
        print(f"\n{'='*80}")
        print("GENERATION SUMMARY")
        print(f"{'='*80}\n")

        import pandas as pd
        output_path = os.path.join("outputs", args.output_file)
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"Output file: {output_path}")
            print(f"Total rows generated: {len(df):,}")
            print(f"Total customers: {df['SN'].nunique()}")
            print(f"Total sequences: {len(df.groupby(['SN', 'sample_idx']))}")
            print(f"Average sequence length: {df.groupby(['SN', 'sample_idx']).size().mean():.1f}")

            # Compare with input
            preprocessed_path = os.path.join("data", "preprocessed", "all_preprocessed.csv")
            if os.path.exists(preprocessed_path):
                input_df = pd.read_csv(preprocessed_path)
                # Remove START and END tokens from input for fair comparison
                input_df_filtered = input_df[~input_df['sourceID'].isin([11, 14])]
                input_rows = len(input_df_filtered)
                print(f"\nInput data rows (excluding START/END): {input_rows:,}")
                print(f"Generated data rows: {len(df):,}")
                print(f"Ratio (generated/input): {len(df)/input_rows:.2f}x")

    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")

    if run_generate:
        print(f"Generated sequences saved to: outputs/{args.output_file}")
        print("\nYou can now evaluate the results with:")
        print(f"    python main_pipeline.py evaluate --generated-file {args.output_file}")

    print()


if __name__ == "__main__":
    main()
