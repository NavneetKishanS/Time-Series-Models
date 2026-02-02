"""
Resamples a pool of generated sequences to match the true distribution of a
categorical feature, such as 'BodyGroup_from'.

This script is used for post-processing calibration to correct distributional bias
in the generated output.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, DATA_DIR

def resample_sequences(
    generated_pool_path,
    true_data_path,
    output_path,
    feature_column='BodyGroup_from',
    generated_id_cols=['SN', 'sample_idx']
):
    """
    Resamples a generated sequence pool to match the true data distribution.

    Args:
        generated_pool_path (str): Path to the CSV of generated sequences (the pool).
        true_data_path (str): Path to the true data CSV to derive the target distribution.
        output_path (str): Path to save the resampled CSV file.
        feature_column (str): The categorical column to match the distribution of.
        generated_id_cols (list): Columns that uniquely identify a sequence in the generated pool.
    """
    print("--- Starting Post-Processing Resampling ---")
    
    # 1. Load data
    print(f"Loading generated pool from: {generated_pool_path}")
    generated_df = pd.read_csv(generated_pool_path)
    if generated_df.empty:
        print("Warning: Generated pool is empty. No resampling will be performed.")
        return

    print(f"Loading true data from: {true_data_path}")
    true_df = pd.read_csv(true_data_path)
    if true_df.empty:
        print("Warning: True data is empty. Cannot determine target distribution.")
        return

    print(f"Feature to match: '{feature_column}'")

    # 2. Calculate true distribution
    # We care about the distribution of sequences, so we first need to get unique sequences from true data
    true_sequences = true_df.drop_duplicates(subset=['dataset_id', 'SeqOrder'])
    true_distribution = true_sequences[feature_column].value_counts(normalize=True)
    total_true_sequences = len(true_sequences)
    
    print("\nTarget Distribution (from true data):")
    print(true_distribution)
    print(f"Total true sequences: {total_true_sequences}")

    # 3. Prepare generated pool for sampling
    # Get a DataFrame of unique sequences from the generated pool
    generated_sequences = generated_df.drop_duplicates(subset=generated_id_cols)
    print("\nGenerated Pool Overview:")
    print(f"Total generated sequences available in the pool: {len(generated_sequences)}")
    print("Sequences per category in the pool:")
    print(generated_sequences[feature_column].value_counts())

    # 4. Perform stratified sampling
    resampled_indices = []
    for category, proportion in true_distribution.items():
        num_to_sample = int(round(proportion * total_true_sequences))
        
        # Get all available sequences for this category from the pool
        category_sequences = generated_sequences[generated_sequences[feature_column] == category]
        
        if len(category_sequences) == 0:
            print(f"Warning: No sequences generated for category '{category}'. Cannot sample.")
            continue
            
        # Sample with replacement if we need more than we have
        should_replace = num_to_sample > len(category_sequences)
        if should_replace:
            print(f"Warning: Sampling with replacement for category '{category}' (need {num_to_sample}, have {len(category_sequences)}).")

        sampled_subset = category_sequences.sample(n=num_to_sample, replace=should_replace, random_state=RANDOM_SEED)
        resampled_indices.append(sampled_subset)
        print(f"  - Sampled {num_to_sample} sequences for category '{category}'")

    if not resampled_indices:
        print("Error: No sequences were resampled. Exiting.")
        return
        
    resampled_sequences_df = pd.concat(resampled_indices)
    
    # 5. Filter the original full generated_df to only include the resampled sequences
    resampled_full_df = pd.merge(resampled_sequences_df[generated_id_cols], generated_df, on=generated_id_cols, how='left')

    # 6. Save the final resampled dataframe
    print(f"\nTotal sequences in final calibrated set: {len(resampled_sequences_df)}")
    print("Final distribution:")
    print(resampled_sequences_df[feature_column].value_counts(normalize=True))
    
    resampled_full_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Calibrated sequences saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Resample generated sequences to match true data distribution.")
    parser.add_argument(
        '--generated-pool',
        type=str,
        default=os.path.join(OUTPUT_DIR, 'generated_sequences_pool.csv'),
        help='Path to the generated sequences pool CSV file.'
    )
    parser.add_argument(
        '--true-data',
        type=str,
        default=os.path.join(DATA_DIR, 'preprocessed', 'all_preprocessed.csv'),
        help='Path to the true data CSV file.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=os.path.join(OUTPUT_DIR, 'generated_sequences_calibrated.csv'),
        help='Path to save the final calibrated CSV file.'
    )
    parser.add_argument(
        '--feature',
        type=str,
        default='BodyGroup_from',
        help="The feature column to align distributions on."
    )
    
    args = parser.parse_args()

    resample_sequences(
        generated_pool_path=args.generated_pool,
        true_data_path=args.true_data,
        output_path=args.output_file,
        feature_column=args.feature
    )

if __name__ == '__main__':
    # Set a seed for reproducibility of sampling
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    main()
