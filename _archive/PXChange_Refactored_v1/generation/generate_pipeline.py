"""
Complete generation pipeline combining both models
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import secrets
import string

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEQUENCE_SAMPLING_CONFIG, COUNTS_SAMPLING_CONFIG,
    OUTPUT_DIR, END_TOKEN_ID, PAD_TOKEN_ID, START_TOKEN_ID,
    NUM_BODY_REGIONS, CONDITIONING_FEATURES
)
from preprocessing.sequence_encoder import decode_sequences, sequence_to_text


def generate_patient_id(length=40):
    """
    Generate a unique patient ID as a 40-character hex string.

    Args:
        length: Length of the ID (default: 40)

    Returns:
        40-character hex string
    """
    # Generate random hex string
    return ''.join(secrets.choice(string.hexdigits.lower()) for _ in range(length))


def decode_bodygroup(encoded_value):
    """
    Decode body group integer back to text.

    Args:
        encoded_value: Integer encoding of body group

    Returns:
        Body group name as string
    """
    bodygroup_map = {
        0: 'HEAD',
        1: 'NECK',
        2: 'CHEST',
        3: 'ABDOMEN',
        4: 'PELVIS',
        5: 'SPINE',
        6: 'ARM',
        7: 'LEG',
        8: 'HAND',
        9: 'FOOT',
        10: 'KNEE',
        11: 'UNKNOWN' # Added for completeness
    }
    return bodygroup_map.get(encoded_value, 'UNKNOWN')


def remove_excessive_repetitions(tokens, max_consecutive_repeats=2):
    """
    Remove excessive consecutive repetitions of the same token.

    Args:
        tokens: List or array of token IDs
        max_consecutive_repeats: Maximum allowed consecutive repeats (default: 2)

    Returns:
        Filtered token list
    """
    if len(tokens) == 0:
        return tokens

    filtered = [tokens[0]]
    consecutive_count = 1

    for token in tokens[1:]:
        if token == filtered[-1]:
            consecutive_count += 1
            if consecutive_count <= max_consecutive_repeats:
                filtered.append(token)
        else:
            filtered.append(token)
            consecutive_count = 1

    return filtered


def generate_sequence_pool(
    sequence_model,
    counts_model,
    num_samples_per_category=500,
    remove_repetitions=True,
    device='cpu',
    verbose=True
):
    """
    Generates a large pool of sequences, conditioned only on the body region.
    This creates a dataset for subsequent resampling to match true distributions.

    Args:
        sequence_model: Trained ConditionalSequenceGenerator
        counts_model: Trained ConditionalCountsGenerator
        num_samples_per_category: Number of sequences to generate per body region.
        remove_repetitions: Whether to filter excessive consecutive token repetitions.
        device: torch device
        verbose: Whether to print progress

    Returns:
        results_df: DataFrame with the generated sequence pool.
    """
    sequence_model.eval()
    counts_model.eval()

    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SEQUENCE POOL")
        print(f"{'='*70}\n")
        print(f"Number of categories (body regions): {NUM_BODY_REGIONS}")
        print(f"Samples per category: {num_samples_per_category}")
        print(f"Total sequences to generate: {NUM_BODY_REGIONS * num_samples_per_category}")
        print(f"Remove repetitions: {remove_repetitions}\n")

    # Create a dummy conditioning vector (using zeros).
    # The original conditioning features were found to be weak predictors,
    # so we focus on the 'BodyGroup_from' feature which is explicitly handled.
    conditioning_dim = len(CONDITIONING_FEATURES)
    # The model expects BodyGroup_from to be part of the conditioning features, so we create a dummy one.
    # The actual body group conditioning will be passed separately.
    dummy_conditioning_array = np.zeros(conditioning_dim, dtype=np.float32)

    with torch.no_grad():
        for bodygroup_from_id in tqdm(range(NUM_BODY_REGIONS), desc="Generating for each body region", disable=not verbose):
            
            # Set the body group in the dummy conditioning vector
            # This assumes 'BodyGroup_from' is one of the features. We find its index.
            try:
                bg_index = CONDITIONING_FEATURES.index('BodyGroup_from')
                dummy_conditioning_array[bg_index] = bodygroup_from_id
            except ValueError:
                # If 'BodyGroup_from' is not in the features, we can't set it.
                # This is a fallback, assuming the model handles it separately or doesn't use it.
                pass

            # Repeat conditioning array for batch generation
            conditioning_tensor = torch.tensor(dummy_conditioning_array, dtype=torch.float32, device=device).repeat(num_samples_per_category, 1)

            # Step 1: Generate symbolic sequences
            max_gen_length = min(SEQUENCE_SAMPLING_CONFIG['max_length'], sequence_model.max_seq_len - 1)
            generated_tokens = sequence_model.generate(
                conditioning_tensor,
                max_length=max_gen_length,
                temperature=SEQUENCE_SAMPLING_CONFIG['temperature'],
                top_k=SEQUENCE_SAMPLING_CONFIG['top_k'],
                top_p=SEQUENCE_SAMPLING_CONFIG['top_p']
            )

            # Step 2: For each generated sequence, predict counts
            for sample_idx in range(num_samples_per_category):
                tokens = generated_tokens[sample_idx]
                token_list = tokens.cpu().numpy()
                
                if END_TOKEN_ID in token_list:
                    end_idx = np.where(token_list == END_TOKEN_ID)[0][0] + 1
                else:
                    end_idx = len(token_list)

                if remove_repetitions:
                    token_list_filtered = remove_excessive_repetitions(token_list[:end_idx], max_consecutive_repeats=2)
                else:
                    token_list_filtered = token_list[:end_idx]
                end_idx = len(token_list_filtered)

                if end_idx <= 1: # Skip empty sequences
                    continue

                tokens_trimmed = torch.tensor(token_list_filtered, dtype=torch.long, device=device).unsqueeze(0)
                seq_features = torch.zeros(1, end_idx, 2, device=device)
                mask = torch.ones(1, end_idx, dtype=torch.bool, device=device)

                mu, sigma = counts_model(
                    conditioning_tensor[sample_idx:sample_idx+1],
                    tokens_trimmed,
                    seq_features,
                    mask
                )

                counts_sampled = counts_model.sample_counts(mu, sigma, num_samples=1).squeeze(-1)
                token_strings = decode_sequences(tokens_trimmed[0].cpu().numpy(), remove_special_tokens=False)

                patient_id_from = generate_patient_id()
                patient_id_to = generate_patient_id()

                filtered_step = 0
                for step_idx in range(end_idx):
                    token_id = int(token_list_filtered[step_idx])
                    if token_id in [START_TOKEN_ID, END_TOKEN_ID]:
                        continue
                    
                    token_name = token_strings[step_idx] if step_idx < len(token_strings) else 'PAD'
                    if token_name in ['START', 'END']:
                        continue

                    results.append({
                        'SN': f"pool_{bodygroup_from_id}", # Use a pool identifier as the SN
                        'customer_idx': bodygroup_from_id, # Store body group id
                        'sample_idx': sample_idx,
                        'step': filtered_step,
                        'token_id': token_id,
                        'token_name': token_name,
                        'BodyGroup_from': bodygroup_from_id,
                        'BodyGroup_to': -1, # Placeholder
                        'BodyGroup_from_text': decode_bodygroup(bodygroup_from_id),
                        'BodyGroup_to_text': 'UNKNOWN',
                        'PatientID_from': patient_id_from,
                        'PatientID_to': patient_id_to,
                        'predicted_mu': mu[0, step_idx].item(),
                        'predicted_sigma': sigma[0, step_idx].item(),
                        'sampled_duration': counts_sampled[0, step_idx].item()
                    })
                    filtered_step += 1

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("Warning: No sequences were generated!")
        return results_df

    sequence_totals = results_df.groupby(['SN', 'sample_idx'])['sampled_duration'].sum().reset_index()
    sequence_totals.columns = ['SN', 'sample_idx', 'total_time']
    results_df = results_df.merge(sequence_totals, on=['SN', 'sample_idx'])

    if verbose:
        print(f"\n[OK] Generation complete!")
        print(f"  Total sequences generated in pool: {len(results_df.groupby(['SN', 'sample_idx']))}")

    return results_df


def generate_sequences_and_counts(
    sequence_model,
    counts_model,
    conditioning_data,
    conditioning_scaler=None,
    num_samples_per_customer=15,
    sequences_per_customer=None,
    remove_repetitions=True,
    device='cpu',
    verbose=True
):
    """
    Complete generation pipeline:
    1. Generate symbolic sequences using sequence model (per customer)
    2. Predict counts with uncertainty using counts model
    3. Sample realistic counts from Gamma distributions

    Args:
        sequence_model: Trained ConditionalSequenceGenerator
        counts_model: Trained ConditionalCountsGenerator
        conditioning_data: DataFrame with conditioning features and dataset_id
        conditioning_scaler: Scaler for conditioning features
        num_samples_per_customer: Number of sequences to generate per customer (default: 15)
        sequences_per_customer: Dict mapping dataset_id to number of sequences (overrides num_samples_per_customer if provided)
        remove_repetitions: Whether to filter excessive consecutive token repetitions (default: True)
        device: torch device
        verbose: Whether to print progress

    Returns:
        results_df: DataFrame with generated sequences and counts (with SN column, START/END removed)
    """
    sequence_model.eval()
    counts_model.eval()

    results = []

    # Prepare conditioning data - must be a DataFrame with dataset_id
    if not isinstance(conditioning_data, pd.DataFrame):
        raise ValueError("conditioning_data must be a DataFrame with dataset_id column for per-customer generation")

    if 'dataset_id' not in conditioning_data.columns:
        raise ValueError("conditioning_data must contain 'dataset_id' column to identify customers")

    # Group by dataset_id (customer)
    customers = conditioning_data['dataset_id'].unique()
    num_customers = len(customers)

    # Calculate total sequences to generate
    if sequences_per_customer is not None:
        total_sequences = sum(sequences_per_customer.get(str(c), num_samples_per_customer) for c in customers)
        avg_samples = total_sequences / num_customers
    else:
        total_sequences = num_customers * num_samples_per_customer
        avg_samples = num_samples_per_customer

    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATING SEQUENCES AND COUNTS (PER CUSTOMER)")
        print(f"{'='*70}\n")
        print(f"Number of customers: {num_customers}")
        print(f"Average samples per customer: {avg_samples:.1f}")
        print(f"Total sequences to generate: {total_sequences}")
        print(f"Remove repetitions: {remove_repetitions}\n")

    with torch.no_grad():
        for customer_idx, dataset_id in enumerate(tqdm(customers, desc="Generating per customer", disable=not verbose)):
            # Convert dataset_id to string for consistent lookup
            dataset_id_str = str(dataset_id)

            # Get conditioning data for this customer (use first row as representative)
            customer_data = conditioning_data[conditioning_data['dataset_id'] == dataset_id].iloc[0]
            conditioning_array = np.array([customer_data[feat] for feat in CONDITIONING_FEATURES], dtype=np.float32)

            if conditioning_scaler is not None:
                conditioning_array = conditioning_scaler.transform(conditioning_array.reshape(1, -1))[0]

            # Determine how many sequences to generate for this customer
            if sequences_per_customer is not None:
                customer_num_samples = sequences_per_customer.get(dataset_id_str, num_samples_per_customer)
            else:
                customer_num_samples = num_samples_per_customer

            # Repeat conditioning array for batch generation
            conditioning_tensor = torch.tensor(conditioning_array, dtype=torch.float32, device=device).repeat(customer_num_samples, 1)

            # Step 1: Generate symbolic sequences
            max_gen_length = min(SEQUENCE_SAMPLING_CONFIG['max_length'], sequence_model.max_seq_len - 1)
            generated_tokens = sequence_model.generate(
                conditioning_tensor,
                max_length=max_gen_length,
                temperature=SEQUENCE_SAMPLING_CONFIG['temperature'],
                top_k=SEQUENCE_SAMPLING_CONFIG['top_k'],
                top_p=SEQUENCE_SAMPLING_CONFIG['top_p']
            )  # [customer_num_samples, seq_len]

            # Extract BodyGroup information from conditioning data
            bodygroup_from = int(customer_data['BodyGroup_from'])
            bodygroup_to = int(customer_data['BodyGroup_to'])

            # Step 2: For each generated sequence, predict counts
            for sample_idx in range(customer_num_samples):
                tokens = generated_tokens[sample_idx]
                token_list = tokens.cpu().numpy()
                if END_TOKEN_ID in token_list:
                    end_idx = np.where(token_list == END_TOKEN_ID)[0][0] + 1
                else:
                    end_idx = len(token_list)

                if remove_repetitions:
                    token_list_filtered = remove_excessive_repetitions(token_list[:end_idx], max_consecutive_repeats=2)
                else:
                    token_list_filtered = token_list[:end_idx]
                end_idx = len(token_list_filtered)

                tokens_trimmed = torch.tensor(token_list_filtered, dtype=torch.long, device=device).unsqueeze(0)
                seq_features = torch.zeros(1, end_idx, 2, device=device)
                mask = torch.ones(1, end_idx, dtype=torch.bool, device=device)

                mu, sigma = counts_model(
                    conditioning_tensor[sample_idx:sample_idx+1],
                    tokens_trimmed,
                    seq_features,
                    mask
                )

                counts_sampled = counts_model.sample_counts(
                    mu, sigma, num_samples=1
                ).squeeze(-1)

                token_strings = decode_sequences(tokens_trimmed[0].cpu().numpy(), remove_special_tokens=False)

                patient_id_from = generate_patient_id()
                patient_id_to = generate_patient_id()

                filtered_step = 0
                for step_idx in range(end_idx):
                    token_id = int(token_list_filtered[step_idx])
                    if token_id in [START_TOKEN_ID, END_TOKEN_ID]:
                        continue
                    
                    token_name = token_strings[step_idx] if step_idx < len(token_strings) else 'PAD'
                    if token_name in ['START', 'END']:
                        continue

                    results.append({
                        'SN': dataset_id_str,
                        'customer_idx': customer_idx,
                        'sample_idx': sample_idx,
                        'step': filtered_step,
                        'token_id': token_id,
                        'token_name': token_name,
                        'BodyGroup_from': bodygroup_from,
                        'BodyGroup_to': bodygroup_to,
                        'BodyGroup_from_text': decode_bodygroup(bodygroup_from),
                        'BodyGroup_to_text': decode_bodygroup(bodygroup_to),
                        'PatientID_from': patient_id_from,
                        'PatientID_to': patient_id_to,
                        'predicted_mu': mu[0, step_idx].item(),
                        'predicted_sigma': sigma[0, step_idx].item(),
                        'sampled_duration': counts_sampled[0, step_idx].item()
                    })
                    filtered_step += 1

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Warning: No sequences were generated!")
        return results_df

    sequence_totals = results_df.groupby(['SN', 'sample_idx'])['sampled_duration'].sum().reset_index()
    sequence_totals.columns = ['SN', 'sample_idx', 'total_time']
    results_df = results_df.merge(sequence_totals, on=['SN', 'sample_idx'])

    if verbose:
        print(f"\n[OK] Generation complete!")
        print(f"  Total customers: {results_df['SN'].nunique()}")
        print(f"  Total sequences generated: {len(results_df.groupby(['SN', 'sample_idx']))}")
        print(f"  Average sequence length: {results_df.groupby(['SN', 'sample_idx'])['step'].max().mean():.1f}")
        print(f"  Average total time: {results_df.groupby(['SN', 'sample_idx'])['total_time'].first().mean():.1f}s")
        print(f"  Total time range: [{results_df['total_time'].min():.1f}s, {results_df['total_time'].max():.1f}s]")

    return results_df


def save_generated_results(results_df, filename='generated_sequences.csv'):
    """
    Save generated results to CSV.
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to {output_path}")
    return output_path


def print_generation_examples(results_df, num_examples=3):
    """
    Print examples of generated sequences.
    """
    print(f"\n{'='*70}")
    print(f"GENERATION EXAMPLES")
    print(f"{'='*70}\n")

    unique_samples = results_df.groupby(['SN', 'sample_idx']).groups.keys()
    examples = list(unique_samples)[:num_examples]

    for i, (sn, sample_idx) in enumerate(examples, 1):
        sample_data = results_df[
            (results_df['SN'] == sn) &
            (results_df['sample_idx'] == sample_idx)
        ]

        token_sequence = sample_data['token_name'].tolist()
        durations = sample_data['sampled_duration'].values
        total_time = sample_data['total_time'].iloc[0]

        print(f"Example {i} (Customer SN={sn}, Sample={sample_idx}):")
        print(f"  Length: {len(token_sequence)} steps")
        print(f"  Sequence: {' -> '.join(token_sequence)}")
        print(f"  Durations (first 10): {durations[:10].round(1)}")
        print(f"  Total time: {total_time:.1f}s\n")


if __name__ == "__main__":
    # Test generation pipeline
    print("Testing generation pipeline...")

    import torch
    from models import ConditionalSequenceGenerator, ConditionalCountsGenerator

    # Create dummy models
    seq_config = {
        'vocab_size': 18,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 64,
        'conditioning_dim': 6
    }

    counts_config = {
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_cross_attention_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 64,
        'conditioning_dim': 6,
        'sequence_feature_dim': 18,
        'min_sigma': 0.1
    }

    sequence_model = ConditionalSequenceGenerator(seq_config)
    counts_model = ConditionalCountsGenerator(counts_config)

    # Create dummy conditioning data
    conditioning_data = np.random.randn(3, 6)

    # Generate
    results_df = generate_sequences_and_counts(
        sequence_model,
        counts_model,
        conditioning_data,
        num_samples=5,
        verbose=True
    )

    # Print examples
    print_generation_examples(results_df)

    print("\n✓ Generation pipeline test complete!")