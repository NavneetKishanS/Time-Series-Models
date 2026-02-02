"""
Performs a detailed error analysis of the generated sequences.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def decode_bodygroup(encoded_value):
    """Decodes body group ID to text."""
    bodygroup_map = {
        0: 'HEAD', 1: 'NECK', 2: 'CHEST', 3: 'ABDOMEN', 4: 'PELVIS',
        5: 'SPINE', 6: 'LOWER_EXTREMITY', 7: 'UPPER_EXTREMITY',
        8: 'SHOULDER', 9: 'UNKNOWN_REGION', 10: 'UNKNOWN'
    }
    return bodygroup_map.get(encoded_value, 'UNKNOWN')

def main():
    # Load data
    print("Loading data...")
    generated_df = pd.read_csv('outputs/generated_sequences.csv')
    true_df = pd.read_csv('data/preprocessed/all_preprocessed.csv')
    print("Data loaded successfully.")

    generated_df['BodyGroup_from_text'] = generated_df['BodyGroup_from'].apply(decode_bodygroup)
    true_df['BodyGroup_from_text'] = true_df['BodyGroup_from'].apply(decode_bodygroup)

    # Get the list of body regions
    body_regions = true_df['BodyGroup_from_text'].unique()

    # Create a directory for the plots
    import os
    output_dir = 'visualizations/error_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # For each body region, compare the token distributions
    for region in body_regions:
        print(f"\nAnalyzing body region: {region}")

        true_region_df = true_df[true_df['BodyGroup_from_text'] == region]
        generated_region_df = generated_df[generated_df['BodyGroup_from_text'] == region]

        if len(generated_region_df) == 0:
            print(f"No generated sequences for {region}")
            continue

        true_token_counts = true_region_df['sourceID'].value_counts(normalize=True)
        generated_token_counts = generated_region_df['token_id'].value_counts(normalize=True)

        comparison_df = pd.DataFrame({
            'true_%': true_token_counts,
            'generated_%': generated_token_counts
        }).fillna(0)

        # Plot the comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title(f'Token Distribution for Body Region: {region}')
        ax.set_ylabel('Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'token_distribution_{region}.png'))
        plt.close()

        print(f"Token distribution plot for {region} saved to {output_dir}")

if __name__ == "__main__":
    main()
