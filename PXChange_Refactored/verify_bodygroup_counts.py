"""
Compares the distribution of body groups in generated vs. true data.
"""
import pandas as pd

def decode_bodygroup(encoded_value):
    """Decodes body group ID to text."""
    bodygroup_map = {
        0: 'HEAD',
        1: 'NECK',
        2: 'CHEST',
        3: 'ABDOMEN',
        4: 'PELVIS',
        5: 'SPINE',
        6: 'LOWER_EXTREMITY',
        7: 'UPPER_EXTREMITY',
        8: 'SHOULDER',
        9: 'UNKNOWN_REGION',
        10: 'UNKNOWN'
    }
    return bodygroup_map.get(encoded_value, 'UNKNOWN')

def main():
    # Load data
    print("Loading data...")
    generated_df = pd.read_csv('outputs/generated_sequences.csv')
    true_df = pd.read_csv('data/preprocessed/all_preprocessed.csv')
    print("Data loaded successfully.")

    # Get unique sequences from true_df
    true_sequences = true_df.drop_duplicates(subset=['SeqOrder'])

    # Get counts for generated data
    generated_counts = generated_df.groupby('BodyGroup_from').size().reset_index(name='generated_count')
    generated_counts['BodyGroup_from_text'] = generated_counts['BodyGroup_from'].apply(decode_bodygroup)
    generated_total = generated_counts['generated_count'].sum()
    generated_counts['generated_%'] = (generated_counts['generated_count'] / generated_total * 100).round(2)

    # Get counts for true data
    true_counts = true_sequences.groupby('BodyGroup_from').size().reset_index(name='true_count')
    true_counts['BodyGroup_from_text'] = true_counts['BodyGroup_from'].apply(decode_bodygroup)
    true_total = true_counts['true_count'].sum()
    true_counts['true_%'] = (true_counts['true_count'] / true_total * 100).round(2)

    # Merge for comparison
    comparison_df = pd.merge(
        generated_counts[['BodyGroup_from_text', 'generated_count', 'generated_%']],
        true_counts[['BodyGroup_from_text', 'true_count', 'true_%']],
        on='BodyGroup_from_text',
        how='outer'
    ).fillna(0)

    # Calculate difference
    comparison_df['difference'] = comparison_df['generated_count'] - comparison_df['true_count']
    comparison_df['difference_%'] = comparison_df['generated_%'] - comparison_df['true_%']


    print("\n" + "="*80)
    print("EXAMINATION EVENT COUNTS PER BODY REGION: GENERATED vs. TRUE")
    print("="*80)
    try:
        from tabulate import tabulate
        headers = ["Body Region", "Generated Count", "Generated %", "True Count", "True %", "Difference", "Difference %"]
        print(tabulate(comparison_df, headers=headers, tablefmt="grid"))
    except ImportError:
        print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()
