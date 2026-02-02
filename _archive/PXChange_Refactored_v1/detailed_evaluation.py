"""
Detailed evaluation of generated sequences
Includes visualizations and statistical comparisons
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    """Load generated and true data"""
    print("Loading data...")
    generated_df = pd.read_csv('outputs/generated_sequences.csv')

    # Load true data
    true_df = pd.read_csv('data/preprocessed/all_preprocessed.csv')

    return generated_df, true_df


def analyze_sequence_lengths(generated_df, true_df):
    """Analyze and compare sequence lengths"""
    print("\n" + "="*70)
    print("SEQUENCE LENGTH ANALYSIS")
    print("="*70)

    # Generated sequences
    gen_lengths = generated_df.groupby(['SN', 'sample_idx']).size()

    # True sequences
    true_lengths = true_df.groupby('SeqOrder').size()

    print("\nGenerated Sequences:")
    print(f"  Count: {len(gen_lengths)}")
    print(f"  Mean: {gen_lengths.mean():.2f}")
    print(f"  Std: {gen_lengths.std():.2f}")
    print(f"  Min: {gen_lengths.min()}")
    print(f"  Max: {gen_lengths.max()}")
    print(f"  Median: {gen_lengths.median():.0f}")

    print("\nTrue Sequences:")
    print(f"  Count: {len(true_lengths)}")
    print(f"  Mean: {true_lengths.mean():.2f}")
    print(f"  Std: {true_lengths.std():.2f}")
    print(f"  Min: {true_lengths.min()}")
    print(f"  Max: {true_lengths.max()}")
    print(f"  Median: {true_lengths.median():.0f}")

    return gen_lengths, true_lengths


def analyze_token_distribution(generated_df, true_df):
    """Analyze token distribution"""
    print("\n" + "="*70)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("="*70)

    # Get token distributions
    gen_tokens = generated_df['token_id'].value_counts().sort_index()
    true_tokens = true_df[~true_df['sourceID'].isin([0, 11, 14])]['sourceID'].value_counts().sort_index()

    # Normalize to percentages
    gen_pct = (gen_tokens / gen_tokens.sum() * 100)
    true_pct = (true_tokens / true_tokens.sum() * 100)

    # Combine for comparison
    comparison = pd.DataFrame({
        'token_id': gen_pct.index,
        'generated_%': gen_pct.values,
        'true_%': [true_pct.get(tid, 0) for tid in gen_pct.index]
    })

    # Get token names
    token_map = dict(zip(generated_df['token_id'], generated_df['token_name']))
    comparison['token_name'] = comparison['token_id'].map(token_map)

    # Calculate difference
    comparison['diff_%'] = comparison['generated_%'] - comparison['true_%']
    comparison = comparison.sort_values('generated_%', ascending=False)

    print("\nTop 10 Tokens (Generated vs True):")
    print(comparison.head(10).to_string(index=False))

    return comparison


def analyze_mri_msr_104(generated_df, true_df):
    """Detailed analysis of MRI_MSR_104 token"""
    print("\n" + "="*70)
    print("MRI_MSR_104 (Token 12) ANALYSIS")
    print("="*70)

    # Generated
    gen_msr104 = generated_df[generated_df['token_id'] == 12]
    gen_counts = gen_msr104.groupby(['SN', 'sample_idx']).size()

    # True
    true_msr104 = true_df[true_df['sourceID'] == 12]
    true_counts = true_msr104.groupby('SeqOrder').size()

    print("\nGenerated:")
    print(f"  Total occurrences: {len(gen_msr104)}")
    print(f"  Sequences with token: {len(gen_counts)}")
    print(f"  Average per sequence: {gen_counts.mean():.2f}")
    print(f"  Distribution:")
    for i in range(1, 6):
        count = (gen_counts == i).sum()
        pct = count / len(gen_counts) * 100
        print(f"    {i} occurrence(s): {count} sequences ({pct:.1f}%)")

    print("\nTrue:")
    print(f"  Total occurrences: {len(true_msr104)}")
    print(f"  Sequences with token: {len(true_counts)}")
    print(f"  Average per sequence: {true_counts.mean():.2f}")
    print(f"  Distribution:")
    for i in range(1, min(6, true_counts.max() + 1)):
        count = (true_counts == i).sum()
        pct = count / len(true_counts) * 100
        print(f"    {i} occurrence(s): {count} sequences ({pct:.1f}%)")


def analyze_bodygroups(generated_df):
    """Analyze body group transitions"""
    print("\n" + "="*70)
    print("BODY GROUP TRANSITION ANALYSIS")
    print("="*70)

    # Get unique transitions per sequence
    transitions = generated_df.groupby(['SN', 'sample_idx']).first()
    transition_counts = transitions.groupby(['BodyGroup_from_text', 'BodyGroup_to_text']).size().sort_values(ascending=False)

    print("\nTop 15 Body Group Transitions:")
    for i, ((from_bg, to_bg), count) in enumerate(transition_counts.head(15).items(), 1):
        pct = count / len(transitions) * 100
        print(f"  {i:2d}. {from_bg:8s} -> {to_bg:8s}: {count:3d} sequences ({pct:5.2f}%)")


def analyze_durations(generated_df, true_df):
    """Analyze step durations"""
    print("\n" + "="*70)
    print("STEP DURATION ANALYSIS")
    print("="*70)

    # Generated
    gen_durations = generated_df['sampled_duration']

    # True
    true_durations = true_df['step_duration']

    print("\nGenerated Durations:")
    print(f"  Mean: {gen_durations.mean():.2f}s")
    print(f"  Std: {gen_durations.std():.2f}s")
    print(f"  Min: {gen_durations.min():.2f}s")
    print(f"  Max: {gen_durations.max():.2f}s")
    print(f"  Median: {gen_durations.median():.2f}s")

    print("\nTrue Durations:")
    print(f"  Mean: {true_durations.mean():.2f}s")
    print(f"  Std: {true_durations.std():.2f}s")
    print(f"  Min: {true_durations.min():.2f}s")
    print(f"  Max: {true_durations.max():.2f}s")
    print(f"  Median: {true_durations.median():.2f}s")


def main():
    print("="*70)
    print("DETAILED EVALUATION OF GENERATED SEQUENCES")
    print("="*70)

    # Load data
    generated_df, true_df = load_data()

    print(f"\nGenerated data: {len(generated_df)} rows, {generated_df.groupby(['SN', 'sample_idx']).ngroups} sequences")
    print(f"True data: {len(true_df)} rows, {true_df['SeqOrder'].nunique()} sequences")

    # Run analyses
    analyze_sequence_lengths(generated_df, true_df)
    analyze_token_distribution(generated_df, true_df)
    analyze_mri_msr_104(generated_df, true_df)
    analyze_bodygroups(generated_df)
    analyze_durations(generated_df, true_df)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nAll analyses saved to console output.")
    print("You can redirect this to a file using:")
    print("  python detailed_evaluation.py > evaluation_report.txt")


if __name__ == "__main__":
    main()
