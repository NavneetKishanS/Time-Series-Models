"""
Comprehensive evaluation of generated sequences with visualizations
Creates plots comparing generated vs true data and saves to visualizations folder
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_DIR, f"evaluation_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)


def load_data():
    """Load generated and true data"""
    print("Loading data...")
    generated_df = pd.read_csv('outputs/generated_sequences.csv')
    true_df = pd.read_csv('data/preprocessed/all_preprocessed.csv')

    print(f"  Generated: {len(generated_df)} rows, {generated_df.groupby(['SN', 'sample_idx']).ngroups} sequences")
    print(f"  True: {len(true_df)} rows, {true_df['SeqOrder'].nunique()} sequences")

    return generated_df, true_df


def plot_sequence_lengths(generated_df, true_df):
    """Plot sequence length distributions"""
    print("\nCreating sequence length visualization...")

    # Calculate lengths
    gen_lengths = generated_df.groupby(['SN', 'sample_idx']).size()
    true_lengths = true_df.groupby('SeqOrder').size()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram comparison
    ax = axes[0, 0]
    bins = np.linspace(0, max(gen_lengths.max(), true_lengths.max()), 30)
    ax.hist(true_lengths, bins=bins, alpha=0.5, label='True', color='blue', density=True)
    ax.hist(gen_lengths, bins=bins, alpha=0.5, label='Generated', color='red', density=True)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Density')
    ax.set_title('Sequence Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[0, 1]
    data_to_plot = [true_lengths, gen_lengths]
    bp = ax.boxplot(data_to_plot, labels=['True', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Sequence Length Box Plot')
    ax.grid(True, alpha=0.3)

    # Statistics text
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = f"""
    SEQUENCE LENGTH STATISTICS

    True Sequences:
      Count: {len(true_lengths):,}
      Mean:  {true_lengths.mean():.2f}
      Std:   {true_lengths.std():.2f}
      Min:   {true_lengths.min()}
      Max:   {true_lengths.max()}
      Median: {true_lengths.median():.0f}

    Generated Sequences:
      Count: {len(gen_lengths):,}
      Mean:  {gen_lengths.mean():.2f}
      Std:   {gen_lengths.std():.2f}
      Min:   {gen_lengths.min()}
      Max:   {gen_lengths.max()}
      Median: {gen_lengths.median():.0f}

    Difference (Gen - True):
      Mean:  {gen_lengths.mean() - true_lengths.mean():.2f}
      Std:   {gen_lengths.std() - true_lengths.std():.2f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center')

    # CDF comparison
    ax = axes[1, 1]
    true_sorted = np.sort(true_lengths)
    gen_sorted = np.sort(gen_lengths)
    true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)
    gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
    ax.plot(true_sorted, true_cdf, label='True', color='blue', linewidth=2)
    ax.plot(gen_sorted, gen_cdf, label='Generated', color='red', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'sequence_lengths_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return gen_lengths, true_lengths


def plot_token_distribution(generated_df, true_df):
    """Plot token distribution comparison"""
    print("\nCreating token distribution visualization...")

    # Get token distributions
    gen_tokens = generated_df['token_id'].value_counts().sort_index()
    true_tokens = true_df[~true_df['sourceID'].isin([0, 11, 14])]['sourceID'].value_counts().sort_index()

    # Normalize to percentages
    gen_pct = (gen_tokens / gen_tokens.sum() * 100)
    true_pct = (true_tokens / true_tokens.sum() * 100)

    # Get all unique tokens
    all_tokens = sorted(set(gen_pct.index) | set(true_pct.index))

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'token_id': all_tokens,
        'generated_%': [gen_pct.get(tid, 0) for tid in all_tokens],
        'true_%': [true_pct.get(tid, 0) for tid in all_tokens]
    })

    # Get token names
    token_map = dict(zip(generated_df['token_id'], generated_df['token_name']))
    comparison['token_name'] = comparison['token_id'].map(token_map)
    comparison['token_label'] = comparison['token_id'].astype(str) + '\n' + comparison['token_name'].fillna('Unknown')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Bar chart comparison (top tokens)
    ax = axes[0, 0]
    top_tokens = comparison.nlargest(15, 'generated_%')
    x = np.arange(len(top_tokens))
    width = 0.35
    ax.bar(x - width/2, top_tokens['true_%'], width, label='True', color='blue', alpha=0.7)
    ax.bar(x + width/2, top_tokens['generated_%'], width, label='Generated', color='red', alpha=0.7)
    ax.set_xlabel('Token')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Top 15 Tokens Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(top_tokens['token_id'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Scatter plot: Generated vs True
    ax = axes[0, 1]
    ax.scatter(comparison['true_%'], comparison['generated_%'], alpha=0.6, s=100)
    max_val = max(comparison['true_%'].max(), comparison['generated_%'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')
    ax.set_xlabel('True Percentage (%)')
    ax.set_ylabel('Generated Percentage (%)')
    ax.set_title('Token Distribution Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference plot
    ax = axes[1, 0]
    comparison['diff'] = comparison['generated_%'] - comparison['true_%']
    top_diff = comparison.nlargest(10, 'diff', keep='all')
    colors = ['green' if d > 0 else 'red' for d in top_diff['diff']]
    ax.barh(range(len(top_diff)), top_diff['diff'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_diff)))
    ax.set_yticklabels([f"Token {tid}" for tid in top_diff['token_id']])
    ax.set_xlabel('Difference (Generated - True) %')
    ax.set_title('Top 10 Tokens with Largest Differences')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    top_10 = comparison.nlargest(10, 'generated_%')
    table_text = "TOP 10 TOKENS\n\n"
    table_text += f"{'ID':<4} {'Name':<20} {'True%':>8} {'Gen%':>8} {'Diff':>8}\n"
    table_text += "-" * 60 + "\n"
    for _, row in top_10.iterrows():
        name = row['token_name'][:18] if pd.notna(row['token_name']) else 'Unknown'
        table_text += f"{int(row['token_id']):<4} {name:<20} {row['true_%']:>7.2f}% {row['generated_%']:>7.2f}% {row['diff']:>7.2f}%\n"

    ax.text(0.1, 0.9, table_text, fontsize=9, family='monospace', va='top')

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'token_distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return comparison


def plot_mri_msr_104_analysis(generated_df, true_df):
    """Detailed analysis of MRI_MSR_104 token"""
    print("\nCreating MRI_MSR_104 visualization...")

    # Generated
    gen_msr104 = generated_df[generated_df['token_id'] == 12]
    gen_counts = gen_msr104.groupby(['SN', 'sample_idx']).size()
    gen_total_seqs = generated_df.groupby(['SN', 'sample_idx']).ngroups

    # True
    true_msr104 = true_df[true_df['sourceID'] == 12]
    true_counts = true_msr104.groupby('SeqOrder').size()
    true_total_seqs = true_df['SeqOrder'].nunique()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution of counts per sequence
    ax = axes[0, 0]
    max_count = max(gen_counts.max(), true_counts.max()) if len(true_counts) > 0 else gen_counts.max()
    bins = np.arange(0, max_count + 2) - 0.5
    ax.hist(true_counts, bins=bins, alpha=0.5, label='True', color='blue', density=True)
    ax.hist(gen_counts, bins=bins, alpha=0.5, label='Generated', color='red', density=True)
    ax.set_xlabel('Number of MRI_MSR_104 occurrences per sequence')
    ax.set_ylabel('Density')
    ax.set_title('MRI_MSR_104 Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar chart comparison
    ax = axes[0, 1]
    max_display = 5
    true_dist = [(true_counts == i).sum() / true_total_seqs * 100 for i in range(max_display + 1)]
    gen_dist = [(gen_counts == i).sum() / gen_total_seqs * 100 for i in range(max_display + 1)]

    x = np.arange(max_display + 1)
    width = 0.35
    ax.bar(x - width/2, true_dist, width, label='True', color='blue', alpha=0.7)
    ax.bar(x + width/2, gen_dist, width, label='Generated', color='red', alpha=0.7)
    ax.set_xlabel('Number of occurrences')
    ax.set_ylabel('Percentage of sequences (%)')
    ax.set_title('MRI_MSR_104 Occurrence Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(max_display)] + [f'{max_display}+'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Statistics
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = f"""
    MRI_MSR_104 (Token 12) STATISTICS

    Generated:
      Total occurrences:     {len(gen_msr104):,}
      Sequences with token:  {len(gen_counts):,}
      Total sequences:       {gen_total_seqs:,}
      Presence rate:         {len(gen_counts)/gen_total_seqs*100:.1f}%
      Avg per sequence:      {gen_counts.mean():.2f}
      Median per sequence:   {gen_counts.median():.1f}

    True:
      Total occurrences:     {len(true_msr104):,}
      Sequences with token:  {len(true_counts):,}
      Total sequences:       {true_total_seqs:,}
      Presence rate:         {len(true_counts)/true_total_seqs*100:.1f}%
      Avg per sequence:      {true_counts.mean():.2f}
      Median per sequence:   {true_counts.median():.1f}

    Comparison:
      Occurrence difference: {len(gen_msr104) - len(true_msr104):+,}
      Presence rate diff:    {(len(gen_counts)/gen_total_seqs - len(true_counts)/true_total_seqs)*100:+.1f}%
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    # Detailed distribution table
    ax = axes[1, 1]
    ax.axis('off')
    table_text = "DETAILED DISTRIBUTION\n\n"
    table_text += f"{'Count':<8} {'True':<15} {'Generated':<15}\n"
    table_text += "-" * 50 + "\n"
    for i in range(min(8, max_count + 1)):
        true_cnt = (true_counts == i).sum()
        gen_cnt = (gen_counts == i).sum()
        true_pct = true_cnt / true_total_seqs * 100 if true_total_seqs > 0 else 0
        gen_pct = gen_cnt / gen_total_seqs * 100
        table_text += f"{i:<8} {true_cnt:5d} ({true_pct:5.1f}%) {gen_cnt:5d} ({gen_pct:5.1f}%)\n"

    ax.text(0.1, 0.9, table_text, fontsize=10, family='monospace', va='top')

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'mri_msr_104_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_duration_analysis(generated_df, true_df):
    """Plot step duration analysis"""
    print("\nCreating duration visualization...")

    gen_durations = generated_df['sampled_duration']
    true_durations = true_df['step_duration']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram
    ax = axes[0, 0]
    max_dur = min(max(gen_durations.quantile(0.99), true_durations.quantile(0.99)), 300)
    bins = np.linspace(0, max_dur, 50)
    ax.hist(true_durations[true_durations <= max_dur], bins=bins, alpha=0.5, label='True', color='blue', density=True)
    ax.hist(gen_durations[gen_durations <= max_dur], bins=bins, alpha=0.5, label='Generated', color='red', density=True)
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Density')
    ax.set_title(f'Step Duration Distribution (up to {max_dur:.0f}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[0, 1]
    # Cap at 99th percentile for better visualization
    true_dur_capped = true_durations[true_durations <= true_durations.quantile(0.99)]
    gen_dur_capped = gen_durations[gen_durations <= gen_durations.quantile(0.99)]
    bp = ax.boxplot([true_dur_capped, gen_dur_capped], labels=['True', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Step Duration Box Plot (99th percentile)')
    ax.grid(True, alpha=0.3)

    # Statistics
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = f"""
    STEP DURATION STATISTICS

    True Durations:
      Count:  {len(true_durations):,}
      Mean:   {true_durations.mean():.2f}s
      Std:    {true_durations.std():.2f}s
      Min:    {true_durations.min():.2f}s
      Max:    {true_durations.max():.2f}s
      Median: {true_durations.median():.2f}s
      25th %: {true_durations.quantile(0.25):.2f}s
      75th %: {true_durations.quantile(0.75):.2f}s
      99th %: {true_durations.quantile(0.99):.2f}s

    Generated Durations:
      Count:  {len(gen_durations):,}
      Mean:   {gen_durations.mean():.2f}s
      Std:    {gen_durations.std():.2f}s
      Min:    {gen_durations.min():.2f}s
      Max:    {gen_durations.max():.2f}s
      Median: {gen_durations.median():.2f}s
      25th %: {gen_durations.quantile(0.25):.2f}s
      75th %: {gen_durations.quantile(0.75):.2f}s
      99th %: {gen_durations.quantile(0.99):.2f}s
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    # CDF comparison
    ax = axes[1, 1]
    true_sorted = np.sort(true_durations[true_durations <= max_dur])
    gen_sorted = np.sort(gen_durations[gen_durations <= max_dur])
    true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_durations)
    gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_durations)
    ax.plot(true_sorted, true_cdf, label='True', color='blue', linewidth=2)
    ax.plot(gen_sorted, gen_cdf, label='Generated', color='red', linewidth=2)
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Duration CDF (up to {max_dur:.0f}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'duration_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_total_time_analysis(generated_df, true_df):
    """Plot total sequence time analysis"""
    print("\nCreating total time visualization...")

    # Generated total times
    gen_total_times = generated_df.groupby(['SN', 'sample_idx'])['total_time'].first()

    # True total times
    if 'true_total_time' in true_df.columns:
        true_total_times = true_df.groupby('SeqOrder')['true_total_time'].first()
    else:
        true_total_times = true_df.groupby('SeqOrder')['step_duration'].sum()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram
    ax = axes[0, 0]
    max_time = max(gen_total_times.quantile(0.99), true_total_times.quantile(0.99))
    bins = np.linspace(0, max_time, 40)
    ax.hist(true_total_times, bins=bins, alpha=0.5, label='True', color='blue', density=True)
    ax.hist(gen_total_times, bins=bins, alpha=0.5, label='Generated', color='red', density=True)
    ax.set_xlabel('Total Time (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('Total Sequence Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[0, 1]
    bp = ax.boxplot([true_total_times, gen_total_times], labels=['True', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Total Sequence Time Box Plot')
    ax.grid(True, alpha=0.3)

    # Statistics
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = f"""
    TOTAL SEQUENCE TIME STATISTICS

    True Sequences:
      Count:  {len(true_total_times):,}
      Mean:   {true_total_times.mean():.2f}s
      Std:    {true_total_times.std():.2f}s
      Min:    {true_total_times.min():.2f}s
      Max:    {true_total_times.max():.2f}s
      Median: {true_total_times.median():.2f}s
      25th %: {true_total_times.quantile(0.25):.2f}s
      75th %: {true_total_times.quantile(0.75):.2f}s

    Generated Sequences:
      Count:  {len(gen_total_times):,}
      Mean:   {gen_total_times.mean():.2f}s
      Std:    {gen_total_times.std():.2f}s
      Min:    {gen_total_times.min():.2f}s
      Max:    {gen_total_times.max():.2f}s
      Median: {gen_total_times.median():.2f}s
      25th %: {gen_total_times.quantile(0.25):.2f}s
      75th %: {gen_total_times.quantile(0.75):.2f}s

    Difference (Gen - True):
      Mean:   {gen_total_times.mean() - true_total_times.mean():.2f}s
      Median: {gen_total_times.median() - true_total_times.median():.2f}s
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')

    # CDF
    ax = axes[1, 1]
    true_sorted = np.sort(true_total_times)
    gen_sorted = np.sort(gen_total_times)
    true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)
    gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
    ax.plot(true_sorted, true_cdf, label='True', color='blue', linewidth=2)
    ax.plot(gen_sorted, gen_cdf, label='Generated', color='red', linewidth=2)
    ax.set_xlabel('Total Time (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Total Time CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'total_time_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_bodygroup_transitions(generated_df):
    """Plot body group transition analysis"""
    print("\nCreating body group transition visualization...")

    # Get transitions per sequence
    transitions = generated_df.groupby(['SN', 'sample_idx']).first()
    transition_counts = transitions.groupby(['BodyGroup_from_text', 'BodyGroup_to_text']).size().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top transitions bar chart
    ax = axes[0]
    top_15 = transition_counts.head(15)
    labels = [f"{from_bg} → {to_bg}" for (from_bg, to_bg) in top_15.index]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
    ax.barh(range(len(top_15)), top_15.values, color=colors)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Sequences')
    ax.set_title('Top 15 Body Group Transitions')
    ax.grid(True, alpha=0.3, axis='x')

    # Percentage breakdown
    ax = axes[1]
    ax.axis('off')
    table_text = "BODY GROUP TRANSITION BREAKDOWN\n\n"
    table_text += f"{'#':<4} {'From':<10} {'To':<10} {'Count':<8} {'%':<8}\n"
    table_text += "-" * 50 + "\n"
    total = len(transitions)
    for i, ((from_bg, to_bg), count) in enumerate(transition_counts.head(20).items(), 1):
        pct = count / total * 100
        from_short = from_bg[:8]
        to_short = to_bg[:8]
        table_text += f"{i:<4} {from_short:<10} {to_short:<10} {count:<8} {pct:>6.2f}%\n"

    ax.text(0.1, 0.95, table_text, fontsize=9, family='monospace', va='top')

    plt.tight_layout()
    output_path = os.path.join(RUN_DIR, 'bodygroup_transitions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_summary_report(generated_df, true_df, gen_lengths, true_lengths, token_comparison):
    """Create a comprehensive summary report"""
    print("\nCreating summary report...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"""
    GENERATED SEQUENCES EVALUATION REPORT
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    ax_title.text(0.5, 0.5, title_text, fontsize=16, ha='center', va='center', weight='bold')

    # Overall statistics
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis('off')
    gen_total_seqs = generated_df.groupby(['SN', 'sample_idx']).ngroups
    true_total_seqs = true_df['SeqOrder'].nunique()
    gen_customers = generated_df['SN'].nunique()

    overall_text = f"""
    OVERALL STATISTICS

    Dataset Information:
      Generated Customers:  {gen_customers:,}
      Generated Sequences:  {gen_total_seqs:,}
      True Sequences:       {true_total_seqs:,}
      Generated Steps:      {len(generated_df):,}
      True Steps:           {len(true_df):,}

    Sequence Length:
      True Mean:            {true_lengths.mean():.2f}
      Generated Mean:       {gen_lengths.mean():.2f}
      Difference:           {gen_lengths.mean() - true_lengths.mean():+.2f}

    Unique Tokens:
      True:                 {true_df['sourceID'].nunique()}
      Generated:            {generated_df['token_id'].nunique()}
    """
    ax1.text(0.1, 0.9, overall_text, fontsize=10, family='monospace', va='top')

    # Key findings
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis('off')

    # Calculate key metrics
    top_token_diff = token_comparison.nlargest(1, 'diff').iloc[0]
    mri_104_gen = (generated_df['token_id'] == 12).sum()
    mri_104_true = (true_df['sourceID'] == 12).sum()

    findings_text = f"""
    KEY FINDINGS

    Token Distribution:
      Most over-represented: Token {int(top_token_diff['token_id'])}
        ({top_token_diff['diff']:.2f}% higher)

    MRI_MSR_104 (Token 12):
      True occurrences:     {mri_104_true:,}
      Generated:            {mri_104_gen:,}
      Difference:           {mri_104_gen - mri_104_true:+,}

    Data Quality:
      Sequence length match: {'Good' if abs(gen_lengths.mean() - true_lengths.mean()) < 2 else 'Needs Review'}
      Token diversity:       {generated_df['token_id'].nunique()} types used
    """
    ax2.text(0.1, 0.9, findings_text, fontsize=10, family='monospace', va='top')

    # Visualizations created
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    viz_text = f"""
    VISUALIZATIONS CREATED

    This evaluation generated the following visualization files in {RUN_DIR}:

      1. sequence_lengths_comparison.png  - Histogram, box plot, and CDF of sequence lengths
      2. token_distribution_comparison.png - Token usage patterns and distribution analysis
      3. mri_msr_104_analysis.png         - Detailed analysis of MRI_MSR_104 token (Token 12)
      4. duration_analysis.png            - Step duration distributions and statistics
      5. total_time_analysis.png          - Total sequence time comparisons
      6. bodygroup_transitions.png        - Body group transition patterns
      7. summary_report.png               - This summary report

    All visualizations are saved as high-resolution PNG files (300 DPI) suitable for presentations.
    """
    ax3.text(0.1, 0.9, viz_text, fontsize=10, family='monospace', va='top')

    output_path = os.path.join(RUN_DIR, 'summary_report.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("COMPREHENSIVE EVALUATION WITH VISUALIZATIONS")
    print("="*70)
    print(f"\nOutput directory: {RUN_DIR}\n")

    # Load data
    generated_df, true_df = load_data()

    # Create visualizations
    gen_lengths, true_lengths = plot_sequence_lengths(generated_df, true_df)
    token_comparison = plot_token_distribution(generated_df, true_df)
    plot_mri_msr_104_analysis(generated_df, true_df)
    plot_duration_analysis(generated_df, true_df)
    plot_total_time_analysis(generated_df, true_df)
    plot_bodygroup_transitions(generated_df)
    create_summary_report(generated_df, true_df, gen_lengths, true_lengths, token_comparison)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nAll visualizations saved to: {RUN_DIR}")
    print("\nVisualization files created:")
    for file in sorted(os.listdir(RUN_DIR)):
        print(f"  - {file}")


if __name__ == "__main__":
    main()
