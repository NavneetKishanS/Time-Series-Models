"""
Show summary of all body groups in the generated sequences
"""
import pandas as pd

# Read the data
df = pd.read_csv('outputs/generated_sequences.csv')

print("="*70)
print("BODY GROUP ANALYSIS")
print("="*70)

print("\nBody Groups FROM (scan starting point):")
print("-" * 70)
from_groups = df.groupby(['BodyGroup_from', 'BodyGroup_from_text']).size().reset_index(name='count')
from_groups = from_groups.sort_values('BodyGroup_from')
from_groups['percentage'] = (from_groups['count'] / from_groups['count'].sum() * 100).round(2)
print(from_groups.to_string(index=False))

print("\n\nBody Groups TO (scan ending point):")
print("-" * 70)
to_groups = df.groupby(['BodyGroup_to', 'BodyGroup_to_text']).size().reset_index(name='count')
to_groups = to_groups.sort_values('BodyGroup_to')
to_groups['percentage'] = (to_groups['count'] / to_groups['count'].sum() * 100).round(2)
print(to_groups.to_string(index=False))

print("\n\nCommon Body Group Transitions (FROM -> TO):")
print("-" * 70)
transitions = df.groupby(['BodyGroup_from_text', 'BodyGroup_to_text']).size().reset_index(name='count')
transitions = transitions.sort_values('count', ascending=False).head(10)
transitions['percentage'] = (transitions['count'] / len(df) * 100).round(2)
print(transitions.to_string(index=False))

print("\n\nSample Sequences with Body Group Info:")
print("-" * 70)
# Show a few complete sequences
for sample_idx in range(3):
    sample_data = df[df['sample_idx'] == sample_idx].iloc[0]
    seq_data = df[(df['SN'] == sample_data['SN']) & (df['sample_idx'] == sample_idx)]

    print(f"\nSample {sample_idx} (Customer {sample_data['SN']}):")
    print(f"  Transition: {sample_data['BodyGroup_from_text']} -> {sample_data['BodyGroup_to_text']}")
    print(f"  Sequence length: {len(seq_data)} steps")
    print(f"  Tokens: {' -> '.join(seq_data['token_name'].head(10).tolist())}")
    if len(seq_data) > 10:
        print(f"           ... ({len(seq_data) - 10} more steps)")
