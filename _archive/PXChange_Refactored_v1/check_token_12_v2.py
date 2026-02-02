import pandas as pd

df = pd.read_csv(r'C:\Users\lukis\Documents\GitHub\Time-Series-Models\PXChange_Refactored\data\preprocessed\all_preprocessed.csv')

# Find sequences with many MRI_MSR_104
token_12_counts = df[df['sourceID'] == 12].groupby('SeqOrder').size()
seq_ids_with_many = token_12_counts[token_12_counts > 10].index[:3]

print('='*80)
print('Sequences with MANY MRI_MSR_104 (> 10):')
print('='*80)

for seq_id in seq_ids_with_many:
    seq_data = df[df['SeqOrder'] == seq_id]
    token_12_data = seq_data[seq_data['sourceID'] == 12]

    print(f'\nSeqOrder {seq_id}: {len(token_12_data)} occurrences of token 12')
    print(f'  Dataset ID: {seq_data["dataset_id"].iloc[0]}')
    print(f'  Steps where token 12 appears: {token_12_data["Step"].tolist()[:20]}')
    print(f'  Durations: {token_12_data["step_duration"].tolist()[:20]}')
