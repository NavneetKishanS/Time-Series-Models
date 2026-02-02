import pandas as pd

df = pd.read_csv(r'C:\Users\lukis\Documents\GitHub\Time-Series-Models\PXChange_Refactored\data\preprocessed\all_preprocessed.csv')

# Find sequences with many MRI_MSR_104
seq_with_many = df[df['sourceID'] == 12].groupby('SeqOrder').size()
seq_ids = seq_with_many[seq_with_many > 10].index[:3]

print('Sequences with many MRI_MSR_104:')
for seq_id in seq_ids:
    print(f'\nSeqOrder {seq_id}:')
    seq_data = df[df['SeqOrder'] == seq_id][['Step', 'sourceID', 'step_duration']].head(15)
    print(seq_data.to_string(index=False))
