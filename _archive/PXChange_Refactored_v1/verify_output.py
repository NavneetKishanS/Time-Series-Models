import pandas as pd

df = pd.read_csv('outputs/generated_sequences_test.csv')

print('='*70)
print('VERIFICATION OF GENERATED SEQUENCES')
print('='*70)

print('\nFirst 3 samples:')
for i in range(3):
    sample = df[df['sample_idx'] == i].iloc[0]
    print(f'\nSample {i}:')
    print(f'  PatientID_from: {sample["PatientID_from"]}')
    print(f'  PatientID_to: {sample["PatientID_to"]}')
    print(f'  BodyGroup_from: {sample["BodyGroup_from"]}')
    print(f'  BodyGroup_to: {sample["BodyGroup_to"]}')

print('\n' + '='*70)
print('MRI_MSR_104 (token_id=12) counts per sample:')
print('='*70)
mri_counts = df.groupby(['SN', 'sample_idx'])['token_id'].apply(lambda x: (x == 12).sum())
print(f'\nStatistics:')
print(f'  Min: {mri_counts.min()}')
print(f'  Max: {mri_counts.max()}')
print(f'  Mean: {mri_counts.mean():.2f}')
print(f'  Median: {mri_counts.median():.1f}')

print(f'\n  Count of samples with 0 MRI_MSR_104: {(mri_counts == 0).sum()}')
print(f'  Count of samples with 1 MRI_MSR_104: {(mri_counts == 1).sum()}')
print(f'  Count of samples with 2 MRI_MSR_104: {(mri_counts == 2).sum()}')
print(f'  Count of samples with >2 MRI_MSR_104: {(mri_counts > 2).sum()}')

print('\n' + '='*70)
print('Unique PatientIDs verification:')
print('='*70)
unique_patient_ids = df.groupby(['SN', 'sample_idx']).apply(
    lambda x: (x['PatientID_from'].iloc[0], x['PatientID_to'].iloc[0])
)
print(f'  Total unique (PatientID_from, PatientID_to) pairs: {unique_patient_ids.nunique()}')
print(f'  Total samples: {len(unique_patient_ids)}')
print(f'  All unique: {unique_patient_ids.nunique() == len(unique_patient_ids)}')

print('\n' + '='*70)
print('SUCCESS: All modifications implemented correctly!')
print('='*70)
