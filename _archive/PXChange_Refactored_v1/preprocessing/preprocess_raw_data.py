"""
Preprocessing script to convert raw MRI scan logs to the format expected by the training pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SOURCEID_VOCAB, DATA_DIR


def encode_position(position):
    """Encode patient position (Supine, Prone, etc.) to integer."""
    position_map = {
        'Supine': 0,
        'Prone': 1,
        'Left': 2,
        'Right': 3,
        'Unknown': 4
    }
    if pd.isna(position):
        return 4
    return position_map.get(position, 4)


def encode_direction(direction):
    """Encode scan direction (Head First, Feet First) to integer."""
    direction_map = {
        'Head First': 0,
        'Feet First': 1,
        'Unknown': 2
    }
    if pd.isna(direction):
        return 2
    return direction_map.get(direction, 2)


def encode_bodygroup(bodygroup):
    """Encode body group to integer."""
    bodygroup_map = {
        'HEAD': 0,
        'NECK': 1,
        'CHEST': 2,
        'ABDOMEN': 3,
        'PELVIS': 4,
        'SPINE': 5,
        'LSPINE': 5,  # Lower spine maps to spine
        'USPINE': 5,  # Upper spine maps to spine
        'ARM': 6,
        'LEG': 7,
        'HAND': 8,
        'FOOT': 9,
        'Unknown': 10
    }
    if pd.isna(bodygroup):
        return 10
    return bodygroup_map.get(bodygroup, 10)


def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if conversion fails."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def preprocess_raw_file(raw_file_path, dataset_id):
    """
    Preprocess a single raw CSV file.

    Args:
        raw_file_path: Path to raw CSV file
        dataset_id: Dataset identifier (e.g., '141049')

    Returns:
        preprocessed_df: Preprocessed DataFrame ready for training
    """
    print(f"\nProcessing dataset {dataset_id}...")

    # Load raw data
    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        print(f"Error loading {raw_file_path}: {str(e)}")
        return None

    print(f"  Loaded {len(df)} raw events")

    # Filter for valid sourceIDs (only keep events we have in vocabulary)
    valid_sourceids = [sid for sid in SOURCEID_VOCAB.keys() if sid not in ['PAD', 'START', 'END', 'UNK']]
    df = df[df['sourceID'].isin(valid_sourceids)].copy()
    print(f"  Filtered to {len(df)} events with valid sourceIDs")

    if len(df) == 0:
        print(f"  Warning: No valid events found in dataset {dataset_id}")
        return None

    # Sort by datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Group by PatientId to create sequences
    sequences = []
    seq_counter = 0

    for patient_id in df['PatientId'].unique():
        patient_df = df[df['PatientId'] == patient_id].copy()

        # Further split by session (gap > 30 minutes = new session)
        patient_df['time_gap'] = patient_df['datetime'].diff().dt.total_seconds() / 60.0
        patient_df['new_session'] = (patient_df['time_gap'] > 30) | (patient_df['time_gap'].isna())
        patient_df['session_id'] = patient_df['new_session'].cumsum()

        # Process each session as a separate sequence
        for session_id in patient_df['session_id'].unique():
            session_df = patient_df[patient_df['session_id'] == session_id].copy()

            # Skip very short sequences (less than 3 events)
            if len(session_df) < 3:
                continue

            # Skip sequences longer than 126 (to leave room for START and END tokens)
            if len(session_df) > 126:
                session_df = session_df.iloc[:126].copy()

            # Create sequence
            seq_data = []

            # Add START token
            first_row = session_df.iloc[0]
            seq_data.append({
                'SeqOrder': seq_counter,
                'Step': 0,
                'sourceID': SOURCEID_VOCAB['START'],
                'Age': safe_float(first_row['Age'], 50.0),
                'Weight': safe_float(first_row['Weight'], 70.0),
                'Height': safe_float(first_row['Height'], 1.70),
                'BodyGroup_from': encode_bodygroup(first_row['BodyGroup_from']),
                'BodyGroup_to': encode_bodygroup(first_row['BodyGroup_to']),
                'PTAB': safe_float(first_row['PTAB'], 0.0),
                'entity_type': 0, # 0 for REAL_PATIENT
                'Position_encoded': encode_position(first_row['Position']),
                'Direction_encoded': encode_direction(first_row['Direction']),
                'timediff': 0.0,
                'step_duration': 0.0
            })

            # Add actual events
            for i, (idx, row) in enumerate(session_df.iterrows(), start=1):
                seq_data.append({
                    'SeqOrder': seq_counter,
                    'Step': i,
                    'sourceID': SOURCEID_VOCAB.get(row['sourceID'], SOURCEID_VOCAB['UNK']),
                    'Age': safe_float(row['Age'], 50.0),
                    'Weight': safe_float(row['Weight'], 70.0),
                    'Height': safe_float(row['Height'], 1.70),
                    'BodyGroup_from': encode_bodygroup(row['BodyGroup_from']),
                    'BodyGroup_to': encode_bodygroup(row['BodyGroup_to']),
                    'PTAB': safe_float(row['PTAB'], 0.0),
                    'entity_type': 0, # 0 for REAL_PATIENT
                    'Position_encoded': encode_position(row['Position']),
                    'Direction_encoded': encode_direction(row['Direction']),
                    'timediff': safe_float(row['timediff'], 0.0),
                    'step_duration': 0.0  # Will calculate below
                })

            # Add END token
            last_row = session_df.iloc[-1]
            seq_data.append({
                'SeqOrder': seq_counter,
                'Step': len(session_df) + 1,
                'sourceID': SOURCEID_VOCAB['END'],
                'Age': safe_float(last_row['Age'], 50.0),
                'Weight': safe_float(last_row['Weight'], 70.0),
                'Height': safe_float(last_row['Height'], 1.70),
                'BodyGroup_from': encode_bodygroup(last_row['BodyGroup_from']),
                'BodyGroup_to': encode_bodygroup(last_row['BodyGroup_to']),
                'PTAB': safe_float(last_row['PTAB'], 0.0),
                'entity_type': 0, # 0 for REAL_PATIENT
                'Position_encoded': encode_position(last_row['Position']),
                'Direction_encoded': encode_direction(last_row['Direction']),
                'timediff': safe_float(last_row['timediff'], 0.0),
                'step_duration': 0.0
            })

            # Calculate step durations
            seq_df = pd.DataFrame(seq_data)
            timediffs = seq_df['timediff'].values
            step_durations = np.zeros(len(timediffs))
            step_durations[1:] = np.diff(timediffs)
            step_durations = np.clip(step_durations, 0, None)  # No negative durations
            seq_df['step_duration'] = step_durations

            sequences.append(seq_df)
            seq_counter += 1

    if not sequences:
        print(f"  Warning: No valid sequences created from dataset {dataset_id}")
        return None

    # Combine all sequences
    preprocessed_df = pd.concat(sequences, ignore_index=True)
    print(f"  Created {seq_counter} sequences with {len(preprocessed_df)} total steps")

    return preprocessed_df


def preprocess_all_datasets(output_subdir='preprocessed'):
    """
    Preprocess all raw CSV files in the data directory.

    Args:
        output_subdir: Subdirectory name for saving preprocessed data
    """
    print("="*80)
    print("PREPROCESSING RAW MRI SCAN DATA")
    print("="*80)

    # Get all CSV files in data directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    if not csv_files:
        print(f"Error: No CSV files found in {DATA_DIR}")
        return

    print(f"\nFound {len(csv_files)} raw CSV files")

    # Create output directory
    output_dir = os.path.join(DATA_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    all_datasets = []

    # Process each file
    for csv_file in tqdm(csv_files, desc="Processing datasets"):
        dataset_id = csv_file.replace('.csv', '')
        raw_file_path = os.path.join(DATA_DIR, csv_file)

        # Preprocess
        preprocessed_df = preprocess_raw_file(raw_file_path, dataset_id)

        if preprocessed_df is not None and len(preprocessed_df) > 0:
            # Add dataset ID
            preprocessed_df['dataset_id'] = dataset_id
            all_datasets.append(preprocessed_df)

            # Save individual preprocessed file
            output_file = os.path.join(output_dir, f'preprocessed_{dataset_id}.csv')
            preprocessed_df.to_csv(output_file, index=False)
            print(f"  Saved to {output_file}")

    # Combine and save all datasets
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_file = os.path.join(output_dir, 'all_preprocessed.csv')
        combined_df.to_csv(combined_file, index=False)

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Total datasets processed: {len(all_datasets)}")
        print(f"Total sequences: {combined_df['SeqOrder'].nunique()}")
        print(f"Total steps: {len(combined_df)}")
        print(f"Average sequence length: {len(combined_df) / combined_df['SeqOrder'].nunique():.1f}")
        print(f"\nAll preprocessed data saved to: {output_dir}")
        print(f"Combined file: {combined_file}")

        # Print some statistics
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"\nConditioning features:")
        for feat in ['Age', 'Weight', 'Height']:
            print(f"  {feat}: mean={combined_df[feat].mean():.1f}, std={combined_df[feat].std():.1f}")

        print(f"\nSourceID distribution (excluding PAD/START/END):")
        sourceid_counts = combined_df[~combined_df['sourceID'].isin([0, 11, 14])]['sourceID'].value_counts()
        reverse_vocab = {v: k for k, v in SOURCEID_VOCAB.items()}
        for sid, count in sourceid_counts.head(10).items():
            print(f"  {reverse_vocab.get(sid, 'UNK')}: {count}")
    else:
        print("\nError: No datasets were successfully preprocessed!")


if __name__ == "__main__":
    preprocess_all_datasets()
