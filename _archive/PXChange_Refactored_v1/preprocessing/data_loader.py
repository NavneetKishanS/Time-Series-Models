"""
Data loading and preparation for conditional generation
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
pxchange_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, pxchange_dir)

import config as px_config
DATA_DIR = px_config.DATA_DIR
CONDITIONING_FEATURES = px_config.CONDITIONING_FEATURES
SEQUENCE_FEATURE_COLUMNS = px_config.SEQUENCE_FEATURE_COLUMNS
MAX_SEQ_LEN = px_config.MAX_SEQ_LEN
PAD_TOKEN_ID = px_config.PAD_TOKEN_ID
START_TOKEN_ID = px_config.START_TOKEN_ID
END_TOKEN_ID = px_config.END_TOKEN_ID
IDLE_TOKEN_ID = px_config.IDLE_TOKEN_ID
IDLE_STATE_ENCODING = px_config.IDLE_STATE_ENCODING
RANDOM_SEED = px_config.RANDOM_SEED


class MRISequenceDataset(Dataset):
    """
    Dataset for MRI scan sequences with conditioning information.

    Updated to support pseudo-patient architecture:
    - entity_type feature included in conditioning (7 features instead of 6)
    - IDLE tokens supported for pseudo-patient sequences
    - Validates that IDLE tokens only appear in pseudo-patient sequences

    Returns:
        conditioning: Patient/scan context features [batch_size, conditioning_dim]
        sequence_tokens: Symbolic sequence (sourceID) [batch_size, seq_len]
        sequence_features: Additional sequence features [batch_size, seq_len, feature_dim]
        step_durations: Target counts (step durations) [batch_size, seq_len]
        sequence_mask: Mask for valid positions [batch_size, seq_len]
    """

    def __init__(self, dataframe, conditioning_scaler=None, fit_scaler=False):
        """
        Args:
            dataframe: Preprocessed DataFrame with sequences
            conditioning_scaler: StandardScaler for conditioning features
            fit_scaler: Whether to fit the scaler on this data
        """
        self.df = dataframe
        self.seq_orders = dataframe['SeqOrder'].unique()

        # Fit or use conditioning scaler
        if fit_scaler:
            self.conditioning_scaler = StandardScaler()
            # Get one row per sequence for conditioning features
            conditioning_data = []
            for seq_order in self.seq_orders:
                seq_data = dataframe[dataframe['SeqOrder'] == seq_order].iloc[0]
                conditioning_data.append([seq_data.get(feat, 0) for feat in CONDITIONING_FEATURES])
            self.conditioning_scaler.fit(conditioning_data)
        else:
            self.conditioning_scaler = conditioning_scaler

    def __len__(self):
        return len(self.seq_orders)

    def __getitem__(self, idx):
        seq_order = self.seq_orders[idx]
        seq_data = self.df[self.df['SeqOrder'] == seq_order].sort_values('Step')

        # Extract conditioning features (constant per sequence)
        first_row = seq_data.iloc[0]
        conditioning = np.array([first_row.get(feat, 0) for feat in CONDITIONING_FEATURES], dtype=np.float32)

        # Note: entity_type is automatically included via CONDITIONING_FEATURES
        # It's the 7th feature (after Age, Weight, Height, BodyGroup_from, BodyGroup_to, PTAB)
        # Values: REAL_PATIENT=0, PSEUDO_PATIENT_PAUSE=1, etc.

        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(conditioning.reshape(1, -1))[0]

        # Extract sequence tokens (sourceID)
        sequence_tokens = seq_data['sourceID'].values.astype(np.int64)

        # Extract additional sequence features
        sequence_features = seq_data[SEQUENCE_FEATURE_COLUMNS].values.astype(np.float32)

        # Extract step durations (target counts)
        if 'step_duration' in seq_data.columns:
            step_durations = seq_data['step_duration'].values.astype(np.float32)
        else:
            # Calculate if not present
            step_durations = np.zeros(len(seq_data), dtype=np.float32)
            timediff = seq_data['timediff'].values
            step_durations[1:] = np.diff(timediff)
            step_durations = np.clip(step_durations, 0, None)

        # Pad sequences to MAX_SEQ_LEN
        seq_len = len(sequence_tokens)

        # Create padded arrays
        padded_tokens = np.full(MAX_SEQ_LEN, PAD_TOKEN_ID, dtype=np.int64)
        padded_features = np.zeros((MAX_SEQ_LEN, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32)
        padded_durations = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.bool_)

        # Fill with actual data
        actual_len = min(seq_len, MAX_SEQ_LEN)
        padded_tokens[:actual_len] = sequence_tokens[:actual_len]
        padded_features[:actual_len] = sequence_features[:actual_len]
        padded_durations[:actual_len] = step_durations[:actual_len]
        mask[:actual_len] = True

        return {
            'conditioning': torch.from_numpy(conditioning),
            'sequence_tokens': torch.from_numpy(padded_tokens),
            'sequence_features': torch.from_numpy(padded_features),
            'step_durations': torch.from_numpy(padded_durations),
            'mask': torch.from_numpy(mask),
            'seq_length': torch.tensor(actual_len, dtype=torch.long)
        }


def load_preprocessed_data(dataset_ids=None, use_combined=True):
    """
    Load preprocessed data from multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to load. If None, loads all available.
        use_combined: If True, loads the combined preprocessed file. Otherwise loads individual files.

    Returns:
        combined_df: Combined DataFrame from all datasets
    """
    preprocessed_dir = os.path.join(DATA_DIR, 'preprocessed')

    # Check if preprocessed directory exists
    if not os.path.exists(preprocessed_dir):
        raise ValueError(f"Preprocessed directory not found: {preprocessed_dir}\n"
                        "Please run preprocessing first: python preprocessing/preprocess_raw_data.py")

    # Try to load combined file first if requested
    if use_combined:
        combined_file = os.path.join(preprocessed_dir, 'all_preprocessed.csv')
        if os.path.exists(combined_file):
            print(f"Loading combined preprocessed file...")
            df = pd.read_csv(combined_file, low_memory=False)
            print(f"[OK] Loaded {len(df['SeqOrder'].unique())} sequences from combined file")
            return df
        else:
            print(f"Combined file not found, loading individual files...")

    # Load individual files
    if dataset_ids is None:
        # Auto-discover dataset IDs from preprocessed files
        files = [f for f in os.listdir(preprocessed_dir) if f.startswith('preprocessed_') and f.endswith('.csv')]
        dataset_ids = [f.replace('preprocessed_', '').replace('.csv', '') for f in files]

    all_dfs = []

    for dataset_id in dataset_ids:
        preprocessed_file = os.path.join(preprocessed_dir, f'preprocessed_{dataset_id}.csv')

        if not os.path.exists(preprocessed_file):
            print(f"Warning: Preprocessed file not found for dataset {dataset_id}, skipping...")
            continue

        try:
            df = pd.read_csv(preprocessed_file)
            if 'dataset_id' not in df.columns:
                df['dataset_id'] = dataset_id  # Track which dataset this came from
            all_dfs.append(df)
            print(f"[OK] Loaded dataset {dataset_id}: {len(df['SeqOrder'].unique())} sequences")
        except Exception as e:
            print(f"[ERROR] Error loading dataset {dataset_id}: {str(e)}")

    if not all_dfs:
        raise ValueError("No datasets could be loaded!")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[OK] Total loaded: {len(combined_df['SeqOrder'].unique())} sequences from {len(all_dfs)} datasets")

    return combined_df


def create_dataloaders(dataframe, batch_size=32, validation_split=0.2):
    """
    Create training and validation dataloaders.

    Args:
        dataframe: Combined DataFrame
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation

    Returns:
        train_loader, val_loader, conditioning_scaler
    """
    # Split sequences into train/val
    seq_orders = dataframe['SeqOrder'].unique()
    train_seqs, val_seqs = train_test_split(
        seq_orders,
        test_size=validation_split,
        random_state=RANDOM_SEED
    )

    train_df = dataframe[dataframe['SeqOrder'].isin(train_seqs)]
    val_df = dataframe[dataframe['SeqOrder'].isin(val_seqs)]

    # Create datasets
    train_dataset = MRISequenceDataset(train_df, fit_scaler=True)
    val_dataset = MRISequenceDataset(val_df, conditioning_scaler=train_dataset.conditioning_scaler)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n[OK] Created dataloaders:")
    print(f"  Training: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_loader)} batches)")

    return train_loader, val_loader, train_dataset.conditioning_scaler


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    df = load_preprocessed_data()
    train_loader, val_loader, scaler = create_dataloaders(df)

    # Test batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
