"""
Examination Dataset for body-region-specific MRI event sequence modeling.

The Examination Model generates: Given conditioning + body region, what MRI events occur?

Training data is extracted from existing preprocessed sequences:
- Each session has a body region (BodyGroup_from) and associated MRI event sequence
- Events are the sourceID tokens with their durations
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

# Body region vocabulary
BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
NUM_BODY_REGIONS = 11

# Import from config
MAX_SEQ_LEN = px_config.MAX_SEQ_LEN
PAD_TOKEN_ID = px_config.PAD_TOKEN_ID
START_TOKEN_ID = px_config.START_TOKEN_ID
END_TOKEN_ID = px_config.END_TOKEN_ID

# Conditioning features for Examination Model
# Includes body region as explicit input
EXAMINATION_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'entity_type'
]


class ExaminationDataset(Dataset):
    """
    Dataset for training the Examination Model.

    Each sample is: (conditioning, body_region) → event_sequence + durations

    Returns:
        conditioning: Patient context features [batch_size, conditioning_dim]
        body_region: Body region being examined [batch_size]
        sequence_tokens: MRI event tokens [batch_size, seq_len]
        step_durations: Duration per event [batch_size, seq_len]
        mask: Valid position mask [batch_size, seq_len]
        seq_length: Actual sequence length [batch_size]
    """

    def __init__(self, dataframe, conditioning_scaler=None, fit_scaler=False,
                 filter_body_region=None):
        """
        Args:
            dataframe: Preprocessed DataFrame with sequences
            conditioning_scaler: StandardScaler for conditioning features
            fit_scaler: Whether to fit the scaler on this data
            filter_body_region: Optional body region ID to filter sequences
        """
        self.df = dataframe
        self.seq_orders = dataframe['SeqOrder'].unique()

        # Filter by body region if specified
        if filter_body_region is not None:
            filtered_seqs = []
            for seq_order in self.seq_orders:
                seq_data = dataframe[dataframe['SeqOrder'] == seq_order].iloc[0]
                if int(seq_data.get('BodyGroup_from', 10)) == filter_body_region:
                    filtered_seqs.append(seq_order)
            self.seq_orders = np.array(filtered_seqs)
            print(f"Filtered to {len(self.seq_orders)} sequences for body region {BODY_REGIONS[filter_body_region]}")

        # Fit or use conditioning scaler
        if fit_scaler:
            self.conditioning_scaler = StandardScaler()
            conditioning_data = []
            for seq_order in self.seq_orders:
                seq_data = dataframe[dataframe['SeqOrder'] == seq_order].iloc[0]
                conditioning_data.append(
                    [seq_data.get(feat, 0) for feat in EXAMINATION_CONDITIONING_FEATURES]
                )
            if conditioning_data:
                self.conditioning_scaler.fit(conditioning_data)
            else:
                # Empty dataset fallback
                self.conditioning_scaler = None
        else:
            self.conditioning_scaler = conditioning_scaler

        self._print_body_region_stats()

    def _print_body_region_stats(self):
        """Print statistics about body region distribution."""
        from collections import Counter

        region_counts = Counter()
        for seq_order in self.seq_orders:
            seq_data = self.df[self.df['SeqOrder'] == seq_order].iloc[0]
            region = int(seq_data.get('BodyGroup_from', 10))
            region_counts[BODY_REGIONS[region]] += 1

        print(f"ExaminationDataset: {len(self.seq_orders)} sequences")
        print("  Body region distribution:")
        for region, count in region_counts.most_common():
            print(f"    {region}: {count}")

    def __len__(self):
        return len(self.seq_orders)

    def __getitem__(self, idx):
        seq_order = self.seq_orders[idx]
        seq_data = self.df[self.df['SeqOrder'] == seq_order].sort_values('Step')

        # Extract conditioning features (constant per sequence)
        first_row = seq_data.iloc[0]
        conditioning = np.array(
            [first_row.get(feat, 0) for feat in EXAMINATION_CONDITIONING_FEATURES],
            dtype=np.float32
        )

        if self.conditioning_scaler is not None:
            conditioning = self.conditioning_scaler.transform(
                conditioning.reshape(1, -1)
            )[0]

        # Extract body region
        body_region = int(first_row.get('BodyGroup_from', 10))

        # Extract sequence tokens (sourceID)
        sequence_tokens = seq_data['sourceID'].values.astype(np.int64)

        # Extract step durations
        if 'step_duration' in seq_data.columns:
            step_durations = seq_data['step_duration'].values.astype(np.float32)
        else:
            step_durations = np.zeros(len(seq_data), dtype=np.float32)
            timediff = seq_data['timediff'].values
            step_durations[1:] = np.diff(timediff)
            step_durations = np.clip(step_durations, 0, None)

        # Pad sequences to MAX_SEQ_LEN
        seq_len = len(sequence_tokens)
        actual_len = min(seq_len, MAX_SEQ_LEN)

        # Create padded arrays
        padded_tokens = np.full(MAX_SEQ_LEN, PAD_TOKEN_ID, dtype=np.int64)
        padded_durations = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.bool_)

        # Fill with actual data
        padded_tokens[:actual_len] = sequence_tokens[:actual_len]
        padded_durations[:actual_len] = step_durations[:actual_len]
        mask[:actual_len] = True

        return {
            'conditioning': torch.from_numpy(conditioning),
            'body_region': torch.tensor(body_region, dtype=torch.long),
            'sequence_tokens': torch.from_numpy(padded_tokens),
            'step_durations': torch.from_numpy(padded_durations),
            'mask': torch.from_numpy(mask),
            'seq_length': torch.tensor(actual_len, dtype=torch.long)
        }


def load_examination_data(dataset_ids=None, use_combined=True):
    """
    Load preprocessed data for Examination Model training.

    Args:
        dataset_ids: List of dataset IDs to load. If None, loads all available.
        use_combined: If True, loads the combined preprocessed file.

    Returns:
        DataFrame with sequence data
    """
    data_dir = px_config.DATA_DIR
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    if not os.path.exists(preprocessed_dir):
        raise ValueError(f"Preprocessed directory not found: {preprocessed_dir}")

    # Try combined file first
    if use_combined:
        combined_file = os.path.join(preprocessed_dir, 'all_preprocessed.csv')
        if os.path.exists(combined_file):
            print(f"Loading combined preprocessed file...")
            df = pd.read_csv(combined_file, low_memory=False)
            print(f"Loaded {len(df['SeqOrder'].unique())} sequences")
            return df

    # Load individual files
    if dataset_ids is None:
        files = [f for f in os.listdir(preprocessed_dir)
                 if f.startswith('preprocessed_') and f.endswith('.csv')]
        dataset_ids = [f.replace('preprocessed_', '').replace('.csv', '') for f in files]

    all_dfs = []
    for dataset_id in dataset_ids:
        preprocessed_file = os.path.join(preprocessed_dir, f'preprocessed_{dataset_id}.csv')
        if os.path.exists(preprocessed_file):
            df = pd.read_csv(preprocessed_file)
            if 'dataset_id' not in df.columns:
                df['dataset_id'] = dataset_id
            all_dfs.append(df)
            print(f"Loaded dataset {dataset_id}: {len(df['SeqOrder'].unique())} sequences")

    if not all_dfs:
        raise ValueError("No datasets could be loaded!")

    return pd.concat(all_dfs, ignore_index=True)


def create_examination_dataloaders(dataframe, batch_size=32, validation_split=0.2,
                                   random_seed=42, filter_body_region=None):
    """
    Create training and validation dataloaders for Examination Model.

    Args:
        dataframe: DataFrame with sequence data
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        filter_body_region: Optional body region ID to filter sequences

    Returns:
        train_loader, val_loader, conditioning_scaler
    """
    # Split by sequences first to prevent data leakage
    seq_orders = dataframe['SeqOrder'].unique()
    train_seqs, val_seqs = train_test_split(
        seq_orders,
        test_size=validation_split,
        random_state=random_seed
    )

    train_df = dataframe[dataframe['SeqOrder'].isin(train_seqs)]
    val_df = dataframe[dataframe['SeqOrder'].isin(val_seqs)]

    # Create datasets
    train_dataset = ExaminationDataset(
        train_df,
        fit_scaler=True,
        filter_body_region=filter_body_region
    )
    val_dataset = ExaminationDataset(
        val_df,
        conditioning_scaler=train_dataset.conditioning_scaler,
        filter_body_region=filter_body_region
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\nCreated Examination dataloaders:")
    print(f"  Training: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} sequences ({len(val_loader)} batches)")

    return train_loader, val_loader, train_dataset.conditioning_scaler


def create_per_region_dataloaders(dataframe, batch_size=32, validation_split=0.2,
                                  random_seed=42):
    """
    Create separate dataloaders for each body region.

    Useful for training region-specific models or analyzing per-region data.

    Args:
        dataframe: DataFrame with sequence data
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping body_region_id → (train_loader, val_loader, scaler)
    """
    region_dataloaders = {}

    for region_id in range(NUM_BODY_REGIONS):
        try:
            train_loader, val_loader, scaler = create_examination_dataloaders(
                dataframe,
                batch_size=batch_size,
                validation_split=validation_split,
                random_seed=random_seed,
                filter_body_region=region_id
            )

            if len(train_loader.dataset) > 0:
                region_dataloaders[region_id] = (train_loader, val_loader, scaler)
                print(f"  Created loaders for {BODY_REGIONS[region_id]}")
        except Exception as e:
            print(f"  Skipping {BODY_REGIONS[region_id]}: {e}")

    return region_dataloaders


if __name__ == "__main__":
    # Test data loading
    print("Testing Examination Dataset loading...")
    print("=" * 60)

    df = load_examination_data()

    # Test all regions
    print("\n--- All Regions Combined ---")
    train_loader, val_loader, scaler = create_examination_dataloaders(df)

    # Test batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    # Test per-region
    print("\n--- Per-Region Dataloaders ---")
    region_loaders = create_per_region_dataloaders(df, batch_size=16)

    for region_id, (train_l, val_l, s) in region_loaders.items():
        print(f"  {BODY_REGIONS[region_id]}: {len(train_l.dataset)} train, {len(val_l.dataset)} val")
