"""
Exchange Dataset for body region transition modeling.

The Exchange Model predicts: Given conditioning + current body region, what is the next body region?

Training data is extracted from existing preprocessed sequences:
- Each session has BodyGroup_from (current region being examined) and BodyGroup_to (next region)
- We model transitions: current_region → next_region
- Special START (ID=11) and END (ID=12) regions for session boundaries
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
NUM_BODY_REGIONS = 11  # 0-10 for actual regions
START_REGION_ID = 11   # Special token for session start
END_REGION_ID = 12     # Special token for session end
NUM_REGION_CLASSES = 13  # Total classes including START and END

# Conditioning features for Exchange Model (patient-level, not body-region specific)
EXCHANGE_CONDITIONING_FEATURES = [
    'Age',
    'Weight',
    'Height',
    'PTAB',
    'entity_type'
]


class ExchangeDataset(Dataset):
    """
    Dataset for training the Exchange Model.

    Each sample is a transition: (conditioning, current_region) → next_region

    Extracts transitions from preprocessed data:
    - START → BodyGroup_from (beginning of each session)
    - BodyGroup_from → BodyGroup_to (if BodyGroup_to != UNKNOWN)
    - BodyGroup_from → END (if BodyGroup_to == UNKNOWN/10)

    Returns:
        conditioning: Patient context features [batch_size, conditioning_dim]
        current_region: Current body region ID [batch_size]
        next_region: Target next body region ID [batch_size]
    """

    def __init__(self, dataframe, conditioning_scaler=None, fit_scaler=False):
        """
        Args:
            dataframe: Preprocessed DataFrame with sequences
            conditioning_scaler: StandardScaler for conditioning features
            fit_scaler: Whether to fit the scaler on this data
        """
        self.df = dataframe
        self.transitions = []

        # Extract unique sequences
        seq_orders = dataframe['SeqOrder'].unique()

        # Extract transitions from each sequence
        for seq_order in seq_orders:
            seq_data = dataframe[dataframe['SeqOrder'] == seq_order].iloc[0]

            # Get conditioning features
            conditioning = [seq_data.get(feat, 0) for feat in EXCHANGE_CONDITIONING_FEATURES]

            # Get body regions
            body_from = int(seq_data.get('BodyGroup_from', 10))  # Default to UNKNOWN
            body_to = int(seq_data.get('BodyGroup_to', 10))

            # Transition 1: START → BodyGroup_from
            self.transitions.append({
                'conditioning': conditioning,
                'current_region': START_REGION_ID,
                'next_region': body_from
            })

            # Transition 2: BodyGroup_from → BodyGroup_to or END
            if body_to == 10:  # UNKNOWN means end of session
                next_region = END_REGION_ID
            else:
                next_region = body_to

            self.transitions.append({
                'conditioning': conditioning,
                'current_region': body_from,
                'next_region': next_region
            })

            # If we have a valid BodyGroup_to (not UNKNOWN), add transition to END
            if body_to != 10:
                self.transitions.append({
                    'conditioning': conditioning,
                    'current_region': body_to,
                    'next_region': END_REGION_ID
                })

        # Convert to numpy arrays for efficiency
        self.conditioning_data = np.array(
            [t['conditioning'] for t in self.transitions],
            dtype=np.float32
        )
        self.current_regions = np.array(
            [t['current_region'] for t in self.transitions],
            dtype=np.int64
        )
        self.next_regions = np.array(
            [t['next_region'] for t in self.transitions],
            dtype=np.int64
        )

        # Fit or use conditioning scaler
        if fit_scaler:
            self.conditioning_scaler = StandardScaler()
            self.conditioning_scaler.fit(self.conditioning_data)
        else:
            self.conditioning_scaler = conditioning_scaler

        # Scale conditioning data
        if self.conditioning_scaler is not None:
            self.conditioning_data = self.conditioning_scaler.transform(
                self.conditioning_data
            ).astype(np.float32)

        print(f"ExchangeDataset: {len(self.transitions)} transitions from {len(seq_orders)} sequences")
        self._print_transition_stats()

    def _print_transition_stats(self):
        """Print statistics about transition distribution."""
        from collections import Counter

        # Count transitions by type
        transition_counts = Counter()
        for i in range(len(self.current_regions)):
            curr = self.current_regions[i]
            next_ = self.next_regions[i]

            curr_name = 'START' if curr == START_REGION_ID else (
                'END' if curr == END_REGION_ID else BODY_REGIONS[curr]
            )
            next_name = 'START' if next_ == START_REGION_ID else (
                'END' if next_ == END_REGION_ID else BODY_REGIONS[next_]
            )

            transition_counts[(curr_name, next_name)] += 1

        print("  Top transitions:")
        for (curr, next_), count in transition_counts.most_common(10):
            print(f"    {curr} -> {next_}: {count}")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return {
            'conditioning': torch.from_numpy(self.conditioning_data[idx]),
            'current_region': torch.tensor(self.current_regions[idx], dtype=torch.long),
            'next_region': torch.tensor(self.next_regions[idx], dtype=torch.long)
        }


def load_exchange_data(dataset_ids=None, use_combined=True):
    """
    Load preprocessed data for Exchange Model training.

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


def create_exchange_dataloaders(dataframe, batch_size=64, validation_split=0.2,
                                random_seed=42):
    """
    Create training and validation dataloaders for Exchange Model.

    Args:
        dataframe: DataFrame with sequence data
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility

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
    train_dataset = ExchangeDataset(train_df, fit_scaler=True)
    val_dataset = ExchangeDataset(
        val_df,
        conditioning_scaler=train_dataset.conditioning_scaler
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

    print(f"\nCreated Exchange dataloaders:")
    print(f"  Training: {len(train_dataset)} transitions ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} transitions ({len(val_loader)} batches)")

    return train_loader, val_loader, train_dataset.conditioning_scaler


if __name__ == "__main__":
    # Test data loading
    print("Testing Exchange Dataset loading...")
    print("=" * 60)

    df = load_exchange_data()
    train_loader, val_loader, scaler = create_exchange_dataloaders(df)

    # Test batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print("\nSample data:")
    print(f"  Conditioning (first 3): {batch['conditioning'][:3]}")
    print(f"  Current regions (first 10): {batch['current_region'][:10]}")
    print(f"  Next regions (first 10): {batch['next_region'][:10]}")
