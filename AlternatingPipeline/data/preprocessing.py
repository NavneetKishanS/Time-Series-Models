"""
Data preprocessing for the Alternating Pipeline.

Extracts exchange and examination events from raw MRI event log CSVs.

Exchange events: Events that occur during body region transitions (patient setup/breakdown)
Examination events: Events that occur during MRI scans (MRI_EXU_95 markers)
"""
import os
import pandas as pd
import numpy as np
from glob import glob
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, BODY_REGION_TO_ID, SOURCEID_VOCAB,
    START_REGION_ID, END_REGION_ID, COIL_COLUMNS,
    EXCHANGE_CONDITIONING_FEATURES, EXAMINATION_CONDITIONING_FEATURES,
    TEMPORAL_FEATURES
)


def extract_temporal_features(df):
    """
    Extract temporal features from datetime column.

    These features help the model understand time-of-day and day-of-week patterns.

    Args:
        df: DataFrame with 'datetime' column already parsed

    Returns:
        DataFrame with temporal features added
    """
    # Basic temporal features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['minute_of_hour'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_morning'] = (df['datetime'].dt.hour < 12).astype(int)
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour (captures continuity: 23:00 is close to 00:00)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Store the date for temporal splitting later
    df['date'] = df['datetime'].dt.date

    return df


def load_raw_csv(filepath):
    """
    Load a raw MRI event log CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(filepath)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract temporal features (NEW: enables time-aware modeling)
    df = extract_temporal_features(df)

    # Clean numeric columns - convert errors to NaN then fill with 0
    numeric_cols = ['Age', 'Weight', 'Height', 'PTAB', 'timediff']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode direction
    df['Direction_encoded'] = df['Direction'].apply(
        lambda x: 0 if x == 'Head First' else 1 if x == 'Feet First' else -1
    )

    # Map body groups to IDs
    df['BodyGroup_from_id'] = df['BodyGroup_from'].apply(
        lambda x: BODY_REGION_TO_ID.get(x, 10)  # 10 = UNKNOWN
    )
    df['BodyGroup_to_id'] = df['BodyGroup_to'].apply(
        lambda x: BODY_REGION_TO_ID.get(x, 10)
    )

    # Map sourceID to token IDs
    df['sourceID_token'] = df['sourceID'].apply(
        lambda x: SOURCEID_VOCAB.get(x, SOURCEID_VOCAB['UNK'])
    )

    return df


def detect_patient_changes(df):
    """
    Detect patient/examination change points in the data.

    Returns indices where:
    - PatientID changes
    - BodyGroup changes (different examination)

    Args:
        df: DataFrame with PatientId and BodyGroup columns

    Returns:
        List of indices where changes occur
    """
    changes = []

    # Detect patient ID changes
    patient_changes = df['PatientId'] != df['PatientId'].shift(1)

    # Detect body group changes
    bodygroup_changes = df['BodyGroup_to'] != df['BodyGroup_to'].shift(1)

    # Combine: any change is a boundary
    any_change = patient_changes | bodygroup_changes

    # Get indices where changes occur
    change_indices = df.index[any_change].tolist()

    return change_indices


def extract_exchange_events(df, verbose=False):
    """
    Extract exchange (transition) event sequences from data.

    Exchange events occur when:
    - Patient ID changes (new patient)
    - Body region changes (different examination on same/different patient)

    For each exchange, we extract:
    - The event sequence during the transition
    - Conditioning features (from the NEW patient/examination)
    - Body region from/to

    Args:
        df: DataFrame from load_raw_csv
        verbose: Print progress

    Returns:
        List of dicts with keys:
        - 'sequence': List of sourceID tokens
        - 'durations': List of durations for each token
        - 'conditioning': Dict of conditioning features
        - 'body_from': Source body region ID
        - 'body_to': Target body region ID
    """
    exchange_sequences = []

    # Find change points
    change_indices = detect_patient_changes(df)

    if verbose:
        print(f"Found {len(change_indices)} change points")

    # Add start and end boundaries
    boundaries = [0] + change_indices + [len(df)]

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]

        segment = df.iloc[start_idx:end_idx]

        if len(segment) == 0:
            continue

        # Determine if this is an exchange segment
        # Exchange segments happen BEFORE the examination starts (before MRI_EXU_95)
        exam_start_mask = segment['sourceID'] == 'MRI_EXU_95'
        exam_start_indices = segment.index[exam_start_mask].tolist()

        if len(exam_start_indices) > 0:
            # There's an examination in this segment
            # Exchange is from start to first MRI_EXU_95
            first_exam_idx = exam_start_indices[0]
            exchange_segment = segment.loc[:first_exam_idx - 1] if first_exam_idx > start_idx else pd.DataFrame()
        else:
            # No examination marker - entire segment is exchange
            exchange_segment = segment

        if len(exchange_segment) < 2:
            continue

        # Extract sequence
        sequence = exchange_segment['sourceID_token'].tolist()

        # Convert cumulative timediff to inter-event durations
        # timediff is cumulative seconds from session start, NOT per-event duration
        timediffs = exchange_segment['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0  # first event has no prior reference

        # Get conditioning from the target (next) patient/examination
        row = segment.iloc[0]  # First row has the target info
        conditioning = {
            # Patient demographics
            'Age': row.get('Age', 0),
            'Weight': row.get('Weight', 0),
            'Height': row.get('Height', 0),
            'PTAB': row.get('PTAB', 0),
            'Direction_encoded': row.get('Direction_encoded', 0),
            # Temporal features (NEW: enables time-aware modeling)
            'hour_of_day': row.get('hour_of_day', 0),
            'day_of_week': row.get('day_of_week', 0),
            'is_morning': row.get('is_morning', 0),
            'hour_sin': row.get('hour_sin', 0.0),
            'hour_cos': row.get('hour_cos', 1.0),
            'dow_sin': row.get('dow_sin', 0.0),
            'dow_cos': row.get('dow_cos', 1.0),
        }

        # Body region transition
        body_from = row.get('BodyGroup_from_id', START_REGION_ID)
        body_to = row.get('BodyGroup_to_id', 10)  # UNKNOWN if not specified

        # Handle first segment (START -> first body region)
        if i == 0:
            body_from = START_REGION_ID

        # Store start datetime for temporal splitting
        start_datetime = exchange_segment.iloc[0]['datetime'] if len(exchange_segment) > 0 else row['datetime']

        # Total duration is the time span of the exchange (last - first timediff)
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        exchange_sequences.append({
            'sequence': sequence,
            'durations': durations,
            'conditioning': conditioning,
            'body_from': body_from,
            'body_to': body_to,
            'total_duration': total_duration,
            'start_datetime': start_datetime,  # NEW: for temporal train/val split
        })

    if verbose:
        print(f"Extracted {len(exchange_sequences)} exchange sequences")

    return exchange_sequences


def extract_examination_events(df, verbose=False):
    """
    Extract examination (scan) event sequences from data.

    Examination events are marked by MRI_EXU_95 (measurement start).
    We extract sequences from MRI_EXU_95 until the next patient/body change.

    For each examination, we extract:
    - The event sequence during the examination
    - Conditioning features
    - Body region being examined
    - Coil configuration

    Args:
        df: DataFrame from load_raw_csv
        verbose: Print progress

    Returns:
        List of dicts with keys:
        - 'sequence': List of sourceID tokens
        - 'durations': List of durations for each token
        - 'conditioning': Dict of conditioning features
        - 'body_region': Body region ID being examined
        - 'coil_config': Dict of coil element states
    """
    examination_sequences = []

    # Find all MRI_EXU_95 markers (examination starts)
    exam_markers = df[df['sourceID'] == 'MRI_EXU_95'].index.tolist()

    if verbose:
        print(f"Found {len(exam_markers)} examination markers")

    # Find change points for boundaries
    change_indices = set(detect_patient_changes(df))

    for i, exam_start in enumerate(exam_markers):
        # Find the end of this examination
        # Either: next examination marker, or next patient/body change
        exam_end = len(df)

        # Check for next exam marker
        if i + 1 < len(exam_markers):
            exam_end = min(exam_end, exam_markers[i + 1])

        # Check for patient/body change
        for change_idx in change_indices:
            if change_idx > exam_start and change_idx < exam_end:
                exam_end = change_idx
                break

        segment = df.iloc[exam_start:exam_end]

        if len(segment) < 2:
            continue

        # Extract sequence
        sequence = segment['sourceID_token'].tolist()

        # Convert cumulative timediff to inter-event durations
        timediffs = segment['timediff'].values.astype(float)
        durations = np.diff(timediffs, prepend=timediffs[0]).tolist()
        durations[0] = 0.0  # first event has no prior reference

        # Get conditioning
        row = segment.iloc[0]
        conditioning = {
            # Patient demographics
            'Age': row.get('Age', 0),
            'Weight': row.get('Weight', 0),
            'Height': row.get('Height', 0),
            'PTAB': row.get('PTAB', 0),
            'Direction_encoded': row.get('Direction_encoded', 0),
            # Temporal features (NEW: enables time-aware modeling)
            'hour_of_day': row.get('hour_of_day', 0),
            'day_of_week': row.get('day_of_week', 0),
            'is_morning': row.get('is_morning', 0),
            'hour_sin': row.get('hour_sin', 0.0),
            'hour_cos': row.get('hour_cos', 1.0),
            'dow_sin': row.get('dow_sin', 0.0),
            'dow_cos': row.get('dow_cos', 1.0),
        }

        # Body region being examined
        body_region = row.get('BodyGroup_to_id', 10)  # Use BodyGroup_to as current region

        # Coil configuration
        coil_config = {}
        for coil in COIL_COLUMNS:
            if coil in row:
                coil_config[coil] = int(row[coil]) if pd.notna(row[coil]) else 0

        # Total duration is the time span of the examination (last - first timediff)
        total_duration = float(timediffs[-1] - timediffs[0]) if len(timediffs) > 1 else 0.0

        examination_sequences.append({
            'sequence': sequence,
            'durations': durations,
            'conditioning': conditioning,
            'body_region': body_region,
            'coil_config': coil_config,
            'total_duration': total_duration,
            'start_datetime': row['datetime'],  # NEW: for temporal train/val split
        })

    if verbose:
        print(f"Extracted {len(examination_sequences)} examination sequences")

    return examination_sequences


def aggregate_by_day(df):
    """
    Group events by day and create daily summaries.

    This is useful for understanding daily patterns and for the temporal
    forecasting model that predicts 1 day based on 2-week history.

    Args:
        df: DataFrame from load_raw_csv with temporal features

    Returns:
        List of daily summary dicts
    """
    daily_summaries = []

    for date, group in df.groupby('date'):
        # Hourly event distribution (0-23)
        hourly_dist = group.groupby('hour_of_day').size().reindex(range(24), fill_value=0).tolist()

        summary = {
            'date': date,
            'day_of_week': group['day_of_week'].iloc[0],
            'is_weekend': int(group['day_of_week'].iloc[0] >= 5),
            'num_patients': group['PatientId'].nunique(),
            'num_events': len(group),
            'start_hour': group['hour_of_day'].min(),
            'end_hour': group['hour_of_day'].max(),
            'total_duration_seconds': group['timediff'].sum(),
            'avg_duration_per_event': group['timediff'].mean(),
            'hourly_distribution': hourly_dist,
            'morning_event_ratio': (group['hour_of_day'] < 12).mean(),
            'body_regions': group['BodyGroup_to'].value_counts().to_dict(),
        }
        daily_summaries.append(summary)

    return sorted(daily_summaries, key=lambda x: x['date'])


def extract_customer_daily_schedules(data_dir=None, verbose=True):
    """
    Extract real daily patient sequences per customer (MRI scanner).

    Each CSV file represents one customer/scanner. For each file, we group
    events by date and extract the ordered patient list for that day.

    Args:
        data_dir: Directory containing raw CSVs (default: config.DATA_DIR)
        verbose: Print progress

    Returns:
        Dict of {customer_id: {date_str: [patient_dicts]}}
        Each patient_dict has: patient_id, body_region, age, weight, height,
        direction, hour_of_day, day_of_week
    """
    if data_dir is None:
        data_dir = DATA_DIR

    csv_files = glob(os.path.join(data_dir, '*.csv'))

    if verbose:
        print(f"Extracting customer daily schedules from {len(csv_files)} files...")

    customer_schedules = {}

    for csv_path in csv_files:
        customer_id = os.path.splitext(os.path.basename(csv_path))[0]

        try:
            df = load_raw_csv(csv_path)
        except Exception as e:
            if verbose:
                print(f"  Error loading {customer_id}: {e}")
            continue

        customer_schedules[customer_id] = {}

        for date, day_group in df.groupby('date'):
            date_str = str(date)
            patients = []
            seen_patients = set()

            # Walk through the day's events in order
            for _, row in day_group.iterrows():
                patient_id = row.get('PatientId', '')
                if not patient_id or patient_id in seen_patients:
                    continue

                seen_patients.add(patient_id)

                body_region_str = row.get('BodyGroup_to', 'UNKNOWN')
                body_region_id = BODY_REGION_TO_ID.get(body_region_str, 10)

                patients.append({
                    'patient_id': patient_id,
                    'body_region': body_region_str,
                    'body_region_id': body_region_id,
                    'age': float(row.get('Age', 0)),
                    'weight': float(row.get('Weight', 0)),
                    'height': float(row.get('Height', 0)),
                    'direction': row.get('Direction', 'Head First'),
                    'hour_of_day': int(row.get('hour_of_day', 8)),
                    'day_of_week': int(row.get('day_of_week', 0)),
                })

            if patients:
                customer_schedules[customer_id][date_str] = patients

        if verbose:
            num_days = len(customer_schedules[customer_id])
            total_patients = sum(len(p) for p in customer_schedules[customer_id].values())
            print(f"  {customer_id}: {num_days} days, {total_patients} total patients")

    return customer_schedules


def preprocess_all_data(data_dir=None, output_dir=None, verbose=True):
    """
    Preprocess all raw CSV files in the data directory.

    Args:
        data_dir: Directory containing raw CSVs (default: config.DATA_DIR)
        output_dir: Directory to save preprocessed data (default: AlternatingPipeline/data/preprocessed)
        verbose: Print progress

    Returns:
        Dict with 'exchange' and 'examination' lists
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'preprocessed')

    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob(os.path.join(data_dir, '*.csv'))

    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

    all_exchange = []
    all_examination = []
    all_daily_summaries = []

    for i, csv_path in enumerate(csv_files):
        if verbose:
            print(f"\nProcessing {i+1}/{len(csv_files)}: {os.path.basename(csv_path)}")

        try:
            df = load_raw_csv(csv_path)

            exchange = extract_exchange_events(df, verbose=False)
            examination = extract_examination_events(df, verbose=False)
            daily_summaries = aggregate_by_day(df)

            all_exchange.extend(exchange)
            all_examination.extend(examination)
            all_daily_summaries.extend(daily_summaries)

            if verbose:
                print(f"  Exchange: {len(exchange)}, Examination: {len(examination)}, Days: {len(daily_summaries)}")

        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            continue

    if verbose:
        print(f"\n{'='*60}")
        print(f"Total exchange sequences: {len(all_exchange)}")
        print(f"Total examination sequences: {len(all_examination)}")
        print(f"Total daily summaries: {len(all_daily_summaries)}")

    # Extract per-customer daily schedules
    if verbose:
        print(f"\nExtracting per-customer daily schedules...")
    customer_schedules = extract_customer_daily_schedules(data_dir, verbose=verbose)

    # Save preprocessed data
    result = {
        'exchange': all_exchange,
        'examination': all_examination,
        'daily_summaries': all_daily_summaries,
        'customer_schedules': customer_schedules,
    }

    output_path = os.path.join(output_dir, 'preprocessed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    if verbose:
        print(f"\nSaved preprocessed data to {output_path}")
        print(f"Customer schedules: {len(customer_schedules)} customers")

    return result


def load_preprocessed_data(path=None):
    """
    Load preprocessed data from pickle file.

    Args:
        path: Path to pickle file (default: AlternatingPipeline/data/preprocessed/preprocessed_data.pkl)

    Returns:
        Dict with 'exchange' and 'examination' lists
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'preprocessed', 'preprocessed_data.pkl')

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    print("Testing data preprocessing...")
    print("=" * 60)

    # Test on a single file
    csv_files = glob(os.path.join(DATA_DIR, '*.csv'))
    if csv_files:
        test_file = csv_files[0]
        print(f"\nTesting on: {os.path.basename(test_file)}")

        df = load_raw_csv(test_file)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)[:10]}...")

        exchange = extract_exchange_events(df, verbose=True)
        examination = extract_examination_events(df, verbose=True)

        if exchange:
            print(f"\nSample exchange event:")
            print(f"  Sequence length: {len(exchange[0]['sequence'])}")
            print(f"  Body: {exchange[0]['body_from']} -> {exchange[0]['body_to']}")
            print(f"  Conditioning: {exchange[0]['conditioning']}")

        if examination:
            print(f"\nSample examination event:")
            print(f"  Sequence length: {len(examination[0]['sequence'])}")
            print(f"  Body region: {examination[0]['body_region']}")
            print(f"  Conditioning: {examination[0]['conditioning']}")
    else:
        print("No CSV files found in data directory")
