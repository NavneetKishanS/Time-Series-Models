"""
Orchestration preprocessing: extract day-level body region sequences
from customer schedules for training the Orchestration Model.

Input: customer_schedules from preprocessed_data.pkl
Output: list of sample dicts with body region token sequences + conditioning
"""
import os
import sys
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    START_REGION_ID, END_REGION_ID, BREAK_TOKEN_ID,
    NUM_BODY_REGIONS, BODY_REGIONS, BODY_REGION_TO_ID,
)


def extract_orchestration_samples(preprocessed_data, break_threshold_hours=1):
    """
    Extract orchestration training samples from preprocessed customer schedules.

    Each sample represents one day at one scanner, encoded as a sequence of
    body region tokens with BREAK tokens inserted where gaps exceed the threshold.

    Args:
        preprocessed_data: Dict from preprocessed_data.pkl with 'customer_schedules'
        break_threshold_hours: Hour gap between consecutive patients that triggers
            a BREAK token insertion (patients store hour_of_day as integer)

    Returns:
        samples: List of dicts with keys:
            - 'tokens': list of body region IDs (without START/END)
            - 'conditioning': 17-dim numpy array
            - 'scanner_idx': int
            - 'start_datetime': datetime for temporal splitting
        scanner_to_idx: dict mapping customer_id -> scanner index
    """
    customer_schedules = preprocessed_data.get('customer_schedules', {})

    # Build scanner index mapping
    scanner_to_idx = {cid: idx for idx, cid in enumerate(sorted(customer_schedules.keys()))}

    # Compute per-scanner historical statistics for conditioning
    scanner_stats = _compute_scanner_stats(customer_schedules)

    samples = []

    for customer_id, daily_schedules in customer_schedules.items():
        scanner_idx = scanner_to_idx[customer_id]
        stats = scanner_stats[customer_id]

        for date_str, patients in daily_schedules.items():
            if not patients:
                continue

            # Build body region token sequence with BREAKs
            tokens = []
            for i, patient in enumerate(patients):
                # Insert BREAK if gap between this and previous patient is large
                if i > 0:
                    prev_hour = patients[i - 1].get('hour_of_day', 8)
                    curr_hour = patient.get('hour_of_day', 8)
                    if curr_hour - prev_hour >= break_threshold_hours:
                        tokens.append(BREAK_TOKEN_ID)

                body_region_id = patient.get('body_region_id', 10)
                tokens.append(body_region_id)

            # Build 17-dim conditioning vector
            day_of_week = patients[0].get('day_of_week', 0)
            conditioning = _build_orchestration_conditioning(
                date_str, day_of_week, stats
            )

            # Parse date for temporal splitting
            try:
                start_datetime = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    start_datetime = datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    continue

            samples.append({
                'tokens': tokens,
                'conditioning': conditioning,
                'scanner_idx': scanner_idx,
                'start_datetime': start_datetime,
                'customer_id': customer_id,
                'date_str': date_str,
                'num_patients': len(patients),
            })

    return samples, scanner_to_idx


def _compute_scanner_stats(customer_schedules):
    """
    Compute per-scanner historical statistics for conditioning.

    Returns dict mapping customer_id -> {
        'avg_patients_per_day': float,
        'region_distribution': np.array of shape [11]
    }
    """
    stats = {}

    for customer_id, daily_schedules in customer_schedules.items():
        patient_counts = []
        region_counts = np.zeros(NUM_BODY_REGIONS, dtype=np.float64)

        for date_str, patients in daily_schedules.items():
            patient_counts.append(len(patients))
            for patient in patients:
                region_id = patient.get('body_region_id', 10)
                if 0 <= region_id < NUM_BODY_REGIONS:
                    region_counts[region_id] += 1

        avg_patients = np.mean(patient_counts) if patient_counts else 0.0
        total_regions = region_counts.sum()
        region_dist = region_counts / total_regions if total_regions > 0 else np.zeros(NUM_BODY_REGIONS)

        stats[customer_id] = {
            'avg_patients_per_day': avg_patients,
            'region_distribution': region_dist,
        }

    return stats


def _build_orchestration_conditioning(date_str, day_of_week, scanner_stats):
    """
    Build 17-dim conditioning vector for an orchestration sample.

    Features:
        [0] dow_sin
        [1] dow_cos
        [2] month_sin
        [3] month_cos
        [4] is_weekend
        [5] avg_patients_per_day
        [6:17] body_region_distribution (11 values)

    Args:
        date_str: Date string (YYYY-MM-DD or YYYYMMDD)
        day_of_week: 0-6 (Monday=0)
        scanner_stats: Dict with 'avg_patients_per_day' and 'region_distribution'

    Returns:
        np.array of shape [17]
    """
    # Parse month from date string
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            dt = datetime(2024, 1, 1)  # fallback

    month = dt.month  # 1-12

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    is_weekend = 1.0 if day_of_week >= 5 else 0.0

    avg_patients = scanner_stats['avg_patients_per_day']
    region_dist = scanner_stats['region_distribution']

    conditioning = np.zeros(17, dtype=np.float32)
    conditioning[0] = dow_sin
    conditioning[1] = dow_cos
    conditioning[2] = month_sin
    conditioning[3] = month_cos
    conditioning[4] = is_weekend
    conditioning[5] = avg_patients
    conditioning[6:17] = region_dist

    return conditioning


def build_demographic_distributions(preprocessed_data):
    """
    Compute per-body-region demographic statistics for sampling patient
    features during orchestrated simulation.

    Args:
        preprocessed_data: Dict from preprocessed_data.pkl

    Returns:
        Dict mapping body_region_id -> {
            'age_mean', 'age_std',
            'weight_mean', 'weight_std',
            'height_mean', 'height_std',
            'direction_prob': probability of Head First (0.0-1.0),
            'count': int
        }
    """
    customer_schedules = preprocessed_data.get('customer_schedules', {})

    region_demographics = defaultdict(lambda: {
        'ages': [], 'weights': [], 'heights': [], 'directions': []
    })

    for customer_id, daily_schedules in customer_schedules.items():
        for date_str, patients in daily_schedules.items():
            for patient in patients:
                region_id = patient.get('body_region_id', 10)

                age = patient.get('age', 0)
                weight = patient.get('weight', 0)
                height = patient.get('height', 0)
                direction = patient.get('direction', 'Head First')

                if age > 0:
                    region_demographics[region_id]['ages'].append(age)
                if weight > 0:
                    region_demographics[region_id]['weights'].append(weight)
                if height > 0:
                    region_demographics[region_id]['heights'].append(height)

                is_head_first = 1.0 if direction == 'Head First' else 0.0
                region_demographics[region_id]['directions'].append(is_head_first)

    distributions = {}

    for region_id in range(NUM_BODY_REGIONS):
        data = region_demographics[region_id]

        ages = np.array(data['ages']) if data['ages'] else np.array([50.0])
        weights = np.array(data['weights']) if data['weights'] else np.array([75.0])
        heights = np.array(data['heights']) if data['heights'] else np.array([1.75])
        directions = np.array(data['directions']) if data['directions'] else np.array([1.0])

        distributions[region_id] = {
            'age_mean': float(np.mean(ages)),
            'age_std': float(np.std(ages)) if len(ages) > 1 else 10.0,
            'weight_mean': float(np.mean(weights)),
            'weight_std': float(np.std(weights)) if len(weights) > 1 else 15.0,
            'height_mean': float(np.mean(heights)),
            'height_std': float(np.std(heights)) if len(heights) > 1 else 0.1,
            'direction_prob': float(np.mean(directions)),
            'count': len(data['ages']),
        }

    return distributions


if __name__ == "__main__":
    from data.preprocessing import load_preprocessed_data

    print("Testing Orchestration Preprocessing...")
    print("=" * 60)

    preprocessed = load_preprocessed_data()

    samples, scanner_to_idx = extract_orchestration_samples(preprocessed)
    print(f"\nExtracted {len(samples)} orchestration samples")
    print(f"Number of scanners: {len(scanner_to_idx)}")

    if samples:
        # Statistics
        seq_lengths = [len(s['tokens']) for s in samples]
        patient_counts = [s['num_patients'] for s in samples]
        break_counts = [s['tokens'].count(BREAK_TOKEN_ID) for s in samples]

        print(f"\nSequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
              f"avg={np.mean(seq_lengths):.1f}")
        print(f"Patients per day: min={min(patient_counts)}, max={max(patient_counts)}, "
              f"avg={np.mean(patient_counts):.1f}")
        print(f"BREAKs per day: min={min(break_counts)}, max={max(break_counts)}, "
              f"avg={np.mean(break_counts):.1f}")

        # Body region distribution across all samples
        all_tokens = []
        for s in samples:
            all_tokens.extend([t for t in s['tokens'] if t < NUM_BODY_REGIONS])
        if all_tokens:
            counts = np.bincount(all_tokens, minlength=NUM_BODY_REGIONS)
            print(f"\nBody region distribution:")
            for i, region in enumerate(BODY_REGIONS):
                if counts[i] > 0:
                    print(f"  {region}: {counts[i]} ({100*counts[i]/len(all_tokens):.1f}%)")

        # Sample conditioning
        print(f"\nSample conditioning shape: {samples[0]['conditioning'].shape}")
        print(f"Sample conditioning: {samples[0]['conditioning']}")

    # Test demographic distributions
    demographics = build_demographic_distributions(preprocessed)
    print(f"\nDemographic distributions for {len(demographics)} body regions:")
    for region_id, stats in sorted(demographics.items()):
        if stats['count'] > 0:
            print(f"  {BODY_REGIONS[region_id]}: n={stats['count']}, "
                  f"age={stats['age_mean']:.1f}+/-{stats['age_std']:.1f}, "
                  f"weight={stats['weight_mean']:.1f}+/-{stats['weight_std']:.1f}")
