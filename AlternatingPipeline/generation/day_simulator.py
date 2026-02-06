"""
Day Simulator: Generate a full day schedule by sampling from pre-generated buckets.

From the meeting transcript:
"To generate a day of data for a given ground truth patient sequence,
you just go through the transitions: start to brain, brain to knee, knee to head...
And for any of these transitions, take either the examination sample or exchange sample
depending on whether there is currently an exchange phase or an examination."

This implements the sequential alternating approach:
Exchange -> Examination -> Exchange -> Examination -> ... -> Exchange (final)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    START_REGION_ID, END_REGION_ID, BODY_REGIONS, ID_TO_SOURCEID,
    BODY_REGION_TO_ID, OUTPUT_DIR
)
from generation.bucket_generator import BucketGenerator


class DaySimulator:
    """
    Simulates a complete day by sampling from pre-generated buckets.

    Usage:
        simulator = DaySimulator(bucket_generator)
        schedule = simulator.simulate_day(ground_truth_patients)
        simulator.save_schedule(schedule, 'generated_day.csv')
    """

    def __init__(self, bucket_generator=None, buckets_dir=None):
        """
        Initialize the day simulator.

        Args:
            bucket_generator: BucketGenerator instance with loaded buckets
            buckets_dir: Directory to load buckets from (if bucket_generator is None)
        """
        if bucket_generator is not None:
            self.buckets = bucket_generator
        else:
            self.buckets = BucketGenerator()
            if buckets_dir is not None:
                self.buckets.load_buckets(buckets_dir)

    def simulate_day(self, ground_truth_patients, start_time=None):
        """
        Generate a full day schedule from a ground truth patient sequence.

        Args:
            ground_truth_patients: List of patient dicts with keys:
                - 'patient_id': Unique patient identifier
                - 'body_region': Body region to be examined (str or int)
                - Optional: 'age', 'weight', 'height', 'direction'
            start_time: Start datetime (default: today 07:00)

        Returns:
            List of event dicts representing the full day schedule
        """
        if start_time is None:
            today = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
            start_time = today

        schedule = []
        current_time = 0.0  # Time in seconds from start
        previous_body_region = START_REGION_ID
        event_id = 0

        for patient_idx, patient in enumerate(ground_truth_patients):
            # Parse patient info
            patient_id = patient.get('patient_id', f'PAT{patient_idx:03d}')

            # Get body region ID
            body_region = patient.get('body_region')
            if isinstance(body_region, str):
                body_region_id = BODY_REGION_TO_ID.get(body_region.upper(), 10)
            else:
                body_region_id = body_region

            # === EXCHANGE PHASE ===
            # Transition from previous body region to current patient's body region
            exchange_sample = self.buckets.get_exchange_sample(previous_body_region, body_region_id)

            if exchange_sample is not None:
                # Add exchange events to schedule
                exchange_events = self._create_exchange_events(
                    exchange_sample,
                    event_id,
                    current_time,
                    start_time,
                    patient_id,
                    patient_idx,
                    previous_body_region,
                    body_region_id
                )
                schedule.extend(exchange_events)
                event_id += len(exchange_events)
                # Use sum of per-token durations (from real data), fall back to total duration
                exchange_duration = sum(exchange_sample.get('durations', []))
                if exchange_duration == 0:
                    exchange_duration = exchange_sample.get('duration', 60)
                current_time += exchange_duration

            # === EXAMINATION PHASE ===
            # Generate examination events for this body region
            examination_sample = self.buckets.get_examination_sample(body_region_id)

            if examination_sample is not None:
                # Add examination events to schedule
                exam_events = self._create_examination_events(
                    examination_sample,
                    event_id,
                    current_time,
                    start_time,
                    patient_id,
                    patient_idx,
                    body_region_id
                )
                schedule.extend(exam_events)
                event_id += len(exam_events)
                # Use sum of per-token durations, fall back to total_duration
                exam_duration = sum(examination_sample.get('durations', []))
                if exam_duration == 0:
                    exam_duration = examination_sample.get('total_duration', 300)
                current_time += exam_duration

            # Update previous body region for next exchange
            previous_body_region = body_region_id

        # === FINAL EXCHANGE ===
        # Transition from last patient's body region to END
        final_exchange = self.buckets.get_exchange_sample(previous_body_region, END_REGION_ID)

        if final_exchange is not None:
            final_events = self._create_exchange_events(
                final_exchange,
                event_id,
                current_time,
                start_time,
                None,  # No patient for final exchange
                len(ground_truth_patients),
                previous_body_region,
                END_REGION_ID
            )
            schedule.extend(final_events)

        return schedule

    def _create_exchange_events(self, sample, start_event_id, start_time_offset,
                                 day_start, patient_id, session_id,
                                 body_from, body_to):
        """Create event dicts for an exchange phase."""
        events = []
        current_offset = start_time_offset

        # Use sample sequence if available, otherwise create placeholder
        sequence = sample.get('sequence', [])
        durations = sample.get('durations', [])

        # If no per-token durations, distribute total duration evenly
        if not durations or len(durations) == 0:
            total = sample.get('duration', 60)
            seq_len = max(len(sequence), 1)
            durations = [total / seq_len] * seq_len

        # Ensure durations match sequence length
        if len(durations) < len(sequence):
            durations = durations + [5.0] * (len(sequence) - len(durations))

        for i, token in enumerate(sequence):
            # Skip START/END tokens in output
            if token in [START_REGION_ID, END_REGION_ID]:
                continue

            source_id = ID_TO_SOURCEID.get(token, 'UNK') if isinstance(token, int) else token
            duration = durations[i] if i < len(durations) else 5.0

            event = {
                'event_id': start_event_id + len(events),
                'timestamp': current_offset,
                'datetime': (day_start + timedelta(seconds=current_offset)).isoformat(),
                'event_type': 'exchange',
                'patient_id': patient_id,
                'session_id': session_id,
                'sourceID': source_id,
                'scan_sequence': None,
                'body_region': None,
                'body_from': self._region_id_to_name(body_from),
                'body_to': self._region_id_to_name(body_to),
                'duration': duration,
                'cumulative_time': current_offset + duration
            }

            events.append(event)
            current_offset += duration

        return events

    def _create_examination_events(self, sample, start_event_id, start_time_offset,
                                    day_start, patient_id, session_id, body_region):
        """Create event dicts for an examination phase."""
        events = []
        current_offset = start_time_offset

        sequence = sample.get('sequence', [])
        sequence_sourceids = sample.get('sequence_sourceids', [])
        durations = sample.get('durations', [])

        # Ensure we have source IDs
        if not sequence_sourceids and sequence:
            sequence_sourceids = [ID_TO_SOURCEID.get(t, 'UNK') for t in sequence]

        # Ensure durations match
        if len(durations) < len(sequence_sourceids):
            durations = durations + [30.0] * (len(sequence_sourceids) - len(durations))

        for i, source_id in enumerate(sequence_sourceids):
            # Skip padding and special tokens
            if source_id in ['PAD', 'START', 'END', 'UNK']:
                continue

            duration = durations[i] if i < len(durations) else 30.0

            event = {
                'event_id': start_event_id + len(events),
                'timestamp': current_offset,
                'datetime': (day_start + timedelta(seconds=current_offset)).isoformat(),
                'event_type': 'examination',
                'patient_id': patient_id,
                'session_id': session_id,
                'sourceID': source_id,
                'scan_sequence': source_id if source_id == 'MRI_EXU_95' else None,
                'body_region': self._region_id_to_name(body_region),
                'body_from': None,
                'body_to': None,
                'duration': duration,
                'cumulative_time': current_offset + duration
            }

            events.append(event)
            current_offset += duration

        return events

    def _region_id_to_name(self, region_id):
        """Convert body region ID to name."""
        if region_id == START_REGION_ID:
            return 'START'
        elif region_id == END_REGION_ID:
            return 'END'
        elif region_id < len(BODY_REGIONS):
            return BODY_REGIONS[region_id]
        else:
            return f'UNKNOWN_{region_id}'

    def save_schedule(self, schedule, filename=None, output_dir=None):
        """
        Save generated schedule to CSV.

        Args:
            schedule: List of event dicts from simulate_day()
            filename: Output filename (default: generated_schedule_TIMESTAMP.csv)
            output_dir: Output directory (default: OUTPUT_DIR)
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'generated_schedule_{timestamp}.csv'

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        df = pd.DataFrame(schedule)
        df.to_csv(filepath, index=False)

        print(f"Saved schedule to {filepath}")
        print(f"Total events: {len(schedule)}")

        return filepath

    def create_sample_ground_truth(self, num_patients=10):
        """
        Create a sample ground truth patient sequence for testing.

        Args:
            num_patients: Number of patients to generate

        Returns:
            List of patient dicts
        """
        patients = []

        for i in range(num_patients):
            # Random body region
            body_region = np.random.choice(BODY_REGIONS[:6])  # Most common regions

            patient = {
                'patient_id': f'PAT{i:03d}',
                'body_region': body_region,
                'age': np.random.randint(20, 80),
                'weight': np.random.uniform(50, 120),
                'height': np.random.uniform(1.5, 2.0),
                'direction': np.random.choice(['Head First', 'Feet First'])
            }

            patients.append(patient)

        return patients


if __name__ == "__main__":
    print("Testing Day Simulator...")
    print("=" * 60)

    # Initialize simulator (without loaded buckets for testing)
    simulator = DaySimulator()

    # Create sample ground truth
    ground_truth = simulator.create_sample_ground_truth(num_patients=5)

    print("\nSample ground truth patients:")
    for i, patient in enumerate(ground_truth):
        print(f"  {i+1}. {patient['patient_id']}: {patient['body_region']} "
              f"(age={patient['age']}, weight={patient['weight']:.1f}kg)")

    print("\nTo simulate a day, load pre-generated buckets first:")
    print("  1. Train Exchange and Examination models")
    print("  2. Generate buckets with BucketGenerator")
    print("  3. Call simulator.simulate_day(ground_truth)")
