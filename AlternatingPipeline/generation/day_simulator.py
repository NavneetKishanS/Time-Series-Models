"""
Day Simulator: Generate a full day schedule using on-the-fly model inference.

No bucket dependency. Both exchange and examination models generate sequences
directly during simulation.

Flow for each patient:
  1. EXCHANGE: model.generate(cond, {body_from, body_to}, phase_type)
  2. EXAMINATION: model.generate(cond, {body_region})
Final: EXCHANGE shutdown (phase_type=2, body_to=END)
"""
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    START_REGION_ID, END_REGION_ID, BODY_REGIONS, ID_TO_SOURCEID,
    BODY_REGION_TO_ID, OUTPUT_DIR, GENERATION_CONFIG, PHASE_TYPES,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID,
    BREAK_TOKEN_ID, NUM_BODY_REGIONS,
)


def build_conditioning_tensor(patient_info, current_time, day_start):
    """
    Build a 10-dim conditioning tensor from patient info and current simulation time.

    Args:
        patient_info: Dict with patient features (age, weight, height, direction, etc.)
        current_time: Current time offset in seconds from day_start
        day_start: datetime of day start

    Returns:
        torch.Tensor of shape [10]
    """
    # Calculate current datetime
    current_dt = day_start + timedelta(seconds=current_time)
    hour = current_dt.hour + current_dt.minute / 60.0
    dow = current_dt.weekday()

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    is_morning = 1.0 if hour < 12 else 0.0

    direction = patient_info.get('direction', 'Head First')
    if isinstance(direction, str):
        direction_encoded = 0.0 if direction == 'Head First' else 1.0
    else:
        direction_encoded = float(direction)

    return torch.tensor([
        float(patient_info.get('age', 50)),
        float(patient_info.get('weight', 75)),
        float(patient_info.get('height', 1.75)),
        float(patient_info.get('ptab', patient_info.get('PTAB', 0))),
        direction_encoded,
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
        is_morning,
    ], dtype=torch.float32)


class DaySimulator:
    """
    Simulates a complete day using on-the-fly model inference.

    Usage:
        simulator = DaySimulator(exchange_model, examination_model, device)
        schedule = simulator.simulate_day(ground_truth_patients)
    """

    def __init__(self, exchange_model, examination_model, device=None):
        """
        Args:
            exchange_model: Trained SequenceGeneratorModel (exchange config)
            examination_model: Trained SequenceGeneratorModel (examination config)
            device: torch device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.exchange_model = exchange_model.to(device)
        self.exchange_model.eval()

        self.examination_model = examination_model.to(device)
        self.examination_model.eval()

        self.gen_config = GENERATION_CONFIG

    def simulate_day(self, ground_truth_patients, start_time=None):
        """
        Generate a full day schedule from a ground truth patient sequence.

        Args:
            ground_truth_patients: List of patient dicts with keys:
                - 'patient_id', 'body_region' (str or int)
                - Optional: 'age', 'weight', 'height', 'direction'
            start_time: Start datetime (default: today 07:00)

        Returns:
            List of event dicts representing the full day schedule
        """
        if start_time is None:
            today = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
            start_time = today

        schedule = []
        current_time = 0.0
        previous_body_region = START_REGION_ID
        event_id = 0

        for patient_idx, patient in enumerate(ground_truth_patients):
            patient_id = patient.get('patient_id', f'PAT{patient_idx:03d}')

            # Get body region ID
            body_region = patient.get('body_region')
            if isinstance(body_region, str):
                body_region_id = BODY_REGION_TO_ID.get(body_region.upper(), 10)
            else:
                body_region_id = body_region

            # === EXCHANGE PHASE ===
            phase_type = PHASE_TYPES['startup'] if patient_idx == 0 else PHASE_TYPES['between']

            cond = build_conditioning_tensor(patient, current_time, start_time).to(self.device)

            with torch.no_grad():
                exchange_tokens, exchange_durations = self.exchange_model.generate(
                    cond,
                    {'body_from': previous_body_region, 'body_to': body_region_id},
                    phase_type=phase_type,
                    max_length=self.gen_config['max_length'],
                    temperature=self.gen_config['temperature'],
                    top_k=self.gen_config['top_k'],
                    top_p=self.gen_config['top_p'],
                )

            exchange_events = self._create_exchange_events(
                exchange_tokens[0], exchange_durations[0],
                event_id, current_time, start_time,
                patient_id, patient_idx,
                previous_body_region, body_region_id
            )
            schedule.extend(exchange_events)
            event_id += len(exchange_events)
            current_time += exchange_durations[0].sum().item()

            # === EXAMINATION PHASE ===
            cond = build_conditioning_tensor(patient, current_time, start_time).to(self.device)

            with torch.no_grad():
                exam_tokens, exam_durations = self.examination_model.generate(
                    cond,
                    {'body_region': body_region_id},
                    max_length=self.gen_config['max_length'],
                    temperature=self.gen_config['temperature'],
                    top_k=self.gen_config['top_k'],
                    top_p=self.gen_config['top_p'],
                )

            exam_events = self._create_examination_events(
                exam_tokens[0], exam_durations[0],
                event_id, current_time, start_time,
                patient_id, patient_idx, body_region_id
            )
            schedule.extend(exam_events)
            event_id += len(exam_events)
            current_time += exam_durations[0].sum().item()

            previous_body_region = body_region_id

        # === FINAL EXCHANGE (shutdown) ===
        if ground_truth_patients:
            last_patient = ground_truth_patients[-1]
            cond = build_conditioning_tensor(last_patient, current_time, start_time).to(self.device)

            with torch.no_grad():
                shutdown_tokens, shutdown_durations = self.exchange_model.generate(
                    cond,
                    {'body_from': previous_body_region, 'body_to': END_REGION_ID},
                    phase_type=PHASE_TYPES['shutdown'],
                    max_length=self.gen_config['max_length'],
                    temperature=self.gen_config['temperature'],
                    top_k=self.gen_config['top_k'],
                    top_p=self.gen_config['top_p'],
                )

            shutdown_events = self._create_exchange_events(
                shutdown_tokens[0], shutdown_durations[0],
                event_id, current_time, start_time,
                None, len(ground_truth_patients),
                previous_body_region, END_REGION_ID
            )
            schedule.extend(shutdown_events)

        return schedule

    def simulate_day_from_orchestration(self, orchestration_tokens,
                                         demographic_distributions,
                                         start_time=None):
        """
        Generate a full day schedule from orchestration model output.

        No ground truth needed — the orchestration tokens define the patient
        sequence and breaks.

        Args:
            orchestration_tokens: List or tensor of token IDs
                e.g. [START, HEAD, SPINE, BREAK, PELVIS, ..., END]
            demographic_distributions: Dict mapping body_region_id -> {
                'age_mean', 'age_std', 'weight_mean', 'weight_std',
                'height_mean', 'height_std', 'direction_prob'
            }
            start_time: Start datetime (default: today 08:00)

        Returns:
            List of event dicts representing the full day schedule
        """
        if start_time is None:
            today = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            start_time = today

        # Convert to plain list
        if isinstance(orchestration_tokens, torch.Tensor):
            orchestration_tokens = orchestration_tokens.cpu().tolist()

        # Strip START/END/PAD, keep only body region IDs and BREAK tokens
        tokens = [
            t for t in orchestration_tokens
            if t not in (START_REGION_ID, END_REGION_ID)
            and t != self.gen_config.get('orch_pad_token_id', 14)
        ]
        # Also filter out the orchestration PAD token (14)
        tokens = [t for t in tokens if t != 14]

        if not tokens:
            return []

        schedule = []
        current_time = 0.0
        previous_body_region = START_REGION_ID
        event_id = 0
        patient_idx = 0

        for i, token in enumerate(tokens):
            if token == BREAK_TOKEN_ID:
                # === BREAK: run exchange model with phase_type='between' ===
                # Uses previous body region for both from/to
                patient_info = self._sample_patient_demographics(
                    previous_body_region, demographic_distributions
                )
                cond = build_conditioning_tensor(
                    patient_info, current_time, start_time
                ).to(self.device)

                with torch.no_grad():
                    break_tokens, break_durations = self.exchange_model.generate(
                        cond,
                        {'body_from': previous_body_region,
                         'body_to': previous_body_region},
                        phase_type=PHASE_TYPES['between'],
                        max_length=self.gen_config['max_length'],
                        temperature=self.gen_config['temperature'],
                        top_k=self.gen_config['top_k'],
                        top_p=self.gen_config['top_p'],
                    )

                break_events = self._create_exchange_events(
                    break_tokens[0], break_durations[0],
                    event_id, current_time, start_time,
                    None, f'BREAK_{patient_idx}',
                    previous_body_region, previous_body_region,
                )
                schedule.extend(break_events)
                event_id += len(break_events)
                current_time += break_durations[0].sum().item()

            elif 0 <= token < NUM_BODY_REGIONS:
                # === PATIENT: exchange + examination ===
                body_region_id = token
                patient_id = f'PAT{patient_idx:03d}'

                patient_info = self._sample_patient_demographics(
                    body_region_id, demographic_distributions
                )

                # Exchange phase
                phase_type = (
                    PHASE_TYPES['startup'] if patient_idx == 0
                    else PHASE_TYPES['between']
                )
                cond = build_conditioning_tensor(
                    patient_info, current_time, start_time
                ).to(self.device)

                with torch.no_grad():
                    exchange_tokens, exchange_durations = self.exchange_model.generate(
                        cond,
                        {'body_from': previous_body_region,
                         'body_to': body_region_id},
                        phase_type=phase_type,
                        max_length=self.gen_config['max_length'],
                        temperature=self.gen_config['temperature'],
                        top_k=self.gen_config['top_k'],
                        top_p=self.gen_config['top_p'],
                    )

                exchange_events = self._create_exchange_events(
                    exchange_tokens[0], exchange_durations[0],
                    event_id, current_time, start_time,
                    patient_id, patient_idx,
                    previous_body_region, body_region_id,
                )
                schedule.extend(exchange_events)
                event_id += len(exchange_events)
                current_time += exchange_durations[0].sum().item()

                # Examination phase
                cond = build_conditioning_tensor(
                    patient_info, current_time, start_time
                ).to(self.device)

                with torch.no_grad():
                    exam_tokens, exam_durations = self.examination_model.generate(
                        cond,
                        {'body_region': body_region_id},
                        max_length=self.gen_config['max_length'],
                        temperature=self.gen_config['temperature'],
                        top_k=self.gen_config['top_k'],
                        top_p=self.gen_config['top_p'],
                    )

                exam_events = self._create_examination_events(
                    exam_tokens[0], exam_durations[0],
                    event_id, current_time, start_time,
                    patient_id, patient_idx, body_region_id,
                )
                schedule.extend(exam_events)
                event_id += len(exam_events)
                current_time += exam_durations[0].sum().item()

                previous_body_region = body_region_id
                patient_idx += 1

        # === FINAL EXCHANGE (shutdown) ===
        if patient_idx > 0:
            patient_info = self._sample_patient_demographics(
                previous_body_region, demographic_distributions
            )
            cond = build_conditioning_tensor(
                patient_info, current_time, start_time
            ).to(self.device)

            with torch.no_grad():
                shutdown_tokens, shutdown_durations = self.exchange_model.generate(
                    cond,
                    {'body_from': previous_body_region, 'body_to': END_REGION_ID},
                    phase_type=PHASE_TYPES['shutdown'],
                    max_length=self.gen_config['max_length'],
                    temperature=self.gen_config['temperature'],
                    top_k=self.gen_config['top_k'],
                    top_p=self.gen_config['top_p'],
                )

            shutdown_events = self._create_exchange_events(
                shutdown_tokens[0], shutdown_durations[0],
                event_id, current_time, start_time,
                None, patient_idx,
                previous_body_region, END_REGION_ID,
            )
            schedule.extend(shutdown_events)

        return schedule

    def _sample_patient_demographics(self, body_region_id, demographic_distributions):
        """
        Sample patient demographics from distributions for a given body region.

        Args:
            body_region_id: int (0-10)
            demographic_distributions: Dict from build_demographic_distributions()

        Returns:
            Dict with patient features for conditioning
        """
        if body_region_id in demographic_distributions:
            stats = demographic_distributions[body_region_id]
        else:
            # Fallback defaults
            stats = {
                'age_mean': 50.0, 'age_std': 15.0,
                'weight_mean': 75.0, 'weight_std': 15.0,
                'height_mean': 1.75, 'height_std': 0.1,
                'direction_prob': 0.8,
            }

        age = np.clip(np.random.normal(stats['age_mean'], stats['age_std']), 1, 100)
        weight = np.clip(np.random.normal(stats['weight_mean'], stats['weight_std']), 20, 200)
        height = np.clip(np.random.normal(stats['height_mean'], stats['height_std']), 0.5, 2.5)
        direction = 'Head First' if np.random.random() < stats['direction_prob'] else 'Feet First'

        return {
            'age': age,
            'weight': weight,
            'height': height,
            'ptab': 0,
            'direction': direction,
        }

    def _create_exchange_events(self, tokens, durations, start_event_id,
                                start_time_offset, day_start,
                                patient_id, session_id, body_from, body_to):
        """Create event dicts for an exchange phase."""
        events = []
        current_offset = start_time_offset

        tokens = tokens.cpu().tolist()
        durations = durations.cpu().tolist()

        for i, token in enumerate(tokens):
            if token in [START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID]:
                continue

            source_id = ID_TO_SOURCEID.get(token, 'UNK')
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
                'cumulative_time': current_offset + duration,
            }

            events.append(event)
            current_offset += duration

        return events

    def _create_examination_events(self, tokens, durations, start_event_id,
                                    start_time_offset, day_start,
                                    patient_id, session_id, body_region):
        """Create event dicts for an examination phase."""
        events = []
        current_offset = start_time_offset

        tokens = tokens.cpu().tolist()
        durations = durations.cpu().tolist()

        for i, token in enumerate(tokens):
            if token in [START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID]:
                continue

            source_id = ID_TO_SOURCEID.get(token, 'UNK')
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
                'cumulative_time': current_offset + duration,
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
        """Save generated schedule to CSV."""
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
        """Create a sample ground truth patient sequence for testing."""
        patients = []

        for i in range(num_patients):
            body_region = np.random.choice(BODY_REGIONS[:6])
            patient = {
                'patient_id': f'PAT{i:03d}',
                'body_region': body_region,
                'age': np.random.randint(20, 80),
                'weight': np.random.uniform(50, 120),
                'height': np.random.uniform(1.5, 2.0),
                'direction': np.random.choice(['Head First', 'Feet First']),
            }
            patients.append(patient)

        return patients


if __name__ == "__main__":
    print("Testing Day Simulator (on-the-fly generation)...")
    print("=" * 60)
    print("\nTo simulate a day, load trained models first:")
    print("  1. Train Exchange and Examination models")
    print("  2. Load both models")
    print("  3. simulator = DaySimulator(exchange_model, exam_model, device)")
    print("  4. schedule = simulator.simulate_day(ground_truth)")
