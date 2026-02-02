"""
Bucket Generator for pre-generating samples.

From the meeting transcript:
"You can just think about body from->to buckets. Produce 1000 samples per bucket.
To generate a day, you don't need to rerun models - just pick random samples
from already generated buckets."

This module pre-generates samples for:
- Exchange buckets: 1000 samples per body region transition (e.g., HEAD->CHEST)
- Examination buckets: 1000 samples per body region (e.g., HEAD examinations)
"""
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BUCKETS_DIR, BUCKET_SIZE, NUM_REGION_CLASSES, NUM_BODY_REGIONS,
    BODY_REGIONS, START_REGION_ID, END_REGION_ID, ID_TO_SOURCEID,
    GENERATION_CONFIG, DURATION_MULTIPLIER,
    EXCHANGE_DURATION_SHAPE, EXCHANGE_DURATION_SCALE,
    EXAMINATION_DURATION_SHAPE, EXAMINATION_DURATION_SCALE,
    EXCLUDED_BODY_REGION_IDS, VALID_BODY_REGION_IDS
)


class BucketGenerator:
    """
    Generates and manages pre-computed sample buckets.

    Usage:
        generator = BucketGenerator(exchange_model, examination_model)
        generator.generate_all_buckets()
        generator.save_buckets()
    """

    def __init__(self, exchange_model=None, examination_model=None, device=None):
        """
        Initialize the bucket generator.

        Args:
            exchange_model: Trained ExchangeModel (or None to load buckets only)
            examination_model: Trained ExaminationModel (or None to load buckets only)
            device: torch device
        """
        self.exchange_model = exchange_model
        self.examination_model = examination_model

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if exchange_model is not None:
            self.exchange_model = exchange_model.to(device)
            self.exchange_model.eval()

        if examination_model is not None:
            self.examination_model = examination_model.to(device)
            self.examination_model.eval()

        # Bucket storage
        self.exchange_buckets = {}  # {(from_region, to_region): [samples]}
        self.examination_buckets = {}  # {body_region: [samples]}

    def _sample_conditioning(self):
        """
        Sample random conditioning features for generation.

        Returns typical patient demographics.
        """
        return {
            'Age': np.random.uniform(20, 80),
            'Weight': np.random.uniform(50, 120),
            'Height': np.random.uniform(1.5, 2.0),
            'PTAB': np.random.uniform(-2000000, 0),
            'Direction_encoded': np.random.choice([0, 1])  # Head First / Feet First
        }

    def _conditioning_to_tensor(self, conditioning):
        """Convert conditioning dict to tensor."""
        return torch.tensor([
            conditioning['Age'],
            conditioning['Weight'],
            conditioning['Height'],
            conditioning['PTAB'],
            conditioning['Direction_encoded']
        ], dtype=torch.float32)

    def generate_exchange_bucket(self, body_from, body_to, num_samples=None):
        """
        Generate samples for a specific exchange (body region transition).

        Args:
            body_from: Source body region ID
            body_to: Target body region ID
            num_samples: Number of samples (default: BUCKET_SIZE)

        Returns:
            List of sample dicts
        """
        if num_samples is None:
            num_samples = BUCKET_SIZE

        if self.exchange_model is None:
            raise ValueError("Exchange model not loaded")

        samples = []

        for _ in range(num_samples):
            conditioning = self._sample_conditioning()
            cond_tensor = self._conditioning_to_tensor(conditioning).unsqueeze(0).to(self.device)
            current_region = torch.tensor([body_from], device=self.device)

            # Predict next region (should be body_to if model is well-trained)
            # For bucket generation, we force the transition to the target
            with torch.no_grad():
                # Get the model's prediction for the transition
                logits = self.exchange_model(cond_tensor, current_region)
                probs = torch.softmax(logits, dim=-1)

            # Create sample with the forced transition
            # Duration is scaled by DURATION_MULTIPLIER to correct for time compression
            # in training data (model was trained on 1-2 weeks compressed to single day)
            sample = {
                'body_from': body_from,
                'body_to': body_to,
                'conditioning': conditioning,
                'transition_prob': probs[0, body_to].item(),
                'sequence': [START_REGION_ID, body_to],  # Simplified sequence
                'duration': np.random.gamma(EXCHANGE_DURATION_SHAPE, EXCHANGE_DURATION_SCALE) * DURATION_MULTIPLIER
            }

            samples.append(sample)

        return samples

    def generate_examination_bucket(self, body_region, num_samples=None):
        """
        Generate samples for a specific body region examination.

        Args:
            body_region: Body region ID (0-10)
            num_samples: Number of samples (default: BUCKET_SIZE)

        Returns:
            List of sample dicts
        """
        if num_samples is None:
            num_samples = BUCKET_SIZE

        if self.examination_model is None:
            raise ValueError("Examination model not loaded")

        samples = []
        config = GENERATION_CONFIG

        for _ in range(num_samples):
            conditioning = self._sample_conditioning()
            cond_tensor = self._conditioning_to_tensor(conditioning).unsqueeze(0).to(self.device)
            region_tensor = torch.tensor([body_region], device=self.device)

            # Generate sequence
            with torch.no_grad():
                generated = self.examination_model.generate(
                    cond_tensor,
                    region_tensor,
                    max_length=config['max_length'],
                    temperature=config['temperature'],
                    top_k=config['top_k'],
                    top_p=config['top_p']
                )

            # Convert to list and decode
            sequence = generated[0].cpu().tolist()

            # Convert token IDs to sourceIDs
            sequence_sourceids = [ID_TO_SOURCEID.get(t, 'UNK') for t in sequence]

            # Generate random durations for each token
            # Duration is scaled by DURATION_MULTIPLIER to correct for time compression
            durations = [np.random.gamma(EXAMINATION_DURATION_SHAPE, EXAMINATION_DURATION_SCALE) * DURATION_MULTIPLIER
                         for _ in sequence]

            sample = {
                'body_region': body_region,
                'conditioning': conditioning,
                'sequence': sequence,
                'sequence_sourceids': sequence_sourceids,
                'durations': durations,
                'total_duration': sum(durations)
            }

            samples.append(sample)

        return samples

    def generate_all_buckets(self, num_samples=None, verbose=True):
        """
        Generate all exchange and examination buckets.

        Args:
            num_samples: Samples per bucket (default: BUCKET_SIZE)
            verbose: Show progress
        """
        if num_samples is None:
            num_samples = BUCKET_SIZE

        # Generate exchange buckets for all valid transitions
        if verbose:
            print("Generating exchange buckets...")
            if EXCLUDED_BODY_REGION_IDS:
                excluded_names = [BODY_REGIONS[i] for i in EXCLUDED_BODY_REGION_IDS]
                print(f"  Excluding body regions: {excluded_names}")

        exchange_transitions = []

        # START -> any valid body region (excluding filtered regions)
        for to_region in VALID_BODY_REGION_IDS:
            exchange_transitions.append((START_REGION_ID, to_region))

        # Any valid body region -> any valid body region (including same)
        for from_region in VALID_BODY_REGION_IDS:
            for to_region in VALID_BODY_REGION_IDS:
                exchange_transitions.append((from_region, to_region))

        # Any valid body region -> END
        for from_region in VALID_BODY_REGION_IDS:
            exchange_transitions.append((from_region, END_REGION_ID))

        if self.exchange_model is not None:
            for body_from, body_to in tqdm(exchange_transitions, disable=not verbose):
                key = (body_from, body_to)
                self.exchange_buckets[key] = self.generate_exchange_bucket(
                    body_from, body_to, num_samples
                )

        # Generate examination buckets for each valid body region
        if verbose:
            print("\nGenerating examination buckets...")

        if self.examination_model is not None:
            for body_region in tqdm(VALID_BODY_REGION_IDS, disable=not verbose):
                self.examination_buckets[body_region] = self.generate_examination_bucket(
                    body_region, num_samples
                )

        if verbose:
            print(f"\nGenerated {len(self.exchange_buckets)} exchange buckets")
            print(f"Generated {len(self.examination_buckets)} examination buckets")

    def save_buckets(self, output_dir=None):
        """
        Save generated buckets to disk.

        Args:
            output_dir: Directory to save buckets (default: BUCKETS_DIR)
        """
        if output_dir is None:
            output_dir = BUCKETS_DIR

        # Save exchange buckets
        exchange_dir = os.path.join(output_dir, 'exchange')
        os.makedirs(exchange_dir, exist_ok=True)

        for (body_from, body_to), samples in self.exchange_buckets.items():
            filename = f"{body_from}_to_{body_to}.pkl"
            filepath = os.path.join(exchange_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(samples, f)

        # Save examination buckets
        examination_dir = os.path.join(output_dir, 'examination')
        os.makedirs(examination_dir, exist_ok=True)

        for body_region, samples in self.examination_buckets.items():
            region_name = BODY_REGIONS[body_region] if body_region < len(BODY_REGIONS) else f"REGION_{body_region}"
            filename = f"{region_name}.pkl"
            filepath = os.path.join(examination_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(samples, f)

        print(f"Saved buckets to {output_dir}")

    def load_buckets(self, input_dir=None):
        """
        Load pre-generated buckets from disk.

        Args:
            input_dir: Directory containing buckets (default: BUCKETS_DIR)
        """
        if input_dir is None:
            input_dir = BUCKETS_DIR

        # Load exchange buckets
        exchange_dir = os.path.join(input_dir, 'exchange')
        if os.path.exists(exchange_dir):
            for filename in os.listdir(exchange_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(exchange_dir, filename)
                    # Parse filename: "11_to_0.pkl" -> (11, 0)
                    parts = filename.replace('.pkl', '').split('_to_')
                    if len(parts) == 2:
                        body_from, body_to = int(parts[0]), int(parts[1])
                        with open(filepath, 'rb') as f:
                            self.exchange_buckets[(body_from, body_to)] = pickle.load(f)

        # Load examination buckets
        examination_dir = os.path.join(input_dir, 'examination')
        if os.path.exists(examination_dir):
            for filename in os.listdir(examination_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(examination_dir, filename)
                    # Parse filename: "HEAD.pkl" -> 0
                    region_name = filename.replace('.pkl', '')
                    if region_name in BODY_REGIONS:
                        body_region = BODY_REGIONS.index(region_name)
                    else:
                        # Try to parse as REGION_X format
                        try:
                            body_region = int(region_name.split('_')[1])
                        except:
                            continue

                    with open(filepath, 'rb') as f:
                        self.examination_buckets[body_region] = pickle.load(f)

        print(f"Loaded {len(self.exchange_buckets)} exchange buckets")
        print(f"Loaded {len(self.examination_buckets)} examination buckets")

    def get_exchange_sample(self, body_from, body_to):
        """
        Get a random sample from an exchange bucket.

        Args:
            body_from: Source body region ID
            body_to: Target body region ID

        Returns:
            Sample dict or None if bucket doesn't exist
        """
        key = (body_from, body_to)
        if key not in self.exchange_buckets or len(self.exchange_buckets[key]) == 0:
            return None

        return np.random.choice(self.exchange_buckets[key])

    def get_examination_sample(self, body_region):
        """
        Get a random sample from an examination bucket.

        Args:
            body_region: Body region ID

        Returns:
            Sample dict or None if bucket doesn't exist
        """
        if body_region not in self.examination_buckets or len(self.examination_buckets[body_region]) == 0:
            return None

        return np.random.choice(self.examination_buckets[body_region])


if __name__ == "__main__":
    print("Testing Bucket Generator...")
    print("=" * 60)

    # Test without models (bucket loading only)
    generator = BucketGenerator()

    print("\nBucket generator initialized (no models loaded)")
    print(f"Exchange buckets: {len(generator.exchange_buckets)}")
    print(f"Examination buckets: {len(generator.examination_buckets)}")

    # Try loading existing buckets
    try:
        generator.load_buckets()
    except Exception as e:
        print(f"No existing buckets found: {e}")

    print("\nTo generate buckets, load trained models and call generate_all_buckets()")
