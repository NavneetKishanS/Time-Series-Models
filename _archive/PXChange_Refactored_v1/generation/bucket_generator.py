"""
Bucket Generator for Pre-generating MRI Event Sequences

Generates N samples per body region using the Examination and Duration models,
then saves them as pickle files for fast sampling during day simulation.

Usage:
    python bucket_generator.py --num-samples 1000 --output-dir buckets/customer_141049
"""
import os
import sys
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BODY_REGIONS, NUM_BODY_REGIONS, MODEL_SAVE_DIR,
    SOURCEID_VOCAB, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID,
    BUCKET_SIZE
)
from models.examination_model import create_examination_model, ExaminationModel
from models.conditional_counts_generator import ConditionalCountsGenerator


class BucketGenerator:
    """
    Generates pre-computed event sequences and durations for each body region.

    Buckets are stored as pickle files containing lists of (events, durations) tuples.
    """

    def __init__(self, examination_model_path=None, duration_model_path=None,
                 device=None):
        """
        Initialize the bucket generator.

        Args:
            examination_model_path: Path to trained Examination Model checkpoint
            duration_model_path: Path to trained Duration Model checkpoint
            device: Torch device to use
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load models
        self.examination_model = self._load_examination_model(examination_model_path)
        self.duration_model = self._load_duration_model(duration_model_path)

        # Create reverse vocab mapping for token decoding
        self.id_to_token = {v: k for k, v in SOURCEID_VOCAB.items()}

    def _load_examination_model(self, checkpoint_path):
        """Load Examination Model from checkpoint or create new one."""
        model = create_examination_model()
        model = model.to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Examination Model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  Loaded successfully")
        else:
            print("No Examination Model checkpoint found, using random weights")

        model.eval()
        return model

    def _load_duration_model(self, checkpoint_path):
        """Load Duration Model from checkpoint or create new one."""
        model = ConditionalCountsGenerator()
        model = model.to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading Duration Model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  Loaded successfully")
        else:
            print("No Duration Model checkpoint found, using random weights")

        model.eval()
        return model

    def generate_sample(self, body_region, conditioning=None,
                        temperature=1.0, top_k=10, max_length=128):
        """
        Generate a single event sequence with durations for a body region.

        Args:
            body_region: Body region ID (0-10)
            conditioning: Optional conditioning tensor. If None, samples random.
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_length: Maximum sequence length

        Returns:
            events: List of event token IDs (excluding START, including END)
            durations: List of durations in seconds for each event
        """
        # Generate random conditioning if not provided
        if conditioning is None:
            # Conditioning: [Age, Weight, Height, PTAB, entity_type]
            # Normalized values: age~0.5, weight~0.5, height~0.5, ptab~0, entity~0
            conditioning = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.0],
                                         dtype=torch.float32, device=self.device)

        # Ensure conditioning is on device
        if isinstance(conditioning, np.ndarray):
            conditioning = torch.tensor(conditioning, dtype=torch.float32)
        conditioning = conditioning.to(self.device)

        # Generate event sequence
        with torch.no_grad():
            # Generate tokens
            generated = self.examination_model.generate(
                conditioning,
                body_region,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )

            # Remove START token, keep rest including END
            events = generated[0, 1:].cpu().numpy()

            # Remove PAD tokens
            events = [e for e in events if e != PAD_TOKEN_ID]

            if len(events) == 0:
                events = [END_TOKEN_ID]

            # Generate durations
            # Need to prepare inputs for duration model
            event_tensor = torch.tensor([events], dtype=torch.long, device=self.device)

            # Pad to match sequence length
            seq_len = len(events)

            # Create conditioning for duration model (7 features)
            # Add body_region_from and body_region_to to match CONDITIONING_FEATURES
            duration_conditioning = torch.cat([
                conditioning[:3],  # Age, Weight, Height
                torch.tensor([body_region, body_region], dtype=torch.float32, device=self.device),  # from, to
                conditioning[3:5]  # PTAB, entity_type
            ]).unsqueeze(0)

            # Create sequence features (Position_encoded, Direction_encoded) - zeros for now
            sequence_features = torch.zeros(1, seq_len, 2, device=self.device)

            # Create mask
            mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

            # Get duration predictions
            mu, sigma = self.duration_model(
                duration_conditioning,
                event_tensor,
                sequence_features,
                mask
            )

            # Sample durations from Gamma distribution
            samples = self.duration_model.sample_counts(mu, sigma, num_samples=1)
            durations = samples[0, :, 0].cpu().numpy()

            # Scale durations to realistic range (multiply by factor)
            # Typical MRI events are 10-300 seconds
            durations = durations * 100  # Adjust scaling as needed
            durations = np.clip(durations, 0.1, 600)  # Clip to realistic range

        return events.tolist() if isinstance(events, np.ndarray) else events, durations.tolist()

    def generate_bucket(self, body_region, num_samples, conditioning_samples=None,
                        temperature=1.0, top_k=10, max_length=128):
        """
        Generate a bucket of samples for a body region.

        Args:
            body_region: Body region ID (0-10)
            num_samples: Number of samples to generate
            conditioning_samples: Optional list of conditioning arrays to use
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_length: Maximum sequence length

        Returns:
            samples: List of (events, durations) tuples
        """
        samples = []
        region_name = BODY_REGIONS[body_region] if body_region < len(BODY_REGIONS) else f"REGION_{body_region}"

        print(f"Generating {num_samples} samples for {region_name}...")

        for i in tqdm(range(num_samples), desc=f"  {region_name}"):
            # Use provided conditioning or generate random
            if conditioning_samples is not None and i < len(conditioning_samples):
                conditioning = conditioning_samples[i]
            else:
                conditioning = None

            events, durations = self.generate_sample(
                body_region,
                conditioning=conditioning,
                temperature=temperature,
                top_k=top_k,
                max_length=max_length
            )
            samples.append((events, durations))

        return samples

    def generate_all_buckets(self, num_samples=BUCKET_SIZE, output_dir=None,
                             body_regions=None, temperature=1.0, top_k=10):
        """
        Generate buckets for all body regions and save to disk.

        Args:
            num_samples: Number of samples per body region
            output_dir: Directory to save bucket files
            body_regions: Optional list of region IDs to generate. If None, generates all.
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Dict mapping region_name -> bucket file path
        """
        if output_dir is None:
            output_dir = os.path.join(MODEL_SAVE_DIR, 'buckets')
        os.makedirs(output_dir, exist_ok=True)

        if body_regions is None:
            body_regions = list(range(NUM_BODY_REGIONS))

        bucket_paths = {}

        for region_id in body_regions:
            region_name = BODY_REGIONS[region_id] if region_id < len(BODY_REGIONS) else f"REGION_{region_id}"

            # Generate bucket
            samples = self.generate_bucket(
                region_id,
                num_samples,
                temperature=temperature,
                top_k=top_k
            )

            # Save to file
            bucket_path = os.path.join(output_dir, f"{region_name}_bucket.pkl")
            with open(bucket_path, 'wb') as f:
                pickle.dump({
                    'region_id': region_id,
                    'region_name': region_name,
                    'num_samples': len(samples),
                    'samples': samples
                }, f)

            print(f"  Saved {len(samples)} samples to {bucket_path}")
            bucket_paths[region_name] = bucket_path

        return bucket_paths

    def decode_events(self, events):
        """Decode event token IDs to human-readable names."""
        return [self.id_to_token.get(e, f"UNK_{e}") for e in events]


def load_bucket(bucket_path):
    """
    Load a pre-generated bucket from disk.

    Args:
        bucket_path: Path to bucket pickle file

    Returns:
        Dict with 'region_id', 'region_name', 'num_samples', 'samples'
    """
    with open(bucket_path, 'rb') as f:
        return pickle.load(f)


def sample_from_bucket(bucket_data, n=1):
    """
    Sample random entries from a loaded bucket.

    Args:
        bucket_data: Loaded bucket dict
        n: Number of samples to return

    Returns:
        List of (events, durations) tuples
    """
    samples = bucket_data['samples']
    if n == 1:
        idx = np.random.randint(len(samples))
        return samples[idx]
    else:
        indices = np.random.choice(len(samples), size=n, replace=True)
        return [samples[i] for i in indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate event sequence buckets")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples per body region")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for bucket files")
    parser.add_argument("--examination-model", type=str, default=None,
                        help="Path to Examination Model checkpoint")
    parser.add_argument("--duration-model", type=str, default=None,
                        help="Path to Duration Model checkpoint")
    parser.add_argument("--body-regions", type=int, nargs='+', default=None,
                        help="Specific body region IDs to generate (default: all)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k sampling parameter")

    args = parser.parse_args()

    # Create bucket generator
    generator = BucketGenerator(
        examination_model_path=args.examination_model,
        duration_model_path=args.duration_model
    )

    # Generate buckets
    bucket_paths = generator.generate_all_buckets(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        body_regions=args.body_regions,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print(f"\nGenerated {len(bucket_paths)} buckets:")
    for region, path in bucket_paths.items():
        print(f"  {region}: {path}")

    # Test loading and sampling
    print("\nTesting bucket loading...")
    for region, path in list(bucket_paths.items())[:2]:
        bucket = load_bucket(path)
        events, durations = sample_from_bucket(bucket)
        event_names = generator.decode_events(events)
        print(f"  {region}: {len(events)} events, total duration: {sum(durations):.1f}s")
        print(f"    Events: {event_names[:5]}...")
