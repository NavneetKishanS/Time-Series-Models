"""
Calibrate examination sequence durations using archive scan statistics.

Replaces near-zero per-token durations (artifacts of rapid sequential scanner events)
with archive-derived realistic values: archive_mean_for_body_region / num_tokens per token.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ID_TO_BODY_REGION
from data.archive_duration_priors import load_examination_priors


def calibrate_examination_durations(sequences: list, archive_priors: dict = None) -> list:
    """
    Args:
        sequences: list of examination sequence dicts (keys: body_region, sequence, durations, ...)
        archive_priors: {BodyGroup_str: {mean, std, count}} from load_examination_priors()
                        If None, loads automatically.
    Returns:
        New list of sequence dicts with calibrated durations.
    """
    if archive_priors is None:
        archive_priors = load_examination_priors()

    calibrated = []
    for seq in sequences:
        region_id = seq.get('body_region', 10)
        region_name = ID_TO_BODY_REGION.get(region_id, 'UNKNOWN').capitalize()
        prior = archive_priors.get(region_name)

        tokens = seq.get('sequence', [])
        n_tokens = max(1, len(tokens))
        existing = [max(0.0, d) for d in seq.get('durations', [0.0] * n_tokens)]

        if prior is not None:
            # Distribute archive mean uniformly; keep ratio structure from existing if non-zero
            total_existing = sum(existing)
            archive_mean = prior['mean']
            if total_existing > 0:
                # Scale existing duration ratios to hit archive target total
                scale = archive_mean / total_existing
                new_durations = [d * scale for d in existing]
            else:
                # All zeros: distribute uniformly
                per_token = archive_mean / n_tokens
                new_durations = [per_token] * n_tokens
        else:
            new_durations = existing  # no archive data for this region

        new_seq = dict(seq)
        new_seq['durations'] = new_durations
        calibrated.append(new_seq)

    return calibrated


if __name__ == '__main__':
    from data.preprocessing import load_preprocessed_data
    d = load_preprocessed_data()
    cal = calibrate_examination_durations(d['examination'][:50])
    durs = [v for s in cal for v in s['durations']]
    print(f"calibrated mean: {sum(durs)/len(durs):.2f}s, min: {min(durs):.2f}s")
