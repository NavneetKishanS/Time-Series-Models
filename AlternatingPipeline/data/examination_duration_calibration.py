"""
Calibrate examination sequence durations using archive scan statistics.

ONLY fills in genuinely missing/near-zero per-token durations. The in-pipeline
per-token durations are REAL — their sum equals the true span duration and
varies ~74x by scan type (scout ~19 s, tse ~93 s, space ~232 s). The previous
implementation rescaled EVERY sequence's total to the body-region archive mean;
because body_region is currently 100% UNKNOWN, that flattened all durations to
the single 'Unknown' prior (~49 s) and destroyed the scan-type variation the
examination model is supposed to reproduce. We now PRESERVE any sequence whose
durations already sum to a plausible total and only fall back to the archive
prior for empty/near-zero sequences.
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
        total_existing = sum(existing)

        # Plausible-total threshold (seconds). Sequences at/above this keep their
        # real durations; only empty/near-zero ones fall back to the prior.
        MIN_PLAUSIBLE_TOTAL = 3.0

        if total_existing >= MIN_PLAUSIBLE_TOTAL:
            # Real durations — PRESERVE them so scan-type variation survives.
            new_durations = existing
        elif prior is not None:
            archive_mean = prior['mean']
            if total_existing > 0:
                # Tiny but non-zero: scale the existing ratios up to the prior total.
                scale = archive_mean / total_existing
                new_durations = [d * scale for d in existing]
            else:
                # All zeros: distribute the prior mean uniformly.
                per_token = archive_mean / n_tokens
                new_durations = [per_token] * n_tokens
        else:
            new_durations = existing  # no real durations and no archive fallback

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
