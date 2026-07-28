"""Protocol vocabulary for examination-duration conditioning.

Every MRI_MSR_100 message carries a `Protocol: '<name>'` field naming the
protocol the operator selected before the scan ran. It has never reached the
model, which conditions instead on a 12-value `sequence_type` bucket derived
from the same message's `Sequence` field.

Measured held out on the real exam CSVs (40,921 rows, 10 serials, group means
fitted on 80% and scored on the other 20%):

    sequence_type (12 values)     R2 31.6%   MAE 53.1s   <- what the model uses
    Sequence raw string (44)      R2 42.2%   MAE 46.6s
    Protocol identity (2,999)     R2 81.7%   MAE 16.2s
    (the trained model itself)               MAE 50.3s

Protocols are site-specific: ~324 per serial, only 4.1% of names appear on more
than one scanner, so the vocabulary is effectively a union of per-site
catalogues and the model needs its existing `serial_idx` conditioning to
disambiguate.

The raw name is deliberately kept alongside the id wherever this is used. It is
what the synthetic CSV's `Protocol` column must carry so Qlik can put real and
synthetic rows on the same dimension, and it is what a later name-text feature
would read without forcing another preprocessing rebuild.
"""

import math
import re
from collections import Counter

import numpy as np

# Reserved for protocols below the frequency floor and for anything unseen at
# serve time. Never assigned to a real protocol, so an embedding row is always
# available as the fallback bucket.
RARE_PROTOCOL_ID = 0
RARE_PROTOCOL_NAME = '<rare>'

_WHITESPACE = re.compile(r'\s+')


def normalize_protocol_name(raw):
    """Collapse whitespace and case so operator-typed variants share one id.

    86 real protocol groups differ only in case (`T2_TSE_SAG` / `t2_tse_SAG` /
    `t2_tse_sag`), which would otherwise split one protocol's training rows
    across three embedding entries. Returns '' for anything missing.
    """
    if raw is None:
        return ''
    if isinstance(raw, float) and math.isnan(raw):
        return ''
    return _WHITESPACE.sub(' ', str(raw)).strip().casefold()


def build_protocol_vocab(protocol_names, min_count=3):
    """Map normalized protocol name -> id, ids starting at 1.

    Protocols seen fewer than `min_count` times stay out of the vocabulary and
    resolve to RARE_PROTOCOL_ID, whose duration signal comes from the
    `sequence_type` conditioning that is deliberately kept alongside this. On
    the real corpus min_count=3 keeps 1,620 protocols covering 95.6% of rows;
    min_count=5 keeps 1,177 covering 91.9%, which is the knob to turn if rare
    protocols overfit.

    Ordering is by descending count then name, so the same corpus always
    produces the same ids regardless of row order — a checkpoint's embedding
    rows stay meaningful across preprocessing re-runs.
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")

    counts = Counter(
        name for name in map(normalize_protocol_name, protocol_names) if name
    )
    kept = sorted(
        (name for name, count in counts.items() if count >= min_count),
        key=lambda name: (-counts[name], name),
    )
    return {name: idx for idx, name in enumerate(kept, start=1)}


def protocol_id(raw_name, vocab):
    """Resolve a raw protocol string to its id, or RARE_PROTOCOL_ID."""
    return vocab.get(normalize_protocol_name(raw_name), RARE_PROTOCOL_ID)


def heldout_group_r2(labels, values, holdout_frac=0.2, seed=0, repeats=5):
    """Score a per-group-mean predictor on data it was not fitted on.

    In-sample variance-explained is meaningless for a grouping this fine: 2,999
    protocols over 40,921 rows scores 89.7% in sample but 81.7% held out, and a
    grouping with one row per group scores a perfect 100% in sample while
    predicting nothing. Group means are therefore fitted on the training split
    only; groups absent from it fall back to the global mean and are counted in
    `coverage`.

    Returns (r2_percent, mae, coverage_percent), averaged over `repeats` splits.
    """
    labels = np.asarray(labels)
    values = np.asarray(values, dtype=float)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("labels and values must be the same length")
    if labels.shape[0] < 2:
        raise ValueError("need at least 2 rows to hold any out")

    rng = np.random.default_rng(seed)
    r2s, maes, coverages = [], [], []
    for _ in range(repeats):
        is_train = rng.random(labels.shape[0]) >= holdout_frac
        if not is_train.any() or is_train.all():
            continue

        train_labels, train_values = labels[is_train], values[is_train]
        order = np.argsort(train_labels, kind='stable')
        sorted_labels = train_labels[order]
        group_names, starts = np.unique(sorted_labels, return_index=True)
        group_means = np.add.reduceat(train_values[order], starts) / np.diff(
            np.append(starts, sorted_labels.shape[0])
        )

        test_labels, test_values = labels[~is_train], values[~is_train]
        slot = np.searchsorted(group_names, test_labels)
        slot_in_range = slot < group_names.shape[0]
        seen = np.zeros(test_labels.shape[0], dtype=bool)
        seen[slot_in_range] = group_names[slot[slot_in_range]] == test_labels[slot_in_range]

        predictions = np.full(test_values.shape[0], train_values.mean())
        predictions[seen] = group_means[slot[seen]]

        residual = ((test_values - predictions) ** 2).sum()
        total = ((test_values - test_values.mean()) ** 2).sum()
        r2s.append(100.0 if total <= 0 else 100.0 * (1.0 - residual / total))
        maes.append(np.abs(test_values - predictions).mean())
        coverages.append(100.0 * seen.mean())

    if not r2s:
        raise ValueError("no usable train/test split was produced")
    return float(np.mean(r2s)), float(np.mean(maes)), float(np.mean(coverages))
