"""Sequence-parameter analysis for the examination-duration model.

Görtler (2026-07-31) ruled the customer-authored protocol name out as a feature
for the general model: a protocol is a saved state — one sequence plus ~50
parameter settings — and customers copy Siemens presets, rename them, and edit
them, destroying the link to the original preset id. The database holds ~3.8M
protocol names across ~15,000 customers against 3,800 Siemens presets, so a
protocol-keyed model learns one site and transfers to none.

What he wants instead is **sequence + sequence parameters**, targeting the same
±15s accuracy. Our own held-out measurement puts scanner + protocol at 15.3s
MAE, which both confirms his estimate and hands us the benchmark: protocol
identity is now the oracle to score against, not an input.

His sharper point is that ±15s is not really a ceiling. If protocols ran fixed,
the jitter would be ~1s; it is 15s because the operator adjusts parameters after
loading the protocol (slice count 15→17 after the localizer, SAR conflicts on
low-weight patients forcing fewer slices or a longer measurement). The name is
the pre-adjustment label; the parameters are what actually ran. So executed
parameters should *beat* the protocol lookup — provided the SUT message records
executed values rather than protocol defaults, which `within_group_variation`
below is what measures.

Everything here is held out on the same split convention as
`protocol_vocab.heldout_group_r2`, so the categorical and continuous numbers in
the same report are directly comparable.
"""

import math
from datetime import timedelta

import numpy as np

# Missing numeric parameters are NaN, never 0.0. Field sets differ by sequence
# family — `TF` (turbo factor) is on the TSE-family messages and absent from
# ep2d_diff, which uses an echo factor instead — and a turbo factor of 0 is not
# a real value, it means "does not apply to this sequence". Collapsing that to
# 0.0 (as the model path's _safe_float does) invents a numeric difference
# between sequence families that does not exist.
# HistGradientBoostingRegressor handles NaN natively, so the analysis can keep
# the distinction the model path currently cannot.
MISSING = float('nan')


def _to_float(raw):
    """Parse a raw SUT token value, returning NaN rather than 0.0 on failure."""
    if raw is None:
        return MISSING
    if isinstance(raw, (int, float)):
        return MISSING if isinstance(raw, float) and math.isnan(raw) else float(raw)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return MISSING


def numeric_field_inventory(sequences, source='sut_raw'):
    """Per-key presence rate, distinct count and percentiles across the corpus.

    This is what sets the EXAMINATION_SEQPARAM_SCALE divisors. An entry there is
    REQUIRED before a name is added to EXAMINATION_SEQPARAM_FEATURES: unscaled
    large-magnitude numerics silently erase categorical conditioning through
    LayerNorm, a bug that has cost this project three separate multi-week
    flat-duration incidents. Divisors are picked from the observed p99, not from
    a handful of sampled messages.

    Returns a list of dicts sorted by descending presence, one per key.
    """
    total = len(sequences)
    if not total:
        return []

    values_by_key = {}
    for seq in sequences:
        for key, raw in (seq.get(source) or {}).items():
            values_by_key.setdefault(key, []).append(raw)

    rows = []
    for key, raws in values_by_key.items():
        numeric = np.array([_to_float(r) for r in raws], dtype=float)
        finite = numeric[np.isfinite(numeric)]
        row = {
            'key': key,
            'present': len(raws),
            'presence_pct': 100.0 * len(raws) / total,
            'distinct': len(set(map(str, raws))),
            'numeric_pct': 100.0 * finite.size / max(1, len(raws)),
            'p50': float(np.percentile(finite, 50)) if finite.size else MISSING,
            'p99': float(np.percentile(finite, 99)) if finite.size else MISSING,
            'max': float(finite.max()) if finite.size else MISSING,
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (-r['presence_pct'], r['key']))


def numeric_matrix(sequences, field_names, source='sut_raw'):
    """Stack the named SUT keys into a float matrix, missing entries as NaN.

    Shape (len(sequences), len(field_names)). Column order matches
    `field_names`, so permutation-importance results map back by index.
    """
    matrix = np.full((len(sequences), len(field_names)), MISSING, dtype=float)
    for row, seq in enumerate(sequences):
        raw = seq.get(source) or {}
        for col, name in enumerate(field_names):
            matrix[row, col] = _to_float(raw.get(name))
    return matrix


def weighting_bucket(tr, te, fa):
    """Approximate the T1/T2/PD contrast weighting from TR, TE and flip angle.

    Görtler described a Siemens-generated protocol name already present in the
    Cubes, built from ~3-4 sequence parameters and reading like "transversal,
    T1-weighted, short TR" — the customer-agnostic substitute for the customer's
    own protocol name. We cannot read that field, but its inputs are all in the
    SUT message, so this reconstructs the contrast component from the textbook
    thresholds (short TR < 800ms, long TR > 2000ms; short TE < 30ms, long TE >
    80ms).

    Deliberately approximate. Single-shot sequences are a known edge: haste
    carries TR:866 / TE:101 and is a T2 sequence, but its TR is not a
    conventional repetition time at all. The bucket exists to test whether a
    coarse parameter-derived descriptor recovers protocol-level signal — the
    regressor gets the raw values regardless, so a misfiled edge case costs the
    descriptor, not the model.
    """
    tr, te, fa = _to_float(tr), _to_float(te), _to_float(fa)
    if not math.isfinite(tr) or not math.isfinite(te):
        return 'unknown'
    long_te, short_te = te > 80, te < 30
    if tr < 800:
        # Short TR: T1 by default, but a low flip angle makes it T2*.
        if math.isfinite(fa) and fa < 40 and not long_te:
            return 't2star'
        return 't1'
    if tr > 2000:
        return 't2' if long_te else ('pd' if short_te else 'intermediate')
    return 't2' if long_te else 'intermediate'


def generated_protocol_name(seq, source='sut_raw'):
    """Our reconstruction of Görtler's parameter-derived protocol descriptor.

    `sequence_binary` + orientation + contrast weighting — all Siemens-standard,
    none customer-authored, so unlike the protocol name this transfers across
    customers. Scored against protocol identity's 82.0% in the Stage B report:
    the gap is how much of the protocol's signal a customer-agnostic descriptor
    can recover.
    """
    raw = seq.get(source) or {}
    binary = seq.get('sequence_binary') or '?'
    orientation = seq.get('orientation') or raw.get('OR') or '?'
    weighting = weighting_bucket(raw.get('TR'), raw.get('TE'), raw.get('FA'))
    return f"{binary}|{orientation}|{weighting}"


def heldout_regressor_score(features, values, holdout_frac=0.2, seed=0,
                            repeats=3, max_iter=200):
    """Gradient-boosted MAE on held-out rows, comparable to heldout_group_r2.

    Group means are not a valid estimator for continuous features, so the
    categorical baselines and the parameter model need different estimators —
    but the same split convention, or the report compares numbers that were
    never measured the same way. Same holdout_frac/seed semantics as
    `protocol_vocab.heldout_group_r2`; fewer repeats because a GBM fit is
    orders of magnitude more expensive than a group mean.

    NaN is passed through to the model on purpose: HistGradientBoostingRegressor
    learns a split direction for missing values, which is the correct treatment
    for a parameter that does not apply to a sequence family.

    Returns (r2_percent, mae).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    features = np.asarray(features, dtype=float)
    values = np.asarray(values, dtype=float)
    if features.shape[0] != values.shape[0]:
        raise ValueError("features and values must be the same length")
    if features.shape[0] < 2:
        raise ValueError("need at least 2 rows to hold any out")

    rng = np.random.default_rng(seed)
    r2s, maes = [], []
    for repeat in range(repeats):
        is_train = rng.random(features.shape[0]) >= holdout_frac
        if not is_train.any() or is_train.all():
            continue
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, random_state=seed + repeat,
        )
        model.fit(features[is_train], values[is_train])
        predictions = model.predict(features[~is_train])
        held = values[~is_train]
        residual = ((held - predictions) ** 2).sum()
        total = ((held - held.mean()) ** 2).sum()
        r2s.append(100.0 if total <= 0 else 100.0 * (1.0 - residual / total))
        maes.append(float(np.abs(held - predictions).mean()))

    if not r2s:
        raise ValueError("no usable train/test split was produced")
    return float(np.mean(r2s)), float(np.mean(maes))


def permutation_importance_mae(features, values, field_names=None,
                               holdout_frac=0.2, seed=0, shuffles=3,
                               max_iter=200):
    """Per-column MAE cost of destroying one feature. ONE fit, N shuffles.

    `heldout_regressor_score` refits for every question asked of it, which is
    the right shape for comparing feature SETS and the wrong shape for ranking
    89 columns — that would be 89 fits. Here the model is fitted once and each
    column is shuffled in the held-out matrix, so the cost is one fit plus
    `len(field_names) * shuffles` predictions.

    This is also the leak audit. `MUID` was caught by suspecting one field and
    testing it by hand (03d); a ranking over the whole vector catches the ones
    nobody thought to suspect. A parameter that is really a duration in disguise
    does not sit mid-table — it sits at the top, well clear of the physics.

    Importance is reported in SECONDS of MAE, not a normalised score, so it
    reads on the same axis as every other number in these reports.

    Correlated columns share credit and each looks individually cheap — TR and
    PEL move together, so neither scores its full worth. That is a property of
    permutation importance, not a defect; use the nested-prefix curve to decide
    how many fields to keep, and this to decide the ORDER.

    Returns a list of dicts sorted by descending `importance_s`:
        {'name', 'index', 'importance_s', 'baseline_mae'}
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    features = np.asarray(features, dtype=float)
    values = np.asarray(values, dtype=float)
    if features.shape[0] != values.shape[0]:
        raise ValueError("features and values must be the same length")
    if features.shape[0] < 2:
        raise ValueError("need at least 2 rows to hold any out")
    if field_names is None:
        field_names = [str(i) for i in range(features.shape[1])]
    if len(field_names) != features.shape[1]:
        raise ValueError("field_names must name every column")

    rng = np.random.default_rng(seed)
    is_train = rng.random(features.shape[0]) >= holdout_frac
    if not is_train.any() or is_train.all():
        raise ValueError("no usable train/test split was produced")

    model = HistGradientBoostingRegressor(max_iter=max_iter, random_state=seed)
    model.fit(features[is_train], values[is_train])

    held_x, held_y = features[~is_train].copy(), values[~is_train]
    baseline_mae = float(np.abs(held_y - model.predict(held_x)).mean())

    rows = []
    for index, name in enumerate(field_names):
        original = held_x[:, index].copy()
        costs = []
        for shuffle in range(shuffles):
            # A fresh generator per shuffle keeps the result reproducible
            # regardless of how many columns were scored before this one.
            order = np.random.default_rng(seed + 1 + shuffle).permutation(
                held_x.shape[0])
            held_x[:, index] = original[order]
            costs.append(float(np.abs(held_y - model.predict(held_x)).mean()))
        held_x[:, index] = original
        rows.append({
            'name': name,
            'index': index,
            'importance_s': float(np.mean(costs)) - baseline_mae,
            'baseline_mae': baseline_mae,
        })

    return sorted(rows, key=lambda r: -r['importance_s'])


def terminator_clusters(sequences):
    """Find segments that end on the SAME terminator event.

    `03_build_preprocessed_pkl.py:676` binds every MRI_MSR_100 to the next
    MRI_MSR_104/34 in the log, so two MSR_100 events before one terminator
    produce two OVERLAPPING segments ending at the same instant, both claiming
    the same measurement. `csv_pipeline/02_exam_preprocessing.py:220` handles
    the same case differently — it emits one row per finish event, dated from
    the *most recent* MSR_100 — which is one reason the pkl carries far more
    rows than the exam CSVs the ±15s benchmark was measured on.

    The shorter segment is the one step 02 would have kept: the later MSR_100 is
    the measurement that actually ran, and the longer segment has swallowed
    whatever preceded it. So `is_primary` marks the shortest member of each
    cluster, giving the caller a one-segment-per-terminator view to compare
    against.

    Returns a dict of arrays aligned with `sequences`:
      size        — how many segments share this segment's terminator (1 = none)
      is_primary  — True for the shortest member of each cluster, exactly once
    """
    def _duration(seq):
        value = _to_float(seq.get('total_duration'))
        return math.inf if math.isnan(value) else value

    keys = []
    for index, seq in enumerate(sequences):
        start, duration = seq.get('start_datetime'), _duration(seq)
        if start is None or math.isinf(duration):
            # Cannot be placed in time — must stand alone rather than merge
            # with every other unplaceable segment.
            keys.append(('unplaced', index))
        else:
            keys.append((seq.get('serial_idx', 0),
                         start + timedelta(seconds=duration)))

    members = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)

    size = np.ones(len(sequences), dtype=int)
    is_primary = np.zeros(len(sequences), dtype=bool)
    for rows in members.values():
        size[rows] = len(rows)
        # min() is stable, so ties resolve to the earliest row.
        is_primary[min(rows, key=lambda i: _duration(sequences[i]))] = True

    return {'size': size, 'is_primary': is_primary}


def within_group_variation(group_labels, values, min_group_size=10):
    """How much a parameter moves *inside* one group. The Stage B decision gate.

    Görtler's argument that executed parameters can beat the ±15s protocol
    lookup rests entirely on operators adjusting parameters after loading the
    protocol. If a parameter is constant within (serial, protocol), then the SUT
    message is recording the protocol's stored defaults, the post-load
    adjustment is invisible to us, and no amount of parameter modelling can
    reach below the protocol lookup's own error.

    Returns a dict with the share of qualifying groups in which the parameter
    varies at all, and the mean within-group sd — computed only over groups with
    at least `min_group_size` rows, since a 2-row group tells us nothing.
    """
    labels = np.asarray(group_labels)
    values = np.asarray(values, dtype=float)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("group_labels and values must be the same length")

    finite = np.isfinite(values)
    labels, values = labels[finite], values[finite]

    varying, sds, sizes = 0, [], 0
    for group in np.unique(labels):
        member = values[labels == group]
        if member.size < min_group_size:
            continue
        sizes += 1
        if np.unique(member).size > 1:
            varying += 1
        sds.append(float(member.std()))

    return {
        'groups': sizes,
        'varying_pct': 100.0 * varying / sizes if sizes else MISSING,
        'mean_within_sd': float(np.mean(sds)) if sds else MISSING,
        'coverage_pct': 100.0 * finite.mean() if finite.size else 0.0,
    }
