"""Held-out split construction for the parameter and protocol reports.

WHY THIS EXISTS
---------------
Every held-out number this project has quoted — `protocol_vocab.heldout_group_r2`,
`parameter_analysis.heldout_regressor_score`, and therefore every MAE in 03b
through 03f — came from a RANDOM row split:

    is_train = rng.random(n) >= holdout_frac

That is the wrong split for the question being asked, in two specific ways.

1. ROWS FROM ONE EXAM LAND ON BOTH SIDES. A measurement's neighbours in the same
   examination share the patient, the operator, the coil setup, the day's
   calibration and usually the protocol family. Predicting a row whose siblings
   were in the training set is close to a lookup, so the score reads high for a
   reason that will not exist at serve time.

2. EVERY SCANNER IN TEST IS ALSO IN TRAIN. The model carries a `serial_idx`
   embedding, so a random split lets it memorise per-install behaviour and
   collect the credit as if it had learned physics. The whole point of the
   general model is the customer whose scanner it has never seen.

Both inflate the score in the same direction, and both inflate it most on the
tail — the rare body group with few rows is exactly the case where "a sibling
was in train" does the most work. Görtler's target is the extraordinary 1% and a
transfer to new customers; a random split can measure neither.

WHAT TO GROUP BY
----------------
`groups` is deliberately caller-supplied rather than inferred here, because the
right grouping depends on the claim being made:

    serial      "does this transfer to a customer we have never seen?"
    serial+day  "does this transfer to a session we have never seen?" — the
                closest available stand-in for an exam, since the pkl carries no
                exam or patient id (see parameter_analysis.exam_group_labels)
    protocol    "does this work on a protocol we have never seen?"

Holding out whole groups makes the score strictly harder and strictly more
honest. A number that drops when you switch to a grouped split was never real.

This module is pure numpy on purpose — no knowledge of the sequence dict shape,
so both `data.protocol_vocab` and `data.parameter_analysis` can import it
without either importing the other.
"""

import numpy as np


def holdout_mask(n_rows, holdout_frac, rng, groups=None):
    """Boolean `is_train` mask of length `n_rows`.

    With `groups=None` this is the historical random-row split, kept so existing
    numbers stay reproducible and so the two conventions can be printed side by
    side (the gap between them is itself a reportable quantity).

    With `groups` supplied, whole groups go to one side or the other. Groups are
    drawn in a random order and accumulated into the held-out side until the ROW
    target is met, so the split lands near `holdout_frac` of rows even when group
    sizes are wildly uneven — which they are: a busy scanner contributes many
    times the rows of a quiet one, and picking `holdout_frac` of GROUPS would
    then hold out anywhere from a few percent to half the corpus.

    Args:
        n_rows: total number of rows.
        holdout_frac: target fraction of ROWS to hold out.
        rng: a `numpy.random.Generator`, consumed once per call.
        groups: optional array-like of length `n_rows`. Rows sharing a value are
            never split across the train/test boundary.

    Returns:
        np.ndarray of bool, True where the row is in the training split.

    Note this can return an all-True or all-False mask when there is only one
    group, or when one group holds nearly every row. Callers already guard with
    `if not is_train.any() or is_train.all(): continue` — that guard is the
    correct response here too, and it is how a degenerate grouping announces
    itself rather than silently scoring on nothing.
    """
    if groups is None:
        return rng.random(n_rows) >= holdout_frac

    groups = np.asarray(groups)
    if groups.shape[0] != n_rows:
        raise ValueError(
            f"groups has {groups.shape[0]} entries for {n_rows} rows — it must "
            f"label every row"
        )

    # np.unique returns sorted values, which is what makes searchsorted a valid
    # inverse below.
    unique, counts = np.unique(groups, return_counts=True)
    codes = np.searchsorted(unique, groups)

    # Greedy closest-approach: walk the groups in random order and take one only
    # when taking it lands the running total CLOSER to the row target than
    # leaving it. Accumulating until the target is merely exceeded looks
    # equivalent and is not — draw a scanner holding 71% of the corpus first and
    # it overshoots a 20% target to 86%, leaving a seventh of the data to train
    # on. The same rule also declines to hold out a group that is simply too big
    # to hold out at this fraction, which is the honest answer: `split_summary`
    # then shows the group counts, so "the busiest scanner never lands in test"
    # is visible in the report instead of being silently baked into the number.
    target_rows = holdout_frac * n_rows
    is_held_group = np.zeros(unique.shape[0], dtype=bool)
    accumulated = 0
    for index in rng.permutation(unique.shape[0]):
        count = counts[index]
        if abs(accumulated + count - target_rows) >= abs(accumulated - target_rows):
            continue
        is_held_group[index] = True
        accumulated += count

    return ~is_held_group[codes]


def split_summary(is_train, groups=None):
    """Describe a split, for the report to print rather than assert silently.

    Returns a dict with the realised row fraction and, when `groups` is given,
    the group counts on each side and a `leaked_groups` count that must be 0.
    That last field is the actual regression test: it is 0 by construction here,
    and non-zero would mean someone reintroduced a row-level split.
    """
    is_train = np.asarray(is_train, dtype=bool)
    summary = {
        'n_rows': int(is_train.shape[0]),
        'n_train': int(is_train.sum()),
        'n_test': int((~is_train).sum()),
        'test_row_frac': float((~is_train).mean()) if is_train.size else 0.0,
    }
    if groups is not None:
        groups = np.asarray(groups)
        train_groups = set(groups[is_train].tolist())
        test_groups = set(groups[~is_train].tolist())
        summary.update({
            'n_train_groups': len(train_groups),
            'n_test_groups': len(test_groups),
            'leaked_groups': len(train_groups & test_groups),
        })
    return summary
