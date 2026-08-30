"""Calibrate synthetic examination finish events to the observed corpus rate.

The token decoder is responsible for the *workflow* leading to a finish.  A
rare operational outcome such as a user abort is not reliably calibrated by a
sampled terminal softmax, particularly after rare-event oversampling.  This
module therefore samples the terminal outcome from a smoothed empirical rate,
using the most specific adequately-supported scanner/region/sequence stratum.
"""

from collections import Counter


_LEVELS = (
    ('serial_region_type', 50, lambda s: (int(s.get('serial_idx', 0)),
                                           int(s.get('body_region', 10)),
                                           int(s.get('sequence_type', 0)))),
    ('region_type', 100, lambda s: (int(s.get('body_region', 10)),
                                    int(s.get('sequence_type', 0)))),
    ('region', 250, lambda s: (int(s.get('body_region', 10)),)),
    ('global', 1, lambda s: ()),
)


def build_abort_rate_table(sequences, abort_token_id):
    """Return supported, Laplace-smoothed abort rates for generation strata."""
    counts = {name: Counter() for name, _, _ in _LEVELS}
    aborts = {name: Counter() for name, _, _ in _LEVELS}
    for sequence in sequences:
        is_abort = int(abort_token_id) in sequence.get('sequence', ())
        for name, _, key_fn in _LEVELS:
            key = key_fn(sequence)
            counts[name][key] += 1
            aborts[name][key] += int(is_abort)

    table = {}
    for name, minimum, _ in _LEVELS:
        for key, total in counts[name].items():
            if total >= minimum:
                # The small prior prevents zero-probability strata while being
                # negligible for the global corpus.
                table[(name, key)] = {
                    'rate': (aborts[name][key] + 1.0) / (total + 2.0),
                    'aborts': int(aborts[name][key]),
                    'total': int(total),
                }
    if ('global', ()) not in table:
        raise ValueError('cannot calibrate aborts from an empty examination corpus')
    return table


def select_abort_rate(table, serial_idx, body_region, sequence_type):
    """Pick the most-specific supported empirical rate."""
    keys = (
        ('serial_region_type', (int(serial_idx), int(body_region), int(sequence_type))),
        ('region_type', (int(body_region), int(sequence_type))),
        ('region', (int(body_region),)),
        ('global', ()),
    )
    for key in keys:
        if key in table:
            meta = table[key]
            return float(meta['rate']), key, meta
    raise RuntimeError('abort-rate table has no global fallback')
