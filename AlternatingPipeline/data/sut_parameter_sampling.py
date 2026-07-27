"""Sampling MRI_SUT_1005 sequence parameters for synthetic scans.

TR and num_slices are model INPUTS read off a real scanner message. A
synthetic scan has no such message, so generation has to draw them from the
real empirical distribution — the same approach step 05 already uses for
sequence types (`_sample_sequence_type`) and demographics
(`build_demographic_distributions`).

Two properties matter:

  * Draw the parameters JOINTLY, as one observed tuple, rather than each
    marginally. TR and num_slices correlate at 0.47 in the training data;
    independent draws would manufacture protocol combinations that no scanner
    ever produced.
  * Condition on (body_region, sequence_type), because that is what the
    parameters actually depend on — a haste and a space protocol have very
    different TRs. Fall back to sequence_type alone, then to the global pool,
    so a rare region/type combination still yields a plausible value.

Values are returned RAW (TR in ms, num_slices as a count). Scaling to O(1)
happens inside the model via its `conditioning_scale` buffer, exactly as it
does for the real values at training time.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

MIN_POOL = 5  # below this a pool is too thin to be representative


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result == result else 0.0  # drop NaN


class SUTParameterSampler:
    """Draws (TR, num_slices, ...) tuples from the real training distribution."""

    def __init__(
        self,
        examination_sequences: Sequence[dict],
        feature_names: Sequence[str],
        rng=None,
    ):
        if not feature_names:
            raise ValueError("feature_names must be non-empty")

        self.feature_names = list(feature_names)
        self._rng = rng
        self._pools: Dict[tuple, List[Tuple[float, ...]]] = defaultdict(list)

        for seq in examination_sequences or []:
            conditioning = seq.get('conditioning') or {}
            values = tuple(
                _safe_float(conditioning.get(name, 0.0)) for name in self.feature_names
            )
            region = int(seq.get('body_region', 10) or 0)
            seq_type = int(seq.get('sequence_type', 0) or 0)

            self._pools[('region_type', region, seq_type)].append(values)
            self._pools[('type', seq_type)].append(values)
            self._pools[('global',)].append(values)

        self._observations = len(self._pools.get(('global',), []))

    @property
    def observations(self) -> int:
        return self._observations

    def _random_index(self, size: int) -> int:
        if self._rng is not None:
            return int(self._rng.integers(size))
        import random
        return random.randrange(size)

    def sample(self, body_region, sequence_type) -> Dict[str, float]:
        """Return one {feature_name: raw_value} draw for this scan context."""
        keys = (
            ('region_type', int(body_region), int(sequence_type)),
            ('type', int(sequence_type)),
            ('global',),
        )
        for key in keys:
            pool = self._pools.get(key)
            if not pool:
                continue
            if len(pool) >= MIN_POOL or key == ('global',):
                values = pool[self._random_index(len(pool))]
                return dict(zip(self.feature_names, values))

        # No training observations at all. Zero is what build_conditioning_tensor
        # already substitutes for an absent key, so the model has seen it.
        return {name: 0.0 for name in self.feature_names}

    def describe(self) -> str:
        contexts = sum(1 for k in self._pools if k[0] == 'region_type')
        if not self._observations:
            return (
                "SUT parameter sampler: NO observations found — every synthetic scan "
                "will use 0.0. Check that the pkl carries "
                f"{self.feature_names} in each sequence's 'conditioning' dict."
            )
        lines = [
            f"SUT parameter sampler: {self._observations:,} real observations of "
            f"{self.feature_names} across {contexts} (body_region, sequence_type) contexts."
        ]
        for name in self.feature_names:
            column = [row[self.feature_names.index(name)]
                      for row in self._pools[('global',)]]
            non_zero = [v for v in column if v != 0.0]
            mean = sum(column) / len(column)
            lines.append(
                f"  {name:<12} mean={mean:>9.2f}  min={min(column):>8.2f}  "
                f"max={max(column):>9.2f}  non-zero={len(non_zero):,}/{len(column):,}"
            )
        return "\n".join(lines)


def build_sut_sampler(
    preprocessed_data: dict,
    feature_names: Sequence[str],
    rng=None,
) -> Optional[SUTParameterSampler]:
    """Build a sampler from a preprocessed pkl, or None when not applicable."""
    if not feature_names:
        return None
    return SUTParameterSampler(
        preprocessed_data.get('examination', []), feature_names, rng=rng
    )
