"""Tests for sampling SUT sequence parameters at generation time."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlternatingPipeline.data.sut_parameter_sampling import (  # noqa: E402
    SUTParameterSampler,
    build_sut_sampler,
)

FEATURES = ['TR', 'num_slices']


def _seq(region, seq_type, tr, slices):
    return {
        'body_region': region,
        'sequence_type': seq_type,
        'conditioning': {'Age': 50, 'TR': tr, 'num_slices': slices},
    }


def _haste_pool(n=10):
    """Region 0 / type 5, TR always 800, slices always 9."""
    return [_seq(0, 5, 800.0, 9.0) for _ in range(n)]


def _space_pool(n=10):
    """Region 3 / type 4, TR always 3000, slices always 40."""
    return [_seq(3, 4, 3000.0, 40.0) for _ in range(n)]


def test_samples_from_the_matching_region_and_type():
    sampler = SUTParameterSampler(
        _haste_pool() + _space_pool(), FEATURES, rng=np.random.default_rng(0)
    )

    assert sampler.sample(body_region=0, sequence_type=5) == {'TR': 800.0, 'num_slices': 9.0}
    assert sampler.sample(body_region=3, sequence_type=4) == {'TR': 3000.0, 'num_slices': 40.0}


def test_values_are_drawn_jointly_not_marginally():
    """TR and num_slices correlate; independent draws would invent protocols."""
    sampler = SUTParameterSampler(
        _haste_pool(50) + _space_pool(50), FEATURES, rng=np.random.default_rng(1)
    )

    # Sampling the GLOBAL pool (unknown region/type) must still return one of
    # the two real tuples, never a mix like TR=800 with 40 slices.
    seen = {
        (d['TR'], d['num_slices'])
        for d in (sampler.sample(body_region=99, sequence_type=99) for _ in range(60))
    }
    assert seen <= {(800.0, 9.0), (3000.0, 40.0)}
    assert len(seen) == 2, "expected both real protocols to appear across 60 draws"


def test_falls_back_to_sequence_type_when_region_pool_is_thin():
    # One observation for region 7 / type 5 — below MIN_POOL, so the sampler
    # must widen to type 5 rather than trust a single scan.
    sequences = _haste_pool(20) + [_seq(7, 5, 999.0, 99.0)]
    sampler = SUTParameterSampler(sequences, FEATURES, rng=np.random.default_rng(2))

    draws = [sampler.sample(body_region=7, sequence_type=5) for _ in range(30)]
    # The thin region pool's outlier may appear via the type pool, but the
    # dominant haste values must come through.
    assert any(d['TR'] == 800.0 for d in draws)


def test_falls_back_to_global_for_unseen_type():
    sampler = SUTParameterSampler(_haste_pool(20), FEATURES, rng=np.random.default_rng(3))

    drawn = sampler.sample(body_region=42, sequence_type=42)
    assert drawn == {'TR': 800.0, 'num_slices': 9.0}


def test_empty_training_data_yields_zeros_not_a_crash():
    sampler = SUTParameterSampler([], FEATURES, rng=np.random.default_rng(4))

    assert sampler.observations == 0
    assert sampler.sample(body_region=0, sequence_type=0) == {'TR': 0.0, 'num_slices': 0.0}
    assert 'NO observations' in sampler.describe()


def test_missing_and_malformed_values_become_zero():
    sequences = [
        {'body_region': 0, 'sequence_type': 1, 'conditioning': {}},
        {'body_region': 0, 'sequence_type': 1, 'conditioning': {'TR': 'not-a-number'}},
        {'body_region': 0, 'sequence_type': 1, 'conditioning': {'TR': float('nan')}},
    ]
    sampler = SUTParameterSampler(sequences, FEATURES, rng=np.random.default_rng(5))

    for _ in range(10):
        drawn = sampler.sample(body_region=0, sequence_type=1)
        assert drawn == {'TR': 0.0, 'num_slices': 0.0}


def test_describe_reports_real_ranges():
    sampler = SUTParameterSampler(
        _haste_pool(10) + _space_pool(10), FEATURES, rng=np.random.default_rng(6)
    )

    described = sampler.describe()
    assert '20 real observations' in described
    assert 'TR' in described and 'num_slices' in described
    assert 'non-zero=20/20' in described


def test_build_sut_sampler_returns_none_without_features():
    assert build_sut_sampler({'examination': _haste_pool()}, []) is None


def test_build_sut_sampler_reads_examination_key():
    sampler = build_sut_sampler(
        {'examination': _haste_pool()}, FEATURES, rng=np.random.default_rng(7)
    )
    assert sampler is not None
    assert sampler.observations == 10


def test_sampler_requires_feature_names():
    with pytest.raises(ValueError):
        SUTParameterSampler(_haste_pool(), [])
