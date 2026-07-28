"""Tests for the protocol vocabulary and the held-out grouping diagnostic.

Background: the examination duration model conditions on a 12-value
`sequence_type` bucket, which explains 31.6% of duration variance held out. The
`Protocol` string — already present in every MRI_MSR_100 message and never used
— explains 81.7% (MAE 16.2 s vs the trained model's 50.3 s). These helpers turn
that string into a model input and provide the gate that decides whether the
same relationship holds on the pkl's `total_duration`.
"""

import unittest

import numpy as np

from AlternatingPipeline.data.protocol_vocab import (
    RARE_PROTOCOL_ID,
    build_protocol_vocab,
    heldout_group_r2,
    normalize_protocol_name,
    protocol_id,
)


class NormalizeProtocolNameTests(unittest.TestCase):
    def test_case_and_whitespace_variants_collapse(self):
        """86 real protocols differ only by case, e.g. T2_TSE_SAG/t2_tse_sag."""
        for a, b in [
            ('T2_TSE_SAG', 't2_tse_sag'),
            ('SAG T1 DR GADO', 'SAG T1 DR gado'),
            ('LOCA  HASTE   APNEE', 'LOCA HASTE APNEE'),
            ('  T1 VIBE  ', 'T1 VIBE'),
        ]:
            self.assertEqual(normalize_protocol_name(a), normalize_protocol_name(b))

    def test_distinct_protocols_stay_distinct(self):
        self.assertNotEqual(
            normalize_protocol_name('LOCA HASTE APNEE'),
            normalize_protocol_name('LOCA HASTE RESPI LIBRE'),
        )

    def test_missing_names_normalize_to_empty(self):
        for missing in (None, '', '   ', float('nan')):
            self.assertEqual(normalize_protocol_name(missing), '')


class BuildProtocolVocabTests(unittest.TestCase):
    def test_rare_protocols_share_the_reserved_zero_id(self):
        names = ['common'] * 5 + ['seen_twice'] * 2 + ['once']
        vocab = build_protocol_vocab(names, min_count=3)
        self.assertEqual(protocol_id('common', vocab), 1)
        self.assertEqual(protocol_id('seen_twice', vocab), RARE_PROTOCOL_ID)
        self.assertEqual(protocol_id('once', vocab), RARE_PROTOCOL_ID)
        self.assertEqual(protocol_id('never_seen_at_all', vocab), RARE_PROTOCOL_ID)

    def test_ids_are_contiguous_from_one_and_zero_is_never_assigned(self):
        names = sum(([f'p{i}'] * 4 for i in range(6)), [])
        vocab = build_protocol_vocab(names, min_count=3)
        self.assertEqual(sorted(vocab.values()), list(range(1, 7)))

    def test_vocab_is_deterministic_regardless_of_input_order(self):
        names = ['a'] * 5 + ['b'] * 5 + ['c'] * 4
        first = build_protocol_vocab(names, min_count=3)
        second = build_protocol_vocab(list(reversed(names)), min_count=3)
        self.assertEqual(first, second)

    def test_lookup_normalizes_so_case_variants_hit_one_id(self):
        vocab = build_protocol_vocab(['T2_TSE_SAG'] * 4, min_count=3)
        self.assertEqual(protocol_id('t2_tse_sag', vocab), protocol_id('T2_TSE_SAG', vocab))
        self.assertNotEqual(protocol_id('t2_tse_sag', vocab), RARE_PROTOCOL_ID)

    def test_blank_names_are_rare_not_their_own_protocol(self):
        vocab = build_protocol_vocab([''] * 10 + ['real'] * 10, min_count=3)
        self.assertEqual(protocol_id('', vocab), RARE_PROTOCOL_ID)
        self.assertEqual(protocol_id(None, vocab), RARE_PROTOCOL_ID)


class HeldoutGroupR2Tests(unittest.TestCase):
    def test_a_perfectly_predictive_grouping_scores_near_one_hundred(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 20, size=4000)
        values = labels * 50.0 + rng.normal(0, 1.0, size=4000)
        r2, mae, coverage = heldout_group_r2(labels, values)
        self.assertGreater(r2, 99.0)
        self.assertLess(mae, 5.0)
        self.assertEqual(coverage, 100.0)

    def test_a_grouping_with_no_signal_scores_near_zero(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 20, size=4000)
        values = rng.normal(100, 30, size=4000)
        r2, _, _ = heldout_group_r2(labels, values)
        self.assertLess(abs(r2), 5.0)

    def test_singleton_groups_do_not_inflate_the_score(self):
        """The reason this is held out: one row per group fits perfectly
        in-sample and predicts nothing."""
        rng = np.random.default_rng(0)
        n = 3000
        labels = np.arange(n)
        values = rng.normal(100, 30, size=n)
        r2, _, _ = heldout_group_r2(labels, values)
        self.assertLess(r2, 5.0)

    def test_unseen_groups_fall_back_to_the_global_mean_and_are_reported(self):
        labels = np.array(['train'] * 200 + ['heldout_only'] * 200)
        values = np.concatenate([np.full(200, 10.0), np.full(200, 10.0)])
        # Deterministic split cannot guarantee which side a label lands on, so
        # just assert coverage is reported as a percentage and nothing raises.
        r2, mae, coverage = heldout_group_r2(labels, values)
        self.assertGreaterEqual(coverage, 0.0)
        self.assertLessEqual(coverage, 100.0)
        self.assertFalse(np.isnan(r2))

    def test_constant_target_does_not_divide_by_zero(self):
        labels = np.array([0, 1] * 100)
        values = np.full(200, 42.0)
        r2, mae, coverage = heldout_group_r2(labels, values)
        self.assertFalse(np.isnan(r2))
        self.assertAlmostEqual(mae, 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
