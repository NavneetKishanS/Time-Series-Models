import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlternatingPipeline.data.parameter_analysis import (  # noqa: E402
    generated_protocol_name,
    heldout_regressor_score,
    numeric_field_inventory,
    numeric_matrix,
    weighting_bucket,
    within_group_variation,
)


def _seq(sut_raw, **extra):
    seq = {'sut_raw': dict(sut_raw)}
    seq.update(extra)
    return seq


class NumericMatrixTests(unittest.TestCase):
    def test_missing_fields_are_nan_not_zero(self):
        """The whole point: 0.0 is a real value for most SUT parameters.

        TF is on the TSE-family messages and absent from ep2d_diff, which uses
        an echo factor instead. Collapsing that absence to 0.0 — as the model
        path's _safe_float does — invents a numeric difference between sequence
        families that does not exist, and a gradient booster would happily
        learn it.
        """
        matrix = numeric_matrix(
            [_seq({'TR': '866', 'TF': '256'}), _seq({'TR': '4300'})],
            ['TR', 'TF'],
        )
        self.assertEqual(matrix[0, 1], 256.0)
        self.assertTrue(math.isnan(matrix[1, 1]))
        self.assertFalse(math.isnan(matrix[1, 0]))

    def test_column_order_matches_field_names(self):
        matrix = numeric_matrix([_seq({'A': '1', 'B': '2'})], ['B', 'A'])
        self.assertEqual(list(matrix[0]), [2.0, 1.0])

    def test_unparseable_values_become_nan(self):
        matrix = numeric_matrix(
            [_seq({'CS': 'BY1-3.SP3-5', 'DLL': '%SiemensSeq%\\haste'})],
            ['CS', 'DLL'],
        )
        self.assertTrue(np.isnan(matrix).all())

    def test_missing_sut_raw_key_does_not_raise(self):
        matrix = numeric_matrix([{}, {'sut_raw': None}], ['TR'])
        self.assertEqual(matrix.shape, (2, 1))
        self.assertTrue(np.isnan(matrix).all())


class FieldInventoryTests(unittest.TestCase):
    def test_reports_presence_and_percentiles(self):
        rows = numeric_field_inventory([
            _seq({'TR': '866', 'TF': '256'}),
            _seq({'TR': '1000', 'TF': '256'}),
            _seq({'TR': '4300'}),
        ])
        by_key = {row['key']: row for row in rows}
        self.assertAlmostEqual(by_key['TR']['presence_pct'], 100.0)
        self.assertAlmostEqual(by_key['TF']['presence_pct'], 200.0 / 3)
        self.assertEqual(by_key['TR']['distinct'], 3)
        self.assertAlmostEqual(by_key['TR']['p50'], 1000.0)

    def test_non_numeric_fields_are_flagged_not_dropped(self):
        """A string field must still appear — that is how you find DLL/OR."""
        rows = numeric_field_inventory([_seq({'OR': 'CT'}), _seq({'OR': 'SCT'})])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['key'], 'OR')
        self.assertEqual(rows[0]['numeric_pct'], 0.0)
        self.assertEqual(rows[0]['distinct'], 2)

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(numeric_field_inventory([]), [])


class WeightingBucketTests(unittest.TestCase):
    def test_short_tr_short_te_is_t1(self):
        self.assertEqual(weighting_bucket(500, 10, 90), 't1')

    def test_long_tr_long_te_is_t2(self):
        self.assertEqual(weighting_bucket(4000, 100, 90), 't2')

    def test_long_tr_short_te_is_pd(self):
        self.assertEqual(weighting_bucket(4000, 15, 90), 'pd')

    def test_short_tr_low_flip_angle_is_t2star(self):
        self.assertEqual(weighting_bucket(500, 15, 20), 't2star')

    def test_missing_inputs_are_unknown_not_a_wrong_guess(self):
        self.assertEqual(weighting_bucket(None, 100, 90), 'unknown')
        self.assertEqual(weighting_bucket(866, None, 90), 'unknown')
        self.assertEqual(weighting_bucket('n/a', 'n/a', 'n/a'), 'unknown')


class GeneratedProtocolNameTests(unittest.TestCase):
    def test_combines_binary_orientation_and_weighting(self):
        name = generated_protocol_name(_seq(
            {'TR': '4000', 'TE': '100', 'FA': '90'},
            sequence_binary='tse', orientation='T',
        ))
        self.assertEqual(name, 'tse|T|t2')

    def test_falls_back_to_the_raw_or_token_and_marks_gaps(self):
        name = generated_protocol_name(_seq({'OR': 'SCT'}))
        self.assertTrue(name.startswith('?|SCT|'))

    def test_is_customer_agnostic(self):
        """Two sites running the same sequence must get the same descriptor.

        This is the entire reason the descriptor exists: the protocol name would
        differ between these two, which is why Görtler ruled it out.
        """
        left = _seq({'TR': '500', 'TE': '10', 'FA': '90'},
                    sequence_binary='haste', orientation='CT',
                    protocol_name='Kopf nativ')
        right = _seq({'TR': '500', 'TE': '10', 'FA': '90'},
                     sequence_binary='haste', orientation='CT',
                     protocol_name='HEAD routine 3mm')
        self.assertEqual(generated_protocol_name(left),
                         generated_protocol_name(right))


class WithinGroupVariationTests(unittest.TestCase):
    def test_constant_parameter_reports_zero_varying(self):
        """The 'protocol defaults' case — parameters never move within a group."""
        labels = np.repeat(['a', 'b'], 20)
        values = np.repeat([9.0, 15.0], 20)
        stats = within_group_variation(labels, values)
        self.assertEqual(stats['groups'], 2)
        self.assertEqual(stats['varying_pct'], 0.0)
        self.assertEqual(stats['mean_within_sd'], 0.0)

    def test_adjusted_parameter_reports_full_varying(self):
        """The 'executed values' case — slice count moves 15->17 per patient."""
        labels = np.repeat(['a', 'b'], 20)
        values = np.concatenate([
            np.tile([15.0, 17.0], 10), np.tile([20.0, 15.0], 10),
        ])
        stats = within_group_variation(labels, values)
        self.assertEqual(stats['varying_pct'], 100.0)
        self.assertGreater(stats['mean_within_sd'], 0.0)

    def test_small_groups_are_excluded(self):
        """A 2-row group says nothing about whether a parameter is adjusted."""
        labels = np.array(['a'] * 3 + ['b'] * 12)
        values = np.array([1.0, 2.0, 3.0] + [7.0] * 12)
        stats = within_group_variation(labels, values, min_group_size=10)
        self.assertEqual(stats['groups'], 1)
        self.assertEqual(stats['varying_pct'], 0.0)

    def test_nan_rows_are_excluded_and_reported_as_coverage(self):
        labels = np.array(['a'] * 20)
        values = np.array([5.0] * 10 + [float('nan')] * 10)
        stats = within_group_variation(labels, values)
        self.assertAlmostEqual(stats['coverage_pct'], 50.0)
        self.assertEqual(stats['groups'], 1)

    def test_no_qualifying_groups_returns_nan_not_a_crash(self):
        stats = within_group_variation(np.array(['a', 'b']), np.array([1.0, 2.0]))
        self.assertEqual(stats['groups'], 0)
        self.assertTrue(math.isnan(stats['varying_pct']))


class HeldoutRegressorScoreTests(unittest.TestCase):
    def test_recovers_a_learnable_signal(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(600, 3))
        values = 100 + 40 * features[:, 0] + rng.normal(scale=2.0, size=600)
        r2, mae = heldout_regressor_score(features, values, repeats=2)
        self.assertGreater(r2, 90.0)
        self.assertLess(mae, 10.0)

    def test_reports_near_zero_r2_on_pure_noise(self):
        """Guards the report against a feature set that carries nothing."""
        rng = np.random.default_rng(1)
        features = rng.normal(size=(400, 3))
        values = rng.normal(loc=100, scale=20, size=400)
        r2, _ = heldout_regressor_score(features, values, repeats=2)
        self.assertLess(r2, 20.0)

    def test_nan_features_are_passed_through_not_rejected(self):
        """Sequence-scoped parameters are NaN for families they do not apply to.

        HistGradientBoostingRegressor learns a split direction for missing
        values, which is the correct treatment — the score must not blow up.
        """
        rng = np.random.default_rng(2)
        features = rng.normal(size=(400, 2))
        features[::2, 1] = float('nan')
        values = 50 + 10 * features[:, 0] + rng.normal(scale=1.0, size=400)
        r2, mae = heldout_regressor_score(features, values, repeats=2)
        self.assertTrue(math.isfinite(mae))
        self.assertGreater(r2, 50.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            heldout_regressor_score(np.zeros((5, 2)), np.zeros(4))


if __name__ == '__main__':
    unittest.main()
