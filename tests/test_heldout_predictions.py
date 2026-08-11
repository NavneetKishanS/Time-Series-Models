"""Per-row held-out predictions, so a stratum can be scored without a refit.

WHY THIS EXISTS, stated plainly because it is the whole point of the 2026-08-10
work: the evidence used to reject Görtler's sub-1% parameters was that AGGREGATE
MSE rose when they were included. That measurement cannot detect the claim he
made. A field that helps only the ~1% of rows it appears on moves aggregate MAE
by a fraction of a second — inside seed noise. The prior result did not disprove
the mechanism, it was blind to it.

Fixing that needs residuals per row, not a pre-averaged (r2, mae) pair, so the
same fitted model can be scored on 'rows where PDM is present' and on
'everything else'. `heldout_regressor_score` averages before returning and
cannot answer that.

The subtle requirement is that every row must be predicted by a model that did
NOT train on it, across repeats — otherwise a stratum's score silently mixes
held-out and in-sample rows, and the rare stratum (few rows, so most likely to
be in-sample on any given repeat) is exactly where that corrupts most.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from AlternatingPipeline.data.parameter_analysis import (  # noqa: E402
    heldout_predictions, heldout_regressor_score, stratified_mae,
)


def _linear_problem(n=400, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    values = 3.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(0, 0.1, n)
    return features, values


class HeldoutPredictionTests(unittest.TestCase):

    def test_every_row_gets_a_prediction_and_a_count(self):
        features, values = _linear_problem()
        result = heldout_predictions(features, values, repeats=3)
        self.assertEqual(result['predictions'].shape, values.shape)
        self.assertEqual(result['held_out_count'].shape, values.shape)
        self.assertTrue((result['held_out_count'] >= 0).all())

    def test_a_row_is_only_ever_predicted_out_of_sample(self):
        # The requirement that makes a stratum score honest. If a row could be
        # predicted by a model that trained on it, the rare stratum — few rows,
        # so most often in-sample — is where the contamination lands hardest.
        features, values = _linear_problem(n=200)
        result = heldout_predictions(features, values, repeats=4, seed=1)
        # With 20% held out over 4 repeats, some rows are never held out at all.
        # Those must be reported as uncovered rather than filled in.
        self.assertTrue((result['held_out_count'] == 0).any())
        self.assertTrue(np.isnan(result['predictions'][result['held_out_count'] == 0]).all())

    def test_covered_rows_carry_a_finite_prediction(self):
        features, values = _linear_problem()
        result = heldout_predictions(features, values, repeats=5)
        covered = result['held_out_count'] > 0
        self.assertTrue(covered.any())
        self.assertTrue(np.isfinite(result['predictions'][covered]).all())

    def test_it_agrees_with_the_averaged_estimator_it_replaces(self):
        # Not a reimplementation with different behaviour. Both fit the same
        # model on the same splits, so their overall MAE has to land in the same
        # place or one of them is measuring something else.
        features, values = _linear_problem(n=600)
        result = heldout_predictions(features, values, repeats=3, seed=0)
        covered = result['held_out_count'] > 0
        pooled = float(np.mean(np.abs(
            values[covered] - result['predictions'][covered])))
        _, reference = heldout_regressor_score(features, values, repeats=3, seed=0)
        self.assertAlmostEqual(pooled, reference, delta=0.25 * max(reference, 0.05))

    def test_groups_are_honoured(self):
        # A grouped split has to hold whole GROUPS out, and the way to prove it
        # did is a problem a random split can memorise: the group id is IN the
        # features, so a random split sees every test row's group during
        # training and looks up the answer, while a grouped split faces groups
        # it has never seen and cannot extrapolate to.
        #
        # This is the same optimism the 2026-08-07 split change was about, made
        # deliberately extreme so it cannot be confused with variance.
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(40), 10)
        features = np.column_stack([
            groups.astype(float) + rng.normal(0, 0.01, 400),
            rng.normal(size=400),
        ])
        values = groups.astype(float) * 5.0 + rng.normal(0, 0.1, 400)

        grouped = heldout_predictions(features, values, repeats=3, seed=0,
                                      groups=groups)
        random_split = heldout_predictions(features, values, repeats=3, seed=0)

        def mae(result):
            covered = result['held_out_count'] > 0
            return float(np.mean(np.abs(values[covered] - result['predictions'][covered])))

        self.assertGreater(mae(grouped), 3 * mae(random_split))


class StratifiedMaeTests(unittest.TestCase):

    def test_it_splits_by_label_and_counts_rows(self):
        values = np.array([10.0, 10.0, 20.0, 20.0])
        predictions = np.array([11.0, 9.0, 25.0, 15.0])
        covered = np.array([True, True, True, True])
        labels = np.array(['a', 'a', 'b', 'b'])

        rows = {r['stratum']: r
                for r in stratified_mae(values, predictions, covered, labels)}
        self.assertAlmostEqual(rows['a']['mae_s'], 1.0)
        self.assertAlmostEqual(rows['b']['mae_s'], 5.0)
        self.assertEqual(rows['a']['rows'], 2)

    def test_uncovered_rows_are_excluded_not_counted_as_perfect(self):
        # A NaN prediction treated as a 0 residual would make an uncovered
        # stratum look like the best-predicted one in the report.
        values = np.array([10.0, 10.0])
        predictions = np.array([12.0, np.nan])
        covered = np.array([True, False])
        labels = np.array(['a', 'a'])

        row = stratified_mae(values, predictions, covered, labels)[0]
        self.assertEqual(row['rows'], 1)
        self.assertAlmostEqual(row['mae_s'], 2.0)

    def test_a_stratum_with_no_covered_rows_is_reported_not_dropped(self):
        # Silence would read as "no problem here". An empty stratum is a
        # statement about the split, and on the rare rows it is THE statement.
        values = np.array([10.0, 20.0])
        predictions = np.array([11.0, np.nan])
        covered = np.array([True, False])
        labels = np.array(['a', 'b'])

        rows = {r['stratum']: r
                for r in stratified_mae(values, predictions, covered, labels)}
        self.assertIn('b', rows)
        self.assertEqual(rows['b']['rows'], 0)
        self.assertTrue(np.isnan(rows['b']['mae_s']))

    def test_strata_come_back_largest_first(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        covered = np.ones(4, dtype=bool)
        labels = np.array(['small', 'big', 'big', 'big'])
        rows = stratified_mae(values, predictions, covered, labels)
        self.assertEqual(rows[0]['stratum'], 'big')

    def test_it_is_usable_on_a_boolean_presence_mask(self):
        # The stratification the whole exercise is about: rows where a rare
        # field is present, against everything else.
        values = np.array([10.0, 10.0, 10.0, 10.0])
        predictions = np.array([10.0, 10.0, 4.0, 16.0])
        covered = np.ones(4, dtype=bool)
        present = np.array([False, False, True, True])

        rows = {r['stratum']: r
                for r in stratified_mae(values, predictions, covered, present)}
        self.assertAlmostEqual(rows[False]['mae_s'], 0.0)
        self.assertAlmostEqual(rows[True]['mae_s'], 6.0)


if __name__ == '__main__':
    unittest.main()
