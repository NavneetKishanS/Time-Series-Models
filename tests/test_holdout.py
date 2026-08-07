"""Grouped held-out splits — the thing that decides whether a score is real.

Every MAE this project has quoted came from a random row split, which lets a
measurement's siblings sit in the training set and lets every scanner in test
also appear in train. Both inflate the score, and both inflate it most on the
tail — which is exactly the population the ±15s goal is about.

The tests here pin two properties:

  1. A grouped split NEVER puts one group on both sides. That is the whole
     contract; if it ever breaks, every number downstream is quietly wrong again
     and nothing else in the suite would notice.
  2. A feature that only memorises a group loses its score under grouping. That
     is the leak signal the report relies on, and it is worth having a test that
     demonstrates the mechanism rather than merely asserting the plumbing.
"""

import os
import sys
import unittest
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlternatingPipeline.data.holdout import (  # noqa: E402
    holdout_mask,
    split_summary,
)
from AlternatingPipeline.data.parameter_analysis import (  # noqa: E402
    exam_group_labels,
    heldout_regressor_score,
    permutation_importance_mae,
)
from AlternatingPipeline.data.protocol_vocab import heldout_group_r2  # noqa: E402


class HoldoutMaskTests(unittest.TestCase):
    def test_without_groups_it_is_the_historical_random_split(self):
        # Same generator, same consumption pattern — so every number measured
        # before this module existed stays reproducible.
        expected = np.random.default_rng(7).random(500) >= 0.2
        actual = holdout_mask(500, 0.2, np.random.default_rng(7))
        np.testing.assert_array_equal(actual, expected)

    def test_no_group_is_ever_split_across_the_boundary(self):
        groups = np.repeat(np.arange(40), 25)
        for seed in range(10):
            is_train = holdout_mask(groups.size, 0.2, np.random.default_rng(seed),
                                    groups)
            leaked = set(groups[is_train]) & set(groups[~is_train])
            self.assertEqual(leaked, set(), f"seed {seed} split a group")

    def test_a_dominant_group_cannot_hijack_the_split(self):
        # One scanner with 5,000 rows and forty with 50 each — 71% of the corpus
        # in a single indivisible group. "Accumulate until the target is
        # exceeded" overshoots a 20% target to 86% whenever the big group is
        # drawn first, leaving a seventh of the data to train on. Closest-
        # approach declines it instead and fills up from the small groups.
        groups = np.concatenate([np.zeros(5000), np.repeat(np.arange(1, 41), 50)])
        for seed in range(20):
            is_train = holdout_mask(groups.size, 0.2, np.random.default_rng(seed),
                                    groups)
            self.assertAlmostEqual(1.0 - is_train.mean(), 0.2, places=6,
                                   msg=f"seed {seed}")

    def test_even_group_sizes_land_on_the_target(self):
        groups = np.repeat(np.arange(40), 25)
        for seed in range(10):
            is_train = holdout_mask(groups.size, 0.25, np.random.default_rng(seed),
                                    groups)
            self.assertAlmostEqual(1.0 - is_train.mean(), 0.25, places=6,
                                   msg=f"seed {seed}")

    def test_different_seeds_hold_out_different_groups(self):
        # Landing on the target every time must not mean picking the same rows
        # every time, or `repeats` would average one split with itself.
        groups = np.repeat(np.arange(40), 25)
        held = {
            frozenset(np.unique(groups[~holdout_mask(
                groups.size, 0.2, np.random.default_rng(s), groups)]).tolist())
            for s in range(10)
        }
        self.assertGreater(len(held), 5)

    def test_a_mislabelled_groups_array_raises_rather_than_silently_truncating(self):
        with self.assertRaises(ValueError) as caught:
            holdout_mask(100, 0.2, np.random.default_rng(0), np.arange(99))
        self.assertIn("label every row", str(caught.exception))

    def test_a_single_group_produces_a_degenerate_mask_rather_than_pretending(self):
        # One group cannot be split. Returning all-train is the correct answer;
        # the callers' `is_train.all()` guard is what turns it into an error.
        is_train = holdout_mask(100, 0.2, np.random.default_rng(0), np.zeros(100))
        self.assertTrue(is_train.all() or not is_train.any())


class SplitSummaryTests(unittest.TestCase):
    def test_grouped_split_reports_zero_leaked_groups(self):
        groups = np.repeat(np.arange(30), 20)
        is_train = holdout_mask(groups.size, 0.25, np.random.default_rng(3), groups)
        self.assertEqual(split_summary(is_train, groups)['leaked_groups'], 0)

    def test_a_row_split_is_caught_as_leaking(self):
        # The regression this whole module exists to prevent: if someone routes
        # around holdout_mask and splits by row, split_summary says so.
        groups = np.repeat(np.arange(30), 20)
        is_train = np.random.default_rng(3).random(groups.size) >= 0.25
        self.assertGreater(split_summary(is_train, groups)['leaked_groups'], 0)

    def test_counts_add_up(self):
        is_train = np.array([True, True, False, False, False])
        summary = split_summary(is_train)
        self.assertEqual(summary['n_rows'], 5)
        self.assertEqual(summary['n_train'], 2)
        self.assertEqual(summary['n_test'], 3)
        self.assertAlmostEqual(summary['test_row_frac'], 0.6)


class ExamGroupLabelTests(unittest.TestCase):
    def _seq(self, serial, day=None, protocol='p'):
        seq = {'serial_idx': serial, 'protocol_name': protocol}
        if day is not None:
            seq['start_datetime'] = datetime(2024, 1, day, 9, 0)
        return seq

    def test_serial_day_separates_two_days_on_one_scanner(self):
        labels = exam_group_labels(
            [self._seq(3, 1), self._seq(3, 1), self._seq(3, 2)], by='serial_day')
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_serial_day_separates_two_scanners_on_one_day(self):
        labels = exam_group_labels([self._seq(3, 1), self._seq(4, 1)],
                                   by='serial_day')
        self.assertNotEqual(labels[0], labels[1])

    def test_serial_ignores_the_date(self):
        labels = exam_group_labels([self._seq(3, 1), self._seq(3, 9)], by='serial')
        self.assertEqual(labels[0], labels[1])

    def test_undated_sequences_stand_alone_rather_than_merging(self):
        # Bucketing every undated row under one label would build a single giant
        # group out of unrelated scans, which is worse than not grouping at all.
        labels = exam_group_labels([self._seq(3), self._seq(3)], by='serial_day')
        self.assertNotEqual(labels[0], labels[1])

    def test_protocol_groups_by_name(self):
        labels = exam_group_labels(
            [self._seq(3, 1, 'A'), self._seq(9, 2, 'A'), self._seq(3, 1, 'B')],
            by='protocol')
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_an_unknown_grouping_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            exam_group_labels([self._seq(1, 1)], by='patient')


class GroupedScoringTests(unittest.TestCase):
    """A memorised group id must not survive a grouped split."""

    def _memorisation_corpus(self, n_groups=40, per_group=40, seed=0):
        rng = np.random.default_rng(seed)
        groups = np.repeat(np.arange(n_groups), per_group)
        # Duration depends ONLY on which group the row came from. Nothing about
        # the row itself predicts it, so the only way to score is to have seen
        # the group before — the exact shape of per-site memorisation.
        offsets = rng.normal(0, 100, n_groups)
        values = offsets[groups] + rng.normal(0, 1, groups.size)
        features = groups.reshape(-1, 1).astype(float)
        return features, values, groups

    def test_random_split_rewards_memorisation_and_grouped_split_does_not(self):
        features, values, groups = self._memorisation_corpus()

        _, random_mae = heldout_regressor_score(features, values, seed=0)
        _, grouped_mae = heldout_regressor_score(features, values, seed=0,
                                                 groups=groups)

        # The random split sees every group in training and nearly reproduces
        # the target; the grouped split has never seen the held-out groups.
        self.assertLess(random_mae, 10.0)
        self.assertGreater(grouped_mae, 4 * random_mae)

    def test_permutation_importance_deflates_a_memorised_column(self):
        features, values, groups = self._memorisation_corpus()

        random_rank = permutation_importance_mae(features, values, ['group_id'],
                                                 seed=0)
        grouped_rank = permutation_importance_mae(features, values, ['group_id'],
                                                  seed=0, groups=groups)

        self.assertGreater(random_rank[0]['importance_s'],
                           grouped_rank[0]['importance_s'])

    def test_group_mean_oracle_loses_coverage_when_groups_are_held_out(self):
        # The protocol oracle scored on a serial-grouped split: protocols unique
        # to a held-out scanner fall back to the global mean, so coverage drops.
        # This is the cost Görtler's objection predicts, made measurable.
        rng = np.random.default_rng(1)
        serials = np.repeat(np.arange(10), 200)
        # Each scanner has its own protocol catalogue — 4.1% overlap in reality.
        protocols = np.array([f"s{s}_p{rng.integers(0, 20)}" for s in serials])
        values = rng.normal(100, 30, serials.size)

        _, _, random_cov = heldout_group_r2(protocols, values, seed=0)
        _, _, grouped_cov = heldout_group_r2(protocols, values, seed=0,
                                             groups=serials)

        self.assertGreater(random_cov, 90.0)
        self.assertLess(grouped_cov, 5.0)

    def test_grouped_scoring_raises_a_readable_error_on_one_group(self):
        features = np.arange(100, dtype=float).reshape(-1, 1)
        values = np.arange(100, dtype=float)
        with self.assertRaises(ValueError) as caught:
            heldout_regressor_score(features, values, groups=np.zeros(100))
        self.assertIn("nothing to transfer to", str(caught.exception))


if __name__ == '__main__':
    unittest.main()
