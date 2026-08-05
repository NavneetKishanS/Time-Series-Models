"""Segment construction and SUT join in csv_pipeline_seqparams/03.

Both behaviours under test were shown to be broken by
`03c_join_and_target_diagnostics.py` against real data on 2026-08-03:

  * Section A — the SUT-derived sequence agreed with the actual scanned
    sequence on only 45.7% of rows. Adjustment sequences were near-zero
    (AdjGreSeq 0.0%, tfl_b1map 1.5%, AALScout 6.9%), i.e. SUT fires for an
    adjustment step and the old "most recent event strictly before the
    segment" join carried it forward onto the *next*, different measurement.

  * Section B2/C — 16.9% of segments shared a terminator with another
    segment, because every MRI_MSR_100 was bound to the next MSR_104/34.
    Keeping one segment per terminator moved held-out serial+protocol MAE
    from 24.0s to 15.4s, recovering the ±15s benchmark.

These tests pin the fixed behaviour of the two helpers that implement it.
"""

import ast
import os
import unittest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STEP03_PATH = os.path.join(
    _REPO_ROOT, 'DatabricksPipeline', 'csv_pipeline_seqparams',
    '03_build_preprocessed_pkl.py',
)


def _load_segment_helpers():
    """Extract just the segment/join helpers from the Databricks notebook.

    Same approach as test_sut_parser.py: the rest of the file queries Spark at
    module level and cannot be imported outside Databricks, so the functions
    are pulled out via `ast` and exec'd. Using the real source keeps the test
    tied to the actual implementation rather than a hand-copied duplicate.
    """
    import bisect

    with open(_STEP03_PATH) as f:
        source = f.read()
    wanted = {'_segment_bounds', '_choose_sut_row'}
    namespace = {'bisect': bisect}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.get_source_segment(source, node), _STEP03_PATH, 'exec'),
                 namespace)
    missing = wanted - set(namespace)
    if missing:
        raise AssertionError(f"step 03 is missing {sorted(missing)}")
    return namespace


class SegmentBoundsTests(unittest.TestCase):
    """One measurement per terminator, dated from the most recent MSR_100."""

    def setUp(self):
        self.bounds = _load_segment_helpers()['_segment_bounds']

    def test_each_start_is_bound_to_the_next_terminator(self):
        # Rows 0 and 10 start; rows 5 and 15 terminate. Nothing is shared, so
        # both segments survive untouched.
        self.assertEqual(
            self.bounds(start_rows=[0, 10], end_rows=[5, 15], n_rows=20),
            [(0, 5), (10, 15)],
        )

    def test_two_starts_before_one_terminator_keep_only_the_later_start(self):
        # THE BUG: rows 0 and 3 both bind to the terminator at row 8, producing
        # two overlapping segments that both claim the same measurement. The
        # later MSR_100 is the one that actually ran (csv_pipeline/02 dates the
        # row from the most recent MSR_100), so only (3, 8) is real.
        self.assertEqual(
            self.bounds(start_rows=[0, 3], end_rows=[8], n_rows=12),
            [(3, 8)],
        )

    def test_dedupe_disabled_reproduces_the_overlapping_segments(self):
        # The flag exists so the 16.9% of rows this drops can be measured again
        # rather than taken on faith.
        self.assertEqual(
            self.bounds(start_rows=[0, 3], end_rows=[8], n_rows=12, dedupe=False),
            [(0, 8), (3, 8)],
        )

    def test_start_without_a_terminator_ends_before_the_next_start(self):
        # Row 4 has no terminator left after it, so it must stop before row 8's
        # measurement rather than swallowing it.
        self.assertEqual(
            self.bounds(start_rows=[0, 4, 8], end_rows=[2], n_rows=12),
            [(0, 2), (4, 7), (8, 11)],
        )

    def test_final_start_without_a_terminator_runs_to_the_last_row(self):
        self.assertEqual(
            self.bounds(start_rows=[6], end_rows=[], n_rows=10),
            [(6, 9)],
        )

    def test_three_starts_sharing_a_terminator_collapse_to_one(self):
        self.assertEqual(
            self.bounds(start_rows=[0, 2, 4], end_rows=[7], n_rows=10),
            [(4, 7)],
        )


class ChooseSutRowTests(unittest.TestCase):
    """The SUT event that describes THIS measurement, not the previous one."""

    def setUp(self):
        self.choose = _load_segment_helpers()['_choose_sut_row']

    def test_prefers_the_sut_event_inside_the_segment(self):
        # Row 2 belongs to whatever ran before this segment; row 6 is inside
        # the measurement and is the one describing it.
        self.assertEqual(
            self.choose(sut_rows=[2, 6], start_row=5, end_row=9),
            (6, 'inside', False),
        )

    def test_takes_the_first_sut_event_when_several_are_inside(self):
        # The sequence is loaded once at the start of the measurement; later
        # events in the same segment are re-reports.
        self.assertEqual(
            self.choose(sut_rows=[6, 7, 8], start_row=5, end_row=9),
            (6, 'inside', False),
        )

    def test_falls_back_to_the_most_recent_event_before_the_segment(self):
        # Coverage matters more than purity: with no event inside, the old
        # join is still the best guess — but it is labelled so the next
        # diagnostic can score the two scopes separately.
        self.assertEqual(
            self.choose(sut_rows=[1, 2], start_row=5, end_row=9),
            (2, 'before', False),
        )

    def test_reports_none_when_no_sut_event_precedes_the_segment(self):
        self.assertEqual(
            self.choose(sut_rows=[20], start_row=5, end_row=9),
            (None, 'none', False),
        )

    def test_an_event_on_the_segment_start_row_counts_as_inside(self):
        # MRI_SUT_1005 and MRI_MSR_100 can share a timestamp; the boundary must
        # not push the segment's own event into the 'before' bucket.
        self.assertEqual(
            self.choose(sut_rows=[5], start_row=5, end_row=9),
            (5, 'inside', False),
        )


class SkipAdjustmentSequenceTests(unittest.TestCase):
    """A gradient adjustment's parameters are never the measurement's.

    The 2026-08-04 run of 03c section A2 put in-segment agreement at 94.8%,
    and `AdjGreSeq` accounts for essentially ALL of the remaining error: 2,061
    rows at 0.0% agreement against ~2,019 disagreeing in-segment rows in total.
    Every one of them says the actual scan was `tse`. So the adjustment runs at
    the very start of a real measurement (in-segment offset is p50 0.0s), emits
    its own MRI_SUT_1005 first, and wins the 'first event inside' rule.
    """

    def setUp(self):
        self.choose = _load_segment_helpers()['_choose_sut_row']

    def test_skips_an_adjustment_event_and_takes_the_next_one_inside(self):
        # Row 6 is the adjustment, row 8 is the measurement's own event.
        self.assertEqual(
            self.choose(sut_rows=[6, 8], start_row=5, end_row=9, skip={6}),
            (8, 'inside', True),
        )

    def test_falls_back_to_before_when_only_adjustments_are_inside(self):
        # No usable in-segment event, so the old rule applies. Better to label
        # the row 'before' — a scope we already know to distrust — than to hand
        # the model an adjustment's parameters as if they were the scan's.
        self.assertEqual(
            self.choose(sut_rows=[2, 6], start_row=5, end_row=9, skip={6}),
            (2, 'before', True),
        )

    def test_the_before_fallback_also_skips_adjustments(self):
        # Row 4 is the most recent before the segment but is an adjustment, so
        # the search continues backwards to row 2.
        self.assertEqual(
            self.choose(sut_rows=[2, 4], start_row=5, end_row=9, skip={4}),
            (2, 'before', False),
        )

    def test_reports_none_when_every_candidate_is_an_adjustment(self):
        self.assertEqual(
            self.choose(sut_rows=[2, 6], start_row=5, end_row=9, skip={2, 6}),
            (None, 'none', True),
        )


class InsideSkippedFlagTests(unittest.TestCase):
    """`skipped_inside` separates two populations that look identical in the
    scope counts and need opposite handling.

    Step 03 reports 80.3% 'inside' / 18.7% 'before' / 1.0% 'none'. A segment in
    the last two either lost its message (a coverage deficit a wider join rule
    could fix) or never had one because the segment IS the adjustment (a
    population that does not belong in the coverage denominator at all). The
    scope alone cannot tell them apart; this flag can.
    """

    def setUp(self):
        self.choose = _load_segment_helpers()['_choose_sut_row']

    def test_false_when_nothing_was_passed_over(self):
        """The plain missing-coverage case: no in-segment event existed."""
        _, scope, skipped = self.choose(
            sut_rows=[2], start_row=5, end_row=9, skip=set())
        self.assertEqual(scope, 'before')
        self.assertFalse(skipped)

    def test_true_when_the_only_in_segment_event_was_an_adjustment(self):
        """The segment is very likely the adjustment itself."""
        _, scope, skipped = self.choose(
            sut_rows=[2, 6], start_row=5, end_row=9, skip={6})
        self.assertEqual(scope, 'before')
        self.assertTrue(skipped)

    def test_true_even_when_a_later_in_segment_event_wins(self):
        """Recorded regardless of scope. It is only READ on non-'inside' rows —
        making it conditional on the outcome would hide the mechanism from
        anyone reading the flag on a row that joined successfully."""
        _, scope, skipped = self.choose(
            sut_rows=[6, 8], start_row=5, end_row=9, skip={6})
        self.assertEqual(scope, 'inside')
        self.assertTrue(skipped)

    def test_an_adjustment_before_the_segment_does_not_set_it(self):
        """Only events INSIDE the segment count. An adjustment that ran before
        the measurement says nothing about whether this segment is one."""
        _, scope, skipped = self.choose(
            sut_rows=[2, 4], start_row=5, end_row=9, skip={4})
        self.assertEqual(scope, 'before')
        self.assertFalse(skipped)

    def test_an_adjustment_after_the_segment_does_not_set_it(self):
        """The in-segment scan stops at end_row, so a later adjustment is never
        examined and must not be counted as passed over."""
        _, scope, skipped = self.choose(
            sut_rows=[2, 12], start_row=5, end_row=9, skip={12})
        self.assertEqual(scope, 'before')
        self.assertFalse(skipped)


if __name__ == '__main__':
    unittest.main()
