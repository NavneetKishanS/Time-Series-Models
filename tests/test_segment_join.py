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
            (6, 'inside'),
        )

    def test_takes_the_first_sut_event_when_several_are_inside(self):
        # The sequence is loaded once at the start of the measurement; later
        # events in the same segment are re-reports.
        self.assertEqual(
            self.choose(sut_rows=[6, 7, 8], start_row=5, end_row=9),
            (6, 'inside'),
        )

    def test_falls_back_to_the_most_recent_event_before_the_segment(self):
        # Coverage matters more than purity: with no event inside, the old
        # join is still the best guess — but it is labelled so the next
        # diagnostic can score the two scopes separately.
        self.assertEqual(
            self.choose(sut_rows=[1, 2], start_row=5, end_row=9),
            (2, 'before'),
        )

    def test_reports_none_when_no_sut_event_precedes_the_segment(self):
        self.assertEqual(
            self.choose(sut_rows=[20], start_row=5, end_row=9),
            (None, 'none'),
        )

    def test_an_event_on_the_segment_start_row_counts_as_inside(self):
        # MRI_SUT_1005 and MRI_MSR_100 can share a timestamp; the boundary must
        # not push the segment's own event into the 'before' bucket.
        self.assertEqual(
            self.choose(sut_rows=[5], start_row=5, end_row=9),
            (5, 'inside'),
        )


if __name__ == '__main__':
    unittest.main()
