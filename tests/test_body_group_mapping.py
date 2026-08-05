"""Body-group resolution in csv_pipeline_seqparams/03.

The 2026-08-05 run printed this, for every one of the ten scanners:

    [serial 175670]
      total_labels=11,880  known=11,823  unknown=57  unknown_pct=0.5%
      KNOWN:
        None                            11,823  ->  LEG  (exam_workflow)

Every exam on that scanner resolved to LEG, every exam on 176227 to HEAD, every
exam on 182625 to ARM — and the summary called it KNOWN while the raw label it
printed was `None`. No scanner images one body part for a month.

Three chained defects produced it, and these tests pin the two that are fixable
without Spark:

  1. `BodyPartExamined` is null on essentially every exam row — the
     WorkflowValues map does not carry any of the three key names step 03
     coalesces over. Step 03 now says so loudly and enumerates the keys that
     ARE present; that one needs the next run to close.
  2. The MRI_EXU_95 fallback forward-filled across a whole serial, so a single
     successful regex match painted every exam after it. Now bounded to the
     exam. -> FillBodyGroupWithinExamTests
  3. The sanity summary bucketed on whether the FINAL group was UNKNOWN, so a
     value invented by that forward fill counted as KNOWN. Now bucketed by
     provenance. -> BodyGroupSanityTests
"""

import ast
import os
import re
import unittest

import pandas as pd


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STEP03_PATH = os.path.join(
    _REPO_ROOT, 'DatabricksPipeline', 'csv_pipeline_seqparams',
    '03_build_preprocessed_pkl.py',
)


def _load_body_helpers():
    """Extract the body-group helpers from the notebook via `ast`.

    Same approach as test_segment_join.py / test_sut_parser.py: the rest of the
    file queries Spark at module level, so the functions are pulled out and
    exec'd against the real source rather than duplicated here.
    """
    with open(_STEP03_PATH) as f:
        source = f.read()
    wanted = {
        '_normalize_body_label', '_canonical_body_label', '_bg_from_evu95',
        '_fill_body_group_within_exam', '_record_body_group_sanity',
        '_print_body_group_sanity',
    }
    namespace = {'re': re, 'pd': pd, '_BODY_GROUP_SANITY': {}}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.get_source_segment(source, node), _STEP03_PATH, 'exec'),
                 namespace)
        # BODY_GROUP_SOURCES / BODY_GROUP_SOURCE_NOTES are read by the printer.
        if isinstance(node, ast.Assign) and any(
            getattr(t, 'id', '').startswith('BODY_GROUP_SOURCE') for t in node.targets
        ):
            exec(compile(ast.get_source_segment(source, node), _STEP03_PATH, 'exec'),
                 namespace)
    missing = wanted - set(namespace)
    if missing:
        raise AssertionError(f"step 03 is missing {sorted(missing)}")
    return namespace


class FillBodyGroupWithinExamTests(unittest.TestCase):
    """A body part observed in one exam says nothing about the next one."""

    def setUp(self):
        self.fill = _load_body_helpers()['_fill_body_group_within_exam']

    def _frame(self, exams, values):
        return pd.DataFrame({
            'WorkflowStartRefDateTime': exams,
            '_bg_msg': values,
        })

    def test_carries_a_value_forward_inside_one_exam(self):
        """The fallback still has to work — this is not a revert."""
        frame = self._frame(['e1', 'e1', 'e1'], ['HEAD', None, None])
        self.assertEqual(list(self.fill(frame)), ['HEAD', 'HEAD', 'HEAD'])

    def test_does_not_leak_across_an_exam_boundary(self):
        """The defect, stated directly: exam 2 saw no body part and must not
        inherit exam 1's."""
        frame = self._frame(['e1', 'e1', 'e2', 'e2'], ['HEAD', None, None, None])
        filled = list(self.fill(frame))
        self.assertEqual(filled[:2], ['HEAD', 'HEAD'])
        self.assertTrue(pd.isna(filled[2]) and pd.isna(filled[3]))

    def test_each_exam_keeps_its_own_value(self):
        frame = self._frame(['e1', 'e1', 'e2', 'e2'],
                            ['HEAD', None, 'KNEE', None])
        self.assertEqual(list(self.fill(frame)), ['HEAD', 'HEAD', 'KNEE', 'KNEE'])

    def test_rows_before_the_first_exam_stay_unresolved(self):
        """Events preceding any exam have a null key. Leaving them unresolved
        is the point: they have no exam whose body part they could share."""
        frame = self._frame([None, None, 'e1'], [None, None, 'HEAD'])
        filled = list(self.fill(frame))
        self.assertTrue(pd.isna(filled[0]) and pd.isna(filled[1]))
        self.assertEqual(filled[2], 'HEAD')

    def test_one_match_cannot_paint_a_whole_serial(self):
        """The shape of the real failure, at scale: one hit in the first exam
        of a hundred. Under the old unbounded ffill all hundred came back
        HEAD."""
        exams = [f'e{i}' for i in range(100) for _ in range(5)]
        values = ['HEAD'] + [None] * 499
        resolved = self.fill(self._frame(exams, values)).notna().sum()
        self.assertEqual(resolved, 5)


class BgFromEvu95Tests(unittest.TestCase):
    """The regex that reads a body part out of an MRI_EXU_95 message."""

    def setUp(self):
        self.parse = _load_body_helpers()['_bg_from_evu95']
        self.mapping = {'HEAD': 'HEAD', 'KNEE': 'LEG', 'LOWER LEG': 'LEG'}

    def test_reads_a_single_word_body_part(self):
        message = "Examination started with body part < HEAD > at 08:00"
        self.assertEqual(self.parse(message, self.mapping), 'HEAD')

    def test_reads_a_multi_word_body_part(self):
        message = "Examination started with body part < LOWER LEG > < x >"
        self.assertEqual(self.parse(message, self.mapping), 'LEG')

    def test_unmapped_body_part_returns_none_not_a_guess(self):
        message = "Examination started with body part < ELBOW >"
        self.assertIsNone(self.parse(message, self.mapping))

    def test_message_without_the_pattern_returns_none(self):
        self.assertIsNone(self.parse("Examination started", self.mapping))

    def test_non_string_input_returns_none(self):
        self.assertIsNone(self.parse(None, self.mapping))
        self.assertIsNone(self.parse(float('nan'), self.mapping))


class BodyGroupSanityTests(unittest.TestCase):
    """The summary has to make a degraded mapping visible, not tidy it away."""

    def setUp(self):
        self.helpers = _load_body_helpers()
        self.record = self.helpers['_record_body_group_sanity']
        self.state = self.helpers['_BODY_GROUP_SANITY']
        self.state.clear()

    def _print(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.helpers['_print_body_group_sanity']()
        return buffer.getvalue()

    def test_counts_are_kept_separate_by_provenance(self):
        self.record('175670', 'HEAD', 'bodypart', 10)
        self.record('175670', 'HEAD', 'exu95', 90)
        output = self._print()
        self.assertIn('bodypart=10', output)
        self.assertIn('exu95=90', output)

    def test_flags_a_serial_that_collapsed_to_one_group(self):
        """The 2026-08-05 signature. It must be called out, not summarised as
        'no UNKNOWN labels recorded'."""
        self.record('175670', 'LEG', 'exu95', 11823)
        self.assertIn('COLLAPSED', self._print())

    def test_does_not_flag_a_serial_with_a_real_spread(self):
        for group in ('HEAD', 'LEG', 'ABDOMEN', 'SPINE'):
            self.record('176227', group, 'bodypart', 500)
        self.assertNotIn('COLLAPSED', self._print())

    def test_a_small_serial_is_not_flagged_on_one_group(self):
        """A scanner with a handful of exams can legitimately image one region;
        the flag is for a month of them."""
        self.record('999999', 'HEAD', 'bodypart', 12)
        self.assertNotIn('COLLAPSED', self._print())

    def test_unknown_groups_do_not_count_toward_the_spread(self):
        """UNKNOWN is not a body group. A serial that is one real group plus a
        pile of UNKNOWNs has still collapsed."""
        self.record('175670', 'LEG', 'exu95', 11000)
        self.record('175670', 'UNKNOWN', 'none', 800)
        self.assertIn('COLLAPSED', self._print())

    def test_reports_the_provenance_mix_so_a_guessed_mapping_is_visible(self):
        """The diagnostic question: how much of this came from the scanner?"""
        self.record('175670', 'LEG', 'exu95', 9000)
        self.record('175670', 'HEAD', 'protocol', 1000)
        output = self._print()
        self.assertIn('exu95', output)
        self.assertIn('protocol', output)
        self.assertNotIn('bodypart=', output)


if __name__ == '__main__':
    unittest.main()
