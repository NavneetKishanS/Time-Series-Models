"""Is a rare parameter's PRESENCE tied to one sequence family? — Görtler, 2026-08-10.

His argument for keeping sub-1% parameters was mechanistic, not statistical:
`PDM` occurs almost only on cardiac sequences, so it may be a near-perfect
indicator of that family even though it is vanishingly rare.

That is a checkable claim, and checking it costs no training run — which matters,
because the alternative is spending GPU hours on an arm whose premise nobody has
verified. `presence_concentration` is the check. It separates three cases that
"the field is rare" does not distinguish:

  * DIFFUSE       — presence is spread across families. Görtler's mechanism does
                    not hold for this field; its flag carries little.
  * REDUNDANT     — presence is tied to a family AND covers essentially all of
                    it, so `sequence_type` already tells the model everything the
                    flag would. This is the most likely way the arm dies, and
                    finding it cheaply is the point.
  * INFORMATIVE   — presence is tied to a family but marks only a SUBSET of it.
                    The flag is a refinement `sequence_type` cannot express, and
                    this is the case worth training on.

The distinction is two conditional probabilities in opposite directions, and
getting them the wrong way round would make a redundant field look informative.
Hence this file.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from AlternatingPipeline.data.parameter_analysis import (  # noqa: E402
    presence_concentration,
)


def _sequences(spec):
    """Build rows from {sequence_type: [list of per-row sut_raw dicts]}."""
    rows = []
    for seq_type, raws in spec.items():
        for raw in raws:
            rows.append({'sequence_type': seq_type, 'sut_raw': dict(raw)})
    return rows


class ConcentrationTests(unittest.TestCase):

    def test_a_field_confined_to_one_family_reads_as_fully_concentrated(self):
        # PDM on cardiac only — Görtler's claim, in its strongest form.
        sequences = _sequences({
            'cardiac': [{'PDM': '1'}] * 20,
            'tse': [{'TR': '500'}] * 480,
        })
        rows = {r['name']: r for r in presence_concentration(sequences, ['PDM'])}
        self.assertAlmostEqual(rows['PDM']['top_share'], 1.0)
        self.assertEqual(rows['PDM']['top_category'], 'cardiac')
        self.assertEqual(rows['PDM']['presence_rows'], 20)

    def test_a_field_spread_evenly_reads_as_diffuse(self):
        sequences = _sequences({
            'cardiac': [{'X': '1'}] * 25,
            'tse': [{'X': '1'}] * 25,
            'epi': [{'X': '1'}] * 25,
            'haste': [{'X': '1'}] * 25,
        })
        rows = {r['name']: r for r in presence_concentration(sequences, ['X'])}
        self.assertAlmostEqual(rows['X']['top_share'], 0.25)
        self.assertEqual(rows['X']['verdict'], 'diffuse')

    def test_the_two_directions_are_not_confused(self):
        # THE test. Both fields are 100% concentrated in 'cardiac', so top_share
        # alone cannot tell them apart — and top_share alone is what a naive
        # reading of "PDM only occurs in heart sequences" would check.
        #
        #   FULL  is on every cardiac row     -> sequence_type already says it
        #   HALF  is on half the cardiac rows -> marks a subset, genuinely new
        sequences = _sequences({
            'cardiac': [{'FULL': '1', 'HALF': '1'}] * 10 + [{'FULL': '1'}] * 10,
            'tse': [{'TR': '500'}] * 480,
        })
        rows = {r['name']: r
                for r in presence_concentration(sequences, ['FULL', 'HALF'])}

        self.assertAlmostEqual(rows['FULL']['top_share'], 1.0)
        self.assertAlmostEqual(rows['HALF']['top_share'], 1.0)

        self.assertAlmostEqual(rows['FULL']['coverage_in_top'], 1.0)
        self.assertAlmostEqual(rows['HALF']['coverage_in_top'], 0.5)

        self.assertEqual(rows['FULL']['verdict'], 'redundant')
        self.assertEqual(rows['HALF']['verdict'], 'informative')

    def test_a_redundant_field_is_named_as_such_not_praised_for_concentration(self):
        # The failure this whole helper exists to prevent: reporting "PDM is
        # 100% concentrated in cardiac!" as support for adding its flag, when
        # sequence_type already carries exactly that information.
        sequences = _sequences({
            'cardiac': [{'PDM': '1'}] * 30,
            'tse': [{'TR': '500'}] * 470,
        })
        row = presence_concentration(sequences, ['PDM'])[0]
        self.assertEqual(row['verdict'], 'redundant')
        self.assertGreater(row['nmi'], 0.9)

    def test_nmi_is_zero_when_presence_is_independent_of_family(self):
        # Half of every family carries the field.
        sequences = _sequences({
            'cardiac': [{'X': '1'}] * 50 + [{}] * 50,
            'tse': [{'X': '1'}] * 50 + [{}] * 50,
        })
        row = presence_concentration(sequences, ['X'])[0]
        self.assertLess(row['nmi'], 0.01)
        self.assertEqual(row['verdict'], 'diffuse')

    def test_nmi_is_one_when_presence_exactly_identifies_a_family(self):
        sequences = _sequences({
            'cardiac': [{'X': '1'}] * 100,
            'tse': [{}] * 100,
        })
        row = presence_concentration(sequences, ['X'])[0]
        self.assertAlmostEqual(row['nmi'], 1.0, places=6)


class EdgeCaseTests(unittest.TestCase):
    """A report gate must not take down a 20-minute notebook run."""

    def test_a_field_absent_from_every_row_does_not_divide_by_zero(self):
        sequences = _sequences({'tse': [{'TR': '500'}] * 10})
        row = presence_concentration(sequences, ['GHOST'])[0]
        self.assertEqual(row['presence_rows'], 0)
        self.assertEqual(row['verdict'], 'absent')
        self.assertEqual(row['nmi'], 0.0)

    def test_a_field_on_every_row_is_not_called_informative(self):
        # Present everywhere carries no information about family, and its flag
        # would be a constant column.
        sequences = _sequences({
            'cardiac': [{'X': '1'}] * 50,
            'tse': [{'X': '1'}] * 50,
        })
        row = presence_concentration(sequences, ['X'])[0]
        self.assertEqual(row['nmi'], 0.0)
        self.assertEqual(row['verdict'], 'diffuse')

    def test_an_empty_corpus_returns_empty_rather_than_raising(self):
        self.assertEqual(presence_concentration([], ['X']), [])

    def test_rows_come_back_ordered_by_how_concentrated_they_are(self):
        # The report prints the head of this list, so the ordering IS which
        # fields a reader sees.
        sequences = _sequences({
            'cardiac': [{'TIGHT': '1'}] * 10 + [{'LOOSE': '1'}] * 10,
            'tse': [{'LOOSE': '1'}] * 10,
        })
        names = [r['name'] for r in presence_concentration(
            sequences, ['LOOSE', 'TIGHT'])]
        self.assertEqual(names[0], 'TIGHT')

    def test_a_missing_sequence_type_key_is_its_own_category_not_a_crash(self):
        sequences = [{'sut_raw': {'X': '1'}}, {'sut_raw': {}}]
        row = presence_concentration(sequences, ['X'])[0]
        self.assertEqual(row['presence_rows'], 1)


class CategoryKeyTests(unittest.TestCase):
    def test_the_categorical_is_selectable(self):
        # The same question is worth asking of body_group, which is the other
        # stratification the 08-10 meeting asked for.
        sequences = [
            {'body_group': 'HEAD', 'sut_raw': {'X': '1'}},
            {'body_group': 'LEG', 'sut_raw': {}},
        ]
        row = presence_concentration(sequences, ['X'],
                                     category_key='body_group')[0]
        self.assertEqual(row['top_category'], 'HEAD')


if __name__ == '__main__':
    unittest.main()
