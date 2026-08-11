"""The vendor parameter table — MR sequence development, 2026-08-11.

188 parameters with a description and an authoritative `isNumeric` flag, sent
after the 08-11 review. Two things it fixes, and the tests are split that way:

  * MEANING. The 08-11 report had to say "122 admitted fields have no entry in
    SUT_FIELD_MAP — we USE them but cannot say what they mean". That is fine for
    training and blocking for step 07, which must synthesise every field it
    conditions on. The #2 field in the whole importance ranking was `IPS` and
    nobody in the meeting could say what it was.

  * NUMERACY, which is the one with teeth. Our own numeric test is empirical —
    "does this parse as a float on 90% of rows" — and a category code written as
    an integer passes it trivially. `N0`/`N1` (nucleus) and `CSC` are exactly
    that, and with the presence floor now at 0.0 they would enter the
    conditioning vector as scaled numerics. Nothing in the corpus can tell them
    apart from a measurement; only the vendor flag can.

The table is REFERENCE, not identity: it must not become SUT_FIELD_MAP, because
that map decides a column's stable name and renaming a column repoints every
learned weight in a trained checkpoint.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_sut_parser import _load_config  # noqa: E402


class TableShapeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_the_table_is_loaded_and_the_right_size(self):
        # If the table ever shrinks silently, every assertion below softens
        # without failing.
        self.assertGreaterEqual(len(self.config.SUT_FIELD_DESCRIPTIONS), 180)

    def test_every_entry_is_a_description_and_a_numeric_flag(self):
        for name, value in self.config.SUT_FIELD_DESCRIPTIONS.items():
            with self.subTest(field=name):
                self.assertEqual(len(value), 2)
                description, is_numeric = value
                self.assertTrue(description)
                self.assertIsInstance(is_numeric, bool)

    def test_the_answers_from_the_08_11_review_are_recorded(self):
        # The two questions that were blocking, both answered in that meeting.
        self.assertEqual(self.config.SUT_FIELD_DESCRIPTIONS['IPS'][0],
                         'ImagesPerSlab')
        self.assertEqual(self.config.SUT_FIELD_DESCRIPTIONS['BHD'][0],
                         'breathholdDuration')

    def test_it_confirms_the_denylists_we_reasoned_our_way_to(self):
        # We denied these from evidence, before anybody had the vendor table.
        # The table agreeing is a real check on that reasoning.
        for field, expected in (('ST', 'scan time'),
                                ('TST', 'total scan time'),
                                ('MUID', 'MeasUID'),
                                ('VER', 'version number'),
                                ('SNR', 'SNR estimate'),
                                ('TEU', 'TE in us')):
            with self.subTest(field=field):
                self.assertEqual(self.config.SUT_FIELD_DESCRIPTIONS[field][0],
                                 expected)


class LookupTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_it_resolves_a_raw_mnemonic(self):
        self.assertEqual(self.config.seqparam_description('TF'), 'turbo factor')

    def test_it_resolves_a_mapped_stable_name(self):
        # The classifier and the report work in STABLE names (`num_slices`),
        # while the vendor ships raw mnemonics (`SLC`). A lookup that only
        # handled one of them would silently report every curated field as
        # undocumented — the exact fields we understand best.
        self.assertEqual(self.config.seqparam_description('num_slices'), 'slices')
        self.assertEqual(self.config.seqparam_description('turbo_factor'),
                         'turbo factor')

    def test_an_unknown_field_returns_None_rather_than_inventing_one(self):
        self.assertIsNone(self.config.seqparam_description('NOT_A_REAL_FIELD'))


class VendorNumeracyOverridesOurSampleTests(unittest.TestCase):
    """The case our own test cannot see."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_the_six_non_numeric_fields_are_identified(self):
        self.assertEqual(
            set(self.config.SUT_VENDOR_NON_NUMERIC),
            {'CS', 'DLL', 'OR', 'N0', 'N1', 'CSC'})

    def test_a_category_code_that_parses_as_a_number_is_still_refused(self):
        # THE test. N0/N1/CSC are enumerations written as integers: 100% numeric
        # by our empirical measure, and meaningless as a scaled column. Before
        # the vendor table they would have been admitted.
        for field in ('N0', 'N1', 'CSC'):
            with self.subTest(field=field):
                verdict = self.config.classify_seqparam_field(
                    field, {'presence_pct': 99.0, 'numeric_pct': 100.0,
                            'presence_rows': 50000})
                self.assertEqual(verdict.verdict, 'excluded')
                self.assertEqual(verdict.category, 'non_numeric')
                self.assertIn('vendor', verdict.reason)

    def test_the_vendor_refusal_survives_a_zero_floor(self):
        # The floor is 0.0 as of 2026-08-11, so "too rare" no longer excludes
        # anything. Nothing but the vendor flag stands between a category code
        # and the conditioning vector.
        for field in ('N0', 'N1', 'CSC'):
            with self.subTest(field=field):
                self.assertNotIn(field, self.config.SEQPARAM_ADMISSIBLE_DISCOVERED)
                self.assertNotIn(field, self.config.EXAMINATION_SEQPARAM_FEATURES)

    def test_a_real_measurement_is_not_caught_by_the_override(self):
        # The counterweight: the vendor flag must not be eating physics.
        for field in ('TR', 'TF', 'PEL', 'SLC', 'FOV', 'IPS', 'BHD'):
            with self.subTest(field=field):
                self.assertNotIn(field, self.config.SUT_VENDOR_NON_NUMERIC)


class ReferenceNotIdentityTests(unittest.TestCase):
    """The table must not start renaming columns."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_the_vendor_table_does_not_change_any_stable_name(self):
        # A stable name IS the identity of a column in the conditioning vector
        # and in the divisor table. Renaming `IPS` to `images_per_slab` for
        # readability would repoint every learned weight in a trained
        # checkpoint. Identity lives in SUT_FIELD_MAP; meaning lives in the
        # vendor table; they must stay separate.
        for name in self.config.SUT_FIELD_DESCRIPTIONS:
            with self.subTest(field=name):
                expected = self.config.SUT_FIELD_MAP.get(name, name)
                self.assertEqual(self.config.seqparam_stable_name(name), expected)

    def test_documented_but_unmapped_fields_keep_their_raw_name(self):
        # IPS is documented and deliberately NOT in SUT_FIELD_MAP.
        self.assertNotIn('IPS', self.config.SUT_FIELD_MAP)
        self.assertEqual(self.config.seqparam_stable_name('IPS'), 'IPS')
        self.assertIsNotNone(self.config.seqparam_description('IPS'))


class MeetingDecisionsTests(unittest.TestCase):
    """What the 2026-08-11 review actually decided, pinned so a later edit has
    to argue with it rather than drift past it."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_the_presence_floor_is_zero(self):
        # "Keep the 1% floor cases in the dataset" — the four arms in gate 4b
        # were indistinguishable, so the meeting took the option that keeps the
        # data. ~170 fields where the 1% floor admitted 133.
        self.assertEqual(self.config.SEQPARAM_MIN_PRESENCE_PCT, 0.0)

    def test_the_parameter_set_is_still_all(self):
        self.assertEqual(self.config.PARAM_SET, 'all')

    def test_the_corpus_is_21_serials(self):
        self.assertEqual(len(self.config.SERIAL_NUMBERS), 21)
        self.assertEqual(len(set(self.config.SERIAL_NUMBERS)), 21,
                         "a duplicated serial would silently shrink the corpus")

    def test_it_matches_the_old_pipeline_for_a_like_for_like_comparison(self):
        # The whole point of the seqparams fork is comparability against the
        # csv_pipeline model on identical scanners and dates.
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))),
            'DatabricksPipeline', 'csv_pipeline', 'config.py')
        namespace = {}
        with open(path) as handle:
            source = handle.read()
        # config.py's serial list is a plain literal at the top; exec'ing the
        # whole module would need Databricks, so take just that statement.
        start = source.index('SERIAL_NUMBERS')
        end = source.index(']', start) + 1
        exec(source[start:end], namespace)
        self.assertEqual(sorted(namespace['SERIAL_NUMBERS']),
                         sorted(self.config.SERIAL_NUMBERS))

    def test_bhd_is_cleared_not_denied(self):
        # Görtler: it is an upper LIMIT per breath-hold step, not the step
        # duration — a cause, not a copy. It must not be sitting in a denylist.
        self.assertNotIn('BHD', self.config.SUT_ALL_DENYLISTS)
        self.assertIn('BHD', self.config.SUT_SUSPECT_WATCHLIST)
        self.assertIn('UPPER LIMIT', self.config.SUT_SUSPECT_WATCHLIST['BHD'])


if __name__ == '__main__':
    unittest.main()
