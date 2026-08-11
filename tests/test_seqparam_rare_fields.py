"""The rare-parameter admission rule — Görtler's <1% challenge, 2026-08-10.

He argued the 1% presence floor is the wrong gate, with a concrete mechanism:
`PDM` occurs almost only on cardiac sequences, so despite being vanishingly rare
it may be a near-perfect INDICATOR of that family. And rare parameters belong to
new sequence types whose share grows, so a static ratio ages badly.

Two things follow, and this file pins both:

  * His example is about a field's PRESENCE, not its VALUE. A 0/1 flag is
    learnable from far fewer rows than a continuous slope, so "drop the field"
    and "admit the field" are not the only two options — `presence_only` is a
    third, and it is the arm that costs almost nothing.
  * Learnability tracks the ABSOLUTE row count; the floor tracks a ratio that
    stays flat as the corpus grows. So the rule gains a count escape hatch,
    which is what makes it self-correcting as serial numbers are added rather
    than something somebody has to remember to revisit.

The one thing that must NOT move: no rare mode may readmit a denylisted field.
`ST` is the target. Widening admission is exactly the kind of change that
quietly reopens a leak, so that is asserted against every mode rather than once.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_sut_parser import _load_config  # noqa: E402


def _config_with(**env):
    """Load config.py under a temporary environment.

    config.py resolves its thresholds at import, so a rare-mode test has to
    re-exec the module rather than poke an attribute — poking would leave
    SEQPARAM_ADMISSIBLE_DISCOVERED and EXAMINATION_SEQPARAM_FEATURES resolved
    against the OLD threshold, which is the bug this file is most likely to miss.
    """
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return _load_config()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Stand-in field stats. Percentages are what the rule reads; presence_rows is
# what the new escape hatch reads. Kept as plain dicts because that is what step
# 03 hands the classifier.
def _stats(presence_pct, presence_rows, numeric_pct=100.0):
    return {'presence_pct': presence_pct, 'presence_rows': presence_rows,
            'numeric_pct': numeric_pct}


_RARE_MODES = ('drop', 'present_only')

# A synthetic divisor table standing in for the one step 03 emits to /dbfs.
#
# Not a nicety. SEQPARAM_PRESENCE_ONLY is derived from SEQPARAM_DISCOVERED, which
# is empty when no table exists — so every assertion that iterates it would pass
# VACUOUSLY against a bare checkout, which is the failure mode most likely to let
# a broken present_only arm ship. `TR` is common, `PDM` is Görtler's rare cardiac
# field, and `RARE_BUT_MANY` is the count-hatch case.
_TABLE = {
    'fingerprint': 'test',
    'fields': {
        'TR': {'divisor': 1000.0, 'presence_pct': 99.0, 'presence_rows': 50000,
               'numeric_pct': 100.0, 'p99': 866.0, 'magnitude': 866.0},
        'PDM': {'divisor': 1.0, 'presence_pct': 0.01, 'presence_rows': 5,
                'numeric_pct': 100.0, 'p99': 1.0, 'magnitude': 1.0},
        'RARE_BUT_MANY': {'divisor': 1.0, 'presence_pct': 0.08,
                          'presence_rows': 40000, 'numeric_pct': 100.0,
                          'p99': 1.0, 'magnitude': 1.0},
    },
}


def _config_with_table(**env):
    """Load config against a real (synthetic) divisor table on disk."""
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    with handle:
        json.dump(_TABLE, handle)
    env.setdefault('SEQPARAM_DIVISOR_TABLE', handle.name)
    return _config_with(**env), handle.name


class VerdictShapeTests(unittest.TestCase):
    """The classifier answers three questions, not one."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_a_common_numeric_field_is_admitted_in_full(self):
        verdict = self.config.classify_seqparam_field(
            'turbo_factor', _stats(42.0, 21000))
        self.assertEqual(verdict.verdict, 'full')
        self.assertEqual(verdict.category, 'admitted')

    def test_the_verdict_carries_a_reason_for_the_report(self):
        # The human half of the parameter report prints this. "Why are we not
        # using this parameter" is the question this pipeline keeps being asked,
        # and it must be answered by the code that made the decision.
        verdict = self.config.classify_seqparam_field('PDM', _stats(0.01, 5))
        self.assertTrue(verdict.reason)
        self.assertIn('0.01', verdict.reason)


class CountEscapeHatchTests(unittest.TestCase):
    """presence_pct OR presence_rows — Görtler's 'the ratio changes' concern.

    A field on 40,000 rows is learnable whatever fraction of the corpus that is.
    The percentage floor alone gets HARDER to justify as serials are added,
    because the ratio stays flat while the count grows.
    """

    def test_a_field_below_the_percentage_floor_survives_on_row_count(self):
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_MIN_PRESENCE_ROWS=500)
        verdict = config.classify_seqparam_field('PDM', _stats(0.08, 40000))
        self.assertEqual(verdict.verdict, 'full')

    def test_a_field_below_both_floors_does_not_survive(self):
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_MIN_PRESENCE_ROWS=500,
                              SEQPARAM_RARE_MODE='drop')
        verdict = config.classify_seqparam_field('PDM', _stats(0.01, 5))
        self.assertEqual(verdict.verdict, 'excluded')
        self.assertEqual(verdict.category, 'rare')

    def test_the_percentage_floor_alone_still_admits(self):
        # The hatch is an OR, not a replacement. A field clearing the percentage
        # floor on a small corpus must not start failing because its absolute
        # count is low.
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_MIN_PRESENCE_ROWS=10 ** 9)
        verdict = config.classify_seqparam_field('TR', _stats(99.0, 50000))
        self.assertEqual(verdict.verdict, 'full')

    def test_missing_presence_rows_falls_back_to_the_percentage_rule(self):
        # Step 03 emits presence_rows, but a divisor table written before this
        # change does not carry it. That table must still classify, not crash.
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_MIN_PRESENCE_ROWS=500)
        verdict = config.classify_seqparam_field(
            'PDM', {'presence_pct': 0.01, 'numeric_pct': 100.0})
        self.assertEqual(verdict.verdict, 'excluded')


class PresenceOnlyTests(unittest.TestCase):
    """The arm nobody proposed in the meeting: keep the flag, drop the value.

    Georg's own words were "PDM only occurs in heart sequences so this might help
    the model IDENTIFY heart sequences". That is a binary indicator. The current
    floor discards the field wholesale, so it throws away the cheap, robust flag
    along with the thin, noisy value.
    """

    def test_a_rare_field_becomes_presence_only(self):
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_RARE_MODE='present_only')
        verdict = config.classify_seqparam_field('PDM', _stats(0.01, 5))
        self.assertEqual(verdict.verdict, 'presence_only')
        self.assertEqual(verdict.category, 'rare')

    def test_presence_only_contributes_a_flag_and_no_value_column(self):
        # The whole point. A value column would reintroduce the thin, noisy
        # measurement the floor exists to keep out.
        #
        # The row hatch is turned ON here so the two rare fields separate: PDM
        # (5 rows) is genuinely thin and becomes a flag, RARE_BUT_MANY (40,000
        # rows) is learnable and keeps its value. With the hatch off — the
        # shipped default — both would be flags, which is correct but would not
        # test the distinction.
        config, _ = _config_with_table(PARAM_SET='all',
                                       SEQPARAM_RARE_MODE='present_only',
                                       SEQPARAM_MIN_PRESENCE_ROWS=500)
        self.assertEqual(config.SEQPARAM_PRESENCE_ONLY, ['PDM'])
        self.assertIn('PDM__present', config.EXAMINATION_SEQPARAM_FEATURES)
        self.assertNotIn('PDM', config.EXAMINATION_SEQPARAM_FEATURES)

    def test_with_the_hatch_off_every_rare_field_becomes_a_flag(self):
        # The shipped default. Row count cannot rescue a field into a value
        # column unless somebody turns the hatch on deliberately.
        config, _ = _config_with_table(PARAM_SET='all',
                                       SEQPARAM_RARE_MODE='present_only')
        self.assertEqual(config.SEQPARAM_PRESENCE_ONLY,
                         ['PDM', 'RARE_BUT_MANY'])

    def test_presence_only_flags_are_not_rescaled(self):
        # A p99-derived divisor on a boolean would make "absent" a nonzero value.
        config, _ = _config_with_table(PARAM_SET='all',
                                       SEQPARAM_RARE_MODE='present_only')
        scale = dict(zip(config.EXAMINATION_SEQPARAM_FEATURES,
                         config.EXAMINATION_SEQPARAM_SCALE))
        for name in config.SEQPARAM_PRESENCE_ONLY:
            with self.subTest(field=name):
                self.assertEqual(scale[config.presence_name(name)], 1.0)

    def test_presence_only_flags_have_a_missing_default(self):
        # SEQPARAM_MISSING_DEFAULTS is derived from SEQPARAM_ALL_CANDIDATES. A
        # flag missing from that list reaches step 03's write pass with no
        # default and takes down a 4-minute Spark job at the last step.
        config, _ = _config_with_table(PARAM_SET='all',
                                       SEQPARAM_RARE_MODE='present_only')
        for name in config.SEQPARAM_PRESENCE_ONLY:
            with self.subTest(field=name):
                flag = config.presence_name(name)
                self.assertIn(flag, config.SEQPARAM_ALL_CANDIDATES)
                self.assertEqual(config.SEQPARAM_MISSING_DEFAULTS[flag], 0.0)

    def test_presence_only_is_empty_when_the_mode_is_off(self):
        config, _ = _config_with_table(PARAM_SET='all', SEQPARAM_RARE_MODE='drop')
        self.assertEqual(config.SEQPARAM_PRESENCE_ONLY, [])
        self.assertNotIn('PDM__present', config.EXAMINATION_SEQPARAM_FEATURES)

    def test_the_count_hatch_admits_in_full_not_as_a_flag(self):
        # RARE_BUT_MANY is 0.08% but 40,000 rows — Görtler's actual example.
        # It should get a VALUE column: 40k rows is enough to learn a slope, and
        # demoting it to a flag would throw away the measurement for no reason.
        config, _ = _config_with_table(PARAM_SET='all',
                                       SEQPARAM_RARE_MODE='present_only',
                                       SEQPARAM_MIN_PRESENCE_ROWS=500)
        self.assertIn('RARE_BUT_MANY', config.EXAMINATION_SEQPARAM_FEATURES)
        self.assertNotIn('RARE_BUT_MANY', config.SEQPARAM_PRESENCE_ONLY)


class DenylistsSurviveEveryRareModeTests(unittest.TestCase):
    """The guard rail. Widening admission must not reopen a leak.

    `ST` is the scan time. Including it is what produced the 5.0s figure that
    must never leave the team, so this is asserted against every mode and every
    floor rather than once at the default.
    """

    def test_target_fields_stay_excluded_under_every_rare_mode(self):
        for mode in _RARE_MODES:
            config = _config_with(SEQPARAM_RARE_MODE=mode,
                                  SEQPARAM_MIN_PRESENCE_PCT=0.0,
                                  SEQPARAM_MIN_PRESENCE_ROWS=1)
            for field in ('ST', 'TST', 'scanning_time'):
                with self.subTest(mode=mode, field=field):
                    verdict = config.classify_seqparam_field(
                        field, _stats(99.0, 50000))
                    self.assertEqual(verdict.verdict, 'excluded')
                    self.assertEqual(verdict.category, 'denied')

    def test_identity_and_planner_fields_stay_excluded_under_every_rare_mode(self):
        for mode in _RARE_MODES:
            config = _config_with(SEQPARAM_RARE_MODE=mode,
                                  SEQPARAM_MIN_PRESENCE_PCT=0.0,
                                  SEQPARAM_MIN_PRESENCE_ROWS=1)
            for field in ('MUID', 'VER', 'SNR'):
                with self.subTest(mode=mode, field=field):
                    self.assertEqual(
                        config.classify_seqparam_field(
                            field, _stats(99.0, 50000)).verdict,
                        'excluded')

    def test_a_zero_floor_never_smuggles_a_denied_field_into_the_vector(self):
        # The end-to-end version: not the classifier in isolation but what the
        # resolved feature list actually contains. assert_no_leakage is the same
        # gate step 03 runs.
        for mode in _RARE_MODES:
            config = _config_with(PARAM_SET='all', SEQPARAM_RARE_MODE=mode,
                                  SEQPARAM_MIN_PRESENCE_PCT=0.0)
            with self.subTest(mode=mode):
                config.assert_no_leakage(config.EXAMINATION_SEQPARAM_FEATURES)
                config.assert_no_leakage(config.SEQPARAM_ALL_CANDIDATES)

    def test_a_non_numeric_field_is_never_rescued_by_a_rare_mode(self):
        # DLL is '%SiemensSeq%\\haste'. Rare-mode is about THIN fields, not about
        # text fields — those need an embedding, and admitting one as a flag
        # would silently ship the bucket-3 work as though it were done.
        for mode in _RARE_MODES:
            config = _config_with(SEQPARAM_RARE_MODE=mode)
            with self.subTest(mode=mode):
                verdict = config.classify_seqparam_field(
                    'DLL', _stats(90.0, 45000, numeric_pct=2.0))
                self.assertEqual(verdict.verdict, 'excluded')
                self.assertEqual(verdict.category, 'non_numeric')


class WriteSelectSplitTests(unittest.TestCase):
    """Step 03 writes wide; config selects narrow.

    Without this split every threshold arm costs a Spark rebuild, because
    03_build_preprocessed_pkl.py applies the SELECTION floor when deciding which
    columns to WRITE. One rebuild at the write floor then serves every arm as a
    config-side column selection.
    """

    def test_the_write_floor_defaults_to_admitting_everything(self):
        config = _load_config()
        self.assertEqual(config.SEQPARAM_WRITE_MIN_PRESENCE_PCT, 0.0)

    def test_the_write_floor_is_independent_of_the_selection_floor(self):
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=5.0,
                              SEQPARAM_WRITE_MIN_PRESENCE_PCT=0.0)
        self.assertEqual(config.SEQPARAM_WRITE_MIN_PRESENCE_PCT, 0.0)
        self.assertEqual(config.SEQPARAM_MIN_PRESENCE_PCT, 5.0)

    def test_the_write_rule_still_refuses_denied_and_non_numeric_fields(self):
        # "Write everything" means every THIN field, not every field. A denied
        # field in the pkl is one config typo away from being trained on.
        config = _config_with(SEQPARAM_WRITE_MIN_PRESENCE_PCT=0.0)
        for field in ('ST', 'TST', 'MUID', 'SNR'):
            with self.subTest(field=field):
                self.assertEqual(
                    config.classify_seqparam_field_for_write(
                        field, _stats(99.0, 50000)).verdict,
                    'excluded')
        self.assertEqual(
            config.classify_seqparam_field_for_write(
                'DLL', _stats(90.0, 45000, numeric_pct=2.0)).verdict,
            'excluded')

    def test_the_write_rule_admits_a_field_the_selection_rule_would_drop(self):
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_WRITE_MIN_PRESENCE_PCT=0.0,
                              SEQPARAM_RARE_MODE='drop')
        thin = _stats(0.01, 5)
        self.assertEqual(
            config.classify_seqparam_field_for_write('PDM', thin).verdict, 'full')
        self.assertEqual(
            config.classify_seqparam_field('PDM', thin).verdict, 'excluded')


class TheIncumbentArmDoesNotMoveTests(unittest.TestCase):
    """The baseline must be bit-identical, or every arm is measured against a
    moving target and the whole comparison is worthless."""

    def test_defaults_are_unchanged(self):
        config = _load_config()
        self.assertEqual(config.SEQPARAM_MIN_PRESENCE_PCT, 1.0)
        self.assertEqual(config.SEQPARAM_MIN_NUMERIC_PCT, 90.0)
        self.assertEqual(config.SEQPARAM_RARE_MODE, 'drop')

    def test_the_count_hatch_is_off_by_default(self):
        # It must not widen the incumbent silently. At ~51k sequences the 1%
        # floor is ~510 rows, so ANY row floor below that admits fields the
        # percentage floor rejects — and an arm sweep whose baseline moved is
        # measuring two things at once. The hatch is a knob the sweep turns on.
        config = _load_config()
        self.assertEqual(config.SEQPARAM_MIN_PRESENCE_ROWS, 0)

    def test_a_zero_row_floor_admits_nothing_on_count(self):
        # 0 means "off", not "every field clears it" — the difference between a
        # disabled hatch and one that admits the entire message.
        config = _config_with(SEQPARAM_MIN_PRESENCE_PCT=1.0,
                              SEQPARAM_MIN_PRESENCE_ROWS=0)
        self.assertEqual(
            config.classify_seqparam_field('PDM', _stats(0.01, 40000)).verdict,
            'excluded')

    def test_an_unknown_rare_mode_fails_loudly_at_import(self):
        with self.assertRaises(ValueError):
            _config_with(SEQPARAM_RARE_MODE='sometimes')


if __name__ == '__main__':
    unittest.main()
