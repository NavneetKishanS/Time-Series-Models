"""The "pass every parameter" decision path, run against real SUT messages.

Görtler, 2026-08-07: pass every parameter and let the model work out which
matter, because a digital twin aiming at the extraordinary 1% cannot hide fields
from itself. Step 03 executes that by DISCOVERING fields from the corpus rather
than reading a hand-written list, which means the field set is now decided by
code that nobody can eyeball the output of without a Databricks cluster.

So it gets eyeballed here instead. These tests run the actual config rules —
seqparam_stable_name, classify_seqparam_field, suggest_divisor — over the three
real MRI_SUT_1005 messages pinned in test_sut_parser.py, and assert the things
that would otherwise only be discovered by an expensive Spark run producing a
quietly wrong pkl:

  * the denylisted fields really are absent from what gets written, on real
    messages that plainly contain them;
  * a field that is genuinely sequence-scoped (TF is on haste, absent from
    ep2d_diff) gets a presence flag of 0 rather than a fabricated value;
  * every admitted field lands inside the [0.05, 20] scaling band, which is the
    LayerNorm-erasure guard that has cost this project three multi-week
    flat-duration incidents.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_sut_parser import (  # noqa: E402
    _EP2D_DIFF_MSG,
    _HASTE_MSG_1,
    _HASTE_MSG_2,
    _load_config,
    _load_sut_parsers,
)

_MESSAGES = [_HASTE_MSG_1, _HASTE_MSG_2, _EP2D_DIFF_MSG]


def _build_field_stats(config, parsers, messages):
    """Step 03's discovery pass, over the pinned messages instead of a corpus.

    Mirrors the "--- 1. discover ---" block of 03_build_preprocessed_pkl.py. It
    is a re-implementation rather than an extraction because that block is
    top-level notebook code between two Spark queries; keeping it short and
    keeping the RULES in config (which this calls) is what stops the two from
    meaning different things.
    """
    raws = [parsers['_parse_sut_raw'](m) for m in messages]

    values_by_name = {}
    for raw in raws:
        for key, value in raw.items():
            values_by_name.setdefault(config.seqparam_stable_name(key), []).append(value)

    stats = {}
    for name, values in values_by_name.items():
        numeric = []
        for value in values:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                pass
        ordered = sorted(numeric)
        stats[name] = {
            'presence_pct': 100.0 * len(values) / len(messages),
            'numeric_pct': 100.0 * len(numeric) / max(1, len(values)),
            'p99': ordered[-1] if ordered else 0.0,
            'p50': ordered[len(ordered) // 2] if ordered else 0.0,
        }
    return stats, raws


class _RealMessageBase(unittest.TestCase):
    """Shared fixture: parse the pinned messages once, decide once."""

    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()
        cls.parsers = _load_sut_parsers()
        cls.stats, cls.raws = _build_field_stats(cls.config, cls.parsers, _MESSAGES)
        cls.verdicts = {
            name: cls.config.classify_seqparam_field(name, spec)
            for name, spec in cls.stats.items()
        }
        cls.admitted = sorted(n for n, (ok, _) in cls.verdicts.items() if ok)


class RealMessageSelectionTests(_RealMessageBase):

    def test_the_messages_are_rich_enough_to_be_worth_testing(self):
        # If the parse ever narrows, every assertion below becomes vacuous.
        self.assertGreater(len(self.stats), 50)

    def test_admits_far_more_than_the_hand_picked_sets(self):
        # The whole point of PARAM_SET='all'. If discovery admitted seven
        # fields, nothing would have changed on 2026-08-07 but the wording.
        for name in ('luke', 'navneet'):
            with self.subTest(param_set=name):
                self.assertGreater(len(self.admitted),
                                   2 * len(self.config.PARAM_SETS[name]))

    def test_target_equivalent_fields_are_refused_on_real_messages(self):
        # The messages plainly carry ST:8 / TST:9. "Pass everything" must not
        # reach the field that IS the answer.
        for field in ('ST', 'TST'):
            with self.subTest(field=field):
                self.assertIn(field, self.stats, "the message stopped carrying it")
                ok, why = self.verdicts[field]
                self.assertFalse(ok)
                self.assertIn('duration target', why)

    def test_identity_fields_are_refused_on_real_messages(self):
        for field in ('MUID', 'VER'):
            with self.subTest(field=field):
                self.assertIn(field, self.stats)
                ok, why = self.verdicts[field]
                self.assertFalse(ok)
                self.assertIn('cannot transfer', why)

    def test_planner_derived_fields_are_refused_on_real_messages(self):
        # SNR moves monotonically AGAINST scan time across these three messages
        # (87/69/39 as ST goes 8/15/400), which is what a quantity computed from
        # the acquisition time looks like.
        for field in sorted(self.config.SUT_PLANNER_DERIVED_DENYLIST):
            with self.subTest(field=field):
                self.assertIn(field, self.stats)
                ok, why = self.verdicts[field]
                self.assertFalse(ok)
                self.assertIn('timing', why)

    def test_string_valued_fields_are_refused_as_numerics(self):
        # DLL is '%SiemensSeq%\\haste' — a category. It needs an embedding, and
        # it already has one; what it must not become is a scaled numeric column.
        ok, why = self.verdicts['DLL']
        self.assertFalse(ok)
        self.assertIn('embedding', why)

    def test_the_physics_parameters_survive(self):
        # The counterweight to the four tests above: the denylists must not be
        # quietly eating the things Görtler actually wants passed through.
        for field in ('TR', 'num_slices', 'phase_encoding_lines', 'averages',
                      'turbo_factor', 'base_resolution', 'echo_spacing',
                      'bandwidth', 'flip_angle', 'field_of_view'):
            with self.subTest(field=field):
                self.assertIn(field, self.admitted)

    def test_watchlisted_fields_are_admitted_not_blocked(self):
        # The watchlist is a reading aid. A field on it that is present in the
        # corpus must still reach the model, or the review it exists to prompt
        # never happens.
        for field in self.config.SUT_SUSPECT_WATCHLIST:
            if field not in self.stats:
                continue        # BHD/ACQW/TGD ride gated sequences, absent here
            with self.subTest(field=field):
                self.assertIn(self.config.seqparam_stable_name(field),
                              self.admitted)


class RealMessageScalingTests(_RealMessageBase):
    def test_every_admitted_field_lands_in_the_layernorm_safe_band(self):
        # The guard that has cost this project three multi-week flat-duration
        # incidents. A field far above O(1) erases the categorical conditioning;
        # one far below it is invisible. Neither raises anything at train time.
        for name in self.admitted:
            spec = self.stats[name]
            if spec['p99'] <= 0:
                continue
            divisor = (self.config.SEQPARAM_CANDIDATES[name][0]
                       if name in self.config.SEQPARAM_CANDIDATES
                       else self.config.suggest_divisor(spec['p99']))
            with self.subTest(field=name, p99=spec['p99'], divisor=divisor):
                self.assertGreaterEqual(spec['p99'] / divisor, 0.05)
                self.assertLessEqual(spec['p99'] / divisor, 20.0)

    def test_curated_divisors_win_over_discovered_ones(self):
        # PPF and SPF are Siemens enum codes that must share ONE scale. A
        # per-field p99 would give them 1.0 and 20.0 and silently separate them.
        for name in ('phase_partial_fourier', 'slice_partial_fourier'):
            with self.subTest(field=name):
                self.assertEqual(self.config.SEQPARAM_CANDIDATES[name][0], 16.0)
                self.assertEqual(self.config._divisor_for(name), 16.0)

    def test_the_divisor_ladder_is_stable_against_small_drift(self):
        # An exact-p99 divisor would move on every rebuild, changing every
        # column's scale and making two training runs incomparable for no reason.
        for p99, expected in ((866.0, 1000.0), (900.0, 1000.0), (1000.0, 1000.0)):
            with self.subTest(p99=p99):
                self.assertEqual(self.config.suggest_divisor(p99), expected)


class RealMessageWriteTests(_RealMessageBase):
    """What step 03's "--- 3. write ---" pass actually puts in `conditioning`."""

    def _conditioning_for(self, raw):
        stable = {self.config.seqparam_stable_name(k): v for k, v in raw.items()}
        cond = {}
        for name in self.admitted:
            present = name in stable
            default = (self.config.SEQPARAM_CANDIDATES[name][1]
                       if name in self.config.SEQPARAM_CANDIDATES else 0.0)
            cond[name] = float(stable[name]) if present else default
            cond[self.config.presence_name(name)] = 1.0 if present else 0.0
        return cond

    def test_a_sequence_scoped_field_is_flagged_absent_not_invented(self):
        # TF:256 is on both haste messages and ABSENT from ep2d_diff, which uses
        # an echo factor instead. This is the case the whole presence-flag design
        # exists for: without the flag, ep2d_diff would be handed turbo_factor
        # 1.0 as though the scanner had measured it.
        haste = self._conditioning_for(self.raws[0])
        diff = self._conditioning_for(self.raws[2])

        self.assertEqual(haste['turbo_factor'], 256.0)
        self.assertEqual(haste['turbo_factor__present'], 1.0)

        self.assertEqual(diff['turbo_factor'], 1.0)   # the neutral default...
        self.assertEqual(diff['turbo_factor__present'], 0.0)   # ...declared as such

    def test_every_value_column_has_its_flag(self):
        cond = self._conditioning_for(self.raws[0])
        for name in self.admitted:
            with self.subTest(field=name):
                self.assertIn(self.config.presence_name(name), cond)

    def test_gate_2_passes_on_what_is_actually_written(self):
        # The runtime dict keys, not the static config list — on a real message
        # that carries ST, TST, SNR, MUID and VER.
        for index, raw in enumerate(self.raws):
            with self.subTest(message=index):
                self.config.assert_no_leakage(self._conditioning_for(raw).keys())

    def test_present_flags_track_the_message_not_the_value(self):
        # REP:0 is present and zero. A flag derived from "is the value nonzero"
        # would call it absent, which is exactly the confusion the flag removes.
        haste = self._conditioning_for(self.raws[0])
        self.assertEqual(haste['repetitions'], 0.0)
        self.assertEqual(haste['repetitions__present'], 1.0)


if __name__ == '__main__':
    unittest.main()
