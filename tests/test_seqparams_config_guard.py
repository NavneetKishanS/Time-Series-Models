import contextlib
import importlib.util
import os
import unittest


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'DatabricksPipeline', 'csv_pipeline_seqparams', 'config.py',
)


def _load_seqparams_config():
    """Load csv_pipeline_seqparams/config.py by path.

    This folder is a Databricks-notebook-source tree (%run-based, no
    __init__.py), not a Python package, so it is loaded directly by file
    path rather than via a normal import statement.
    """
    spec = importlib.util.spec_from_file_location(
        'csv_pipeline_seqparams_config', _CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _param_set(name):
    """Reload the config with PARAM_SET set, restoring the environment after.

    PARAM_SET is read at module import, which is the whole point of it — the
    Databricks notebooks %run this file and read module-level names — so the
    only way to test a second set is to re-import under a different env.
    """
    previous = os.environ.get('PARAM_SET')
    os.environ['PARAM_SET'] = name
    try:
        yield _load_seqparams_config()
    finally:
        if previous is None:
            os.environ.pop('PARAM_SET', None)
        else:
            os.environ['PARAM_SET'] = previous


class LeakageGuardTests(unittest.TestCase):
    def test_assert_no_leakage_passes_for_clean_feature_list(self):
        module = _load_seqparams_config()
        module.assert_no_leakage(['TR', 'num_slices'])  # must not raise

    def test_assert_no_leakage_rejects_scanning_time(self):
        module = _load_seqparams_config()
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            module.assert_no_leakage(['TR', 'scanning_time'])

    def test_assert_no_leakage_rejects_st_and_tst(self):
        # The 2026-08-04 call overrode the 07-31 "planned value, admissible"
        # reading of ST: 03c section E measured it as exact on ~86-87% of rows,
        # so a duration model handed ST learns identity rather than physics.
        module = _load_seqparams_config()
        for field in ('ST', 'TST'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    module.assert_no_leakage(['TR', field])

    def test_assert_no_leakage_rejects_identifiers_with_its_own_message(self):
        # 03d ACQUITTED MUID empirically (29.8% purity, -0.0s when dropped) but
        # its verdict was still "the identifier fields must leave the candidate
        # pool". The objection is different from the target-equivalence one —
        # an id does not transfer between customers — so the message differs.
        module = _load_seqparams_config()
        for field in ('MUID', 'VER'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, 'Identifier guard'):
                    module.assert_no_leakage(['TR', field])

    def test_assert_no_leakage_rejects_planner_derived_with_its_own_message(self):
        # The third objection, added after the 2026-08-07 Görtler meeting. His
        # instruction was to pass everything and let the model sort it out, and
        # the reason a denylist survives that is that these fields are not scan
        # physics — the scanner computed them FROM the timing it was about to
        # run. Distinct message because the remedy is distinct: this one carries
        # a measured price tag and can be removed if the report says so.
        module = _load_seqparams_config()
        for field in sorted(module.SUT_PLANNER_DERIVED_DENYLIST):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, 'Planner-derived guard'):
                    module.assert_no_leakage(['TR', field])

    def test_the_three_objections_do_not_overlap(self):
        # Each denylist answers a different question, and a field appearing in
        # two of them would report whichever guard happens to run first — which
        # is how a field ends up removed for the wrong stated reason.
        module = _load_seqparams_config()
        lists = {
            'target': module.SUT_LEAKAGE_DENYLIST,
            'identity': module.SUT_IDENTIFIER_DENYLIST,
            'planner': module.SUT_PLANNER_DERIVED_DENYLIST,
        }
        for a, b in (('target', 'identity'), ('target', 'planner'),
                     ('identity', 'planner')):
            with self.subTest(pair=f"{a}/{b}"):
                self.assertEqual(lists[a] & lists[b], set())

    def test_the_union_covers_every_objection(self):
        # SUT_ALL_DENYLISTS is what gate #3 enforces at tensor-construction
        # time. If a fourth objection is added and not folded in here, it would
        # be enforced at config import and silently unenforced at the point that
        # actually builds the model's input.
        module = _load_seqparams_config()
        self.assertEqual(
            module.SUT_ALL_DENYLISTS,
            module.SUT_LEAKAGE_DENYLIST | module.SUT_IDENTIFIER_DENYLIST
            | module.SUT_PLANNER_DERIVED_DENYLIST,
        )

    def test_watchlist_fields_stay_admissible(self):
        # The watchlist is a reading aid, not a denylist. A field that drifts
        # into both is being both flagged for review and silently blocked, which
        # is the worst of the two — the review never happens because nothing
        # measures it.
        module = _load_seqparams_config()
        self.assertEqual(
            module.SUT_ALL_DENYLISTS.intersection(module.SUT_SUSPECT_WATCHLIST),
            set(),
        )

    def test_assert_no_leakage_checks_runtime_dict_keys_too(self):
        # Gate #2's actual use pattern: check the real conditioning dict's
        # keys after SUT parsing + filtering, not just the static config list.
        module = _load_seqparams_config()
        conditioning = {'Age': 40, 'TR': 1500.0, 'scanning_time': 59.0}
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            module.assert_no_leakage(conditioning.keys())


class ParamSetSwitchTests(unittest.TestCase):
    """PARAM_SET is the single place model inputs are chosen (Görtler action
    item, 2026-08-04). These assert the MECHANISM rather than one set's
    contents, so adding or re-tuning a set does not break the suite — but
    breaking the switch does."""

    def test_defaults_to_all(self):
        # Görtler, 2026-08-07: pass everything and let the model sort it out.
        # `luke`/`navneet` survive as the control group, not as the default.
        previous = os.environ.pop('PARAM_SET', None)
        try:
            module = _load_seqparams_config()
            self.assertEqual(module.PARAM_SET, 'all')
        finally:
            if previous is not None:
                os.environ['PARAM_SET'] = previous

    def test_env_var_selects_a_named_set(self):
        for name in ('luke', 'navneet', 'all'):
            with self.subTest(param_set=name), _param_set(name) as module:
                self.assertEqual(module.PARAM_SET, name)
                # The VALUE block leads the feature list; flags and derived
                # features follow it.
                values = module.PARAM_SETS[name]
                self.assertEqual(
                    module.EXAMINATION_SEQPARAM_FEATURES[:len(values)], values)

    def test_the_named_sets_are_actually_different(self):
        # A switch between identical lists would pass every other test here and
        # answer nothing on Friday.
        module = _load_seqparams_config()
        self.assertNotEqual(module.PARAM_SETS['luke'],
                            module.PARAM_SETS['navneet'])

    def test_all_is_a_superset_of_the_hand_picked_sets(self):
        # The whole claim being tested by keeping the control group is "more is
        # better". That only means anything if `all` actually contains more.
        module = _load_seqparams_config()
        for name in ('luke', 'navneet'):
            with self.subTest(param_set=name):
                self.assertLessEqual(set(module.PARAM_SETS[name]),
                                     set(module.PARAM_SETS['all']))

    def test_every_value_feature_gets_a_presence_flag(self):
        # The flag is what lets the model tell "this sequence has value X" from
        # "this concept does not apply to this sequence". Without it, safe_float
        # turns every absent field into a fabricated measurement.
        for name in ('luke', 'navneet', 'all'):
            with self.subTest(param_set=name), _param_set(name) as module:
                features = module.EXAMINATION_SEQPARAM_FEATURES
                for value_name in module.PARAM_SETS[name]:
                    self.assertIn(module.presence_name(value_name), features)

    def test_flags_can_be_ablated_back_to_the_historical_vector(self):
        # Gate 4 has to PRICE the flags, not assume them, so the pre-08-07
        # vector must remain reachable.
        previous = os.environ.get('SEQPARAM_USE_PRESENCE_FLAGS')
        os.environ['SEQPARAM_USE_PRESENCE_FLAGS'] = '0'
        try:
            with _param_set('luke') as module:
                self.assertEqual(module.EXAMINATION_SEQPARAM_FEATURES,
                                 module.PARAM_SETS['luke'])
        finally:
            if previous is None:
                os.environ.pop('SEQPARAM_USE_PRESENCE_FLAGS', None)
            else:
                os.environ['SEQPARAM_USE_PRESENCE_FLAGS'] = previous

    def test_unknown_set_fails_loudly_at_import(self):
        with self.assertRaisesRegex(ValueError, 'not a known parameter set'):
            with _param_set('nuffnet'):
                pass

    def test_scale_is_derived_not_hand_maintained(self):
        # The two used to be hand-maintained parallel lists that could desync.
        for name in ('luke', 'navneet', 'all'):
            with self.subTest(param_set=name), _param_set(name) as module:
                self.assertEqual(len(module.EXAMINATION_SEQPARAM_FEATURES),
                                 len(module.EXAMINATION_SEQPARAM_SCALE))
                for feature, divisor in zip(module.EXAMINATION_SEQPARAM_FEATURES,
                                            module.EXAMINATION_SEQPARAM_SCALE):
                    if module.is_presence_name(feature):
                        continue
                    self.assertEqual(divisor, module._divisor_for(feature),
                                     msg=feature)

    def test_presence_flags_are_never_rescaled(self):
        # A flag is already 0/1. Handing it a p99-derived divisor would make
        # "absent" a nonzero value and quietly destroy the distinction the flag
        # exists to carry.
        for name in ('luke', 'navneet', 'all'):
            with self.subTest(param_set=name), _param_set(name) as module:
                for feature, divisor in zip(module.EXAMINATION_SEQPARAM_FEATURES,
                                            module.EXAMINATION_SEQPARAM_SCALE):
                    if module.is_presence_name(feature):
                        self.assertEqual(divisor, 1.0, msg=feature)

    def test_a_feature_without_a_divisor_is_refused(self):
        # The LayerNorm-erasure failure mode is invisible in a trained model and
        # has cost this project three multi-week incidents, so it is refused at
        # import rather than warned about.
        module = _load_seqparams_config()
        with self.assertRaisesRegex(ValueError, 'no scale divisor'):
            module._divisor_for('a_field_nobody_calibrated')

    def test_models_and_analysis_dirs_are_namespaced_by_param_set(self):
        # Both sets widen base_conditioning_dim to the same number while
        # meaning different things per column. A shared directory would
        # silently overwrite one checkpoint with the other.
        with _param_set('luke') as luke, _param_set('navneet') as navneet:
            self.assertNotEqual(luke.MODELS_DIR, navneet.MODELS_DIR)
            self.assertNotEqual(luke.ANALYSIS_DIR, navneet.ANALYSIS_DIR)
            self.assertIn('luke', luke.MODELS_DIR)
            self.assertIn('navneet', navneet.MODELS_DIR)

    def test_the_pkl_path_is_NOT_namespaced(self):
        # One pkl serves every set — step 03 writes the whole candidate union
        # and training reads only what it needs. Namespacing this would put
        # a Spark rebuild back in front of every switch.
        with _param_set('luke') as luke, _param_set('navneet') as navneet:
            self.assertEqual(luke.PKL_OUTPUT, navneet.PKL_OUTPUT)


class CandidateTableTests(unittest.TestCase):
    def test_every_named_feature_has_a_candidate_entry(self):
        module = _load_seqparams_config()
        for name, features in module.PARAM_SETS.items():
            for feature in features:
                with self.subTest(param_set=name, feature=feature):
                    self.assertIn(feature, module.SEQPARAM_CANDIDATES)

    def test_every_candidate_is_something_the_parser_produces(self):
        # A misspelled candidate would take its default on every row forever
        # and read as a dead field rather than as a typo.
        module = _load_seqparams_config()
        produced = set(module.SUT_FIELD_MAP.values())
        for name in module.SEQPARAM_CANDIDATES:
            with self.subTest(candidate=name):
                self.assertIn(name, produced)

    def test_no_named_set_contains_a_banned_field(self):
        module = _load_seqparams_config()
        banned = module.SUT_ALL_DENYLISTS
        for name, features in module.PARAM_SETS.items():
            with self.subTest(param_set=name):
                self.assertEqual(banned.intersection(features), set())

    def test_one_based_factors_default_to_one_not_zero(self):
        # The reason SEQPARAM_CANDIDATES is keyed rather than two lists. TF is
        # absent on ep2d_diff (which uses an echo factor); defaulting it to 0.0
        # invents a numeric difference between sequence families that does not
        # exist. 1.0 is the identity element of TA ~= TR x PEL x AVG x CONC /
        # (PAT x TF), i.e. "does not apply to this sequence".
        module = _load_seqparams_config()
        for name in ('averages', 'concatenations', 'parallel_imaging_factor',
                     'turbo_factor', 'phase_partial_fourier',
                     'slice_partial_fourier'):
            with self.subTest(candidate=name):
                self.assertEqual(module.SEQPARAM_CANDIDATES[name][1], 1.0)

    def test_repetitions_defaults_to_zero_because_REP_is_zero_based(self):
        # Not a multiplicative factor despite reading like one: REP counts
        # ADDITIONAL measurements, and all three real messages pinned in
        # test_sut_parser.py carry REP:0 for a single-measurement scan. So the
        # absent value must match the common present value, not 1.
        module = _load_seqparams_config()
        self.assertEqual(module.SEQPARAM_CANDIDATES['repetitions'][1], 0.0)

    def test_measurements_default_to_zero(self):
        # A measurement has no neutral value — absent means "not recorded",
        # and the model should be able to learn the missing-ness.
        module = _load_seqparams_config()
        for name in ('TR', 'num_slices', 'phase_encoding_lines',
                     'base_resolution'):
            with self.subTest(candidate=name):
                self.assertEqual(module.SEQPARAM_CANDIDATES[name][1], 0.0)

    # Measured p99 per candidate over the full corpus — 51,321 sequences, 10
    # serials, 2024-01, from the 2026-08-08 step-03 run.
    #
    # THIS REPLACED A THREE-MESSAGE CALIBRATION, and the replacement is the
    # whole point. The previous version of this test pinned the values in the
    # haste / ep2d_diff messages in test_sut_parser.py, which carry CONC:1,
    # PAT:2 and REP:0. Those readings are correct and they are p50s. The corpus
    # p99s are 23, 256 and 89, so divisors calibrated to the samples put three
    # features at 23x, 256x and 89x — the LayerNorm-erasure mode, arriving
    # through the guard meant to prevent it. Three samples cannot see a p99.
    CORPUS_P99 = {
        'TR': 9000.0, 'num_slices': 58.0, 'phase_encoding_lines': 936.0,
        'base_resolution': 576.0, 'averages': 3.0, 'concatenations': 23.0,
        'parallel_imaging_factor': 256.0, 'turbo_factor': 512.0,
        'repetitions': 89.0, 'phase_partial_fourier': 32.0,
        'slice_partial_fourier': 16.0,
    }

    def test_divisors_land_the_measured_corpus_at_order_one(self):
        module = _load_seqparams_config()
        self.assertEqual(set(self.CORPUS_P99), set(module.SEQPARAM_CANDIDATES),
                         "a candidate was added or removed without a measured "
                         "corpus p99 to calibrate its divisor against")
        low, high = module.SEQPARAM_SCALE_BAND
        for name, p99 in self.CORPUS_P99.items():
            divisor = module.SEQPARAM_CANDIDATES[name][0]
            with self.subTest(candidate=name, p99=p99, divisor=divisor):
                self.assertLessEqual(p99 / divisor, high)
                self.assertGreaterEqual(p99 / divisor, low)

    def test_the_sampled_values_are_not_left_oversized(self):
        # The samples are still worth checking in ONE direction. A typical value
        # landing below the band just means most scans sit near zero for that
        # field, which is true of CONC and REP and is fine. A typical value
        # landing ABOVE it means the divisor is too small for the common case,
        # which is never fine.
        module = _load_seqparams_config()
        sampled = {
            'TR': 4300, 'num_slices': 18, 'phase_encoding_lines': 333,
            'base_resolution': 320, 'averages': 1, 'concatenations': 1,
            'parallel_imaging_factor': 2, 'turbo_factor': 256,
            'repetitions': 0, 'phase_partial_fourier': 8,
            'slice_partial_fourier': 16,
        }
        _, high = module.SEQPARAM_SCALE_BAND
        for name, value in sampled.items():
            with self.subTest(candidate=name):
                self.assertLessEqual(value / module.SEQPARAM_CANDIDATES[name][0],
                                     high)

    def test_a_negative_field_gets_a_real_divisor(self):
        # TP (table position) runs about -1900 to -989. A p99-only rule read
        # -989, decided "not positive, nothing to scale", and handed back 1.0 —
        # putting the field into the conditioning vector at ~1500x. suggest_divisor
        # takes a MAGNITUDE now, and the abs() is a second line of defence.
        module = _load_seqparams_config()
        self.assertEqual(module.suggest_divisor(-989.0), 1000.0)
        self.assertEqual(module.suggest_divisor(989.0), 1000.0)
        # A genuinely empty field still needs the identity, not a division by 0.
        self.assertEqual(module.suggest_divisor(0.0), 1.0)
        self.assertEqual(module.suggest_divisor(float('nan')), 1.0)

    def test_the_corpus_can_overrule_a_curated_divisor(self):
        # "Curated wins" protects the PPF/SPF enum pairing, which a per-field
        # percentile would split. It must NOT protect a curated value the data
        # has falsified — that is how three out-of-band divisors survived into a
        # build.
        import json
        import tempfile

        table = {'fields': {
            # Curated at 30.0 against a magnitude of 58: sound, left alone even
            # though the table itself carries a stale value.
            'num_slices': {'divisor': 999.0, 'magnitude': 58.0,
                           'presence_pct': 99.0, 'numeric_pct': 100.0},
            # Curated at 500.0 against a magnitude of 400,000: the corpus has
            # moved past the hand-calibration, so the hand-calibration loses.
            'parallel_imaging_factor': {'divisor': 500.0, 'magnitude': 400000.0,
                                        'presence_pct': 99.0, 'numeric_pct': 100.0},
        }, 'fingerprint': 'test'}

        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(table, f)
            path = f.name
        previous = os.environ.get('SEQPARAM_DIVISOR_TABLE')
        os.environ['SEQPARAM_DIVISOR_TABLE'] = path
        try:
            module = _load_seqparams_config()
            # Curated wins over a stale table entry when it is in band...
            self.assertEqual(module._divisor_for('num_slices'), 30.0)
            # ...and loses to the corpus when it is not.
            self.assertEqual(module._divisor_for('parallel_imaging_factor'),
                             500000.0)
            overrides = {o['name']: o for o in module.SEQPARAM_DIVISOR_OVERRIDES}
            self.assertIn('parallel_imaging_factor', overrides)
            self.assertEqual(overrides['parallel_imaging_factor']['source'],
                             'curated')
            self.assertNotIn('num_slices', overrides)
        finally:
            os.unlink(path)
            if previous is None:
                os.environ.pop('SEQPARAM_DIVISOR_TABLE', None)
            else:
                os.environ['SEQPARAM_DIVISOR_TABLE'] = previous

    def test_every_candidate_has_a_positive_divisor(self):
        # A zero or negative divisor either divides by zero or flips the
        # feature's sign, both silently.
        module = _load_seqparams_config()
        for name, (divisor, _) in module.SEQPARAM_CANDIDATES.items():
            with self.subTest(candidate=name):
                self.assertGreater(divisor, 0.0)

    def test_curated_missing_defaults_are_never_overridden(self):
        # The hand-curated defaults encode physics a p99 cannot: 1.0 for the
        # 1-based multiplicands of the TA formula ("does not apply to this
        # sequence"), 0.0 for a measurement ("not recorded"). A discovered field
        # may only ADD to this table, never rewrite an entry in it.
        module = _load_seqparams_config()
        for name, (_, default) in module.SEQPARAM_CANDIDATES.items():
            with self.subTest(candidate=name):
                self.assertEqual(module.SEQPARAM_MISSING_DEFAULTS[name], default)

    def test_presence_flags_default_to_absent(self):
        # A flag defaulting to 1.0 would claim every unwritten field was
        # present, which is worse than having no flag at all.
        module = _load_seqparams_config()
        flags = [n for n in module.SEQPARAM_ALL_CANDIDATES
                 if module.is_presence_name(n)]
        self.assertTrue(flags)
        for name in flags:
            with self.subTest(flag=name):
                self.assertEqual(module.SEQPARAM_MISSING_DEFAULTS[name], 0.0)

    def test_every_written_key_has_a_missing_default(self):
        # Step 03 looks the default up by name for every SEQPARAM_ALL_CANDIDATES
        # entry, so a gap here is a KeyError in the middle of a Spark job.
        module = _load_seqparams_config()
        for name in module.SEQPARAM_ALL_CANDIDATES:
            with self.subTest(candidate=name):
                self.assertIn(name, module.SEQPARAM_MISSING_DEFAULTS)

    def test_all_candidates_covers_every_set(self):
        # Step 03 writes SEQPARAM_ALL_CANDIDATES; anything a set names that is
        # not in it would be absent from the pkl and silently read as 0.0 at
        # training time.
        module = _load_seqparams_config()
        for name, features in module.PARAM_SETS.items():
            with self.subTest(param_set=name):
                self.assertTrue(
                    set(features).issubset(module.SEQPARAM_ALL_CANDIDATES)
                )


class StepDefinitionTests(unittest.TestCase):
    """Step 03 reads these names out of the %run'd config namespace, and
    nothing else would catch a rename until a Spark job fails an hour in."""

    def _step_03_source(self):
        path = os.path.join(os.path.dirname(_CONFIG_PATH),
                            '03_build_preprocessed_pkl.py')
        with open(path) as f:
            return f.read()

    def test_step_03_writes_every_admissible_field_not_the_selected_set(self):
        # One Spark rebuild has to serve every parameter set, or switching sets
        # costs hours instead of a training run.
        source = self._step_03_source()
        self.assertIn('_written_names = [n for n, _, _ in _admitted]', source)

    def test_step_03_decides_what_to_WRITE_on_the_write_rule(self):
        # The 2026-08-10 split. Deciding the written columns on the SELECTION
        # rule is what made every threshold arm of Görtler's <1% question cost
        # another Spark job: a pkl built at a 1% floor has no column for a 0.3%
        # field, and no config setting can conjure one.
        source = self._step_03_source()
        self.assertIn('classify_seqparam_field_for_write(_name, _field_stats[_name])',
                      source)

    def test_step_03_records_the_absolute_row_count(self):
        # Learnability tracks the count, not the ratio. Without this the report
        # cannot answer "how many rows does PDM appear on" — the most basic
        # question about the rare-parameter debate — without another Spark run.
        source = self._step_03_source()
        self.assertIn("'presence_rows': len(_raws)", source)

    def test_step_03_persists_stats_for_EXCLUDED_fields_too(self):
        # Until 2026-08-10 the emitted table carried {name: reason} for excluded
        # fields and threw the stats away, so the corpus could not be asked
        # about its own exclusions after the fact.
        source = self._step_03_source()
        self.assertIn("'excluded': _excluded_payload", source)
        self.assertIn("'presence_rows': _stats['presence_rows']", source)

    def test_step_03_uses_the_per_name_missing_default(self):
        source = self._step_03_source()
        self.assertIn('_defaults[_name]', source)

    def test_step_03_writes_a_presence_flag_per_value(self):
        # Without this, safe_float turns every absent field into a fabricated
        # measurement, and "pass all parameters" makes the model worse.
        source = self._step_03_source()
        self.assertIn('_cond[presence_name(_name)]', source)

    def test_step_03_writes_the_join_scope_flag(self):
        # ~20% of rows carry a NEIGHBOUR's parameters, 71.5% of which name a
        # different sequence. The flag is how the model can tell.
        source = self._step_03_source()
        self.assertIn("_cond['sut_in_segment']", source)

    def test_step_03_uses_the_shared_field_rules(self):
        # The admissibility rule and the stable-name rule live in config so the
        # report can print the same reasons the build acted on. A second copy
        # here is how the two silently diverge.
        source = self._step_03_source()
        for symbol in ('classify_seqparam_field', 'seqparam_stable_name',
                       'suggest_divisor'):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, source)

    def test_step_03_emits_the_divisor_table(self):
        # This is what lets PARAM_SET='all' resolve ~89 calibrated divisors
        # without anybody hand-writing them.
        source = self._step_03_source()
        self.assertIn('SEQPARAM_DIVISOR_TABLE', source)
        self.assertIn("json.dump(_divisor_payload", source)

    def test_step_03_still_runs_leakage_gate_2(self):
        # The runtime dict keys, not just the static config list.
        source = self._step_03_source()
        self.assertIn('assert_no_leakage(_cond.keys())', source)


class SegmentDedupeFlagTests(unittest.TestCase):
    """03's segment loop reads DEDUPE_SHARED_TERMINATOR out of the %run'd
    config namespace, and nothing else would catch its removal until a Spark
    job fails an hour in."""

    def test_dedupe_flag_exists_and_defaults_on(self):
        module = _load_seqparams_config()
        self.assertIs(module.DEDUPE_SHARED_TERMINATOR, True)

    def test_step_03_reads_the_flag_by_name(self):
        step03 = os.path.join(
            os.path.dirname(_CONFIG_PATH), '03_build_preprocessed_pkl.py',
        )
        with open(step03) as f:
            self.assertIn('dedupe=DEDUPE_SHARED_TERMINATOR', f.read())


class ModelConfigAssemblyTests(unittest.TestCase):
    """build_seqparams_model_config — the single source of truth for
    combining AlternatingPipeline.config.EXAMINATION_MODEL_CONFIG with this
    pipeline's SUT additions, so 04/05/06 can't silently diverge from each
    other by duplicating the assembly recipe three times."""

    def test_widens_dims_by_confirmed_feature_count(self):
        module = _load_seqparams_config()
        base = {
            'base_conditioning_dim': 10,
            'conditioning_scale': [100.0] * 10,
            'model_type': 'examination',
        }
        # Simulate confirmed features for this test, independent of the
        # real (currently empty) placeholder state.
        module.EXAMINATION_SEQPARAM_FEATURES = ['TR', 'num_slices']
        module.EXAMINATION_SEQPARAM_SCALE = [1000.0, 30.0]

        result = module.build_seqparams_model_config(base)

        self.assertEqual(result['base_conditioning_dim'], 12)
        self.assertEqual(result['conditioning_scale'], [100.0] * 10 + [1000.0, 30.0])
        self.assertTrue(result['use_sut_conditioning'])
        self.assertEqual(result['num_trigger_modes'], module.NUM_TRIGGER_MODES)
        self.assertEqual(result['model_type'], 'examination')  # base fields preserved

    def test_is_a_no_op_widening_when_features_list_is_empty(self):
        module = _load_seqparams_config()
        # Explicitly empty, independent of the real (now confirmed non-empty)
        # module defaults — this asserts the general no-op-when-empty
        # invariant of build_seqparams_model_config itself.
        module.EXAMINATION_SEQPARAM_FEATURES = []
        module.EXAMINATION_SEQPARAM_SCALE = []
        base = {'base_conditioning_dim': 10, 'conditioning_scale': [1.0] * 10}

        result = module.build_seqparams_model_config(base)

        self.assertEqual(result['base_conditioning_dim'], 10)
        self.assertEqual(result['conditioning_scale'], [1.0] * 10)

    def test_does_not_mutate_the_base_config_dict(self):
        module = _load_seqparams_config()
        base = {'base_conditioning_dim': 10, 'conditioning_scale': [1.0] * 10}
        module.EXAMINATION_SEQPARAM_FEATURES = ['TR']
        module.EXAMINATION_SEQPARAM_SCALE = [1000.0]

        module.build_seqparams_model_config(base)

        self.assertEqual(base['base_conditioning_dim'], 10)  # untouched
        self.assertEqual(base['conditioning_scale'], [1.0] * 10)  # untouched


if __name__ == '__main__':
    unittest.main()
