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

    def test_defaults_to_luke(self):
        with _param_set('luke') as module:
            self.assertEqual(module.PARAM_SET, 'luke')
            self.assertEqual(module.EXAMINATION_SEQPARAM_FEATURES,
                             module.PARAM_SETS['luke'])

    def test_env_var_selects_the_other_set(self):
        with _param_set('navneet') as module:
            self.assertEqual(module.PARAM_SET, 'navneet')
            self.assertEqual(module.EXAMINATION_SEQPARAM_FEATURES,
                             module.PARAM_SETS['navneet'])

    def test_the_two_sets_are_actually_different(self):
        # A switch between two identical lists would pass every other test
        # here and answer nothing on Friday.
        module = _load_seqparams_config()
        self.assertNotEqual(module.PARAM_SETS['luke'],
                            module.PARAM_SETS['navneet'])

    def test_unknown_set_fails_loudly_at_import(self):
        with self.assertRaisesRegex(ValueError, 'not a known parameter set'):
            with _param_set('nuffnet'):
                pass

    def test_scale_is_derived_from_the_candidate_table(self):
        # The two used to be hand-maintained parallel lists that could desync.
        for name in ('luke', 'navneet'):
            with self.subTest(param_set=name), _param_set(name) as module:
                self.assertEqual(len(module.EXAMINATION_SEQPARAM_FEATURES),
                                 len(module.EXAMINATION_SEQPARAM_SCALE))
                self.assertEqual(
                    module.EXAMINATION_SEQPARAM_SCALE,
                    [module.SEQPARAM_CANDIDATES[n][0]
                     for n in module.EXAMINATION_SEQPARAM_FEATURES],
                )

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
        banned = module.SUT_LEAKAGE_DENYLIST | module.SUT_IDENTIFIER_DENYLIST
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

    def test_divisors_land_the_real_sampled_values_at_order_one(self):
        # The LayerNorm-erasure guard, pinned to real data rather than to a
        # comment. Values are the haste / ep2d_diff messages in
        # test_sut_parser.py; step 03 runs the same check across the full
        # corpus at write time.
        module = _load_seqparams_config()
        observed = {
            'TR': (866, 4300), 'num_slices': (9, 18),
            'phase_encoding_lines': (80, 333), 'base_resolution': (130, 320),
            'averages': (1, 1), 'concatenations': (1, 1),
            'parallel_imaging_factor': (2, 2), 'turbo_factor': (256, 256),
            'repetitions': (0, 0),
            'phase_partial_fourier': (1, 8), 'slice_partial_fourier': (16, 16),
        }
        self.assertEqual(set(observed), set(module.SEQPARAM_CANDIDATES),
                         "a candidate was added or removed without a real "
                         "observed value to calibrate its divisor against")
        for name, (low, high) in observed.items():
            divisor = module.SEQPARAM_CANDIDATES[name][0]
            with self.subTest(candidate=name):
                self.assertLessEqual(high / divisor, 20.0)
                if high > 0:
                    self.assertGreaterEqual(high / divisor, 0.05)

    def test_every_candidate_has_a_positive_divisor(self):
        # A zero or negative divisor either divides by zero or flips the
        # feature's sign, both silently.
        module = _load_seqparams_config()
        for name, (divisor, _) in module.SEQPARAM_CANDIDATES.items():
            with self.subTest(candidate=name):
                self.assertGreater(divisor, 0.0)

    def test_missing_defaults_mirror_the_candidate_table(self):
        module = _load_seqparams_config()
        self.assertEqual(
            module.SEQPARAM_MISSING_DEFAULTS,
            {n: d for n, (_, d) in module.SEQPARAM_CANDIDATES.items()},
        )

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

    def test_step_03_writes_the_union_not_the_selected_set(self):
        source = self._step_03_source()
        self.assertIn('for name in SEQPARAM_ALL_CANDIDATES', source)

    def test_step_03_uses_the_per_name_missing_default(self):
        source = self._step_03_source()
        self.assertIn('default=SEQPARAM_MISSING_DEFAULTS[name]', source)


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
