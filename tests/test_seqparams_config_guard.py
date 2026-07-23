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


class LeakageGuardTests(unittest.TestCase):
    def test_module_imports_cleanly_with_placeholder_empty_features(self):
        # The current placeholder EXAMINATION_SEQPARAM_FEATURES is empty, so
        # the module-level gate #1 check must not raise.
        module = _load_seqparams_config()
        self.assertEqual(module.EXAMINATION_SEQPARAM_FEATURES, [])

    def test_assert_no_leakage_passes_for_clean_feature_list(self):
        module = _load_seqparams_config()
        module.assert_no_leakage(['TR', 'num_slices'])  # must not raise

    def test_assert_no_leakage_rejects_scanning_time(self):
        module = _load_seqparams_config()
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            module.assert_no_leakage(['TR', 'scanning_time'])

    def test_assert_no_leakage_checks_runtime_dict_keys_too(self):
        # Gate #2's actual use pattern: check the real conditioning dict's
        # keys after SUT parsing + filtering, not just the static config list.
        module = _load_seqparams_config()
        conditioning = {'Age': 40, 'TR': 1500.0, 'scanning_time': 59.0}
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            module.assert_no_leakage(conditioning.keys())


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

    def test_is_a_no_op_widening_when_features_still_placeholder_empty(self):
        module = _load_seqparams_config()
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
