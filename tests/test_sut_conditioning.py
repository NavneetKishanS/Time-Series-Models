import unittest

import torch

from AlternatingPipeline.config import SOURCEID_VOCAB, START_TOKEN_ID
from AlternatingPipeline.models.sequence_generator import SequenceGeneratorModel
from AlternatingPipeline.training.train_examination import ExaminationDataset
from AlternatingPipeline.training.utils import build_conditioning_tensor


def _examination_config(**overrides):
    config = {
        'model_type': 'examination',
        'vocab_size': len(SOURCEID_VOCAB),
        'd_model': 8,
        'nhead': 2,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_duration_encoder_layers': 1,
        'dim_feedforward': 16,
        'dropout': 0.0,
        'max_seq_len': 8,
        'base_conditioning_dim': 2,
        'num_body_regions': 2,
        'num_region_classes': 4,
        'has_phase_type': False,
        'body_region_mode': 'single',
        'duration_mode': 'gaussian',
    }
    config.update(overrides)
    return config


class TriggerModeConditioningTests(unittest.TestCase):
    """The new opt-in `use_sut_conditioning` flag in SequenceGeneratorModel."""

    def test_old_config_creates_no_sut_conditioning_machinery(self):
        model = SequenceGeneratorModel(_examination_config())
        self.assertFalse(getattr(model, 'use_sut_conditioning', False))
        self.assertFalse(hasattr(model, 'trigger_mode_embedding'))

    def test_trigger_mode_changes_duration_estimate(self):
        """The historical landmine: a new conditioning input must actually move
        the model's output, not get erased by LayerNorm (see conditioning_scale
        comments in sequence_generator.py)."""
        config = _examination_config(use_sut_conditioning=True, num_trigger_modes=4)
        model = SequenceGeneratorModel(config)
        self.assertTrue(hasattr(model, 'trigger_mode_embedding'))

        torch.manual_seed(0)
        tokens = torch.tensor([[
            START_TOKEN_ID, SOURCEID_VOCAB['MRI_MSR_100'], SOURCEID_VOCAB['MRI_MSR_104'],
        ]])
        conditioning = torch.zeros(1, 2)
        region = {'body_region': torch.tensor([0])}

        mu_a, _ = model.estimate_durations(
            tokens, conditioning, {**region, 'trigger_mode': torch.tensor([0])}
        )
        mu_b, _ = model.estimate_durations(
            tokens, conditioning, {**region, 'trigger_mode': torch.tensor([3])}
        )
        self.assertGreater((mu_a - mu_b).abs().max().item(), 1e-4)

    def test_missing_trigger_mode_defaults_to_zero_without_error(self):
        """Old call sites that don't know about trigger_mode must keep working —
        same default-to-zero pattern already used for sequence_type/serial_idx."""
        config = _examination_config(use_sut_conditioning=True, num_trigger_modes=4)
        model = SequenceGeneratorModel(config)
        tokens = torch.tensor([[START_TOKEN_ID, SOURCEID_VOCAB['MRI_MSR_100']]])
        conditioning = torch.zeros(1, 2)

        mu_default, _ = model.estimate_durations(
            tokens, conditioning, {'body_region': torch.tensor([0])}
        )
        mu_explicit, _ = model.estimate_durations(
            tokens, conditioning,
            {'body_region': torch.tensor([0]), 'trigger_mode': torch.tensor([0])},
        )
        self.assertTrue(torch.allclose(mu_default, mu_explicit))


class ConditioningTensorExtensionTests(unittest.TestCase):
    """The new optional extra_feature_names parameter on build_conditioning_tensor."""

    def test_default_call_matches_original_10dim_output(self):
        cond = {
            'Age': 40, 'Weight': 70, 'Height': 170, 'PTAB': 0, 'Direction_encoded': 1,
            'hour_sin': 0.5, 'hour_cos': 0.5, 'dow_sin': 0.1, 'dow_cos': 0.9, 'is_morning': 1,
        }
        tensor = build_conditioning_tensor(cond)
        self.assertEqual(tuple(tensor.shape), (10,))

    def test_extra_feature_names_appends_values_in_order(self):
        cond = {'Age': 40, 'TR': 1500.0, 'num_slices': 24.0}
        tensor = build_conditioning_tensor(cond, extra_feature_names=['TR', 'num_slices'])
        self.assertEqual(tuple(tensor.shape), (12,))
        self.assertAlmostEqual(tensor[10].item(), 1500.0)
        self.assertAlmostEqual(tensor[11].item(), 24.0)

    def test_extra_feature_names_missing_key_defaults_to_zero(self):
        tensor = build_conditioning_tensor({}, extra_feature_names=['TR'])
        self.assertEqual(tuple(tensor.shape), (11,))
        self.assertEqual(tensor[10].item(), 0.0)

    def test_denylist_none_is_a_no_op(self):
        # Default behavior (no denylist passed) must be unaffected.
        tensor = build_conditioning_tensor({'TR': 1500.0}, extra_feature_names=['TR'])
        self.assertEqual(tuple(tensor.shape), (11,))

    def test_denylist_rejects_overlapping_feature_at_the_point_of_construction(self):
        # This is leakage gate #3 (training-tensor-build-time): it must fire
        # here, independent of whatever gate #1/#2 already checked upstream —
        # a caller could construct extra_feature_names by any means at all,
        # and this must still catch it.
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            build_conditioning_tensor(
                {'scanning_time': 59.0},
                extra_feature_names=['scanning_time'],
                denylist={'scanning_time'},
            )

    def test_denylist_passes_for_clean_feature_list(self):
        tensor = build_conditioning_tensor(
            {'TR': 1500.0}, extra_feature_names=['TR'], denylist={'scanning_time'},
        )
        self.assertEqual(tuple(tensor.shape), (11,))


class ExaminationDatasetTriggerModeTests(unittest.TestCase):
    """ExaminationDataset threading trigger_mode + extra numeric conditioning."""

    def _fake_sequence(self, trigger_mode=None):
        seq = {
            'conditioning': {
                'Age': 40, 'Weight': 70, 'Height': 170, 'PTAB': 0,
                'Direction_encoded': 0, 'hour_sin': 0, 'hour_cos': 1,
                'dow_sin': 0, 'dow_cos': 1, 'is_morning': 1,
            },
            'body_region': 0,
            'sequence_type': 3,
            'serial_idx': 0,
            'sequence': [SOURCEID_VOCAB['MRI_MSR_100'], SOURCEID_VOCAB['MRI_MSR_104']],
            'durations': [0.0, 45.0],
        }
        if trigger_mode is not None:
            seq['trigger_mode'] = trigger_mode
        return seq

    def test_item_carries_trigger_mode_as_eighth_field(self):
        """Position matters: make_pad_collate trims by index (seq_indices=(4,5,6)),
        so per-sequence categoricals must stay at the tail. protocol was appended
        as the ninth field for exactly that reason — see test_protocol_conditioning.
        """
        dataset = ExaminationDataset([self._fake_sequence(trigger_mode=2)], max_seq_len=8)
        item = dataset[0]
        self.assertEqual(len(item), 9)
        self.assertEqual(item[7].item(), 2)

    def test_missing_trigger_mode_defaults_to_zero(self):
        dataset = ExaminationDataset([self._fake_sequence()], max_seq_len=8)
        item = dataset[0]
        self.assertEqual(item[7].item(), 0)

    def test_extra_conditioning_features_widen_the_tensor(self):
        seq = self._fake_sequence()
        seq['conditioning']['TR'] = 1500.0
        dataset = ExaminationDataset(
            [seq], max_seq_len=8, extra_conditioning_features=['TR'],
        )
        item = dataset[0]
        self.assertEqual(tuple(item[0].shape), (11,))
        self.assertAlmostEqual(item[0][10].item(), 1500.0)

    def test_leakage_denylist_is_enforced_at_dataset_construction(self):
        # Leakage gate #3, exercised through the actual training entry point
        # a caller would use — must fire here even if some future caller
        # assembled extra_conditioning_features by a path that never went
        # through csv_pipeline_seqparams/config.py's own gate #1.
        seq = self._fake_sequence()
        seq['conditioning']['scanning_time'] = 59.0
        with self.assertRaisesRegex(ValueError, 'scanning_time'):
            ExaminationDataset(
                [seq], max_seq_len=8,
                extra_conditioning_features=['scanning_time'],
                leakage_denylist={'scanning_time'},
            )


if __name__ == '__main__':
    unittest.main()
