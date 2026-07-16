import unittest

import torch

from AlternatingPipeline.config import (
    BREAK_TOKEN_ID,
    END_REGION_ID,
    END_TOKEN_ID,
    ORCH_PAD_TOKEN_ID,
    PAD_TOKEN_ID,
    SOURCEID_VOCAB,
    START_REGION_ID,
    START_TOKEN_ID,
)
from AlternatingPipeline.generation.output_integrity import (
    GenerationIntegrityError,
    repair_examination_sequence,
    validate_examination_sequence,
    validate_orchestration_sequence,
    validate_rendered_output,
)
from AlternatingPipeline.models.orchestration_model import OrchestrationModel
from AlternatingPipeline.models.sequence_generator import SequenceGeneratorModel


def _sequence_config(model_type):
    config = {
        'model_type': model_type,
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
    if model_type == 'exchange':
        config['body_region_mode'] = 'from_to'
    return config


def _set_constant_logits(model, scores):
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(-100.0)
        for token_id, score in scores.items():
            model.output_projection.bias[token_id] = score


class SequenceGeneratorConstraintTests(unittest.TestCase):
    def test_examination_decoder_enforces_measurement_grammar(self):
        model = SequenceGeneratorModel(_sequence_config('examination'))
        finish_id = SOURCEID_VOCAB['MRI_MSR_104']
        _set_constant_logits(model, {
            END_TOKEN_ID: 100.0,
            PAD_TOKEN_ID: 90.0,
            START_TOKEN_ID: 80.0,
            SOURCEID_VOCAB['UNK']: 70.0,
            finish_id: 60.0,
        })

        tokens, _ = model.generate(
            torch.zeros(2),
            {'body_region': 0},
            max_length=8,
            top_k=1,
            top_p=1.0,
        )

        self.assertEqual(tokens[0].tolist(), [
            START_TOKEN_ID,
            SOURCEID_VOCAB['MRI_MSR_100'],
            finish_id,
            END_TOKEN_ID,
        ])
        self.assertEqual(
            validate_examination_sequence(
                tokens[0],
                START_TOKEN_ID,
                END_TOKEN_ID,
                PAD_TOKEN_ID,
                SOURCEID_VOCAB['MRI_MSR_100'],
                [finish_id, SOURCEID_VOCAB['MRI_MSR_34']],
                SOURCEID_VOCAB['UNK'],
            ),
            finish_id,
        )

    def test_exchange_decoder_cannot_emit_empty_or_unknown_block(self):
        model = SequenceGeneratorModel(_sequence_config('exchange'))
        event_id = SOURCEID_VOCAB['MRI_CCS_11']
        _set_constant_logits(model, {
            END_TOKEN_ID: 100.0,
            PAD_TOKEN_ID: 90.0,
            START_TOKEN_ID: 80.0,
            SOURCEID_VOCAB['UNK']: 70.0,
            event_id: 60.0,
        })

        tokens, _ = model.generate(
            torch.zeros(2),
            {'body_from': 0, 'body_to': 1},
            max_length=5,
            top_k=1,
            top_p=1.0,
        )

        self.assertEqual(tokens[0].tolist(), [
            START_TOKEN_ID,
            event_id,
            END_TOKEN_ID,
        ])


class OrchestrationConstraintTests(unittest.TestCase):
    def _model(self):
        config = {
            'vocab_size': 15,
            'd_model': 8,
            'nhead': 2,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 16,
            'dropout': 0.0,
            'max_seq_len': 8,
            'base_conditioning_dim': 3,
            'num_scanners': 1,
            'scanner_emb_dim': 4,
            'pad_token_id': ORCH_PAD_TOKEN_ID,
            'start_token_id': START_REGION_ID,
            'end_token_id': END_REGION_ID,
            'break_token_id': BREAK_TOKEN_ID,
        }
        return OrchestrationModel(config)

    def test_day_plan_has_patients_around_breaks_and_respects_support(self):
        model = self._model()
        allowed_region = 2
        _set_constant_logits(model, {
            BREAK_TOKEN_ID: 100.0,
            END_REGION_ID: 90.0,
            allowed_region: 80.0,
        })

        tokens = model.generate(
            torch.zeros(3),
            0,
            max_length=5,
            top_k=1,
            top_p=1.0,
            allowed_tokens={allowed_region, BREAK_TOKEN_ID, END_REGION_ID},
        )

        self.assertEqual(tokens[0].tolist(), [
            START_REGION_ID,
            allowed_region,
            BREAK_TOKEN_ID,
            allowed_region,
            END_REGION_ID,
        ])
        self.assertEqual(
            validate_orchestration_sequence(
                tokens[0],
                START_REGION_ID,
                END_REGION_ID,
                ORCH_PAD_TOKEN_ID,
                BREAK_TOKEN_ID,
                {allowed_region},
            ),
            [allowed_region, allowed_region],
        )

    def test_scanner_support_requires_at_least_one_region(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, 'body-region'):
            model.generate(
                torch.zeros(3),
                0,
                allowed_tokens={BREAK_TOKEN_ID, END_REGION_ID},
            )


class IntegrityValidatorTests(unittest.TestCase):
    def test_repeated_exam_starts_fall_back_to_latest_complete_span(self):
        measurement_start = SOURCEID_VOCAB['MRI_MSR_100']
        finish = SOURCEID_VOCAB['MRI_MSR_104']
        event = SOURCEID_VOCAB['MRI_MSR_21']
        repaired, duration_source_idx = repair_examination_sequence(
            [
                START_TOKEN_ID,
                measurement_start,
                SOURCEID_VOCAB['MRI_EXU_95'],
                measurement_start,
                event,
                finish,
                END_TOKEN_ID,
            ],
            START_TOKEN_ID,
            END_TOKEN_ID,
            PAD_TOKEN_ID,
            measurement_start,
            [finish, SOURCEID_VOCAB['MRI_MSR_34']],
            SOURCEID_VOCAB['UNK'],
        )

        self.assertEqual(repaired, [
            START_TOKEN_ID,
            measurement_start,
            event,
            finish,
            END_TOKEN_ID,
        ])
        # The finish was at generated index 5; duration supervision is shifted.
        self.assertEqual(duration_source_idx, 4)
        self.assertEqual(
            validate_examination_sequence(
                repaired,
                START_TOKEN_ID,
                END_TOKEN_ID,
                PAD_TOKEN_ID,
                measurement_start,
                [finish, SOURCEID_VOCAB['MRI_MSR_34']],
                SOURCEID_VOCAB['UNK'],
            ),
            finish,
        )

    def test_malformed_examination_is_rejected(self):
        with self.assertRaises(GenerationIntegrityError):
            validate_examination_sequence(
                [
                    START_TOKEN_ID,
                    SOURCEID_VOCAB['MRI_MSR_100'],
                    END_TOKEN_ID,
                ],
                START_TOKEN_ID,
                END_TOKEN_ID,
                PAD_TOKEN_ID,
                SOURCEID_VOCAB['MRI_MSR_100'],
                [
                    SOURCEID_VOCAB['MRI_MSR_104'],
                    SOURCEID_VOCAB['MRI_MSR_34'],
                ],
                SOURCEID_VOCAB['UNK'],
            )

    def test_rendered_output_must_match_the_planned_counts(self):
        exchange_rows = [{
            'sample_idx': 0,
            'step': 0,
            'token_name': 'MRI_CCS_11',
            'sampled_duration': 1.0,
            'datetime': '2024-02-01 07:00:00',
        }]
        exam_rows = [{
            'PatientID': 'PAT001',
            'StepCount': 1,
            'duration': 60,
            'startTime': '2024-02-01 07:00:01',
            'endTime': '2024-02-01 07:01:01',
            'FinishEvent': 'Successful',
            'sourceID': 'MRI_MSR_104',
            'BodyPart': 'HEAD',
        }]

        report = validate_rendered_output(
            exchange_rows,
            exam_rows,
            {'PAT001': 1},
            expected_exchange_blocks=1,
            valid_body_regions={'HEAD'},
        )
        self.assertEqual(report['exam_rows'], 1)

        with self.assertRaisesRegex(GenerationIntegrityError, 'planned/rendered'):
            validate_rendered_output(
                exchange_rows,
                exam_rows,
                {'PAT001': 2},
                expected_exchange_blocks=1,
                valid_body_regions={'HEAD'},
            )


if __name__ == '__main__':
    unittest.main()
