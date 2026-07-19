import math
import unittest

import torch

from AlternatingPipeline.config import (
    END_TOKEN_ID,
    SOURCEID_VOCAB,
    START_TOKEN_ID,
)
from AlternatingPipeline.models.sequence_generator import SequenceGeneratorModel


def _mixture_config(num_components=3):
    return {
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
        'duration_mode': 'log',
        'duration_distribution': 'mixture',
        'duration_num_components': num_components,
        'duration_min_sigma': 0.05,
        'duration_component_init_means': [
            0.25 + 0.5 * idx for idx in range(num_components)
        ],
        'duration_component_init_sigmas': [
            0.20 + 0.05 * idx for idx in range(num_components)
        ],
    }


class DurationMixtureTests(unittest.TestCase):
    def test_forward_and_mixture_loss_have_expected_shapes(self):
        model = SequenceGeneratorModel(_mixture_config())
        conditioning = torch.zeros(2, 2)
        body_region = torch.zeros(2, dtype=torch.long)
        tokens = torch.tensor([
            [START_TOKEN_ID, SOURCEID_VOCAB['MRI_MSR_100'], SOURCEID_VOCAB['MRI_MSR_104'], END_TOKEN_ID],
            [START_TOKEN_ID, SOURCEID_VOCAB['MRI_MSR_100'], SOURCEID_VOCAB['MRI_MSR_34'], END_TOKEN_ID],
        ])

        logits, mu, sigma, mixture_logits = model(
            conditioning,
            {'body_region': body_region},
            tokens,
            return_duration_distribution=True,
        )

        self.assertEqual(logits.shape, (2, 4, len(SOURCEID_VOCAB)))
        self.assertEqual(mu.shape, (2, 4, 3))
        self.assertEqual(sigma.shape, (2, 4, 3))
        self.assertEqual(mixture_logits.shape, (2, 4, 3))

        targets = torch.zeros(2, 4)
        targets[:, 1] = torch.tensor([1.0, 2.0])
        ignore_mask = targets <= 0
        loss = model.compute_duration_loss(
            mu,
            sigma,
            targets,
            ignore_mask=ignore_mask,
            mixture_logits=mixture_logits,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.duration_head.mixture_logits_head.weight.grad)

    def test_mixture_nll_rewards_a_component_near_the_target(self):
        model = SequenceGeneratorModel(_mixture_config(num_components=2))
        target = torch.tensor([[1.0]])
        sigma = torch.full((1, 1, 2), 0.1)
        mixture_logits = torch.zeros(1, 1, 2)
        target_log = math.log1p(1.0)

        good_mu = torch.tensor([[[target_log, 3.0]]])
        bad_mu = torch.tensor([[[2.0, 3.0]]])
        good_loss = model.compute_duration_loss(
            good_mu,
            sigma,
            target,
            mixture_logits=mixture_logits,
        )
        bad_loss = model.compute_duration_loss(
            bad_mu,
            sigma,
            target,
            mixture_logits=mixture_logits,
        )

        self.assertLess(good_loss.item(), bad_loss.item())

    def test_duration_moments_respect_mixture_weights(self):
        model = SequenceGeneratorModel(_mixture_config(num_components=2))
        mu = torch.tensor([[[math.log(2.0), math.log(10.0)]]])
        sigma = torch.full((1, 1, 2), 1e-4)
        logits = torch.tensor([[[20.0, -20.0]]])

        mean, std = model.duration_moments(mu, sigma, logits)

        self.assertAlmostEqual(mean.item(), 1.0, places=3)
        self.assertTrue(torch.isfinite(std).all())

    def test_generation_keeps_legacy_two_dimensional_stats(self):
        model = SequenceGeneratorModel(_mixture_config())
        finish_id = SOURCEID_VOCAB['MRI_MSR_104']
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-100.0)
            model.output_projection.bias[finish_id] = 100.0

        tokens, durations, mu, sigma = model.generate(
            torch.zeros(2),
            {'body_region': 0},
            max_length=6,
            top_k=1,
            top_p=1.0,
            return_stats=True,
        )

        self.assertEqual(tokens.ndim, 2)
        self.assertEqual(durations.shape, tokens.shape)
        self.assertEqual(mu.shape, tokens.shape)
        self.assertEqual(sigma.shape, tokens.shape)
        self.assertTrue(torch.isfinite(durations).all())
        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(sigma).all())


if __name__ == '__main__':
    unittest.main()
