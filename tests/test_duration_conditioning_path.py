"""Tests for the per-position conditioning path into the duration encoder.

Measured on the 2026-07-24 examination checkpoint (step 06, Criterion 1b),
perturbing each conditioning channel and reading the duration head:

    sequence_type (scout vs space)   mean|delta| = 249.475s   <- per-position bias
    body_region   (ABDOMEN vs SPINE) mean|delta| =   0.000s   <- conditioning token only
    serial_idx    (0 vs 1)           mean|delta| =   0.001s   <- conditioning token only
    TR+num_slices (1x vs 3x)         mean|delta| =   0.002s   <- conditioning token only

Only sequence_type moved the duration head, and it is the only channel with a
per-position path (`duration_seq_type_bias`). Everything arriving solely via
the single conditioning token at position 0 was dead — including body_region,
which has a ~2.3x real duration spread (ABDOMEN ~63s vs SPINE ~144s), so this
was a model defect rather than a property of the data.

`duration_cond_bias` gives the whole conditioning vector the same per-position
path. It is zero-initialised so every existing checkpoint keeps behaving
exactly as before until it is retrained.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "AlternatingPipeline"))

from AlternatingPipeline.config import EXAMINATION_MODEL_CONFIG  # noqa: E402
from AlternatingPipeline.models.examination_model import create_examination_model  # noqa: E402
from AlternatingPipeline.models.checkpoint_compat import (  # noqa: E402
    inspect_checkpoint, load_checkpoint_lenient,
)

TOKENS = torch.tensor([[11, 10, 13, 12]])


def _model(seed=0):
    torch.manual_seed(seed)
    return create_examination_model(EXAMINATION_MODEL_CONFIG).eval()


def _info(body_region=3):
    return {
        'body_region': torch.tensor([body_region]),
        'sequence_type': torch.tensor([3]),
        'serial_idx': torch.tensor([0]),
    }


def _cond(value=1.0):
    dim = EXAMINATION_MODEL_CONFIG['base_conditioning_dim']
    return torch.full((1, dim), value)


def _mu(model, cond, info):
    with torch.no_grad():
        mu, _ = model.estimate_durations(TOKENS, cond, info)
    return mu


def test_duration_cond_bias_exists():
    assert hasattr(_model(), 'duration_cond_bias')


def test_zero_initialised_so_existing_checkpoints_are_unaffected():
    model = _model()
    assert torch.count_nonzero(model.duration_cond_bias.weight) == 0
    assert torch.count_nonzero(model.duration_cond_bias.bias) == 0


def test_zero_init_contributes_exactly_nothing():
    """Backward compatibility: at init the new path must add a zero tensor."""
    model = _model(seed=7)
    with torch.no_grad():
        cond_token = model._get_conditioning_token(_cond(), _info())
        contribution = model.duration_cond_bias(cond_token)

    assert contribution.shape[-1] == model.d_model
    assert torch.count_nonzero(contribution) == 0


def test_conditioning_moves_the_duration_head_once_the_path_is_trained():
    """The defect this fixes: conditioning must be able to reach the head.

    The magnitude here is not meaningful — a randomly-initialised projection is
    not a stand-in for a trained one, and how far it moves the output depends on
    d_model and the init scale. What matters is the direction: enabling the
    per-position route strictly increases how much the conditioning can move the
    duration head. `test_duration_cond_bias_receives_gradient` covers the claim
    that training can actually strengthen it.
    """
    model = _model(seed=11)
    info = _info()

    # Zero-init: the ONLY route is the diluted conditioning token.
    before = (_mu(model, _cond(1.0), info) - _mu(model, _cond(4.0), info)).abs().max()

    torch.nn.init.xavier_uniform_(model.duration_cond_bias.weight)
    after = (_mu(model, _cond(1.0), info) - _mu(model, _cond(4.0), info)).abs().max()

    assert after > before, (
        f"per-position path did not increase conditioning sensitivity "
        f"(before={before:.6f}, after={after:.6f})"
    )


def test_duration_cond_bias_receives_gradient():
    """Zero-init is only safe if the path still trains: d(Wx)/dW = x, not 0.

    This is the property the whole fix rests on. If the layer started at zero
    AND received no gradient, it would stay a no-op forever and the retrain
    would change nothing.
    """
    model = _model(seed=19)
    model.train()

    mu, _ = model.estimate_durations(TOKENS, _cond(2.0), _info())
    mu.sum().backward()

    weight_grad = model.duration_cond_bias.weight.grad
    assert weight_grad is not None, "no gradient reached duration_cond_bias"
    assert torch.count_nonzero(weight_grad) > 0, (
        "duration_cond_bias got an all-zero gradient — it would never train"
    )


def test_body_region_reaches_the_duration_head_through_the_new_path():
    """body_region was the channel proven dead in production (0.000s)."""
    model = _model(seed=13)
    torch.nn.init.xavier_uniform_(model.duration_cond_bias.weight)
    cond = _cond()

    abdomen = _mu(model, cond, _info(body_region=3))
    spine = _mu(model, cond, _info(body_region=5))

    assert (abdomen - spine).abs().max() > 1e-4


def test_old_checkpoint_loads_and_predicts_identically():
    """A checkpoint predating this change must load and not shift its output."""
    model = _model(seed=17)
    legacy_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith('duration_cond_bias')
    }

    report = inspect_checkpoint(model, legacy_state)
    assert report.loadable
    assert 'duration_cond_bias.weight' in report.missing

    cond, info = _cond(), _info()
    before = _mu(model, cond, info)
    load_checkpoint_lenient(model, legacy_state, label="legacy", verbose=False)
    after = _mu(model, cond, info)

    assert torch.allclose(before, after, atol=0.0)


def test_exchange_model_also_gets_the_path():
    """Exchange has the same dead-conditioning shape (Stage 3b: no variability)."""
    from AlternatingPipeline.config import EXCHANGE_MODEL_CONFIG
    from AlternatingPipeline.models.exchange_model import create_exchange_model

    torch.manual_seed(0)
    model = create_exchange_model(EXCHANGE_MODEL_CONFIG)
    assert hasattr(model, 'duration_cond_bias')
    assert torch.count_nonzero(model.duration_cond_bias.weight) == 0
