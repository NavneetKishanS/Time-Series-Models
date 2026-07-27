"""Tests for checkpoint/architecture compatibility diagnosis.

Motivated by a real failure: 06_compare_models.py crashed with a raw
`RuntimeError: Error(s) in loading state_dict` when the baseline examination
checkpoint on DBFS turned out to have been written by an architecture that
does not exist in this repo (a 3-component mixture-density duration head).

The important property proven here is the one that made the original
mitigation useless: `strict=False` does NOT tolerate shape mismatches, so
callers cannot rely on it to survive an architecture change.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AlternatingPipeline.models.checkpoint_compat import (  # noqa: E402
    IncompatibleCheckpointError,
    inspect_checkpoint,
    load_checkpoint_lenient,
)


class _Head(nn.Module):
    """Stand-in for SinglePassDurationHead (1 component)."""

    def __init__(self, n_components=1, with_bias_embedding=False, with_mixture=False):
        super().__init__()
        self.shared_mlp = nn.Linear(8, 16)
        self.mu_head = nn.Linear(16, n_components)
        self.sigma_head = nn.Linear(16, n_components)
        if with_mixture:
            self.mixture_logits_head = nn.Linear(16, n_components)
        if with_bias_embedding:
            self.duration_seq_type_bias = nn.Embedding(4, 8)


def test_identical_architecture_is_clean():
    model, other = _Head(), _Head()
    report = inspect_checkpoint(model, other.state_dict())

    assert report.loadable
    assert report.missing == []
    assert report.unexpected == []
    assert report.shape_mismatched == []


def test_new_param_absent_from_checkpoint_is_loadable():
    """The duration_seq_type_bias case: model gained a param after training."""
    model = _Head(with_bias_embedding=True)
    checkpoint = _Head(with_bias_embedding=False).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert report.loadable
    assert "duration_seq_type_bias.weight" in report.missing
    assert report.shape_mismatched == []

    # And it actually loads, leaving the new parameter at its initialised value.
    before = model.duration_seq_type_bias.weight.clone()
    load_checkpoint_lenient(model, checkpoint, label="test")
    assert torch.equal(model.duration_seq_type_bias.weight, before)
    assert torch.equal(model.mu_head.weight, checkpoint["mu_head.weight"])


def test_extra_param_in_checkpoint_is_loadable():
    model = _Head()
    checkpoint = _Head(with_mixture=True).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert report.loadable
    assert "mixture_logits_head.weight" in report.unexpected
    assert report.shape_mismatched == []


def test_shape_mismatch_is_not_loadable():
    """The real failure: a 3-component mixture head vs. a 1-component head."""
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3, with_mixture=True).state_dict()

    report = inspect_checkpoint(model, checkpoint)

    assert not report.loadable
    names = [name for name, _, _ in report.shape_mismatched]
    assert "mu_head.weight" in names
    assert "sigma_head.weight" in names

    described = report.describe(label="baseline")
    assert "mu_head.weight" in described
    assert "torch.Size([3, 16])" in described
    assert "torch.Size([1, 16])" in described


def test_load_lenient_raises_actionable_error_on_shape_mismatch():
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3, with_mixture=True).state_dict()

    with pytest.raises(IncompatibleCheckpointError) as excinfo:
        load_checkpoint_lenient(model, checkpoint, label="baseline examination")

    message = str(excinfo.value)
    assert "baseline examination" in message
    assert "mu_head.weight" in message
    # Must name the remedy, not just the symptom.
    assert "retrain" in message.lower()


def test_strict_false_does_not_survive_shape_mismatch():
    """Guards the assumption that made the original mitigation ineffective.

    b9f3dd0 added `strict=False` to step 05's checkpoint load specifically so
    a new parameter would not break an existing checkpoint. That works for
    missing/unexpected keys only — PyTorch still raises on shape mismatch
    regardless of `strict`. If this test ever fails, PyTorch changed and
    load_checkpoint_lenient can be simplified.
    """
    model = _Head(n_components=1)
    checkpoint = _Head(n_components=3).state_dict()

    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(checkpoint, strict=False)
