"""Checkpoint/architecture compatibility diagnosis.

This project's checkpoints live on a shared, mutable DBFS FileStore path and
outlive many rounds of architecture changes. Three distinct things can differ
between a checkpoint and the model class trying to load it:

  * MISSING keys      — the model gained a parameter after the checkpoint was
                        trained (e.g. `duration_seq_type_bias`, added in
                        b9f3dd0). Benign: the new parameter keeps its
                        initialised value.
  * UNEXPECTED keys   — the checkpoint has parameters the model no longer has.
                        Benign: they are ignored.
  * SHAPE MISMATCHES  — the same parameter name has a different shape. NOT
                        benign, and, critically, NOT survivable via
                        `strict=False`: PyTorch collects size mismatches into
                        `error_msgs` and raises regardless of `strict`. A
                        shape mismatch means the checkpoint came from a
                        genuinely different architecture, and any "load what
                        fits" workaround would leave that submodule randomly
                        initialised — silently producing meaningless
                        predictions, which is the exact failure mode this
                        project has been bitten by repeatedly.

`inspect_checkpoint` reports all three without loading anything, so callers
can decide whether to proceed, skip, or stop with a message that says what to
do instead of a raw stack trace.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


class IncompatibleCheckpointError(RuntimeError):
    """A checkpoint cannot be loaded into this model at all."""


@dataclass
class CheckpointCompatibility:
    """What differs between a state dict and a model's own parameters."""

    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    shape_mismatched: List[Tuple[str, Any, Any]] = field(default_factory=list)

    @property
    def loadable(self) -> bool:
        """True when `load_state_dict(..., strict=False)` will succeed."""
        return not self.shape_mismatched

    @property
    def clean(self) -> bool:
        """True when the checkpoint matches the model exactly."""
        return not (self.missing or self.unexpected or self.shape_mismatched)

    def describe(self, label: str = "checkpoint", max_names: int = 8) -> str:
        """Human-readable diagnosis, safe to print in a notebook cell."""
        if self.clean:
            return f"{label}: exact architecture match."

        lines = [f"{label}: architecture differs from this checkpoint."]

        if self.shape_mismatched:
            lines.append(
                f"  {len(self.shape_mismatched)} SHAPE MISMATCH(ES) — not loadable "
                f"(strict=False does not help; PyTorch raises on these regardless):"
            )
            for name, ckpt_shape, model_shape in self.shape_mismatched[:max_names]:
                lines.append(
                    f"    - {name}: checkpoint {ckpt_shape} vs. model {model_shape}"
                )
            if len(self.shape_mismatched) > max_names:
                lines.append(f"    ... and {len(self.shape_mismatched) - max_names} more")

        if self.missing:
            lines.append(
                f"  {len(self.missing)} param(s) in the model but not the checkpoint "
                f"(would keep their initialised values): {self.missing[:max_names]}"
            )
        if self.unexpected:
            lines.append(
                f"  {len(self.unexpected)} param(s) in the checkpoint but not the model "
                f"(would be ignored): {self.unexpected[:max_names]}"
            )

        if self.shape_mismatched:
            lines.append(
                "  => This checkpoint was written by a DIFFERENT model architecture, "
                "not merely an older version of this one. Retrain it with the current "
                "code, or point at a checkpoint that this code produced."
            )
        return "\n".join(lines)


def inspect_checkpoint(model, state_dict: Dict[str, Any]) -> CheckpointCompatibility:
    """Compare `state_dict` against `model`'s parameters without loading it."""
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))

    shape_mismatched = []
    for name in sorted(set(model_state) & set(state_dict)):
        ckpt_tensor, model_tensor = state_dict[name], model_state[name]
        ckpt_shape = getattr(ckpt_tensor, "shape", None)
        model_shape = getattr(model_tensor, "shape", None)
        if ckpt_shape is not None and model_shape is not None and ckpt_shape != model_shape:
            shape_mismatched.append((name, ckpt_shape, model_shape))

    return CheckpointCompatibility(
        missing=missing, unexpected=unexpected, shape_mismatched=shape_mismatched
    )


def load_checkpoint_lenient(
    model,
    state_dict: Dict[str, Any],
    label: str = "checkpoint",
    verbose: bool = True,
) -> CheckpointCompatibility:
    """Load `state_dict` into `model`, tolerating added/removed parameters.

    Raises IncompatibleCheckpointError — with a diagnosis naming the offending
    parameters and what to do about it — when the checkpoint comes from a
    different architecture. Never leaves a partially-initialised model behind:
    on a shape mismatch nothing is loaded at all.
    """
    report = inspect_checkpoint(model, state_dict)

    if not report.loadable:
        raise IncompatibleCheckpointError(report.describe(label=label))

    if verbose and not report.clean:
        print(report.describe(label=label))

    model.load_state_dict(state_dict, strict=False)
    return report
