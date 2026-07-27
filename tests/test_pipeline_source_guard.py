"""Tests for the stale-/tmp-copy guard used by steps 05 and 06.

Regression: on 2026-07-27 step 06 died with a bare
`ModuleNotFoundError: No module named 'AlternatingPipeline.models.checkpoint_compat'`
because TMP_ROOT had been copied by step 04 during the 07-24 training run,
before that module existed, on a cluster that had been up ever since. The error
named the module, not the stale copy, which is the wrong thing to look at.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "DatabricksPipeline", "csv_pipeline_seqparams", "config.py",
)


@pytest.fixture(scope="module")
def seqparams_config():
    """Load the notebook-style config as a module (pure constants + funcs)."""
    module = types.ModuleType("seqparams_config_under_test")
    with open(CONFIG_PATH) as handle:
        exec(compile(handle.read(), CONFIG_PATH, "exec"), module.__dict__)
    return module


def _make_source_tree(root, dotted_modules):
    for dotted in dotted_modules:
        path = os.path.join(root, *dotted.split(".")) + ".py"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("# stub\n")


def test_passes_when_every_required_module_is_present(seqparams_config, tmp_path):
    required = [
        "AlternatingPipeline.config",
        "AlternatingPipeline.models.checkpoint_compat",
    ]
    _make_source_tree(str(tmp_path), required)

    assert seqparams_config.assert_pipeline_source_fresh(
        str(tmp_path), required_modules=required, purge=False
    ) is True


def test_raises_naming_the_missing_module_and_the_remedy(seqparams_config, tmp_path):
    _make_source_tree(str(tmp_path), ["AlternatingPipeline.config"])

    with pytest.raises(RuntimeError) as excinfo:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path),
            required_modules=[
                "AlternatingPipeline.config",
                "AlternatingPipeline.models.checkpoint_compat",
            ],
            purge=False,
        )

    message = str(excinfo.value)
    assert "AlternatingPipeline.models.checkpoint_compat" in message
    assert "AlternatingPipeline.config" not in message.split("missing:")[1].split("\n")[0]
    # Must point at the stale copy and the fix, not just the symptom.
    assert str(tmp_path) in message
    assert "04_train_models.py" in message
    assert "do NOT run the training cell" in message


def test_missing_directory_is_reported_not_crashed(seqparams_config, tmp_path):
    absent = str(tmp_path / "never_copied")

    with pytest.raises(RuntimeError, match="Stale source copy"):
        seqparams_config.assert_pipeline_source_fresh(
            absent, required_modules=["AlternatingPipeline.config"], purge=False
        )


def test_purge_evicts_stale_namespace_packages(seqparams_config, tmp_path):
    """Top-level namespace packages have __file__ is None; purge by name."""
    required = ["AlternatingPipeline.config"]
    _make_source_tree(str(tmp_path), required)

    stale_pkg = types.ModuleType("AlternatingPipeline")
    stale_pkg.__file__ = None  # namespace package, as in Databricks
    sys.modules["AlternatingPipeline"] = stale_pkg
    sys.modules["AlternatingPipeline.models"] = types.ModuleType("AlternatingPipeline.models")
    sys.modules["csv_pipeline_seqparams.config"] = types.ModuleType("csv_pipeline_seqparams.config")
    sys.modules["unrelated_module"] = types.ModuleType("unrelated_module")

    try:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path), required_modules=required, purge=True
        )

        assert "AlternatingPipeline" not in sys.modules
        assert "AlternatingPipeline.models" not in sys.modules
        assert "csv_pipeline_seqparams.config" not in sys.modules
        assert "unrelated_module" in sys.modules, "purge must not touch other modules"
    finally:
        sys.modules.pop("unrelated_module", None)


def test_purge_evicts_legacy_top_level_modules_by_file_path(seqparams_config, tmp_path):
    """The 2026-07-27 regression: name-prefix matching alone is not enough.

    AlternatingPipeline/models/examination_model.py does
    `from models.sequence_generator import ...` — a legacy TOP-LEVEL name that
    no prefix in ("AlternatingPipeline", "csv_pipeline_seqparams") matches. It
    survived a re-run in a persistent kernel, so a fresh sha in the pre-flight
    sat alongside a stale model class, detectable only via the parameter count.
    """
    required = ["AlternatingPipeline.config"]
    _make_source_tree(str(tmp_path), required)

    legacy = types.ModuleType("models.sequence_generator")
    legacy.__file__ = os.path.join(str(tmp_path), "AlternatingPipeline", "models",
                                   "sequence_generator.py")
    sys.modules["models.sequence_generator"] = legacy
    sys.modules["config"] = types.ModuleType("config")
    sys.modules["config"].__file__ = os.path.join(str(tmp_path), "AlternatingPipeline", "config.py")

    elsewhere = types.ModuleType("elsewhere")
    elsewhere.__file__ = "/usr/lib/python3/elsewhere.py"
    sys.modules["elsewhere"] = elsewhere

    try:
        seqparams_config.assert_pipeline_source_fresh(
            str(tmp_path), required_modules=required, purge=True
        )

        assert "models.sequence_generator" not in sys.modules
        assert "config" not in sys.modules
        assert "elsewhere" in sys.modules, "purge must not evict modules outside tmp_root"
    finally:
        for name in ("models.sequence_generator", "config", "elsewhere"):
            sys.modules.pop(name, None)


def test_no_required_modules_is_a_purge_only_noop(seqparams_config, tmp_path):
    assert seqparams_config.assert_pipeline_source_fresh(str(tmp_path)) is True
