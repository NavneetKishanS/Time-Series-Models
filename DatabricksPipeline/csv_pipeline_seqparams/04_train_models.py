# Databricks notebook source
# Databricks notebook — Train Examination model (feature-rich SUT pipeline)
#
# Forked from DatabricksPipeline/csv_pipeline/04_train_models.py, TRIMMED to
# examination-only training (decision: exchange/orchestration are out of
# scope for this pipeline). Reuses AlternatingPipeline/training/train_examination.py
# UNCHANGED — it already accepts a config override and an
# extra_conditioning_features override, so no new trainer script is needed,
# only a different config + data path + save dir.
#
# Prerequisites:
#   - Run csv_pipeline_seqparams/03_build_preprocessed_pkl.py first
#   - Ideally, DatabricksPipeline/sut_parameter_discovery.py has already
#     confirmed real slot numbers in csv_pipeline_seqparams/config.py (this
#     will still run with the empty placeholder — it just won't have any
#     new numeric features to learn from, equivalent to the old model plus
#     a (currently untrained-on) trigger_mode embedding)
#   - Repo must be attached via Databricks Repos (Git integration)
#   - Use a GPU cluster (e.g. ML Runtime with GPU)
#
# Output:
#   /dbfs/FileStore/csv_pipeline_seqparams/models/examination/examination_model_best.pt

# COMMAND ----------

# MAGIC %pip install tqdm

# COMMAND ----------

import sys, os, pickle, base64, requests

sys.dont_write_bytecode = True

# ── CONFIGURE THIS PATH to your Databricks Repos clone ─────────────────────
REPO_ROOT = "/Workspace/Shared/Patient Exchange and Examination/Time-Series-Models"
# ───────────────────────────────────────────────────────────────────────────

TMP_ROOT = "/tmp/alternating_pipeline_src"

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_host = "https://" + _ctx.browserHostName().get()
_token = _ctx.apiToken().get()
_headers = {"Authorization": f"Bearer {_token}"}

_SKIP = {"outputs", "__pycache__"}

def _export_source(workspace_path, dst):
    r = requests.get(f"{_host}/api/2.0/workspace/export",
                     headers=_headers,
                     params={"path": workspace_path, "format": "SOURCE"})
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(base64.b64decode(r.json()["content"]))

def _api_copy_py(workspace_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    resp = requests.get(f"{_host}/api/2.0/workspace/list",
                        headers=_headers, params={"path": workspace_dir})
    if resp.status_code == 404:
        return
    resp.raise_for_status()
    for obj in resp.json().get("objects", []):
        name = os.path.basename(obj["path"])
        if name in _SKIP:
            continue
        if obj["object_type"] == "DIRECTORY":
            _api_copy_py(obj["path"], os.path.join(local_dir, name))
        elif obj["object_type"] == "FILE" and name.endswith(".py"):
            _export_source(obj["path"], os.path.join(local_dir, name))
        elif obj["object_type"] == "NOTEBOOK":
            # Databricks auto-detects the "# Databricks notebook source"
            # header and lists these as NOTEBOOK (not FILE) even for plain
            # git-tracked .py files inside a Repo — with the .py extension
            # STRIPPED from `path` (e.g. ".../config", not ".../config.py").
            # Every file in csv_pipeline_seqparams/ carries that header, so
            # without this branch _api_copy_py silently copies nothing at
            # all here (confirmed: local dir came back empty). `format=
            # SOURCE` still returns the original .py text either way, so
            # just re-add the extension when writing the local copy.
            dst_name = name if name.endswith(".py") else f"{name}.py"
            _export_source(obj["path"], os.path.join(local_dir, dst_name))

_api_copy_py(f"{REPO_ROOT}/AlternatingPipeline", f"{TMP_ROOT}/AlternatingPipeline")
_api_copy_py(f"{REPO_ROOT}/DatabricksPipeline/csv_pipeline_seqparams",
             f"{TMP_ROOT}/csv_pipeline_seqparams")
print(f"Copied AlternatingPipeline + csv_pipeline_seqparams to {TMP_ROOT}")
sys.path.insert(0, TMP_ROOT)

# Purge any previously-imported copies — same long-lived-kernel caching gotcha
# as csv_pipeline/04_train_models.py (see that file's comment for the full
# story: a re-run in a persistent Databricks kernel silently re-executes
# whatever code the kernel first imported unless stale modules are purged).
#
# Purge BY NAME PREFIX, not by `__file__`: AlternatingPipeline and
# csv_pipeline_seqparams both lack __init__.py, so Python treats them as
# namespace packages, and a namespace package's module object has
# `__file__ is None` (submodules like csv_pipeline_seqparams.config still
# have a real __file__, but the top-level package entry itself doesn't) — a
# `__file__`-based check silently never evicts that top-level entry, so a
# stale namespace package from an earlier run in this kernel (with an
# incomplete/outdated __path__) can survive this purge and shadow the fresh
# copy made above.
# BOTH conditions are required — each catches what the other misses:
#
#   by NAME PREFIX: namespace packages (no __init__.py) have __file__ is None
#     on the top-level module object, so a __file__ check never evicts them.
#
#   by __file__ under TMP_ROOT: AlternatingPipeline/models/examination_model.py
#     does `from models.sequence_generator import ...` — a LEGACY TOP-LEVEL
#     name, not AlternatingPipeline.models.* — so the prefix list never matches
#     it. This is how a stale model class survived a re-run on 2026-07-27: the
#     pre-flight showed the new sha (it reads files directly) while the
#     executed class was the cached one, betrayed only by the parameter count.
#     Matching on __file__ is exactly what csv_pipeline/04_train_models.py does.
_STALE_PREFIXES = ("AlternatingPipeline", "csv_pipeline_seqparams")
for _name, _mod in list(sys.modules.items()):
    _mod_file = (getattr(_mod, "__file__", None) or "").replace("\\", "/")
    if (any(_name == p or _name.startswith(p + ".") for p in _STALE_PREFIXES)
            or _mod_file.startswith(TMP_ROOT)):
        del sys.modules[_name]

# The files under TMP_ROOT were just overwritten in place; overwriting does not
# change the DIRECTORY mtime, so FileFinder can keep serving a cached listing.
import importlib as _importlib
_importlib.invalidate_caches()

# Verify the purge actually covered the legacy top-level names, rather than
# trusting that it did. This assertion is what would have caught the 07-27
# stale-class run at the copy cell instead of ~5 hours later.
_MUST_BE_ABSENT = (
    "AlternatingPipeline.models.sequence_generator",
    "AlternatingPipeline.models.examination_model",
    "AlternatingPipeline.config",
    "models.sequence_generator",
    "models.examination_model",
    "config",
)
_leftover = [_n for _n in _MUST_BE_ABSENT if _n in sys.modules]
if _leftover:
    raise RuntimeError(
        f"Module purge missed {_leftover} — these would shadow the freshly "
        f"copied source and train the PREVIOUS architecture while the "
        f"pre-flight reports the new sha. Restart Python "
        f"(dbutils.library.restartPython()) and re-run from the top."
    )
print(f"[purge] sys.modules clean — {len(_MUST_BE_ABSENT)} pipeline module names evicted")

import csv_pipeline_seqparams.config as seqparams_config
from AlternatingPipeline.config import EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG

PKL_PATH = seqparams_config.PKL_OUTPUT
MODELS_DIR = seqparams_config.MODELS_DIR
os.makedirs(f"{MODELS_DIR}/examination", exist_ok=True)

print(f"Repo root:  {REPO_ROOT}")
print(f"PKL path:   {PKL_PATH}")
print(f"Models dir: {MODELS_DIR}")

# COMMAND ----------

# =============================================================================
# Assemble EXAMINATION_MODEL_CONFIG_SEQPARAMS
# -----------------------------------------------------------------------------
# Built via csv_pipeline_seqparams.config.build_seqparams_model_config() —
# the single source of truth also used by 05_feature_analysis.py and
# 06_compare_models.py, so the three scripts can't silently diverge. Called
# HERE (not inside config.py itself) because this cross-package import
# (AlternatingPipeline.config) is only safely resolvable after the
# sys.path/Workspace-copy dance above has run — csv_pipeline/'s own
# config.py duplicates constants rather than cross-importing for exactly
# this reason. Zero edits to AlternatingPipeline/config.py itself.
# =============================================================================

EXAMINATION_MODEL_CONFIG_SEQPARAMS = seqparams_config.build_seqparams_model_config(
    EXAMINATION_MODEL_CONFIG
)
EXAMINATION_TRAINING_CONFIG_SEQPARAMS = dict(EXAMINATION_TRAINING_CONFIG)

print("EXAMINATION_MODEL_CONFIG_SEQPARAMS:")
print(f"  use_sut_conditioning: {EXAMINATION_MODEL_CONFIG_SEQPARAMS['use_sut_conditioning']}")
print(f"  num_trigger_modes:    {EXAMINATION_MODEL_CONFIG_SEQPARAMS['num_trigger_modes']}")
print(f"  base_conditioning_dim: {EXAMINATION_MODEL_CONFIG['base_conditioning_dim']} -> "
      f"{EXAMINATION_MODEL_CONFIG_SEQPARAMS['base_conditioning_dim']}")
print(f"  extra numeric features: {seqparams_config.EXAMINATION_SEQPARAM_FEATURES or '(none yet — placeholder)'}")

# COMMAND ----------

# =============================================================================
# Verify GPU (identical to csv_pipeline/04)
# =============================================================================

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n" + "!" * 70)
    print("!! WARNING: NO GPU DETECTED — training will run on CPU.")
    print("!" * 70 + "\n")

if device.type == 'cpu':
    _ncores = os.cpu_count() or 4
    torch.set_num_threads(_ncores)
    try:
        torch.set_num_interop_threads(min(4, _ncores))
    except RuntimeError:
        pass
    os.environ.setdefault('OMP_NUM_THREADS', str(_ncores))
    os.environ.setdefault('MKL_NUM_THREADS', str(_ncores))
    EXAMINATION_TRAINING_CONFIG_SEQPARAMS['early_stopping_patience'] = 8
    EXAMINATION_TRAINING_CONFIG_SEQPARAMS['epochs'] = 60
    print(f"[CPU] {_ncores} intra-op threads; early_stopping_patience=8, epochs<=60")

# COMMAND ----------

# =============================================================================
# PRE-FLIGHT PROVENANCE CHECK — same pattern as csv_pipeline/04, extended to
# fingerprint the seqparams config too (its sha changing across runs is
# EXPECTED as SUT_FIELD_MAP / EXAMINATION_SEQPARAM_FEATURES get filled in).
# =============================================================================

import time, json, hashlib

PROVENANCE_SRC = [
    "AlternatingPipeline/training/train_examination.py",
    "AlternatingPipeline/data/examination_duration_calibration.py",
    "AlternatingPipeline/models/sequence_generator.py",
    "AlternatingPipeline/config.py",
    "csv_pipeline_seqparams/config.py",
]
CHECKPOINT_PATH = f"{MODELS_DIR}/examination/examination_model_best.pt"

def _file_meta(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": path, "exists": True,
        "size_mb": round(st.st_size / 1e6, 2),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "mtime_epoch": st.st_mtime,
        "sha256": h.hexdigest()[:16],
    }

with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
print(f"Examination sequences: {len(data['examination']):,}")

print("=" * 64)
print(" STEP 04-SEQPARAMS PRE-FLIGHT")
print("=" * 64)
_pkl_meta = _file_meta(PKL_PATH)
print(f"\nPKL  {_pkl_meta['path']}  mtime {_pkl_meta.get('mtime')}  "
      f"{_pkl_meta.get('size_mb')} MB  sha {_pkl_meta.get('sha256')}")

print("\nCODE (loaded source under TMP_ROOT):")
for _rel in PROVENANCE_SRC:
    _m = _file_meta(os.path.join(TMP_ROOT, _rel))
    print(f"   {_rel:<58} " + (f"sha {_m['sha256']}  {_m['mtime']}" if _m["exists"] else "!! MISSING"))

_ckpt_meta = _file_meta(CHECKPOINT_PATH)
if _ckpt_meta["exists"]:
    _age_h = (time.time() - _ckpt_meta["mtime_epoch"]) / 3600
    print(f"\nEXISTING checkpoint (this run OVERWRITES it): "
          f"{_ckpt_meta['mtime']}  ({_age_h:.1f}h old)  {_ckpt_meta['size_mb']} MB")
else:
    print("\nEXISTING checkpoint: none yet — fresh train")
print("=" * 64)

# COMMAND ----------

# =============================================================================
# TRAIN EXAMINATION MODEL
# =============================================================================

from AlternatingPipeline.training.train_examination import train_examination_model

print("\n" + "="*60)
print("Training Examination Model (SUT-enriched conditioning)")
print("="*60)

examination_model, examination_history = train_examination_model(
    data_path=PKL_PATH,
    config=EXAMINATION_MODEL_CONFIG_SEQPARAMS,
    training_config=EXAMINATION_TRAINING_CONFIG_SEQPARAMS,
    save_dir=f"{MODELS_DIR}/examination",
    extra_conditioning_features=seqparams_config.EXAMINATION_SEQPARAM_FEATURES,
    # Leakage gate #3 (training-tensor-build-time) — enforced inside
    # build_conditioning_tensor() at the actual point of tensor construction,
    # independent of gate #1 (config import, csv_pipeline_seqparams/config.py)
    # and gate #2 (preprocessing-write-time, step 03). This catches a bad
    # feature list regardless of how extra_conditioning_features was
    # assembled, not just the one path gate #1 already validated.
    leakage_denylist=seqparams_config.SUT_LEAKAGE_DENYLIST,
    verbose=True,
)

print(f"\nBest val loss:       {min(examination_history['val_loss']):.4f}")
print(f"Best val perplexity: {min(examination_history['val_perplexity']):.2f}")

# COMMAND ----------

# =============================================================================
# WRITE MODEL MANIFEST
# =============================================================================

_manifest = {
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "repo_root": REPO_ROOT,
    "pkl": _file_meta(PKL_PATH),
    "code": {rel: _file_meta(os.path.join(TMP_ROOT, rel)).get("sha256") for rel in PROVENANCE_SRC},
    "checkpoint": _file_meta(CHECKPOINT_PATH),
    "val_loss": float(min(examination_history["val_loss"])),
    "use_sut_conditioning": True,
    "num_trigger_modes": EXAMINATION_MODEL_CONFIG_SEQPARAMS['num_trigger_modes'],
    "base_conditioning_dim": EXAMINATION_MODEL_CONFIG_SEQPARAMS['base_conditioning_dim'],
    "extra_conditioning_features": seqparams_config.EXAMINATION_SEQPARAM_FEATURES,
}
_manifest_path = f"{MODELS_DIR}/MODEL_MANIFEST.json"
with open(_manifest_path, "w") as _f:
    json.dump(_manifest, _f, indent=2)
print(f"Wrote model manifest → {_manifest_path}")
print(json.dumps(_manifest, indent=2))

# COMMAND ----------

# =============================================================================
# NEXT STEP: run 05_feature_analysis.py (permutation importance section) or
# 06_compare_models.py (old vs. new comparison).
# =============================================================================
