# Databricks notebook source
# Databricks notebook — Train Exchange, Examination, and Orchestration models
#
# Loads preprocessed_data.pkl from DBFS, trains the three AlternatingPipeline
# models on a GPU cluster, and saves checkpoints back to DBFS.
#
# Prerequisites:
#   - Run 03_build_preprocessed_pkl.py first
#   - Repo must be attached via Databricks Repos (Git integration)
#   - Use a GPU cluster (e.g. ML Runtime with GPU)
#
# Output:
#   /dbfs/FileStore/csv_pipeline/models/exchange/exchange_model_best.pt
#   /dbfs/FileStore/csv_pipeline/models/examination/examination_model_best.pt
#   /dbfs/FileStore/csv_pipeline/models/orchestration/orchestration_model_best.pt

# COMMAND ----------

# MAGIC %pip install tqdm

# COMMAND ----------

import sys, os, pickle, base64, requests

sys.dont_write_bytecode = True  # suppress __pycache__ writes

# ── CONFIGURE THIS PATH to your Databricks Repos clone ─────────────────────
REPO_ROOT = "/Workspace/Shared/Patient Exchange and Examination/Time-Series-Models"
# ───────────────────────────────────────────────────────────────────────────

# Copy .py files via the Databricks Workspace REST API — bypasses FUSE EIO on /Workspace/Shared/
TMP_ROOT = "/tmp/alternating_pipeline_src"

_ctx     = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_host    = "https://" + _ctx.browserHostName().get()
_token   = _ctx.apiToken().get()
_headers = {"Authorization": f"Bearer {_token}"}

_SKIP = {"outputs", "__pycache__"}

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
        dst = os.path.join(local_dir, name)
        if obj["object_type"] == "DIRECTORY":
            _api_copy_py(obj["path"], dst)
        elif obj["object_type"] == "FILE" and name.endswith(".py"):
            r = requests.get(f"{_host}/api/2.0/workspace/export",
                             headers=_headers,
                             params={"path": obj["path"], "format": "SOURCE"})
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(base64.b64decode(r.json()["content"]))

_api_copy_py(f"{REPO_ROOT}/AlternatingPipeline", f"{TMP_ROOT}/AlternatingPipeline")
print(f"Copied AlternatingPipeline to {TMP_ROOT}")
sys.path.insert(0, TMP_ROOT)

PKL_PATH   = "/dbfs/FileStore/csv_pipeline/preprocessed_data.pkl"
MODELS_DIR = "/dbfs/FileStore/csv_pipeline/models"

os.makedirs(f"{MODELS_DIR}/exchange",     exist_ok=True)
os.makedirs(f"{MODELS_DIR}/examination",  exist_ok=True)
os.makedirs(f"{MODELS_DIR}/orchestration", exist_ok=True)

print(f"Repo root:  {REPO_ROOT}")
print(f"PKL path:   {PKL_PATH}")
print(f"Models dir: {MODELS_DIR}")

# COMMAND ----------

# =============================================================================
# Verify GPU
# =============================================================================

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# =============================================================================
# Load preprocessed data
# =============================================================================

with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

print(f"Exchange sequences:    {len(data['exchange']):,}")
print(f"Examination sequences: {len(data['examination']):,}")
print(f"Customer schedules:    {len(data['customer_schedules'])}")
print(f"Daily summaries:       {len(data['daily_summaries']):,}")

# COMMAND ----------

# =============================================================================
# PRE-FLIGHT PROVENANCE CHECK  (run BEFORE training)
# -----------------------------------------------------------------------------
# Confirms WHICH pkl and WHICH code are about to train the models, and how old
# the existing checkpoints are (this run OVERWRITES them). Inspect this up front
# so an accidental stale pkl / stale repo clone is caught here — instead of
# after a full generate + eval cycle. Step 04 also writes a MODEL_MANIFEST.json
# at the end, which step 05 verifies before it generates anything.
# =============================================================================
import os, time, json, hashlib
from collections import Counter

# Source files whose content actually determines model behaviour. We fingerprint
# the *loaded* copies (under TMP_ROOT) so the printout reflects the code this run
# is really using, not whatever happens to be in the repo.
PROVENANCE_SRC = [
    "AlternatingPipeline/training/train_examination.py",
    "AlternatingPipeline/training/train_exchange.py",
    "AlternatingPipeline/training/train_orchestration.py",
    "AlternatingPipeline/data/examination_duration_calibration.py",
    "AlternatingPipeline/models/sequence_generator.py",
    "AlternatingPipeline/config.py",
]
CHECKPOINTS = {
    "exchange":      f"{MODELS_DIR}/exchange/exchange_model_best.pt",
    "examination":   f"{MODELS_DIR}/examination/examination_model_best.pt",
    "orchestration": f"{MODELS_DIR}/orchestration/orchestration_model_best.pt",
}

def _file_meta(path):
    """Lightweight provenance fingerprint for a single file."""
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

def _pkl_fingerprint(d):
    return {
        "examination_seqs": len(d.get("examination", [])),
        "exchange_seqs": len(d.get("exchange", [])),
        "distinct_sequence_types": len({int(s.get("sequence_type", 0)) for s in d.get("examination", [])}),
        "distinct_body_regions": len({int(s.get("body_region", 10)) for s in d.get("examination", [])}),
    }

print("=" * 64)
print(" STEP 04 PRE-FLIGHT — TRAINING INPUTS")
print("=" * 64)

# 1) the pkl that will train the models
_pkl_meta = _file_meta(PKL_PATH)
if not _pkl_meta["exists"]:
    raise FileNotFoundError(f"PKL missing: {PKL_PATH} — run 03_build_preprocessed_pkl.py first")
_fp = _pkl_fingerprint(data)
print(f"\nPKL  {_pkl_meta['path']}")
print(f"     mtime {_pkl_meta['mtime']}   {_pkl_meta['size_mb']} MB   sha {_pkl_meta['sha256']}")
print(f"     examination_seqs={_fp['examination_seqs']:,}  exchange_seqs={_fp['exchange_seqs']:,}  "
      f"sequence_types={_fp['distinct_sequence_types']}  body_regions={_fp['distinct_body_regions']}")
if _fp["distinct_sequence_types"] < 2:
    print("  !! WARNING: pkl carries <2 sequence types — scan-type conditioning will be dead. "
          "Re-run step 03 with current code.")

# 2) the code actually loaded into this run
print("\nCODE (loaded source under TMP_ROOT):")
for _rel in PROVENANCE_SRC:
    _m = _file_meta(os.path.join(TMP_ROOT, _rel))
    print(f"   {_rel:<58} " + (f"sha {_m['sha256']}  {_m['mtime']}" if _m["exists"] else "!! MISSING"))

# 3) existing checkpoints this run will OVERWRITE
print("\nEXISTING checkpoints (this run OVERWRITES them):")
for _name, _path in CHECKPOINTS.items():
    _m = _file_meta(_path)
    if _m["exists"]:
        _age_h = (time.time() - _m["mtime_epoch"]) / 3600
        print(f"   {_name:<13} {_m['mtime']}  ({_age_h:5.1f}h old)  {_m['size_mb']:6.1f} MB  sha {_m['sha256']}")
    else:
        print(f"   {_name:<13} (none yet — fresh train)")
print("=" * 64)

# COMMAND ----------

# =============================================================================
# TRAIN EXCHANGE MODEL
# =============================================================================

from AlternatingPipeline.training.train_exchange import train_exchange_model

print("\n" + "="*60)
print("Training Exchange Model")
print("="*60)

exchange_model, exchange_history = train_exchange_model(
    data_path=PKL_PATH,
    save_dir=f"{MODELS_DIR}/exchange",
    verbose=True,
)

print(f"\nBest val loss:       {min(exchange_history['val_loss']):.4f}")
print(f"Best val perplexity: {min(exchange_history['val_perplexity']):.2f}")

# COMMAND ----------

# =============================================================================
# TRAIN EXAMINATION MODEL
# =============================================================================

from AlternatingPipeline.training.train_examination import train_examination_model

print("\n" + "="*60)
print("Training Examination Model")
print("="*60)

examination_model, examination_history = train_examination_model(
    data_path=PKL_PATH,
    save_dir=f"{MODELS_DIR}/examination",
    verbose=True,
)

print(f"\nBest val loss:       {min(examination_history['val_loss']):.4f}")
print(f"Best val perplexity: {min(examination_history['val_perplexity']):.2f}")

# COMMAND ----------

# =============================================================================
# TRAIN ORCHESTRATION MODEL
# =============================================================================

from AlternatingPipeline.training.train_orchestration import train_orchestration_model
from AlternatingPipeline.data.orchestration_preprocessing import extract_orchestration_samples

print("\n" + "="*60)
print("Training Orchestration Model")
print("="*60)

orch_samples, scanner_to_idx = extract_orchestration_samples(data)
print(f"Orchestration samples: {len(orch_samples):,}")
print(f"Scanners: {scanner_to_idx}")

# Save scanner mapping — needed at inference time
import json
with open(f"{MODELS_DIR}/orchestration/scanner_to_idx.json", 'w') as f:
    json.dump(scanner_to_idx, f, indent=2)

# Temporarily add orch_samples to data dict so train function can access it
data['orch_samples']    = orch_samples
data['scanner_to_idx']  = scanner_to_idx

orchestration_model, orch_history = train_orchestration_model(
    data_path=PKL_PATH,
    save_dir=f"{MODELS_DIR}/orchestration",
    verbose=True,
)

print(f"\nBest val loss: {min(orch_history['val_loss']):.4f}")

# COMMAND ----------

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("ALL MODELS TRAINED")
print("="*60)
print(f"\nModel files saved to: {MODELS_DIR}")
print(f"  exchange/exchange_model_best.pt")
print(f"  examination/examination_model_best.pt")
print(f"  orchestration/orchestration_model_best.pt")
print(f"  orchestration/scanner_to_idx.json")

displayHTML(f'''
<h3>Training complete</h3>
<ul>
  <li>Exchange best val loss: {min(exchange_history["val_loss"]):.4f}</li>
  <li>Examination best val loss: {min(examination_history["val_loss"]):.4f}</li>
  <li>Orchestration best val loss: {min(orch_history["val_loss"]):.4f}</li>
</ul>
<p>Next: run <b>05_generate_synthetic_data.py</b></p>
''')

# COMMAND ----------

# =============================================================================
# WRITE MODEL MANIFEST  (run AFTER training)
# -----------------------------------------------------------------------------
# Stamps WHICH pkl, WHICH code, and WHICH checkpoint hashes this run produced.
# Step 05 reads this manifest before generating and refuses to silently use a
# model that no longer matches the pkl/code it was trained from.
# =============================================================================
_manifest = {
    "trained_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
    "trained_at_epoch":  time.time(),
    "repo_root":         REPO_ROOT,
    "pkl":               _file_meta(PKL_PATH),
    "pkl_fingerprint":   _pkl_fingerprint(data),
    "code":              {rel: _file_meta(os.path.join(TMP_ROOT, rel)).get("sha256")
                          for rel in PROVENANCE_SRC},
    "checkpoints":       {name: _file_meta(path) for name, path in CHECKPOINTS.items()},
    "val_loss": {
        "exchange":      float(min(exchange_history["val_loss"])),
        "examination":   float(min(examination_history["val_loss"])),
        "orchestration": float(min(orch_history["val_loss"])),
    },
}
_manifest_path = f"{MODELS_DIR}/MODEL_MANIFEST.json"
with open(_manifest_path, "w") as _f:
    json.dump(_manifest, _f, indent=2)
print(f"Wrote model manifest → {_manifest_path}")
print(json.dumps(_manifest, indent=2))

# COMMAND ----------

  import os, time                                                                                                                      
                                                                                                                                                                           
  ckpt_dir = "/dbfs/FileStore/csv_pipeline/models/exchange"                                                                                                                
  print("exists:", os.path.isdir(ckpt_dir))                                                                                                                                
  for f in sorted(os.listdir(ckpt_dir)):                                                                                                                                   
      p = os.path.join(ckpt_dir, f)                                                                                                                                        
      sz = os.path.getsize(p) / 1e6                                                                                                                                        
      mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
      print(f"  {mt}  {sz:8.1f} MB  {f}") 

# COMMAND ----------

  import os, time                                                                                                                                                          
  d = "/dbfs/FileStore/csv_pipeline/models/examination"
  print("exists:", os.path.isdir(d))                                                                                                                                       
  if os.path.isdir(d):
      for f in sorted(os.listdir(d)):                                                                                                                                      
          p = os.path.join(d, f)
          sz = os.path.getsize(p) / 1e6                                                                                                                                    
          mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
          print(f"  {mt}  {sz:8.1f} MB  {f}")  
