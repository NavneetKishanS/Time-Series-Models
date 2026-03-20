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
%pip install tqdm

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
