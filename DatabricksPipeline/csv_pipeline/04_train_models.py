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

# Purge any previously-imported copies so a RE-RUN in a long-lived Databricks
# kernel actually picks up the freshly copied source. Databricks kernels persist
# across runs: `import` returns the module cached in sys.modules and ignores the
# file edits _api_copy_py just made — so without this, every re-run silently
# re-executes whatever code the kernel first imported (stale guard/calibration),
# even though the pre-flight (which reads files directly) reports the NEW shas.
# Match by __file__ under TMP_ROOT to catch every loaded copy regardless of the
# import name (AlternatingPipeline.*, but also top-level config/data/models).
for _name, _mod in list(sys.modules.items()):
    _f = getattr(_mod, "__file__", None) or ""
    if _f.startswith(TMP_ROOT):
        del sys.modules[_name]

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
# PERSISTENT RUN LOG  — tee everything to DBFS so a ~24h run is recoverable
# -----------------------------------------------------------------------------
# Databricks drops cell output for very long runs; on refresh the streamed
# training log (pre-flight shas, per-epoch losses, the duration-spread guard,
# and the post-train probe table) is gone. Tee stdout+stderr to a file on
# DBFS from here on. The file is written line-buffered and flushed on every
# write, so even a killed/timed-out run leaves a readable partial log.
# tqdm progress-bar redraws (\r without \n) are skipped so the file stays
# small and readable; the final 100% line (ends in \n) is kept.
# =============================================================================
import sys, time
_LOG_DIR  = "/dbfs/FileStore/csv_pipeline/logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_NAME = f"train_04_{time.strftime('%Y%m%d_%H%M%S')}.log"
_LOG_PATH = f"{_LOG_DIR}/{_LOG_NAME}"

class _Tee:
    def __init__(self, path, stream):
        self._f = open(path, "a", buffering=1)
        self._stream = stream
    def write(self, s):
        # Write to the notebook stream first, but NEVER block on it.  After
        # "Custom TB Handler failed, unregistering" Databricks stops draining
        # its output pipe; the next stream.write() fills the pipe buffer and
        # hangs the Python thread permanently — the log file write on the next
        # line is then never reached and training appears completely frozen for
        # hours.  Catching all exceptions (BrokenPipeError, OSError, …) keeps
        # the log file alive even when the notebook UI is dead.
        try:
            self._stream.write(s)
        except Exception:
            pass
        # Always write non-tqdm lines to the DBFS log file.  tqdm redraw lines
        # contain \r but no \n; everything else (epoch summaries, print() calls)
        # is preserved.
        if s and not ("\r" in s and "\n" not in s):
            try:
                self._f.write(s)
                self._f.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._f.flush()
        except Exception:
            pass
    def __getattr__(self, name):
        return getattr(self._stream, name)

if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(_LOG_PATH, sys.__stdout__)
    sys.stderr = _Tee(_LOG_PATH, sys.__stderr__)

try:
    _WS = spark.conf.get("spark.databricks.workspaceUrl")
    _URL = f"https://{_WS}/files/csv_pipeline/logs/{_LOG_NAME}"
    displayHTML(f'<p><b>Run log (survives refresh):</b> '
                f'<a href="{_URL}" target="_blank">{_LOG_NAME}</a></p>')
except Exception:
    _URL = None
print(f"[run-log] teeing all output to {_LOG_PATH}")

# COMMAND ----------

# =============================================================================
# Verify GPU
# =============================================================================

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    # CPU warning. A GPU cluster is still faster, but with the nested-tensor
    # fix (enable_nested_tensor=False) and torch.compile the CPU run is now
    # viable in hours rather than days.
    print("\n" + "!" * 70)
    print("!! WARNING: NO GPU DETECTED — training will run on CPU.")
    print("!! A GPU cluster (Databricks ML Runtime + GPU) is still faster,")
    print("!! but with the nested-tensor fix a full CPU run takes ~4-8h.")
    print("!! Re-attach a GPU cluster if speed is critical.")
    print("!" * 70 + "\n")
    try:
        displayHTML(
            "<div style='padding:12px;border:2px solid #c00;background:#fee;"
            "color:#900;font-weight:bold'>NO GPU DETECTED — training on CPU. "
            "With nested-tensor fix: ~4-8h total. GPU cluster is faster.</div>"
        )
    except Exception:
        pass

# ── CPU-specific performance setup ───────────────────────────────────────────
if device.type == 'cpu':
    _ncores = os.cpu_count() or 4
    torch.set_num_threads(_ncores)
    torch.set_num_interop_threads(min(4, _ncores))
    os.environ.setdefault('OMP_NUM_THREADS', str(_ncores))
    os.environ.setdefault('MKL_NUM_THREADS', str(_ncores))
    print(f"[CPU] {_ncores} intra-op threads, {min(4, _ncores)} interop threads")
    # Import `config` the same way the training scripts do (they add
    # TMP_ROOT/AlternatingPipeline to sys.path and import bare `config`).
    # Mutating the dict in-place here means any training module that binds
    # EXCHANGE_TRAINING_CONFIG from this same `config` object sees the new values.
    _ap_path = os.path.join(TMP_ROOT, 'AlternatingPipeline')
    if _ap_path not in sys.path:
        sys.path.insert(0, _ap_path)
    import config as _ap_cfg
    # GPU-calibrated patience of 15-20 adds many hours on CPU.  Tighten to
    # ~half without meaningfully sacrificing model quality — the small datasets
    # (3.5K exchange, 45K exam) converge well before the original limits.
    _ap_cfg.EXCHANGE_TRAINING_CONFIG['early_stopping_patience'] = 8
    _ap_cfg.EXCHANGE_TRAINING_CONFIG['epochs'] = 60
    _ap_cfg.EXAMINATION_TRAINING_CONFIG['early_stopping_patience'] = 8
    _ap_cfg.EXAMINATION_TRAINING_CONFIG['epochs'] = 60
    _ap_cfg.ORCHESTRATION_TRAINING_CONFIG['early_stopping_patience'] = 10
    _ap_cfg.ORCHESTRATION_TRAINING_CONFIG['epochs'] = 60
    print("[CPU] early_stopping_patience: exchange/exam → 8, orch → 10")
    print("[CPU] max epochs capped at 60 (early stopping typically fires sooner)")
# ─────────────────────────────────────────────────────────────────────────────

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

# =============================================================================
# RECOVERABLE ARTIFACTS  — clickable links + inline echo of the key results
# -----------------------------------------------------------------------------
# Everything below is read back FROM DBFS, so it is exactly what survives a
# notebook refresh. Click the links to download, or just read the echoed
# probe table / manifest below. Share the duration_probe.json + the run log
# if the printed output was lost.
# =============================================================================
_PROBE_PATH = f"{MODELS_DIR}/examination/duration_probe.json"
_artifacts = {
    "run log":         (_LOG_PATH,      f"csv_pipeline/logs/{_LOG_NAME}"),
    "model manifest":  (_manifest_path, "csv_pipeline/models/MODEL_MANIFEST.json"),
    "duration probe":  (_PROBE_PATH,    "csv_pipeline/models/examination/duration_probe.json"),
}
try:
    _WS = spark.conf.get("spark.databricks.workspaceUrl")
    _links = "".join(
        f'<li><b>{label}:</b> <a href="https://{_WS}/files/{rel}" target="_blank">{os.path.basename(path)}</a></li>'
        for label, (path, rel) in _artifacts.items() if os.path.exists(path)
    )
    displayHTML(f"<h3>Recoverable artifacts (survive refresh)</h3><ul>{_links}</ul>")
except Exception as _e:
    print(f"(could not render artifact links: {_e})")

# Echo the decisive duration probe inline so it is in this cell's output too
if os.path.exists(_PROBE_PATH):
    with open(_PROBE_PATH) as _pf:
        _probe = json.load(_pf)
    print("\n=== DURATION PROBE (the go/no-go gate) ===")
    for _r in _probe.get("rows", []):
        print(f"  {_r['sequence_type']:<8} n={_r['n']:>3}  "
              f"predicted={_r['predicted_s']:>7.1f}s  target={_r['target_s']:>7.1f}s")
    if "spread_x" in _probe:
        print(f"  spread {_probe['spread_x']}x  "
              f"[{_probe['predicted_lo_s']:.0f}s .. {_probe['predicted_hi_s']:.0f}s]  "
              f"flat_warning={_probe.get('flat_warning')}")
    print(f"\nPaste {os.path.basename(_PROBE_PATH)} (or the run log) into the chat if the streamed output was lost.")

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
