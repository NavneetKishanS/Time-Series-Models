# Databricks notebook source
# Databricks notebook — Old vs. new examination model comparison
#
# Key enabler: ExaminationDataset / build_conditioning_tensor only ever read
# named keys via .get(), so the OLD model can run unmodified directly
# against the NEW (superset) pkl — it simply ignores the extra TR/num_slices/
# trigger_mode keys it doesn't know about. Both models are therefore
# evaluated on the EXACT SAME held-out sequences for a clean comparison,
# rather than needing two separately-built held-out sets.
#
# Prerequisites: both csv_pipeline/04_train_models.py (old) and
# csv_pipeline_seqparams/04_train_models.py (new) must have already produced
# their checkpoints.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import sys
import math
import pickle
from collections import defaultdict

import numpy as np
import torch

_AP_PATH = "/tmp/alternating_pipeline_src"
if _AP_PATH not in sys.path:
    sys.path.insert(0, _AP_PATH)

from AlternatingPipeline.config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG,
    START_TOKEN_ID, ID_TO_SEQUENCE_TYPE,
)
from AlternatingPipeline.models.examination_model import create_examination_model
from AlternatingPipeline.models.checkpoint_compat import (
    IncompatibleCheckpointError, load_checkpoint_lenient,
)
from AlternatingPipeline.training.utils import temporal_split, build_conditioning_tensor
from AlternatingPipeline.validation.metrics import compare_real_vs_predicted, print_comparison_report

# Baseline (pre-SUT) examination checkpoint. Overridable so an ABLATION
# checkpoint — same code, same pkl, same split, use_sut_conditioning=False —
# can be used instead. That is the cleaner control: the csv_pipeline default
# below was trained on a different pkl by whatever code was current at the
# time, so a difference between it and the new model is not attributable to
# the SUT features alone.
OLD_CHECKPOINT = os.environ.get(
    "BASELINE_CHECKPOINT",
    "/dbfs/FileStore/csv_pipeline/models/examination/examination_model_best.pt",
)
NEW_CHECKPOINT = f"{MODELS_DIR}/examination/examination_model_best.pt"
DURATION_SCALE = EXAMINATION_TRAINING_CONFIG['duration_scale']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COMMAND ----------

# =============================================================================
# Load both models against the SAME (new, superset) pkl
# =============================================================================

with open(PKL_OUTPUT, 'rb') as f:
    _data = pickle.load(f)
examination_sequences = _data['examination']
_, val_sequences = temporal_split(examination_sequences, val_days=2)
print(f"Held-out validation sequences (shared by both models): {len(val_sequences)}")

if not os.path.exists(NEW_CHECKPOINT):
    raise FileNotFoundError(
        f"New checkpoint not found at {NEW_CHECKPOINT} — run csv_pipeline_seqparams/04_train_models.py first."
    )

# --- baseline model (optional) --------------------------------------------
# The baseline is a nice-to-have: it only powers the old-vs-new deltas in
# criteria 2 and 3. Everything that evaluates the NEW model — criterion 1
# (feature sensitivity), its per-type calibration, and the standard report —
# stands on its own. So a missing or architecturally-incompatible baseline
# degrades this notebook instead of aborting it, rather than throwing away a
# multi-hour training run's evaluation over an artifact this pipeline does
# not even produce.
old_model = None
baseline_status = None
if not os.path.exists(OLD_CHECKPOINT):
    baseline_status = f"not found at {OLD_CHECKPOINT}"
else:
    _candidate = create_examination_model(EXAMINATION_MODEL_CONFIG)
    try:
        load_checkpoint_lenient(
            _candidate,
            torch.load(OLD_CHECKPOINT, map_location=device),
            label=f"baseline examination checkpoint ({OLD_CHECKPOINT})",
        )
    except IncompatibleCheckpointError as _err:
        baseline_status = "architecture mismatch"
        print(_err)
    else:
        old_model = _candidate.to(device).eval()

if old_model is None:
    print(f"\n!! BASELINE SKIPPED — {baseline_status}.")
    print("   Criteria 2 and 3 will report the NEW model only (no old-vs-new delta).")
    print("   To get the comparison, set BASELINE_CHECKPOINT to a checkpoint this")
    print("   code produced — ideally an ablation trained by")
    print("   csv_pipeline_seqparams/04_train_models.py with use_sut_conditioning=False,")
    print("   which isolates the SUT features as the only difference.")

# build_seqparams_model_config comes from this file's own %run ./config —
# single source of truth shared with 04_train_models.py / 05_feature_analysis.py.
EXAMINATION_MODEL_CONFIG_SEQPARAMS = build_seqparams_model_config(EXAMINATION_MODEL_CONFIG)
new_model = create_examination_model(EXAMINATION_MODEL_CONFIG_SEQPARAMS)
load_checkpoint_lenient(
    new_model,
    torch.load(NEW_CHECKPOINT, map_location=device),
    label=f"new examination checkpoint ({NEW_CHECKPOINT})",
)
new_model = new_model.to(device).eval()

if old_model is not None:
    print(f"Old model params: {sum(p.numel() for p in old_model.parameters()):,}")
print(f"New model params: {sum(p.numel() for p in new_model.parameters()):,}")

# COMMAND ----------

# =============================================================================
# Predict with both models on every held-out sequence
# =============================================================================

def _predict_one(model, seq, extra_features):
    toks = seq['sequence'][:model.max_seq_len - 1]
    if not toks:
        return None
    inp = torch.tensor([[START_TOKEN_ID] + toks], dtype=torch.long, device=device)
    cond = build_conditioning_tensor(
        seq['conditioning'], extra_feature_names=extra_features,
        denylist=SUT_LEAKAGE_DENYLIST,
    ).unsqueeze(0).to(device)
    info = {
        'body_region': torch.tensor([seq['body_region']], device=device),
        'sequence_type': torch.tensor([int(seq.get('sequence_type', 0))], device=device),
        'serial_idx': torch.tensor([int(seq.get('serial_idx', 0))], device=device),
        'trigger_mode': torch.tensor([int(seq.get('trigger_mode', 0))], device=device),
    }
    with torch.no_grad():
        mu, _ = model.estimate_durations(inp, cond, info)
    m = mu[0, len(toks) - 1].item()
    return (math.expm1(m) if model.duration_mode == 'log' else m) * DURATION_SCALE


rows = []
for seq in val_sequences:
    new_pred = _predict_one(new_model, seq, extra_features=EXAMINATION_SEQPARAM_FEATURES)
    if new_pred is None:
        continue
    # old model: 10-dim conditioning, ignores the extra SUT keys
    old_pred = (
        _predict_one(old_model, seq, extra_features=None)
        if old_model is not None else None
    )
    if old_model is not None and old_pred is None:
        continue
    target = sum(max(0.0, d) for d in seq.get('durations', []))
    rows.append({
        'sequence_type': ID_TO_SEQUENCE_TYPE.get(seq.get('sequence_type', 0), 'other'),
        'body_region': BODY_REGIONS[seq['body_region']] if seq['body_region'] < len(BODY_REGIONS) else 'UNKNOWN',
        'target_s': target,
        'old_pred_s': old_pred,
        'new_pred_s': new_pred,
    })

print(f"Evaluated {len(rows)} held-out sequences "
      f"({'both models' if old_model is not None else 'new model only'}).")

# COMMAND ----------

# =============================================================================
# CRITERION 1 (mandatory, checked first) — sensitivity: does the new model
# actually respond to the new feature(s) on real data? If this fails, none
# of the aggregate metrics below can be trusted (see the historical
# LayerNorm-erasure / silent-collapse failure mode).
# =============================================================================

print("\n" + "="*60)
print("CRITERION 1 (mandatory): new-feature sensitivity on real data")
print("="*60)

if not EXAMINATION_SEQPARAM_FEATURES:
    print("  SKIPPED — EXAMINATION_SEQPARAM_FEATURES is still empty (placeholder). "
          "Run sut_parameter_discovery.py and retrain before this criterion is meaningful.")
    criterion_1_pass = None
else:
    _probe_seq = val_sequences[0]
    _cond_a = dict(_probe_seq['conditioning'])
    _cond_b = dict(_probe_seq['conditioning'])
    for name in EXAMINATION_SEQPARAM_FEATURES:
        base_val = _cond_a.get(name, 0.0) or 1.0
        _cond_b[name] = base_val * 3.0  # perturb by 3x
    _seq_a = {**_probe_seq, 'conditioning': _cond_a}
    _seq_b = {**_probe_seq, 'conditioning': _cond_b}
    pred_a = _predict_one(new_model, _seq_a, extra_features=EXAMINATION_SEQPARAM_FEATURES)
    pred_b = _predict_one(new_model, _seq_b, extra_features=EXAMINATION_SEQPARAM_FEATURES)
    moved = abs(pred_a - pred_b)
    criterion_1_pass = moved > 0.5  # half a second — well above float noise
    print(f"  Perturbing {EXAMINATION_SEQPARAM_FEATURES} 3x moved the prediction by {moved:.2f}s "
          f"({pred_a:.1f}s -> {pred_b:.1f}s)")
    print(f"  {'PASS' if criterion_1_pass else 'FAIL — STOP, do not trust criteria below'}")

# COMMAND ----------

# =============================================================================
# CRITERION 2 — overall held-out MAE/MAPE improvement
# =============================================================================

print("\n" + "="*60)
print("CRITERION 2: overall MAE/MAPE")
print("="*60)

targets = np.array([r['target_s'] for r in rows])
new_preds = np.array([r['new_pred_s'] for r in rows])

new_mae = np.mean(np.abs(new_preds - targets))
new_mape = np.mean(np.abs(new_preds - targets) / np.maximum(targets, 1.0)) * 100
print(f"  New MAE:  {new_mae:.1f}s  (MAPE {new_mape:.1f}%)")

mae_improvement_pct = None
if old_model is not None:
    old_preds = np.array([r['old_pred_s'] for r in rows])
    old_mae = np.mean(np.abs(old_preds - targets))
    old_mape = np.mean(np.abs(old_preds - targets) / np.maximum(targets, 1.0)) * 100
    mae_improvement_pct = 100 * (old_mae - new_mae) / max(old_mae, 1e-6)
    print(f"  Old MAE:  {old_mae:.1f}s  (MAPE {old_mape:.1f}%)")
    print(f"  Improvement: {mae_improvement_pct:+.1f}%  "
          f"({'PASS (>=10%)' if mae_improvement_pct >= 10 else 'REVIEW (<10%)'})")
else:
    print(f"  Old MAE:  n/a — baseline skipped ({baseline_status})")
    print("  Improvement: not computable without a baseline. Compare the new MAE above "
          "against 05_feature_analysis.py's own baseline MAE for a sanity check.")

# COMMAND ----------

# =============================================================================
# CRITERION 3 — per-sequence-type calibration, especially the previously-
# underpredicted long types (Stage 3a in project memory: space, medic,
# dixon, epi, tfl, tse were systematically compressed).
# =============================================================================

print("\n" + "="*60)
print("CRITERION 3: per-sequence-type calibration (bias = mean(pred)/mean(target))")
print("="*60)

LONG_TYPES = {'space', 'medic', 'dixon', 'epi', 'tfl', 'tse'}
by_type = defaultdict(lambda: {'target': [], 'old': [], 'new': []})
for r in rows:
    by_type[r['sequence_type']]['target'].append(r['target_s'])
    by_type[r['sequence_type']]['new'].append(r['new_pred_s'])
    if old_model is not None:
        by_type[r['sequence_type']]['old'].append(r['old_pred_s'])

improved_long_types = 0
total_long_types = 0
well_calibrated_long_types = 0
for seq_type, d in sorted(by_type.items()):
    mean_target = np.mean(d['target'])
    if mean_target <= 0:
        continue
    new_bias = np.mean(d['new']) / mean_target
    tag = ' <- long/previously-compressed type' if seq_type in LONG_TYPES else ''
    if old_model is not None:
        old_bias = np.mean(d['old']) / mean_target
        print(f"  {seq_type:<10} n={len(d['target']):>5}  old_bias={old_bias:.2f}  "
              f"new_bias={new_bias:.2f}  (1.0=perfect){tag}")
    else:
        print(f"  {seq_type:<10} n={len(d['target']):>5}  new_bias={new_bias:.2f}  "
              f"(1.0=perfect){tag}")
    if seq_type in LONG_TYPES:
        total_long_types += 1
        if old_model is not None and abs(new_bias - 1.0) < abs(old_bias - 1.0):
            improved_long_types += 1
        # Absolute check, meaningful with or without a baseline: Stage 3a's
        # failure mode was long types coming out 2-3x SHORT, so a bias inside
        # [0.7, 1.3] is the thing that historically was not achieved.
        if 0.7 <= new_bias <= 1.3:
            well_calibrated_long_types += 1

if not total_long_types:
    print("\n  No long-type sequences in this held-out set — cannot evaluate criterion 3.")
elif old_model is not None:
    print(f"\n  Long types improved vs. baseline: {improved_long_types}/{total_long_types}  "
          f"({'PASS' if improved_long_types > total_long_types / 2 else 'REVIEW'})")
else:
    print(f"\n  Long types within +/-30% of target: "
          f"{well_calibrated_long_types}/{total_long_types}  "
          f"({'PASS' if well_calibrated_long_types > total_long_types / 2 else 'REVIEW'}) "
          f"— absolute calibration, no baseline needed")

# COMMAND ----------

# =============================================================================
# CRITERION 5 — sanity: token-decoder perplexity roughly comparable (widening
# conditioning shouldn't destabilize the unrelated token-decoder path).
# Criterion 4 (permutation importance ranking) is reported by
# 05_feature_analysis.py Section B, not repeated here.
# =============================================================================

print("\n" + "="*60)
print("CRITERION 5: token-decoder sanity (informational)")
print("="*60)
print("  See training_history.pkl val_perplexity in both models' save dirs — "
      "compare manually; not recomputed here to avoid a second full pass "
      "over the validation set.")

# COMMAND ----------

# =============================================================================
# Standard comparison report via the existing validation harness
# =============================================================================

print("\n" + "="*60)
print("Standard comparison report (AlternatingPipeline/validation/metrics.py)")
print("="*60)

real_schedule = [
    {'event_type': 'examination', 'body_region': r['body_region'], 'duration': r['target_s']}
    for r in rows
]
predicted_schedule_new = [
    {'event_type': 'examination', 'body_region': r['body_region'], 'duration': r['new_pred_s']}
    for r in rows
]
metrics = compare_real_vs_predicted(real_schedule, predicted_schedule_new)
print_comparison_report(metrics)

# COMMAND ----------

# =============================================================================
# SUMMARY VERDICT
# =============================================================================

print("\n" + "="*60)
print("VERDICT SUMMARY")
print("="*60)
if old_model is None:
    print(f"  Baseline: SKIPPED ({baseline_status}) — criteria below are new-model-only")

if criterion_1_pass is None:
    print("  Criterion 1 (sensitivity): SKIPPED — no real SUT features trained yet")
elif not criterion_1_pass:
    print("  Criterion 1 (sensitivity): FAIL — stop, do not trust criteria 2-3 above")
else:
    print("  Criterion 1 (sensitivity): PASS")
    if mae_improvement_pct is None:
        print(f"  Criterion 2 (overall MAE): NO BASELINE — new MAE {new_mae:.1f}s")
    else:
        print(f"  Criterion 2 (overall MAE): {'PASS' if mae_improvement_pct >= 10 else 'REVIEW'} "
              f"({mae_improvement_pct:+.1f}%)")
    if total_long_types and old_model is not None:
        print(f"  Criterion 3 (long-type calibration): "
              f"{'PASS' if improved_long_types > total_long_types / 2 else 'REVIEW'} "
              f"({improved_long_types}/{total_long_types} improved vs. baseline)")
    elif total_long_types:
        print(f"  Criterion 3 (long-type calibration): "
              f"{'PASS' if well_calibrated_long_types > total_long_types / 2 else 'REVIEW'} "
              f"({well_calibrated_long_types}/{total_long_types} within +/-30% of target)")
    print("  Criterion 4 (permutation importance): see 05_feature_analysis.py output")
