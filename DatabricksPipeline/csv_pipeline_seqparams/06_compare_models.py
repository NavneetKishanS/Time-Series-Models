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

# DBTITLE 1,Cell 3
import os
import sys
import math
import pickle
import shutil
from collections import defaultdict

import numpy as np
import torch

_AP_PATH = "/tmp/alternating_pipeline_src"  # matches 04_train_models.py's TMP_ROOT
_REPO_ROOT = "/Workspace/Shared/Patient Exchange and Examination NK/Time-Series-Models"

# Self-heal: refresh the /tmp copy if it's stale or missing modules this
# notebook needs (avoids requiring 04_train_models.py to run first).
_REQUIRED_MODULES = [
    "AlternatingPipeline.config",
    "AlternatingPipeline.models.examination_model",
    "AlternatingPipeline.models.checkpoint_compat",
    "AlternatingPipeline.training.utils",
    "AlternatingPipeline.validation.metrics",
]
_needs_refresh = not os.path.isdir(_AP_PATH) or not all(
    os.path.isfile(os.path.join(_AP_PATH, m.replace('.', '/') + '.py'))
    or os.path.isdir(os.path.join(_AP_PATH, m.replace('.', '/')))
    for m in _REQUIRED_MODULES
)
if _needs_refresh:
    print(f"[refresh] Stale or missing source at {_AP_PATH} — re-copying from repo...")
    _ap_dst = f"{_AP_PATH}/AlternatingPipeline"
    if os.path.exists(_ap_dst):
        shutil.rmtree(_ap_dst)
    os.makedirs(_AP_PATH, exist_ok=True)
    shutil.copytree(f"{_REPO_ROOT}/AlternatingPipeline", _ap_dst)
    # Evict stale cached modules so fresh imports take effect
    for _k in [k for k in sys.modules if k.startswith('AlternatingPipeline')]:
        del sys.modules[_k]
    print("[refresh] Done.")

if _AP_PATH not in sys.path:
    sys.path.insert(0, _AP_PATH)

assert_pipeline_source_fresh(_AP_PATH, required_modules=_REQUIRED_MODULES)

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
    f"{BASE_MODELS_DIR}/examination/examination_model_best.pt",
)
NEW_CHECKPOINT = f"{MODELS_DIR}/examination/examination_model_best.pt"
DURATION_SCALE = EXAMINATION_TRAINING_CONFIG['duration_scale']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COMMAND ----------

# DBTITLE 1,Cell 4
# =============================================================================
# Load both models against the SAME (new, superset) pkl
# =============================================================================

with open(PKL_OUTPUT, 'rb') as f:
    _data = pickle.load(f)
examination_sequences = _data['examination']
# See config.SUT_AUDIT_ONLY_KEYS — none of this notebook's fields are read
# below; stripped in place to avoid carrying them for the rest of the session.
strip_audit_only_fields(examination_sequences)
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
# Override dimensions from the checkpoint itself so this notebook loads whatever
# checkpoint it finds regardless of when it was trained (feature set may have
# expanded since training).
EXAMINATION_MODEL_CONFIG_SEQPARAMS = build_seqparams_model_config(EXAMINATION_MODEL_CONFIG)
_new_ckpt = torch.load(NEW_CHECKPOINT, map_location=device)
if 'conditioning_scale' in _new_ckpt:
    EXAMINATION_MODEL_CONFIG_SEQPARAMS['base_conditioning_dim'] = _new_ckpt['conditioning_scale'].shape[0]
    # Remove pre-computed scale list so model creates buffer from base_conditioning_dim;
    # load_state_dict then overwrites it with the checkpoint's trained values.
    EXAMINATION_MODEL_CONFIG_SEQPARAMS.pop('conditioning_scale', None)
if 'serial_embedding.weight' in _new_ckpt:
    EXAMINATION_MODEL_CONFIG_SEQPARAMS['num_serials'] = _new_ckpt['serial_embedding.weight'].shape[0]
new_model = create_examination_model(EXAMINATION_MODEL_CONFIG_SEQPARAMS)
load_checkpoint_lenient(
    new_model,
    _new_ckpt,
    label=f"new examination checkpoint ({NEW_CHECKPOINT})",
)
new_model = new_model.to(device).eval()

if old_model is not None:
    print(f"Old model params: {sum(p.numel() for p in old_model.parameters()):,}")
print(f"New model params: {sum(p.numel() for p in new_model.parameters()):,}")

# COMMAND ----------

# DBTITLE 1,Cell 5
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
        denylist=SUT_ALL_DENYLISTS,
    ).unsqueeze(0).to(device)
    # Trim to model's trained conditioning dim (feature list may have grown
    # since this checkpoint was trained — new features are appended at end).
    _expected_dim = model.conditioning_scale.shape[0]
    if cond.shape[1] > _expected_dim:
        cond = cond[:, :_expected_dim]
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

# Probed over MANY sequences, not one. The original single-sequence version
# (val_sequences[0]) with an `or 1.0` fallback silently degenerated when that
# one scan happened to carry TR=0: the "3x perturbation" became 0 -> 3.0 raw,
# i.e. 0 -> 0.003 after conditioning_scale, which cannot move any prediction.
# Sequences are also filtered to those with a genuinely non-zero value, so a
# null result means "the model ignores this feature", not "we fed it nothing".
PROBE_N = int(os.environ.get('PROBE_N', 80))


def _probe_delta(seqs, make_a, make_b):
    """Mean/max |prediction delta| between two mutations of the same sequences."""
    deltas = []
    for _s in seqs:
        _pa = _predict_one(new_model, make_a(_s), extra_features=EXAMINATION_SEQPARAM_FEATURES)
        _pb = _predict_one(new_model, make_b(_s), extra_features=EXAMINATION_SEQPARAM_FEATURES)
        if _pa is None or _pb is None:
            continue
        deltas.append(abs(_pa - _pb))
    if not deltas:
        return 0.0, 0.0, 0
    return float(np.mean(deltas)), float(np.max(deltas)), len(deltas)


def _scale_sut(factor):
    """Mutation that multiplies every SUT feature by `factor`."""
    def _mutate(seq):
        cond = dict(seq.get('conditioning', {}))
        for name in EXAMINATION_SEQPARAM_FEATURES:
            cond[name] = float(cond.get(name, 0.0) or 0.0) * factor
        return {**seq, 'conditioning': cond}
    return _mutate


def _force(**overrides):
    """Mutation that overrides top-level categorical conditioning keys."""
    return lambda seq: {**seq, **overrides}


if not EXAMINATION_SEQPARAM_FEATURES:
    print("  SKIPPED — EXAMINATION_SEQPARAM_FEATURES is still empty (placeholder). "
          "Run sut_parameter_discovery.py and retrain before this criterion is meaningful.")
    criterion_1_pass = None
else:
    _nonzero = [
        s for s in val_sequences
        if any(float(s.get('conditioning', {}).get(n, 0) or 0) > 0
               for n in EXAMINATION_SEQPARAM_FEATURES)
    ]
    print(f"  Held-out sequences with a non-zero {EXAMINATION_SEQPARAM_FEATURES}: "
          f"{len(_nonzero):,}/{len(val_sequences):,}")
    if not _nonzero:
        print("  FAIL — no held-out sequence carries a non-zero value. The features never "
              "reached the pkl; fix step 03 before reading anything below.")
        criterion_1_pass = False
    else:
        _sut_mean, _sut_max, _sut_n = _probe_delta(
            _nonzero[:PROBE_N], _scale_sut(1.0), _scale_sut(3.0)
        )
        criterion_1_pass = _sut_mean > 0.5  # half a second — well above float noise
        print(f"  Perturbing {EXAMINATION_SEQPARAM_FEATURES} 3x over {_sut_n} sequences moved "
              f"the prediction by mean {_sut_mean:.3f}s (max {_sut_max:.3f}s)")
        print(f"  {'PASS' if criterion_1_pass else 'FAIL — STOP, do not trust criteria below'}")

# COMMAND ----------

# =============================================================================
# CRITERION 1b — IS THE CONDITIONING TOKEN ALIVE AT ALL?
#
# Criterion 1 failing does not by itself mean the SUT features are useless. It
# could equally mean the duration head cannot see the conditioning token that
# carries them. Those two have completely different remedies, so distinguish
# them here before spending a retrain on either.
#
# In estimate_durations(), sequence_type reaches the duration encoder TWICE:
# through the conditioning token AND through duration_seq_type_bias, which is
# added to every token position. body_region, serial_idx, trigger_mode, TR and
# num_slices reach it ONLY through the single conditioning token at position 0.
#
# So sequence_type is the positive control. If it moves the prediction hard
# while body_region — which has a 2.3x real duration spread (ABDOMEN ~63s vs
# SPINE ~144s) — does not, the conditioning token is effectively dead to the
# duration head, and no amount of SUT feature engineering will help until that
# is fixed.
# =============================================================================

print("\n" + "="*60)
print("CRITERION 1b: which conditioning channels move the duration head?")
print("="*60)

_probe_seqs = val_sequences[:PROBE_N]
_channel_probes = [
    ("sequence_type (scout vs space)",
     _force(sequence_type=SEQUENCE_TYPE_VOCAB['scout']),
     _force(sequence_type=SEQUENCE_TYPE_VOCAB['space']),
     "per-position bias + cond token  <- POSITIVE CONTROL"),
    ("body_region (ABDOMEN vs SPINE)",
     _force(body_region=BODY_REGION_TO_ID['ABDOMEN']),
     _force(body_region=BODY_REGION_TO_ID['SPINE']),
     "conditioning token only"),
    ("serial_idx (0 vs 1)",
     _force(serial_idx=0), _force(serial_idx=1),
     "conditioning token only"),
]
if EXAMINATION_SEQPARAM_FEATURES:
    _channel_probes.append((
        f"{'+'.join(EXAMINATION_SEQPARAM_FEATURES)} (1x vs 3x)",
        _scale_sut(1.0), _scale_sut(3.0),
        "conditioning token only",
    ))

_channel_results = {}
for _label, _mk_a, _mk_b, _path in _channel_probes:
    _mean_d, _max_d, _n = _probe_delta(_probe_seqs, _mk_a, _mk_b)
    _channel_results[_label] = _mean_d
    print(f"  {_label:<38} mean|delta|={_mean_d:>8.3f}s  max={_max_d:>8.3f}s  ({_path})")

_control_label = _channel_probes[0][0]
_control = _channel_results.get(_control_label, 0.0)
_cond_only = [v for k, v in _channel_results.items() if k != _control_label]

print()
if _control < 1.0:
    print("  INCONCLUSIVE — even the positive control barely moved. The duration head is "
          "not responding to ANY conditioning; investigate the checkpoint itself.")
elif _cond_only and max(_cond_only) < 1.0:
    print("  >> CONDITIONING TOKEN IS DEAD. sequence_type moves the duration head only "
          "because duration_seq_type_bias injects it at every token position; every "
          "channel that arrives solely via the conditioning token moves it by <1s.")
    print("     body_region alone has a ~2.3x real duration spread, so this is a model "
          "defect, not a property of the data.")
    print("     REMEDY: inject the conditioning token per-position into the duration "
          "encoder, mirroring duration_seq_type_bias (zero-init keeps existing "
          "checkpoints loadable). That unblocks body_region, serial_idx AND the SUT "
          "features together — retraining for the SUT features alone would not fix it.")
else:
    print("  Conditioning token IS live — at least one cond-token-only channel moves the "
          "duration head. A null result for the SUT features is then about those "
          "features, not about the architecture.")

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

if _channel_results:
    _cond_only_max = max(
        (v for k, v in _channel_results.items() if k != _control_label), default=0.0
    )
    if _control >= 1.0 and _cond_only_max < 1.0:
        print("  Criterion 1b (conditioning token): DEAD — only sequence_type reaches the "
              "duration head. Fix the per-position injection before retraining for features.")
    elif _control < 1.0:
        print("  Criterion 1b (conditioning token): INCONCLUSIVE — positive control did not move.")
    else:
        print(f"  Criterion 1b (conditioning token): LIVE — best cond-token-only channel "
              f"moves {_cond_only_max:.2f}s")

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
