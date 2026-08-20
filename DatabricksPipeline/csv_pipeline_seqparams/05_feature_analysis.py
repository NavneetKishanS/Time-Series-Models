# Databricks notebook source
# Databricks notebook — Feature importance + correlation matrix (in Databricks)
#
# Two independent sections, per the explicit "do a feature importance and
# corr matrix all within databricks" ask:
#
#   Section A (correlation matrix) needs only the preprocessed pkl — run it
#   right after 03_build_preprocessed_pkl.py, before training.
#   Section B (permutation importance) needs the trained checkpoint from
#   04_train_models.py.
#
# Correlation-matrix pattern follows the only precedent in this repo,
# _archive/PXChange_Refactored_v1/analyze_data.py (df[cols].corr() + seaborn
# heatmap), from an old unrelated tabular model. No feature-importance
# tooling (SHAP, permutation_importance, feature_importances_) exists
# anywhere in this repo — permutation importance is implemented here because
# it is model-agnostic (works directly against estimate_durations(), no
# gradient-attribution machinery needed) and answers the actual question
# ("did adding this feature help?") more directly than a correlation number.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import gc
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# COMMAND ----------

# =============================================================================
# SECTION A — Correlation matrix (pkl only, no trained model needed)
# =============================================================================

print("\n" + "="*60)
print("SECTION A: Correlation matrix")
print("="*60)

with open(PKL_OUTPUT, 'rb') as f:
    _data = pickle.load(f)
examination_sequences = _data['examination']
print(f"Examination sequences: {len(examination_sequences):,}")

# See config.SUT_AUDIT_ONLY_KEYS — neither Section A nor B reads these;
# stripped in place so they aren't carried for the rest of the session
# (Section B loads the model onto the same node right after Section A).
strip_audit_only_fields(examination_sequences)

_rows = []
for s in examination_sequences:
    cond = s.get('conditioning', {})
    row = {
        'sequence_type': s.get('sequence_type', 0),
        'body_region': s.get('body_region', 10),
        'serial_idx': s.get('serial_idx', 0),
        'trigger_mode': s.get('trigger_mode', 0),
        'total_duration': s.get('total_duration', 0.0),
        'Age': cond.get('Age', 0.0),
        'Weight': cond.get('Weight', 0.0),
        'Height': cond.get('Height', 0.0),
        'PTAB': cond.get('PTAB', 0.0),
    }
    for name in EXAMINATION_SEQPARAM_FEATURES:
        row[name] = cond.get(name, 0.0)
    _rows.append(row)

df = pd.DataFrame(_rows)

numerical_features = [
    'sequence_type', 'body_region', 'serial_idx', 'trigger_mode',
    'Age', 'Weight', 'Height', 'PTAB', 'total_duration',
] + list(EXAMINATION_SEQPARAM_FEATURES)

print("\nCorrelation matrix of numerical conditioning features vs. total_duration:")
corr_matrix = df[numerical_features].corr()
print(corr_matrix.round(2))

# PARAM_SET='all' at the 0% presence floor (487d9e2) puts ~170 raw SUT fields
# into the model, and with presence flags on (the default) that's ~341 extra
# conditioning columns -> ~350 total here. A FIXED (10, 8) figure with
# annot=True was sized for the original ~14-column feature set; at 350x350
# cells that's ~122,500 text annotations crushed into a few pixels each, which
# renders as a solid black block, not a readable heatmap. Scale the canvas to
# the actual feature count and drop per-cell numbers once they can no longer
# be read anyway — the colorbar still carries the signal.
_n_feat = len(numerical_features)
_ANNOT_MAX_FEATURES = 40  # beyond this, per-cell text is illegible and expensive
_heatmap_annot = _n_feat <= _ANNOT_MAX_FEATURES
_side_in = max(10, _n_feat * 0.35)
plt.figure(figsize=(_side_in, _side_in * 0.8))
sns.heatmap(
    corr_matrix, annot=_heatmap_annot, cmap='coolwarm', fmt=".2f",
    center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'},
    annot_kws={'size': 6} if _heatmap_annot else None,
)
plt.title(f'Correlation Matrix — Examination Conditioning Features (SUT-enriched, n={_n_feat})')
plt.xticks(fontsize=6 if _n_feat > _ANNOT_MAX_FEATURES else 10, rotation=90)
plt.yticks(fontsize=6 if _n_feat > _ANNOT_MAX_FEATURES else 10)
plt.tight_layout()
_corr_path = f"{ANALYSIS_DIR}/correlation_matrix.png"
plt.savefig(_corr_path, dpi=150 if _n_feat > _ANNOT_MAX_FEATURES else 100)
plt.close()  # leaked figures accumulate across reruns in a persistent Databricks kernel
print(f"\nSaved → {_corr_path}" + ("" if _heatmap_annot else
      f"  (annotations dropped: {_n_feat} features > {_ANNOT_MAX_FEATURES}; see the CSV for exact values)"))

_corr_csv = f"{ANALYSIS_DIR}/correlation_matrix.csv"
corr_matrix.to_csv(_corr_csv)
print(f"Saved → {_corr_csv}")

# Readable regardless of feature count: correlation with the actual target,
# ranked, which is usually the number people opened this chart to find anyway.
_target_corr = corr_matrix['total_duration'].drop('total_duration').sort_values(key=abs, ascending=False)
plt.figure(figsize=(9, max(5, len(_target_corr) * 0.18)))
plt.barh(_target_corr.index[::-1], _target_corr.values[::-1],
         color=['#4C78A8' if v >= 0 else '#E45756' for v in _target_corr.values[::-1]])
plt.axvline(0, color='#666', linewidth=0.8)
plt.xlabel('Pearson r with total_duration')
plt.title(f'Feature correlation with total_duration (n={len(_target_corr)}, ranked by |r|)')
plt.tight_layout()
_target_corr_path = f"{ANALYSIS_DIR}/correlation_with_duration.png"
plt.savefig(_target_corr_path, dpi=110)
plt.close()
print(f"Saved → {_target_corr_path}")

# COMMAND ----------

# Categorical breakdown (Pearson correlation on integer category CODES is
# a blunt instrument for sequence_type/trigger_mode/body_region — this table
# is the more honest view for those: mean/std/count of total_duration per
# category, same shape as the existing per-scan-type duration probe).

print("\nPer-category total_duration (mean / std / n):")
for cat_col, id_to_name in [
    ('sequence_type', ID_TO_SEQUENCE_TYPE),
    ('trigger_mode', {v: k for k, v in TRIGGER_MODE_VOCAB.items()}),
    ('body_region', {i: r for i, r in enumerate(BODY_REGIONS)}),
]:
    print(f"\n  -- {cat_col} --")
    grouped = df.groupby(cat_col)['total_duration'].agg(['mean', 'std', 'count'])
    for cat_id, r in grouped.iterrows():
        name = id_to_name.get(cat_id, str(cat_id))
        print(f"    {name:<18} mean={r['mean']:>7.1f}s  std={r['std']:>6.1f}  n={int(r['count']):>6,}")

_cat_csv = f"{ANALYSIS_DIR}/categorical_duration_breakdown.csv"
df.groupby(['sequence_type', 'trigger_mode', 'body_region'])['total_duration'].agg(
    ['mean', 'std', 'count']
).to_csv(_cat_csv)
print(f"\nSaved → {_cat_csv}")

# Free Section A's intermediates before Section B loads the model onto the
# same node. `_rows` and `df` are a ~51k-row x ~350-col Python list-of-dicts
# and DataFrame at PARAM_SET='all' — the same double-liability pattern
# 04_train_models.py Cell 8 was fixed for (96db4e4), unaddressed here.
# `examination_sequences` is NOT freed: Section B's temporal_split() below
# still needs it.
for _var in ('_rows', 'df', 'corr_matrix', '_target_corr', '_data'):
    globals().pop(_var, None)
gc.collect()

# COMMAND ----------

# =============================================================================
# SECTION B — Permutation importance (needs the trained checkpoint)
# =============================================================================

print("\n" + "="*60)
print("SECTION B: Permutation importance")
print("="*60)

import torch
import sys

_AP_PATH = "/tmp/alternating_pipeline_src"  # matches 04_train_models.py's TMP_ROOT
if _AP_PATH not in sys.path:
    sys.path.insert(0, _AP_PATH)

# See 06_compare_models.py — this notebook relies on 04's copy, which /tmp keeps
# alive across the whole cluster lifetime and never refreshes on its own.
assert_pipeline_source_fresh(_AP_PATH, required_modules=[
    "AlternatingPipeline.config",
    "AlternatingPipeline.models.examination_model",
    "AlternatingPipeline.training.utils",
])

from AlternatingPipeline.config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG, START_TOKEN_ID,
)
from AlternatingPipeline.models.examination_model import create_examination_model
from AlternatingPipeline.models.checkpoint_compat import load_checkpoint_lenient, IncompatibleCheckpointError
from AlternatingPipeline.training.utils import temporal_split, build_conditioning_tensor

CHECKPOINT_PATH = f"{MODELS_DIR}/examination/examination_model_best.pt"
if not os.path.exists(CHECKPOINT_PATH):
    print(f"No checkpoint at {CHECKPOINT_PATH} yet — run 04_train_models.py first. "
          f"Skipping Section B.")
else:
    # build_seqparams_model_config comes from this file's own %run ./config —
    # single source of truth shared with 04_train_models.py / 06_compare_models.py.
    EXAMINATION_MODEL_CONFIG_SEQPARAMS = build_seqparams_model_config(EXAMINATION_MODEL_CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_examination_model(EXAMINATION_MODEL_CONFIG_SEQPARAMS)
    # Lenient: tolerates params added since the checkpoint was trained (e.g.
    # duration_cond_bias, which is zero-initialised and therefore a no-op),
    # while still refusing a checkpoint from a different architecture.
    try:
        load_checkpoint_lenient(
            model,
            torch.load(CHECKPOINT_PATH, map_location=device),
            label=f"examination checkpoint ({CHECKPOINT_PATH})",
        )
    except IncompatibleCheckpointError as e:
        print(f"Checkpoint incompatible with current model architecture:\n{e}\n")
        print("The model config has changed since this checkpoint was trained.")
        print("Re-run 04_train_models.py to produce a compatible checkpoint.")
        print("Skipping Section B.")
        CHECKPOINT_PATH = None  # signal to skip the rest

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    model = model.to(device)
    model.eval()

    _, val_sequences = temporal_split(examination_sequences, val_days=2)
    print(f"Held-out validation sequences: {len(val_sequences)}")

    duration_scale = EXAMINATION_TRAINING_CONFIG['duration_scale']

    def _predicted_seconds(seqs, shuffle_field=None, rng=None):
        """Run estimate_durations() over seqs, optionally shuffling one
        conditioning field across the batch first. Returns (predicted_secs,
        target_secs) arrays, one value per sequence (finish-token span total,
        same convention as the existing post-train probe)."""
        conds, regions, seq_types, serials, triggers, tokens_list, targets = [], [], [], [], [], [], []
        for s in seqs:
            toks = s['sequence'][:model.max_seq_len - 1]
            if not toks:
                continue
            conds.append(build_conditioning_tensor(
                s['conditioning'], extra_feature_names=EXAMINATION_SEQPARAM_FEATURES,
                denylist=SUT_ALL_DENYLISTS,
            ))
            regions.append(s['body_region'])
            seq_types.append(int(s.get('sequence_type', 0)))
            serials.append(int(s.get('serial_idx', 0)))
            triggers.append(int(s.get('trigger_mode', 0)))
            tokens_list.append(toks)
            targets.append(sum(max(0.0, d) for d in s.get('durations', [])))

        if rng is None:
            rng = np.random.default_rng(0)

        conds_t = torch.stack(conds)
        regions_t = torch.tensor(regions, dtype=torch.long)
        seq_types_t = torch.tensor(seq_types, dtype=torch.long)
        serials_t = torch.tensor(serials, dtype=torch.long)
        triggers_t = torch.tensor(triggers, dtype=torch.long)

        if shuffle_field is not None:
            if shuffle_field in EXAMINATION_SEQPARAM_FEATURES:
                col = 10 + EXAMINATION_SEQPARAM_FEATURES.index(shuffle_field)
                perm = torch.tensor(rng.permutation(len(conds)))
                conds_t[:, col] = conds_t[perm, col]
            elif shuffle_field == 'sequence_type':
                perm = rng.permutation(len(conds))
                seq_types_t = seq_types_t[perm]
            elif shuffle_field == 'serial_idx':
                perm = rng.permutation(len(conds))
                serials_t = serials_t[perm]
            elif shuffle_field == 'trigger_mode':
                perm = rng.permutation(len(conds))
                triggers_t = triggers_t[perm]
            elif shuffle_field == 'body_region':
                perm = rng.permutation(len(conds))
                regions_t = regions_t[perm]

        preds = []
        with torch.no_grad():
            for i, toks in enumerate(tokens_list):
                inp = torch.tensor([[START_TOKEN_ID] + toks],
                                    dtype=torch.long, device=device)
                info = {
                    'body_region': regions_t[i:i+1].to(device),
                    'sequence_type': seq_types_t[i:i+1].to(device),
                    'serial_idx': serials_t[i:i+1].to(device),
                    'trigger_mode': triggers_t[i:i+1].to(device),
                }
                mu, _ = model.estimate_durations(inp, conds_t[i:i+1].to(device), info)
                m = mu[0, len(toks) - 1].item()
                pred_sec = (math.expm1(m) if model.duration_mode == 'log' else m) * duration_scale
                preds.append(pred_sec)
        return np.array(preds), np.array(targets)

    baseline_preds, targets = _predicted_seconds(val_sequences)
    baseline_mae = np.mean(np.abs(baseline_preds - targets))
    print(f"Baseline MAE (no shuffle): {baseline_mae:.1f}s over {len(targets)} sequences")

    feature_names = ['sequence_type', 'serial_idx', 'trigger_mode', 'body_region'] + list(EXAMINATION_SEQPARAM_FEATURES)
    n_repeats = 5
    results = []
    for name in feature_names:
        degradations = []
        for rep in range(n_repeats):
            rng = np.random.default_rng(rep)
            preds, _ = _predicted_seconds(val_sequences, shuffle_field=name, rng=rng)
            mae = np.mean(np.abs(preds - targets))
            degradations.append(mae - baseline_mae)
        mean_degradation = float(np.mean(degradations))
        results.append({
            'feature': name,
            'baseline_mae_s': round(baseline_mae, 1),
            'shuffled_mae_s': round(baseline_mae + mean_degradation, 1),
            'degradation_s': round(mean_degradation, 1),
            'pct_degradation': round(100 * mean_degradation / max(1e-6, baseline_mae), 1),
        })

    results.sort(key=lambda r: -r['degradation_s'])
    print("\nPermutation importance (higher degradation = more important):")
    for r in results:
        print(f"  {r['feature']:<14} degradation={r['degradation_s']:>7.1f}s "
              f"({r['pct_degradation']:>5.1f}%)  shuffled_mae={r['shuffled_mae_s']:.1f}s")

    _perm_df = pd.DataFrame(results)
    _perm_csv = f"{ANALYSIS_DIR}/permutation_importance.csv"
    _perm_df.to_csv(_perm_csv, index=False)
    print(f"\nSaved → {_perm_csv}")

    # Same feature-count problem as the Section A heatmap: at PARAM_SET='all'
    # this is ~345 bars (4 base + EXAMINATION_SEQPARAM_FEATURES), not the ~10
    # the fixed (9, 5) figure was sized for.
    plt.figure(figsize=(9, max(5, len(_perm_df) * 0.18)))
    plt.barh(_perm_df['feature'], _perm_df['degradation_s'])
    plt.xlabel('MAE degradation when shuffled (s)')
    plt.title(f'Permutation importance — examination duration model (n={len(_perm_df)} features)')
    plt.tick_params(axis='y', labelsize=6 if len(_perm_df) > 40 else 10)
    plt.tight_layout()
    _perm_png = f"{ANALYSIS_DIR}/permutation_importance.png"
    plt.savefig(_perm_png, dpi=150 if len(_perm_df) > 40 else 100)
    plt.close()
    print(f"Saved → {_perm_png}")

# COMMAND ----------

# =============================================================================
# NEXT STEP: run 06_compare_models.py for the old-vs-new head-to-head.
# =============================================================================
