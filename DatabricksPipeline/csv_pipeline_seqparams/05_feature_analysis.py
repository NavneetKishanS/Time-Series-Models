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

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix — Examination Conditioning Features (SUT-enriched)')
plt.tight_layout()
_corr_path = f"{ANALYSIS_DIR}/correlation_matrix.png"
plt.savefig(_corr_path)
print(f"\nSaved → {_corr_path}")

_corr_csv = f"{ANALYSIS_DIR}/correlation_matrix.csv"
corr_matrix.to_csv(_corr_csv)
print(f"Saved → {_corr_csv}")

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

from AlternatingPipeline.config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG, START_TOKEN_ID,
)
from AlternatingPipeline.models.examination_model import create_examination_model
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
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
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
                denylist=SUT_LEAKAGE_DENYLIST,
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

    plt.figure(figsize=(9, 5))
    plt.barh(_perm_df['feature'], _perm_df['degradation_s'])
    plt.xlabel('MAE degradation when shuffled (s)')
    plt.title('Permutation importance — examination duration model')
    plt.tight_layout()
    _perm_png = f"{ANALYSIS_DIR}/permutation_importance.png"
    plt.savefig(_perm_png)
    print(f"Saved → {_perm_png}")

# COMMAND ----------

# =============================================================================
# NEXT STEP: run 06_compare_models.py for the old-vs-new head-to-head.
# =============================================================================
