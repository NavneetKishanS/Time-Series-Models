# Databricks notebook source
"""
03b_parameter_report.py — one report, five gates, two audiences.

WHAT THIS REPLACES
------------------
03b_parameter_decomposition, 03c_join_and_target_diagnostics,
03d_identifier_leakage_check, 03e_param_set_shootout and
03f_coverage_and_masking: 2,321 lines across five notebooks, each written to
resolve a contradiction the previous one raised. That is a research log, and it
did its job — the join question is closed, MUID is acquitted, the parameter
route beats the protocol oracle on the rows it covers. What it is not is
something anybody should have to run five of.

Most of what those notebooks answered is now settled history and lives in
FINDINGS_parameter_investigation.md. What survives is the handful of checks that
must re-run on EVERY build, because each of them can regress:

  GATE 1  build health      how many rows, how much coverage, and where the
                            missing coverage went
  GATE 2  join integrity    does the joined SUT message describe the sequence
                            that actually ran (was 100.0% — a regression test)
  GATE 3  admissibility     which fields exist, which are admitted, and the
                            REASON for every exclusion
  GATE 4  value             do the parameters beat the protocol oracle and the
                            +-15s bar, on an honest split, per body group
  GATE 5  leakage sentinel  is any admitted field predicting duration the way
                            an identifier or a copy of the target would

TWO OUTPUTS, ON PURPOSE
-----------------------
    parameter_report.md     the executive one. How many parameters, why, what
                            more or fewer would buy, and what is blocking.
                            Numbers with sentences around them.
    parameter_report.json   every field, every percentile, every per-group MAE.
                            Readable outside Databricks, which matters because
                            the person reasoning about these numbers between
                            runs cannot start a cluster.

Both land in ANALYSIS_DIR, which is namespaced by PARAM_SET.

THE ONE THING THAT CHANGED IN HOW THESE NUMBERS ARE MADE
--------------------------------------------------------
Every MAE 03b-03f reported came from a RANDOM 80/20 row split. Sequences from
one exam landed on both sides, and every scanner in test was also in train. That
inflates a score in exactly the direction that matters least and hides the one
that matters most: the goal is the extraordinary 1% and transfer to customers we
have never seen, and a random split measures neither.

Everything below is scored on a GROUPED split (data.holdout). The random-split
number is printed alongside it wherever the gap is informative, because the gap
is itself a finding — the bigger it is, the more of the old score was memory.

Run after 03_build_preprocessed_pkl.py. NO TRAINING, NO SPARK. ~10 minutes.
"""

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = "/Workspace/Repos/Time-Series-Models"
for _candidate in (_REPO, "/tmp/Time-Series-Models", os.getcwd()):
    if os.path.isdir(os.path.join(_candidate, "AlternatingPipeline")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

from AlternatingPipeline.data.protocol_vocab import (
    heldout_group_r2, normalize_protocol_name,
)
from AlternatingPipeline.data.parameter_analysis import (
    _to_float, exam_group_labels, heldout_regressor_score,
    numeric_field_inventory, numeric_matrix, permutation_importance_mae,
)
from AlternatingPipeline.data.holdout import holdout_mask, split_summary

PKL_PATH = os.environ.get('PKL_PATH', PKL_OUTPUT)

# Görtler named +-15s as the gold standard; scanner + protocol measured 15.3s
# MAE on the real exam CSVs, so the two agree.
TARGET_MAE_S = float(os.environ.get('TARGET_MAE_S', '15.0'))
REPEATS = int(os.environ.get('REPORT_REPEATS', '2'))

# What holds a group together. 'serial_day' is the tightest grouping guaranteed
# to contain whole exams (the pkl carries no exam id); 'serial' is the
# transfer-to-a-new-customer question and is reported alongside it in gate 4.
GROUP_BY = os.environ.get('REPORT_GROUP_BY', 'serial_day')

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def rule(char='=', width=78):
    print(char * width)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# ---------------------------------------------------------------------------
# The report accumulator. Everything printed is also captured, so the JSON and
# the markdown can never disagree with the stdout — they are the same numbers
# rendered three ways rather than three computations.
# ---------------------------------------------------------------------------
REPORT = {
    'generated': datetime.now().isoformat(timespec='seconds'),
    'pkl': PKL_PATH,
    'param_set': PARAM_SET,
    'presence_flags': SEQPARAM_USE_PRESENCE_FLAGS,
    'divisor_fingerprint': SEQPARAM_DIVISOR_FINGERPRINT,
    'group_by': GROUP_BY,
    'target_mae_s': TARGET_MAE_S,
    'gates': {},
}
CHARTS = []


def save_chart(fig, name, caption):
    path = os.path.join(ANALYSIS_DIR, f"{name}.png")
    fig.savefig(path, dpi=110, bbox_inches='tight')
    CHARTS.append({'name': name, 'path': path, 'caption': caption})
    try:
        display(fig)          # noqa: F821 — Databricks builtin
    except Exception:
        pass
    plt.close(fig)
    return path


# COMMAND ----------

# =============================================================================
# LOAD
# =============================================================================

with open(PKL_PATH, 'rb') as f:
    sequences = pickle.load(f)['examination']

durations = np.array([float(s.get('total_duration', 0.0)) for s in sequences])
keep = durations > 0
sequences = [s for s, k in zip(sequences, keep) if k]
durations = durations[keep]

serials = np.array([int(s.get('serial_idx', 0)) for s in sequences])
seq_types = np.array([int(s.get('sequence_type', 0)) for s in sequences])
regions = np.array([int(s.get('body_region', 0)) for s in sequences])
binaries = np.array([s.get('sequence_binary') or '' for s in sequences])
orientations = np.array([s.get('orientation') or '' for s in sequences])
scopes = np.array([s.get('sut_scope', '') for s in sequences])
offsets = np.array([_to_float(s.get('sut_offset_s')) for s in sequences])
skipped = np.array([bool(s.get('sut_inside_skipped', False)) for s in sequences])
protocols = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])

if not (scopes == 'inside').any():
    raise RuntimeError(
        "This pkl carries no 'sut_scope' — it predates the join fix (dcdd381). "
        "Every gate below is about which rows the in-segment rule covers, so "
        "there is nothing to measure. Re-run 03_build_preprocessed_pkl.py."
    )

# Aborts end on MRI_MSR_34 and bypass the 10s floor, so they are a different
# population and would otherwise flatter every column equally.
is_abort = np.array([
    len(s.get('sequence', [])) > 0
    and int(s['sequence'][-1]) == SOURCEID_VOCAB['MRI_MSR_34']
    for s in sequences
])
clean = ~is_abort & (durations >= 10) & (durations <= 600)
inside = scopes == 'inside'

# The honest split, built once and reused by every gate so the numbers in
# different sections are comparable to each other.
GROUPS = exam_group_labels(sequences, by=GROUP_BY)
SERIAL_GROUPS = exam_group_labels(sequences, by='serial')

print(f"{len(sequences):,} sequences with a positive duration from {PKL_PATH}")
print(f"grouping by {GROUP_BY}: {len(set(GROUPS)):,} groups, "
      f"{len(set(SERIAL_GROUPS))} scanners")

# COMMAND ----------

# =============================================================================
# GATE 1 — BUILD HEALTH
#
# Coverage is upstream of every parameter question. On a row with no in-segment
# SUT event, every parameter is a default no matter how many parameters we
# allow, so no amount of feature work reaches it. This is the number to fix
# first and the number a silent regression would hide.
# =============================================================================

rule()
print(" GATE 1 — BUILD HEALTH")
rule()

scope_counts = Counter(scopes)
gate1 = {
    'rows_total': int(len(sequences)),
    'rows_clean': int(clean.sum()),
    'rows_abort': int(is_abort.sum()),
    'rows_out_of_range': int((~is_abort & ~clean).sum()),
    'scope_counts': {k: int(v) for k, v in scope_counts.items()},
    'coverage_pct': float(pct(inside.sum(), len(sequences))),
    'coverage_pct_clean': float(pct((clean & inside).sum(), clean.sum())),
}

print(f"\n  rows in pkl                     {gate1['rows_total']:>9,}")
print(f"  ... aborted (MRI_MSR_34)        {gate1['rows_abort']:>9,}")
print(f"  ... outside 10-600s             {gate1['rows_out_of_range']:>9,}")
print(f"  ... CLEAN, the scored population{gate1['rows_clean']:>9,}")
print(f"\n  {'join scope':<12} {'rows':>9} {'share':>8}  meaning")
rule('-')
_scope_meaning = {
    'inside': 'the SUT event fired DURING this measurement — trustworthy',
    'before': "a NEIGHBOUR's parameters, carried forward — 71.5% name a "
              "different sequence",
    'none':   'no SUT event at all — every parameter is a default',
}
for _name in ('inside', 'before', 'none'):
    _n = scope_counts.get(_name, 0)
    print(f"  {_name:<12} {_n:>9,} {pct(_n, len(sequences)):>7.1f}%  "
          f"{_scope_meaning[_name]}")

_not_inside = (~inside).sum()
gate1['adjustment_share_of_missing'] = float(pct(skipped.sum(), _not_inside))
print(f"\n  of the {_not_inside:,} rows without an in-segment join, "
      f"{skipped.sum():,} ({gate1['adjustment_share_of_missing']:.1f}%) had one "
      f"that was\n  an ADJUSTMENT — those segments are probably the adjustment "
      f"itself and belong\n  in the population filter rather than counted "
      f"against the join.")

# Sampler-cell thinness. Step 07 must SYNTHESISE every parameter it conditions
# on, drawing a joint tuple per (body_region, sequence_type) from real rows.
# Joint drawing means the WIDTH of the vector is free — a whole real row is
# copied, so correlations survive however many fields there are. What is not
# free is a thin cell: SUTParameterSampler's empty-pool fallback returns 0.0,
# which would put a fabricated parameter into synthetic output.
cell_counts = Counter(zip(regions[clean & inside].tolist(),
                          seq_types[clean & inside].tolist()))
_thin = sorted((n, r, t) for (r, t), n in cell_counts.items())[:8]
gate1['sampler_cells'] = int(len(cell_counts))
gate1['sampler_thinnest'] = [
    {'body_region': int(r), 'sequence_type': int(t), 'rows': int(n)}
    for n, r, t in _thin
]
print(f"\n  step-07 sampler pools: {len(cell_counts)} (body_region, sequence_type) "
      f"cells")
print(f"  thinnest: " + ", ".join(f"({r},{t})={n}" for n, r, t in _thin[:6]))
if _thin and _thin[0][0] < 10:
    print("  !! A cell under ~10 rows cannot support a joint draw. Step 07's "
          "empty-pool\n     fallback writes 0.0, which is a fabricated "
          "parameter in synthetic output.")

REPORT['gates']['build_health'] = gate1

# --- CHART 1: coverage waterfall -------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
_stages = ['in pkl', 'non-abort', 'clean\n(10-600s)', 'clean +\nin-segment']
_values = [len(sequences), int((~is_abort).sum()), int(clean.sum()),
           int((clean & inside).sum())]
_bars = ax.bar(_stages, _values, color=['#4C78A8', '#4C78A8', '#4C78A8', '#54A24B'])
for _b, _v in zip(_bars, _values):
    ax.text(_b.get_x() + _b.get_width() / 2, _v, f"{_v:,}\n{pct(_v, len(sequences)):.0f}%",
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('sequences')
ax.set_title('Where the corpus goes — the green bar is what parameters can reach')
ax.set_ylim(0, max(_values) * 1.18)
ax.spines[['top', 'right']].set_visible(False)
save_chart(fig, 'coverage_waterfall',
           'Every parameter question is capped by the green bar. Rows outside '
           'it get defaults no matter how many fields we allow.')

# COMMAND ----------

# =============================================================================
# GATE 2 — JOIN INTEGRITY
#
# A REGRESSION TEST, not an investigation. 03c settled this: the in-segment rule
# agrees with the actually-scanned sequence on 100.0% of testable rows. If that
# ever falls, every number in gate 4 is describing a different scan than the one
# it is scored against, and nothing else in this report would notice.
# =============================================================================

rule()
print(" GATE 2 — JOIN INTEGRITY")
rule()

# The SUT message names its own sequence in DLL; the MSR_100 message names it in
# `sequence_type`. Classifying the DLL string with the repo's own classifier and
# comparing is a test with no hand-made mapping in it.
testable = inside & (binaries != '')
agree = np.array([
    classify_sequence_type(b) == int(t)
    for b, t in zip(binaries[testable], seq_types[testable])
])
gate2 = {
    'testable_rows': int(testable.sum()),
    'agreement_pct': float(pct(agree.sum(), testable.sum())),
}
print(f"\n  {gate2['testable_rows']:,} in-segment rows name a sequence in both "
      f"messages")
print(f"  agreement: {gate2['agreement_pct']:.1f}%   (was 100.0% at the "
      f"2026-08-05 run)")

# The same test on the fallback rows, which is what says the fallback is not
# merely noisy but wrong.
_fb = (scopes == 'before') & (binaries != '')
if _fb.any():
    _fb_agree = np.array([
        classify_sequence_type(b) == int(t)
        for b, t in zip(binaries[_fb], seq_types[_fb])
    ])
    gate2['fallback_agreement_pct'] = float(pct(_fb_agree.sum(), _fb.sum()))
    print(f"  fallback ('before') rows:  {gate2['fallback_agreement_pct']:.1f}% "
          f"agreement over {_fb.sum():,} rows")
    print(f"  → the gap between those two is why `sut_in_segment` is a feature "
          f"rather than\n    a filter: the model can learn to discount the "
          f"second population.")

if gate2['agreement_pct'] < 99.0:
    print("\n  !! REGRESSION. The in-segment join no longer describes the scan "
          "it is\n     attached to. Stop here — gate 4's numbers are not about "
          "these sequences.")

REPORT['gates']['join_integrity'] = gate2

# COMMAND ----------

# =============================================================================
# GATE 3 — FIELD ADMISSIBILITY
#
# "How many parameters are we using, and why?" answered from the same code that
# made the decision — config.classify_seqparam_field — rather than from a
# comment that can drift away from it.
# =============================================================================

rule()
print(" GATE 3 — FIELD ADMISSIBILITY")
rule()

inventory = numeric_field_inventory(sequences)
field_rows = []
for row in inventory:
    name = seqparam_stable_name(row['key'])
    stats = {'presence_pct': row['presence_pct'], 'numeric_pct': row['numeric_pct']}
    ok, why = classify_seqparam_field(name, stats)
    field_rows.append({
        'name': name,
        'raw_key': row['key'],
        'admissible': bool(ok),
        'reason': why,
        'presence_pct': row['presence_pct'],
        'numeric_pct': row['numeric_pct'],
        'distinct': row['distinct'],
        'p50': row['p50'],
        'p99': row['p99'],
        'curated': name in SEQPARAM_CANDIDATES,
        'in_selected_set': name in EXAMINATION_SEQPARAM_FEATURES,
        'watchlisted': row['key'] in SUT_SUSPECT_WATCHLIST,
    })

admitted = [r for r in field_rows if r['admissible']]
excluded = [r for r in field_rows if not r['admissible']]
gate3 = {
    'fields_seen': len(field_rows),
    'admitted': len(admitted),
    'excluded': len(excluded),
    'selected': len([r for r in admitted if r['in_selected_set']]),
    'unidentified': [r['name'] for r in admitted if not r['curated']],
    'fields': field_rows,
}

print(f"\n  {gate3['fields_seen']} distinct fields in the messages")
print(f"  {gate3['admitted']} admissible  |  {gate3['excluded']} excluded")
print(f"\n  EXCLUDED — with the reason, because this is the question that keeps "
      f"being asked:")
rule('-')
for r in sorted(excluded, key=lambda r: r['name']):
    print(f"  {r['name']:<16} {r['reason']}")
rule('-')

_wl = [r for r in admitted if r['watchlisted']]
if _wl:
    print(f"\n  ON THE WATCHLIST but ADMITTED ({len(_wl)}) — flagged for review, "
          f"not blocked.\n  Gate 5 is what settles each of them with a number:")
    for r in sorted(_wl, key=lambda r: -r['presence_pct']):
        print(f"    {r['name']:<10} {r['presence_pct']:>5.1f}% present   "
              f"{SUT_SUSPECT_WATCHLIST[r['raw_key']]}")

if gate3['unidentified']:
    print(f"\n  {len(gate3['unidentified'])} admitted fields have no entry in "
          f"SUT_FIELD_MAP — we USE them but\n  cannot say what they mean. That "
          f"is fine for training and NOT fine for step 07,\n  which has to "
          f"synthesise every field it conditions on:")
    print("    " + ", ".join(sorted(gate3['unidentified'])[:24]))

REPORT['gates']['admissibility'] = gate3

# --- CHART 2: admissibility map --------------------------------------------
_ordered = sorted(field_rows, key=lambda r: (-r['presence_pct'], r['name']))
fig, ax = plt.subplots(figsize=(9, max(5, len(_ordered) * 0.16)))
_colours, _labels = [], []
for r in _ordered:
    if not r['admissible']:
        _colours.append('#E45756' if 'denied' in r['reason'] else '#BAB0AC')
    elif r['curated']:
        _colours.append('#54A24B')
    else:
        _colours.append('#4C78A8')
    _labels.append(r['name'])
ax.barh(range(len(_ordered)), [r['presence_pct'] for r in _ordered],
        color=_colours)
ax.set_yticks(range(len(_ordered)))
ax.set_yticklabels(_labels, fontsize=6)
ax.invert_yaxis()
ax.set_xlabel('% of rows carrying the field')
ax.set_title(f"Every field, and why it is or is not in the model "
             f"({gate3['admitted']} of {gate3['fields_seen']} admitted)")
ax.axvline(SEQPARAM_MIN_PRESENCE_PCT, color='#666', ls=':', lw=1)
_legend = [
    plt.Rectangle((0, 0), 1, 1, color='#54A24B'),
    plt.Rectangle((0, 0), 1, 1, color='#4C78A8'),
    plt.Rectangle((0, 0), 1, 1, color='#E45756'),
    plt.Rectangle((0, 0), 1, 1, color='#BAB0AC'),
]
ax.legend(_legend, ['admitted, hand-calibrated', 'admitted, discovered',
                    'DENIED (target / identity / planner-derived)',
                    'excluded (not numeric, or below the presence floor)'],
          fontsize=7, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
save_chart(fig, 'field_admissibility',
           'The answer to "how many parameters and why", one row per field.')

# COMMAND ----------

# =============================================================================
# GATE 4 — VALUE
#
# Do the parameters beat the protocol oracle Görtler set as the benchmark, and
# do they clear his +-15s bar — on a split that supports the claim being made?
#
# Everything here is scored on the GROUPED split. Where the random-split number
# is also shown, the GAP is the finding: it is how much of the historical score
# was one exam's siblings telling the model the answer.
# =============================================================================

rule()
print(" GATE 4 — VALUE")
rule()

score_rows = clean & inside
print(f"\n  scored on {score_rows.sum():,} clean in-segment rows, grouped by "
      f"{GROUP_BY}")

_admitted_names = [r['name'] for r in admitted]
_raw_keys = [r['raw_key'] for r in admitted]
parameters = numeric_matrix(sequences, _raw_keys)

_y = durations[score_rows]
_g = GROUPS[score_rows]
_gs = SERIAL_GROUPS[score_rows]


def score(features, groups, label):
    r2, mae = heldout_regressor_score(features, _y, repeats=REPEATS,
                                      groups=groups)
    return {'label': label, 'r2_pct': round(r2, 1), 'mae_s': round(mae, 1)}


# The benchmark. Scored on the SAME split as the parameters — scoring the oracle
# on a random split and the parameters on a grouped one would hand the oracle an
# advantage that is purely an artefact of how the rows were divided.
_, oracle_mae_g, oracle_cov_g = heldout_group_r2(
    protocols[score_rows], _y, groups=_g)
_, oracle_mae_r, oracle_cov_r = heldout_group_r2(protocols[score_rows], _y)

print(f"\n  {'configuration':<44} {'R2':>7} {'MAE':>8} {'vs oracle':>10} "
      f"{'vs +-15s':>9}")
rule('-')
print(f"  {'serial + protocol (ORACLE, grouped)':<44} {'':>7} "
      f"{oracle_mae_g:>7.1f}s {'—':>10} {oracle_mae_g - TARGET_MAE_S:>+8.1f}s")

configs = []
_name_to_raw = {r['name']: r['raw_key'] for r in admitted}


def columns_for(names):
    idx = [_admitted_names.index(n) for n in names if n in _admitted_names]
    return parameters[np.ix_(np.where(score_rows)[0], idx)] if idx else None


_variants = [('all admissible parameters', _admitted_names)]
for _set_name in ('luke', 'navneet'):
    _present = [n for n in PARAM_SETS[_set_name] if n in _admitted_names]
    if _present:
        _variants.append((f"PARAM_SET={_set_name!r} ({len(_present)} fields)",
                          _present))

for _label, _names in _variants:
    _cols = columns_for(_names)
    if _cols is None:
        continue
    _row = score(_cols, _g, _label)
    configs.append(_row)
    print(f"  {_label:<44} {_row['r2_pct']:>6.1f}% {_row['mae_s']:>7.1f}s "
          f"{_row['mae_s'] - oracle_mae_g:>+9.1f}s "
          f"{_row['mae_s'] - TARGET_MAE_S:>+8.1f}s")

# The price tag on the planner-derived denylist. A denylist entry that cannot
# show what it costs is an assertion; one that can is a decision.
_denied_present = [r for r in excluded
                   if r['raw_key'] in SUT_PLANNER_DERIVED_DENYLIST]
if _denied_present:
    _with = _admitted_names + [r['name'] for r in _denied_present]
    _with_keys = _raw_keys + [r['raw_key'] for r in _denied_present]
    _with_cols = numeric_matrix(sequences, _with_keys)[score_rows]
    _row = score(_with_cols, _g, 'all + planner-derived (SNR) — NOT SHIPPED')
    configs.append(_row)
    _cost = [c for c in configs if c['label'].startswith('all admissible')][0]['mae_s'] - _row['mae_s']
    print(f"  {_row['label']:<44} {_row['r2_pct']:>6.1f}% {_row['mae_s']:>7.1f}s "
          f"{_row['mae_s'] - oracle_mae_g:>+9.1f}s "
          f"{_row['mae_s'] - TARGET_MAE_S:>+8.1f}s")
    print(f"\n  → the planner-derived denylist costs {_cost:+.1f}s of MAE. That "
          f"is its price tag.\n    If it is large AND gate 5 shows no "
          f"random-vs-grouped divergence for SNR,\n    the entry should be "
          f"removed from SUT_PLANNER_DERIVED_DENYLIST deliberately.")
    REPORT['planner_denylist_cost_s'] = round(float(_cost), 1)

# What the honest split actually cost us, stated once.
_all_cols = columns_for(_admitted_names)
_random = score(_all_cols, None, 'all admissible parameters (RANDOM split)')
rule('-')
print(f"  {_random['label']:<44} {_random['r2_pct']:>6.1f}% "
      f"{_random['mae_s']:>7.1f}s {_random['mae_s'] - oracle_mae_r:>+9.1f}s "
      f"{_random['mae_s'] - TARGET_MAE_S:>+8.1f}s")
_grouped_mae = configs[0]['mae_s']
_optimism = _grouped_mae - _random['mae_s']
if _optimism > 0.5:
    print(f"\n  → the random split flatters the model by {_optimism:.1f}s. That "
          f"difference was never\n    real accuracy — it was one exam's "
          f"siblings sitting in the training set, and\n    every number "
          f"03b-03f reported carried it.")
elif _optimism < -0.5:
    print(f"\n  → the grouped split scores {abs(_optimism):.1f}s BETTER than the "
          f"random one, which is\n    not the expected direction. With few "
          f"groups that is usually variance rather\n    than a finding — check "
          f"the group count above before reading anything into it.")
else:
    print(f"\n  → the two splits agree to within {abs(_optimism):.1f}s. The "
          f"model is not leaning on\n    within-group memorisation, which is "
          f"the result we want and did not have before.")

# Transfer to a scanner we have never seen — the actual onboarding question,
# and on the 2026-08-08 run the single worst number in the report.
_serial_row = score(_all_cols, _gs, 'all admissible (held out by SCANNER)')
configs.append(_serial_row)
print(f"  {_serial_row['label']:<44} {_serial_row['r2_pct']:>6.1f}% "
      f"{_serial_row['mae_s']:>7.1f}s")

_transfer_gap = _serial_row['mae_s'] - _grouped_mae
if _transfer_gap > 2.0:
    print(f"""
  !! TRANSFER GAP: {_grouped_mae:.1f}s within known scanners, {_serial_row['mae_s']:.1f}s on a scanner the
     model has never seen — {_transfer_gap:+.1f}s. This is the number that decides whether
     the model can be pointed at a new customer, and it is the one a
     session-grouped split cannot see. Everything the parameters buy on familiar
     scanners is spent again on an unfamiliar one.

     With {len(set(SERIAL_GROUPS))} serials in the corpus this split is coarse and noisy, which is
     a fact about the data rather than the method — and an argument for
     onboarding more customers rather than for discounting the number.""")

gate4 = {
    'transfer_mae_s_by_scanner': _serial_row['mae_s'],
    'transfer_gap_s': round(float(_transfer_gap), 1),
    'scored_rows': int(score_rows.sum()),
    'oracle_mae_s_grouped': round(float(oracle_mae_g), 1),
    'oracle_coverage_pct_grouped': round(float(oracle_cov_g), 1),
    'oracle_mae_s_random': round(float(oracle_mae_r), 1),
    'oracle_coverage_pct_random': round(float(oracle_cov_r), 1),
    'configurations': configs,
    'random_split_mae_s': _random['mae_s'],
    'split_optimism_s': round(float(_grouped_mae - _random['mae_s']), 1),
}

print(f"\n  ORACLE COVERAGE: {oracle_cov_g:.1f}% grouped vs {oracle_cov_r:.1f}% "
      f"random. Protocols are\n  site-specific, so holding out whole sessions "
      f"strands them and the oracle falls\n  back to the global mean. That drop "
      f"IS Görtler's objection, measured.")

# COMMAND ----------

# --- the 1% cases: one model, errors partitioned by body group -------------
# Fitting per group would train ten small models and measure their smallness.
# One model, held out once, errors split afterwards, is the thing that answers
# "does this work for the rare body group".

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

_is_train = holdout_mask(int(score_rows.sum()), 0.2, np.random.default_rng(0), _g)
REPORT['split'] = split_summary(_is_train, _g)

_model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
_model.fit(_all_cols[_is_train], _y[_is_train])
_pred = _model.predict(_all_cols[~_is_train])
_err = np.abs(_y[~_is_train] - _pred)
_held_regions = regions[score_rows][~_is_train]

_region_names = {i: r for i, r in enumerate(BODY_REGIONS)}
per_region = []
for _r in sorted(set(_held_regions.tolist())):
    _m = _held_regions == _r
    per_region.append({
        'body_region': _region_names.get(int(_r), str(_r)),
        'rows': int(_m.sum()),
        'share_pct': round(float(pct(_m.sum(), _m.size)), 1),
        'mae_s': round(float(_err[_m].mean()), 1),
    })
per_region.sort(key=lambda r: r['rows'])

rule()
print(" GATE 4b — PER BODY GROUP (the extraordinary 1%)")
rule()
print(f"\n  {'body region':<12} {'rows':>7} {'share':>7} {'MAE':>8}   "
      f"rarest first")
rule('-')
for _r in per_region:
    _flag = '  <-- thin' if _r['rows'] < 100 else ''
    print(f"  {_r['body_region']:<12} {_r['rows']:>7,} {_r['share_pct']:>6.1f}% "
          f"{_r['mae_s']:>7.1f}s{_flag}")
rule('-')
print("  A single overall MAE cannot see these. The goal is the rare rows, and "
      "the\n  rare rows are the top of this table.")

_q = [50, 75, 90, 95, 99]
_quantiles = {f'p{n}': round(float(np.percentile(_err, n)), 1) for n in _q}
print(f"\n  error quantiles: " + "  ".join(f"p{n}={_quantiles[f'p{n}']}s" for n in _q))
print(f"  {pct((_err <= TARGET_MAE_S).sum(), _err.size):.1f}% of held-out rows "
      f"land within +-{TARGET_MAE_S:.0f}s.")

gate4['per_body_region'] = per_region
gate4['error_quantiles_s'] = _quantiles
gate4['within_target_pct'] = round(float(pct((_err <= TARGET_MAE_S).sum(), _err.size)), 1)
REPORT['gates']['value'] = gate4

# --- CHART 3: MAE vs field count -------------------------------------------
ranking_grouped = permutation_importance_mae(
    _all_cols, _y, field_names=_admitted_names, groups=_g)
ranked_names = [r['name'] for r in ranking_grouped]

curve = []
for _k in (3, 5, 8, 12, 20, 30, 50, len(ranked_names)):
    if _k > len(ranked_names):
        continue
    _cols = columns_for(ranked_names[:_k])
    _r2, _mae = heldout_regressor_score(_cols, _y, repeats=REPEATS, groups=_g)
    curve.append({'fields': _k, 'r2_pct': round(_r2, 1), 'mae_s': round(_mae, 1)})
gate4['field_count_curve'] = curve

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot([c['fields'] for c in curve], [c['mae_s'] for c in curve],
        marker='o', color='#4C78A8', label='parameters (grouped split)')
ax.axhline(oracle_mae_g, color='#E45756', ls='--',
           label=f'protocol oracle ({oracle_mae_g:.1f}s)')
ax.axhline(TARGET_MAE_S, color='#54A24B', ls=':',
           label=f'Görtler +-{TARGET_MAE_S:.0f}s bar')
ax.set_xlabel('number of parameter fields (best-first)')
ax.set_ylabel('held-out MAE (s)')
ax.set_title('Does more actually help? — the Görtler question, measured')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
save_chart(fig, 'mae_vs_field_count',
           'Where the curve flattens is what a shorter list costs. Below the '
           'red line beats the protocol name.')

# --- CHART 4: per body group -----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
_names = [r['body_region'] for r in per_region]
_maes = [r['mae_s'] for r in per_region]
_cols_ = ['#E45756' if r['rows'] < 100 else '#4C78A8' for r in per_region]
ax.barh(_names, _maes, color=_cols_)
ax.axvline(oracle_mae_g, color='#E45756', ls='--', lw=1,
           label=f'oracle {oracle_mae_g:.1f}s')
ax.axvline(TARGET_MAE_S, color='#54A24B', ls=':', lw=1,
           label=f'+-{TARGET_MAE_S:.0f}s')
ax.set_xlabel('held-out MAE (s)')
ax.set_title('Per body group, rarest at the top (red = under 100 held-out rows)')
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
save_chart(fig, 'mae_per_body_group',
           'The 1%-cases chart. An overall MAE averages these away.')

# --- CHART 5: error quantiles ----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
_xs = np.linspace(0, 100, 201)
ax.plot(_xs, np.percentile(_err, _xs), color='#4C78A8')
ax.axhline(TARGET_MAE_S, color='#54A24B', ls=':',
           label=f'+-{TARGET_MAE_S:.0f}s bar')
ax.fill_between(_xs, 0, np.percentile(_err, _xs),
                where=np.percentile(_err, _xs) <= TARGET_MAE_S,
                color='#54A24B', alpha=0.15)
ax.set_xlabel('percentile of held-out rows')
ax.set_ylabel('absolute error (s)')
ax.set_title(f"Where the tail is — {gate4['within_target_pct']:.0f}% of rows "
             f"are inside the bar")
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
save_chart(fig, 'error_quantiles',
           '85% -> 99% is a claim about the right-hand side of this curve.')

# COMMAND ----------

# =============================================================================
# GATE 5 — LEAKAGE SENTINEL
#
# 03d caught MUID by suspecting one field and testing it by hand. That does not
# scale to 74 fields, and it only ever catches the ones somebody thought to
# suspect. This is the mechanical version.
#
# THE TEST: an identifier scores a large permutation importance on a RANDOM
# split (its training rows tell it the answer) and near zero on a GROUPED one
# (the held-out group's id was never seen). A real physics parameter scores
# about the same either way, because physics transfers and identity does not.
# The DIVERGENCE is the signal, not the magnitude.
# =============================================================================

rule()
print(" GATE 5 — LEAKAGE SENTINEL")
rule()

ranking_random = permutation_importance_mae(
    _all_cols, _y, field_names=_admitted_names)
_random_by_name = {r['name']: r['importance_s'] for r in ranking_random}

# RETENTION, NOT DIFFERENCE. The first version of this gate ranked by
# (random - grouped) in seconds and flagged anything above 1.0s, which put TR at
# the top of the suspect list on the 2026-08-08 run: 17.24s random, 12.93s
# grouped. TR is repetition time. It is the most causal parameter in the
# message, and it was flagged purely for being important — a big number shrinks
# by a big absolute amount.
#
# What an identifier actually looks like is losing nearly ALL of its importance
# when the group it memorised is held out. So the signal is the FRACTION
# retained, and it is only meaningful for a field that had something to lose.
_MEANINGFUL_S = 0.5

sentinel = []
for r in ranking_grouped:
    _rand = _random_by_name.get(r['name'], 0.0)
    _grouped = r['importance_s']
    _retained = (_grouped / _rand) if _rand > _MEANINGFUL_S else None
    sentinel.append({
        'name': r['name'],
        'importance_grouped_s': round(float(_grouped), 3),
        'importance_random_s': round(float(_rand), 3),
        'retained_pct': None if _retained is None else round(100.0 * _retained, 1),
        'divergence_s': round(float(_rand - _grouped), 3),
        'watchlisted': r['name'] in {seqparam_stable_name(k)
                                     for k in SUT_SUSPECT_WATCHLIST},
    })

# A CONVICTION NEEDS BOTH a low retention AND something actually at stake. The
# 2026-08-10 run flagged UI0 and TOM at 24.7% and 30.7% retained — on total
# importances of 0.13s and 0.21s against a 9.7s MAE. Losing three quarters of
# nothing is not evidence of anything, and a gate that cries leak over 0.4s
# trains the reader to ignore it.
_LOSS_S = 1.0

_scored = [s for s in sentinel if s['retained_pct'] is not None]
_suspects = sorted(_scored, key=lambda s: s['retained_pct'])[:10]
print(f"\n  {'field':<16} {'grouped':>10} {'random':>10} {'lost':>9} "
      f"{'retained':>10}  note")
rule('-')
for s in _suspects:
    _lost = s['importance_random_s'] - s['importance_grouped_s']
    _flag = ('  <-- CHECK' if s['retained_pct'] < 35.0 and _lost > _LOSS_S
             else '')
    print(f"  {s['name']:<16} {s['importance_grouped_s']:>+9.2f}s "
          f"{s['importance_random_s']:>+9.2f}s {_lost:>8.2f}s "
          f"{s['retained_pct']:>9.1f}%{_flag}")
rule('-')
print(f"""
  RETAINED is the share of a field's importance that SURVIVES holding out whole
  groups. A physics parameter keeps most of it, because physics transfers. An
  identifier keeps almost none, because the held-out group's id was never seen.

  A CHECK needs both: under 35% retained AND more than {_LOSS_S:.1f}s actually lost. A
  field with nothing to lose cannot lose it informatively, and low retention on
  a 0.2s field is noise wearing a leak's clothes.

  The test to apply, the same one that convicted ST: is this a re-expression of
  the answer, or a cause of it? A cause stays in however strongly it correlates.
  Move a convicted field to the matching denylist in config.py — never route
  around the guard.""")

print(f"\n  TOP OF THE RANKING (grouped — what actually earns its place):")
for i, r in enumerate(ranking_grouped[:12], 1):
    _note = SUT_SUSPECT_WATCHLIST.get(r['name'], '')
    print(f"    {i:>2}  {r['name']:<16} {r['importance_s']:>+7.2f}s   {_note}")

_dead = [s['name'] for s in sentinel if abs(s['importance_grouped_s']) < 0.05]
print(f"\n  {len(_dead)} of {len(sentinel)} fields move the held-out MAE by less "
      f"than 0.05s.\n  Under 'pass everything' that is FINE — a presence-flagged "
      f"field the model\n  ignores costs two columns, and one of them may be the "
      f"field that carries a\n  rare body group. It is reported, not pruned.")

REPORT['gates']['leakage_sentinel'] = {
    'ranking': sentinel,
    'dead_fields': _dead,
    'top_divergence': _suspects,
}

# COMMAND ----------

# =============================================================================
# THE TWO OUTPUTS
# =============================================================================

_json_path = os.path.join(ANALYSIS_DIR, 'parameter_report.json')
with open(_json_path, 'w') as f:
    json.dump(REPORT, f, indent=2, sort_keys=True, default=str)

_n_values = gate3['admitted']                       # what the corpus offers
_n_selected = len(EXAMINATION_SEQPARAM_FEATURES)    # the whole vector, flags included
# What is actually in the model, which is NOT the same as what is admissible:
# config resolves PARAM_SET against the divisor table on disk, and on the first
# run after a schema change that table is one build behind.
_n_value_cols = len([n for n in EXAMINATION_SEQPARAM_FEATURES
                     if not is_presence_name(n) and n not in SEQPARAM_DERIVED])

# The headline must be the configuration we SHIP, not the best number on the
# table. `min(configs)` would quote the "all + planner-derived (SNR)" row, which
# is measured precisely so it can be refused — headlining it would put a number
# in front of Görtler that no deployed model can reproduce.
_shipped = next(c for c in configs if c['label'].startswith('all admissible p'))
_short = [c for c in configs if 'PARAM_SET' in c['label']]

_md = [
    f"# Parameter report",
    "",
    f"*{REPORT['generated']} · `PARAM_SET={PARAM_SET}` · "
    f"{gate1['rows_total']:,} sequences · grouped by `{GROUP_BY}`*",
    "",
    "## The short answer",
    "",
    f"**The model is conditioned on {_n_value_cols} parameters.** Each one is "
    f"accompanied by a presence flag saying whether the scanner actually "
    f"reported it, plus one flag saying whether the parameters belong to this "
    f"scan or to a neighbour's — so the conditioning vector is "
    f"{_n_selected} columns wide.",
    "",
    f"The messages carry {gate3['fields_seen']} distinct fields, of which "
    f"**{_n_values} are admissible**; {gate3['excluded']} are excluded and every "
    f"exclusion has a stated reason (below).",
    "",
]
if _n_value_cols != _n_values:
    _md += [
        f"> **These two numbers disagree, and that is a finding, not a rounding.** "
        f"`PARAM_SET={PARAM_SET}` resolved to {_n_value_cols} parameters from a "
        f"divisor table that predates this build, while this corpus offers "
        f"{_n_values}. The pkl carries both, so training is safe — but to "
        f"actually use all {_n_values}, reload the config (the table is written "
        f"at the end of step 03) and re-run step 04.",
        "",
    ]
_md += [
    f"On {gate4['scored_rows']:,} clean, in-segment rows the parameters reach "
    f"**{_shipped['mae_s']:.1f}s MAE** against the protocol oracle's "
    f"**{gate4['oracle_mae_s_grouped']:.1f}s** and Görtler's "
    f"**±{TARGET_MAE_S:.0f}s** bar. "
    f"{gate4['within_target_pct']:.1f}% of held-out rows land inside the bar.",
    "",
    (f"**On a scanner the model has never seen, that becomes "
     f"{gate4['transfer_mae_s_by_scanner']:.1f}s** "
     f"({gate4['transfer_gap_s']:+.1f}s). Everything the parameters buy on "
     f"familiar scanners is spent again on an unfamiliar one, and onboarding a "
     f"new customer is the unfamiliar case. This is the number to fix next."
     if gate4['transfer_gap_s'] > 2.0 else
     f"On a scanner the model has never seen it holds at "
     f"{gate4['transfer_mae_s_by_scanner']:.1f}s "
     f"({gate4['transfer_gap_s']:+.1f}s), so nothing here depends on having "
     f"seen the site before — which is what onboarding a new customer needs."),
    "",
    "## Why not fewer",
    "",
    "| fields | MAE | vs oracle |",
    "|---:|---:|---:|",
]
for c in curve:
    _md.append(f"| {c['fields']} | {c['mae_s']:.1f}s | "
               f"{c['mae_s'] - gate4['oracle_mae_s_grouped']:+.1f}s |")
_md += [
    "",
    "Where that curve flattens is what a shorter list costs. The hand-picked "
    "sets are the control group:",
    "",
    "| configuration | fields | MAE |",
    "|---|---:|---:|",
    f"| all admissible (SHIPPED) | {_n_values} | {_shipped['mae_s']:.1f}s |",
]
for _set_name in ('luke', 'navneet'):
    _c = next((c for c in _short if _set_name in c['label']), None)
    if _c:
        _n = len([n for n in PARAM_SETS[_set_name] if n in _admitted_names])
        _md.append(f"| PARAM_SET=`{_set_name}` | {_n} | {_c['mae_s']:.1f}s |")
_md += [
    "",
    "## Why not more",
    "",
    f"{gate3['excluded']} fields are held out, in three categories, and none of "
    f"them is feature selection — they are the difference between a real number "
    f"and a fake one:",
    "",
]
for r in sorted(excluded, key=lambda r: r['name']):
    _md.append(f"- **`{r['name']}`** — {r['reason']}")
if 'planner_denylist_cost_s' in REPORT:
    _md += [
        "",
        f"The planner-derived exclusion carries a measured price tag: allowing "
        f"it back in would change MAE by "
        f"**{REPORT['planner_denylist_cost_s']:+.1f}s**. That is the number to "
        f"weigh against the risk, rather than an assertion either way.",
    ]
_md += [
    "",
    "## Is it working, for the cases we care about",
    "",
    "A single overall MAE cannot see the extraordinary 1%. Per body group, "
    "rarest first:",
    "",
    "| body group | held-out rows | MAE |",
    "|---|---:|---:|",
]
for r in per_region:
    _md.append(f"| {r['body_region']} | {r['rows']:,} | {r['mae_s']:.1f}s |")
_md += [
    "",
    f"Error quantiles: " + ", ".join(
        f"p{n} = {_quantiles[f'p{n}']}s" for n in _q) + ".",
    "",
    "## What is blocking",
    "",
    (f"1. **Transfer to a new scanner — "
     f"{gate4['transfer_mae_s_by_scanner']:.1f}s against {_shipped['mae_s']:.1f}s "
     f"within known ones.** No parameter closes this, because the gap is not "
     f"about which parameters we have; it is about the model never having seen "
     f"the site. It is the number that decides whether the pipeline can be "
     f"pointed at a new customer."
     if gate4['transfer_gap_s'] > 2.0 else
     f"1. **Transfer to a new scanner is holding** "
     f"({gate4['transfer_mae_s_by_scanner']:.1f}s against "
     f"{_shipped['mae_s']:.1f}s within known ones). Worth re-checking on every "
     f"build — with few serials this split is noisy."),
    f"2. **Coverage — {gate1['coverage_pct_clean']:.1f}%.** On the other "
    f"{100 - gate1['coverage_pct_clean']:.1f}% of clean rows there is no "
    f"in-segment parameter message, so every field is a default no matter how "
    f"many we allow. No additional parameter closes this either.",
    (f"3. **The honest split costs {gate4['split_optimism_s']:+.1f}s.** Every "
     f"MAE reported before 2026-08-07 used a random row split, which let one "
     f"exam's siblings sit in training. That gap was never accuracy."
     if gate4['split_optimism_s'] > 0.5 else
     f"3. **The split convention is no longer costing anything** "
     f"({gate4['split_optimism_s']:+.1f}s between the grouped and random "
     f"splits). The model is not leaning on within-group memorisation."),
]
if gate3['unidentified']:
    _md.append(
        f"4. **{len(gate3['unidentified'])} admitted fields are unidentified.** "
        f"Fine for training — the model does not need to know what a field "
        f"means. Not fine for step 07, which must synthesise every field it "
        f"conditions on: "
        + ", ".join(f"`{n}`" for n in sorted(gate3['unidentified'])[:12])
        + ".")
if gate1['sampler_thinnest'] and gate1['sampler_thinnest'][0]['rows'] < 10:
    _md.append(
        f"5. **Thin sampler cells.** The smallest (body_region, sequence_type) "
        f"pool has {gate1['sampler_thinnest'][0]['rows']} rows. Step 07's "
        f"empty-pool fallback writes 0.0, which puts a fabricated parameter "
        f"into synthetic output.")

_md += ["", "## Charts", ""]
for c in CHARTS:
    _md.append(f"- **{c['name']}** — {c['caption']}  \n  `{c['path']}`")
_md += [
    "",
    "---",
    "",
    f"Full detail — every field, percentile and per-group number — is in "
    f"`{_json_path}`. Settled findings from the 03b–03f investigation are in "
    f"`FINDINGS_parameter_investigation.md`.",
    "",
]

_md_path = os.path.join(ANALYSIS_DIR, 'parameter_report.md')
with open(_md_path, 'w') as f:
    f.write("\n".join(_md))

rule()
print(" OUTPUTS")
rule()
print(f"\n  human  → {_md_path}")
print(f"  machine→ {_json_path}")
print(f"  charts → {ANALYSIS_DIR}/*.png  ({len(CHARTS)} written)")

try:
    displayHTML("<pre>" + "\n".join(_md).replace('<', '&lt;') + "</pre>")  # noqa: F821
except Exception:
    print("\n" + "\n".join(_md))
