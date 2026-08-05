# Databricks notebook source
"""
03f_coverage_and_masking.py — the join is right where it fires. What about the rest?

NO TRAINING, NO SPARK. Reads the same pkl 03b/03c/03d/03e read. ~10 minutes.

WHY THIS EXISTS
---------------
The 2026-08-05 run closed the join question and opened a coverage one.

03c section A2: the in-segment rule now agrees with the actual scanned sequence
on 100.0% of the 36,807 testable rows it covers (was 94.8% before the AdjGreSeq
skip). Isolated to the rows where the old and new rules disagree: old 0.5%, new
100.0%. There is nothing left to fix in the rule itself.

03e section 2, on those rows: the 89-field admissible parameter vector scores
9.7s MAE against the protocol oracle's 13.2s. That is Görtler's thesis
confirmed — executed parameters BEAT the customer protocol name, using nothing
customer-specific, and clear his +-15s bar by 5.3s.

But the rule only fires on 80.3% of segments. The other 19.7% — 9,607 'before'
and 509 'none' — fall back to the most recent SUT event preceding the segment,
which 03c scores at 28.5% agreement. Across ALL 49,514 clean rows the same
vector is 24.6s against an oracle of 11.0s. So the gap between "we beat the
oracle" and "we lose to it by 13.6s" is entirely which rows have a parameter
message, and no additional field can close it.

Two questions follow, and this notebook answers both without a rebuild:

  A  WHO are the rows the join misses, and can any be recovered cheaply?
  B  On those rows, is a stale neighbour's parameters WORSE than no parameters?
     71.5% of them name a different sequence than the one that ran, so the model
     is being handed numbers that are not merely noisy but wrong. NaN is an
     option the flat conditioning tensor also has (SEQPARAM_MISSING_DEFAULTS).

And one that the 9.7s makes urgent:

  C  WHICH of the 89 fields earn their place — and is any of them a leak?

Section C is not an optimisation. 89 fields cannot ship: step 07 must synthesise
every feature the model conditions on, and 89 scaled numerics on the flat
conditioning tensor is exactly the LayerNorm-erasure condition config.py's
divisor comments warn about. It is also the audit that 03d ran for MUID by hand,
run over the whole vector this time — R2 94.8% is high enough that a field which
is a duration in disguise has to be ruled out before the number leaves the team.

CAREFUL WITH THE OLD NUMBERS
----------------------------
03b section 5, 03c section D and 03d section 3 all still select fields on
`numeric_pct > 90 and presence_pct > 20` with NO denylist, so `ST` and `TST` sit
inside every one of those vectors. 03e measures the cost of that directly: +4.7s
in-segment, +2.2s over all rows. This notebook applies both denylists
everywhere, so its numbers are comparable to 03e's and NOT to those three.

Run after 03. Independent of 03b/03c/03d/03e — it recomputes what it needs.
"""

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import pickle
import sys
from collections import Counter

import numpy as np

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
    _to_float, heldout_regressor_score, numeric_field_inventory, numeric_matrix,
    permutation_importance_mae,
)

PKL_PATH = os.environ.get('PKL_PATH', PKL_OUTPUT)

# Görtler named +-15s as the gold standard; scanner + protocol measured 15.3s
# MAE on the real exam CSVs, so the two agree.
TARGET_MAE_S = float(os.environ.get('TARGET_MAE_S', '15.0'))

# 2 matches 03d section 3 and 03e, so these numbers sit alongside the ones
# already reported. Raise it if two rows land within noise of each other.
REPEATS = int(os.environ.get('SHOOTOUT_REPEATS', '2'))


def rule(char='=', width=78):
    print(char * width)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# `sut_raw` is keyed by the RAW message mnemonic (TR, SLC, PEL, ...) while
# PARAM_SETS names the stable parameter names (TR, num_slices, ...).
NAME_TO_RAW_KEY = {name: key for key, name in SUT_FIELD_MAP.items()}

with open(PKL_PATH, 'rb') as f:
    sequences = pickle.load(f)['examination']

durations = np.array([float(s.get('total_duration', 0.0)) for s in sequences])
keep = durations > 0
sequences = [s for s, k in zip(sequences, keep) if k]
durations = durations[keep]
print(f"{len(sequences):,} sequences with a positive duration from {PKL_PATH}")

serials = np.array([int(s.get('serial_idx', 0)) for s in sequences])
seq_types = np.array([int(s.get('sequence_type', 0)) for s in sequences])
binaries = np.array([s.get('sequence_binary') or '' for s in sequences])
orientations = np.array([s.get('orientation') or '' for s in sequences])
scopes = np.array([s.get('sut_scope', '') for s in sequences])
offsets = np.array([_to_float(s.get('sut_offset_s')) for s in sequences])
protocols = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])

if not (scopes == 'inside').any():
    raise RuntimeError(
        "This pkl carries no 'sut_scope' — it predates the join fix (dcdd381). "
        "Every question in this notebook is about which rows the in-segment "
        "rule covers, so there is nothing to measure. Re-run "
        "csv_pipeline_seqparams/03_build_preprocessed_pkl.py first."
    )

# 03d section 3's population, rebuilt here so this notebook stands alone.
# Aborts end on MRI_MSR_34 and bypass the 10s floor, so they are a different
# population and would otherwise flatter every column equally.
is_abort = np.array([
    len(s.get('sequence', [])) > 0
    and int(s['sequence'][-1]) == SOURCEID_VOCAB['MRI_MSR_34']
    for s in sequences
])
clean = ~is_abort & (durations >= 10) & (durations <= 600)
inside = scopes == 'inside'

print(f"  non-abort, 10-600s:            {clean.sum():,} rows")
print(f"  ... of which in-segment join:  {(clean & inside).sum():,} "
      f"({pct((clean & inside).sum(), clean.sum()):.1f}%)")

# Written by step 03 only after the sut_inside_skipped change. Where present it
# separates 'this segment IS an adjustment' from 'no SUT event ever fired for
# it' — two populations with opposite remedies. Degrades gracefully.
HAS_SKIP_FLAG = any('sut_inside_skipped' in s for s in sequences)
skipped = np.array([bool(s.get('sut_inside_skipped', False)) for s in sequences])

# COMMAND ----------

# =============================================================================
# A — WHO ARE THE ROWS THE JOIN MISSES?
#
# 03c reports the three scopes as a coverage number and stops there. Whether
# 19.7% missing coverage is a data limit or a rule that can be widened depends
# entirely on WHY those segments have no in-segment SUT event, and the pkl
# already carries enough to tell: the scope, the signed offset of whatever was
# joined instead, the sequence family from the MSR_100 message, and the
# duration.
# =============================================================================

rule()
print(" A. WHO ARE THE ROWS THE JOIN MISSES?")
rule()

print(f"\n  {'scope':<10} {'rows':>8} {'share':>7} {'mean':>8} {'median':>8} "
      f"{'sd':>8} {'abort':>7}")
rule('-')
for scope in ('inside', 'before', 'none'):
    mask = scopes == scope
    if not mask.any():
        continue
    print(f"  {scope:<10} {mask.sum():>8,} {pct(mask.sum(), len(scopes)):>6.1f}% "
          f"{durations[mask].mean():>7.1f}s {np.median(durations[mask]):>7.0f}s "
          f"{durations[mask].std():>7.1f}s {pct(is_abort[mask].sum(), mask.sum()):>6.1f}%")
rule('-')

if HAS_SKIP_FLAG:
    print("\n  Of the rows with no usable in-segment event, how many HAD one "
          "that was\n  passed over as an adjustment?\n")
    print(f"  {'scope':<10} {'rows':>8} {'adjustment inside':>19} "
          f"{'no SUT at all':>15}")
    rule('-')
    for scope in ('before', 'none'):
        mask = scopes == scope
        if not mask.any():
            continue
        print(f"  {scope:<10} {mask.sum():>8,} "
              f"{skipped[mask].sum():>12,} ({pct(skipped[mask].sum(), mask.sum()):>4.1f}%) "
              f"{(~skipped[mask]).sum():>14,}")
    rule('-')
    print("""
  An adjustment fires at the top of a real measurement, so 'adjustment inside'
  used to mean the SEGMENT was real and the adjustment stole its message. Since
  de51099 the adjustment is skipped, so a row landing here means the segment had
  NO other SUT event — which is the signature of a segment that IS the
  adjustment. Those are not missing coverage; they are measurements the
  parameter model has no business predicting, and they belong in the population
  filter next to aborts, not in the fallback.""")
else:
    print("\n  (no 'sut_inside_skipped' in this pkl — cannot separate 'the "
          "segment IS an\n   adjustment' from 'no SUT event ever fired'. That "
          "flag is one line in\n   _choose_sut_row and arrives with the next "
          "rebuild.)")

# ---- A1. Which sequence families lose their parameters? ----

print("\n\n  A1. SEQUENCE FAMILY BY SCOPE — share of each family's rows that "
      "the join covers\n")
print(f"  {'sequence_type':<16} {'rows':>8} {'inside':>8} {'before':>8} "
      f"{'none':>7} {'median dur':>11}")
rule('-')
family_rows = []
for type_id in sorted(set(seq_types.tolist())):
    mask = seq_types == type_id
    family_rows.append((
        mask.sum(), ID_TO_SEQUENCE_TYPE.get(type_id, str(type_id)),
        pct((mask & inside).sum(), mask.sum()),
        pct((mask & (scopes == 'before')).sum(), mask.sum()),
        pct((mask & (scopes == 'none')).sum(), mask.sum()),
        float(np.median(durations[mask])),
    ))
for n, name, p_in, p_before, p_none, med in sorted(family_rows, reverse=True):
    print(f"  {name:<16} {n:>8,} {p_in:>7.1f}% {p_before:>7.1f}% "
          f"{p_none:>6.1f}% {med:>10.0f}s")
rule('-')
print("  A family with uniformly low 'inside' is a family whose scanner does not\n"
      "  emit MRI_SUT_1005 the way the others do — a data limit. A family that\n"
      "  tracks the overall 80/19/1 split is losing rows to something\n"
      "  incidental, which is the case worth chasing.")

# COMMAND ----------

# =============================================================================
# A2 — HOW FAR AWAY IS THE FALLBACK, AND WOULD A TOLERANCE BE SAFE?
#
# 03c prints the 'before' offset distribution at p50 101s / p90 248s / max
# 600s, which reads as "the fallback reaches back a long way". But those are the
# UPPER percentiles. `_choose_sut_row` compares ROW INDICES, not timestamps, so
# a SUT event emitted one row ahead of its own MRI_MSR_100 — same instant, wrong
# side of the boundary — is labelled 'before' at ~0s offset and is a boundary
# artefact rather than a stale message.
#
# Widening the rule by a few seconds is one line of code, so the only question
# is whether the recovered rows are CORRECT. Same test 03c section A uses: the
# DLL from the joined SUT message must classify to the sequence_type the
# MRI_MSR_100 message reported. Coverage that arrives with 30% agreement is not
# coverage.
# =============================================================================

rule()
print(" A2. IS THERE A CHEAP TOLERANCE HIDING IN THE 'before' ROWS?")
rule()

before = scopes == 'before'
before_offsets = np.abs(offsets[before])
before_offsets = before_offsets[np.isfinite(before_offsets)]

if before_offsets.size:
    print("\n  |offset| from the segment start, 'before' rows — the LOW end, "
          "which 03c does not print\n")
    labels = ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'max']
    values = [np.percentile(before_offsets, q) for q in (1, 5, 10, 25, 50, 75, 90)]
    values.append(before_offsets.max())
    print("    " + "  ".join(f"{l}={v:.0f}s" for l, v in zip(labels, values)))

dll_classified = np.array([classify_sequence_type(b) for b in binaries])
other_id = SEQUENCE_TYPE_VOCAB['other']
testable = (binaries != '') & (dll_classified != other_id)
agrees = dll_classified == seq_types

print(f"\n  {'look-back':<12} {'recovered':>10} {'coverage':>9} {'testable':>9} "
      f"{'agreement':>10}")
rule('-')
print(f"  {'0s (today)':<12} {'—':>10} {pct(inside.sum(), len(scopes)):>8.1f}% "
      f"{testable[inside].sum():>9,} "
      f"{pct(agrees[inside & testable].sum(), max(1, testable[inside].sum())):>9.1f}%")
for tolerance in (2.0, 5.0, 15.0, 30.0, 60.0):
    recovered = before & np.isfinite(offsets) & (np.abs(offsets) <= tolerance)
    if not recovered.any():
        print(f"  {f'+{tolerance:.0f}s':<12} {0:>10,} "
              f"{pct(inside.sum(), len(scopes)):>8.1f}% {'—':>9} {'—':>10}")
        continue
    n_testable = testable[recovered].sum()
    agreement = (pct(agrees[recovered & testable].sum(), n_testable)
                 if n_testable else float('nan'))
    print(f"  {f'+{tolerance:.0f}s':<12} {recovered.sum():>10,} "
          f"{pct((inside | recovered).sum(), len(scopes)):>8.1f}% "
          f"{n_testable:>9,} {agreement:>9.1f}%")
rule('-')
print("""
  HOW TO READ THIS

    A tolerance whose agreement stays near the 100% of 'inside'
        Those rows are a boundary artefact of comparing row indices. Widen
        `_choose_sut_row` by that many seconds and the coverage is real.
    Agreement falls off immediately, at every tolerance
        There is no boundary artefact — a 'before' event genuinely belongs to
        an earlier measurement however close it sits. Nothing to recover here,
        and section B's masking is the remedy rather than a wider rule.""")

# COMMAND ----------

# =============================================================================
# A3 — WHY 03c SECTION D's join-INVALID ROW LOOKS BACKWARDS
#
# 03c section D reports, on the 2026-08-05 run:
#
#     join-valid only (A agrees)     38,599    protocol 13.0s   params 15.1s
#     join-INVALID only (A disagrees) 6,102    protocol  0.7s   params  1.7s
#
# and then prints "a large gap between the join-valid and join-invalid rows is
# direct confirmation that the join is the problem". On these numbers the gap
# runs the other way, and the reason is not the join at all: a per-protocol
# group MEAN scoring 0.7s MAE says the durations in that subset are very nearly
# constant within protocol. Anything predicts a constant.
#
# So the join-invalid pool is a degenerate subpopulation, not evidence about
# parameter quality either way. This section measures that directly, so section
# D's wording can be corrected against a number rather than an argument.
# =============================================================================

rule()
print(" A3. IS THE NON-INSIDE POOL JUST EASIER?")
rule()


def within_protocol_sd(mask):
    """Mean per-protocol duration sd — how much room a predictor has at all."""
    sds = []
    for key in np.unique(protocols[mask]):
        member = durations[mask & (protocols == key)]
        if member.size >= 5:
            sds.append(member.std())
    return float(np.mean(sds)) if sds else float('nan')


print(f"\n  {'row subset':<30} {'n':>8} {'sd':>8} {'within-proto sd':>16} "
      f"{'protocol MAE':>13}")
rule('-')
for label, mask in [
    ('clean, in-segment', clean & inside),
    ('clean, NOT in-segment', clean & ~inside),
    ('clean, before-scope only', clean & before),
]:
    if mask.sum() < 1000:
        print(f"  {label:<30} {mask.sum():>8,}   too few rows to score")
        continue
    _, p_mae, _ = heldout_group_r2(protocols[mask], durations[mask])
    print(f"  {label:<30} {mask.sum():>8,} {durations[mask].std():>7.1f}s "
          f"{within_protocol_sd(mask):>15.1f}s {p_mae:>12.1f}s")
rule('-')
print("""
  If the non-inside pool's within-protocol sd is a fraction of the in-segment
  pool's, its low MAE is a property of the ROWS, not of the parameters, and no
  comparison across the two subsets means anything. That is the correction 03c
  section D's closing text needs.""")

# COMMAND ----------

# =============================================================================
# B — WRONG PARAMETERS, OR NO PARAMETERS?
#
# 03c section A2 scores the 'before' fallback at 28.5% agreement: on roughly
# seven of every ten of those rows the model is handed a DIFFERENT scan's
# settings. That is not noise around the right answer, it is a confident wrong
# answer, and a regressor has no way to tell the two apart.
#
# The alternative is to declare them missing. HistGradientBoostingRegressor
# learns a split direction for NaN; the model path has the same option through
# SEQPARAM_MISSING_DEFAULTS. If masking wins, the fix is a config decision — the
# fallback stops being a fallback — and it costs nothing to make.
#
# Row 3 exists because masking alone leaves a fifth of the population with no
# features at all, so the regressor predicts the global mean there and the
# comparison is unfair to masking. The customer-agnostic categoricals
# (sequence_binary, orientation) are available on EVERY row regardless of scope
# and are already admissible under Görtler's rule, so a deployed model would
# certainly carry them.
# =============================================================================

rule()
print(" B. WRONG PARAMETERS, OR NO PARAMETERS?")
rule()

inventory = numeric_field_inventory(sequences)
all_fields_asis = [
    row['key'] for row in inventory
    if row['numeric_pct'] > 90 and row['presence_pct'] > 20
]
_banned = set(SUT_IDENTIFIER_DENYLIST) | set(SUT_LEAKAGE_DENYLIST)
all_fields = [f for f in all_fields_asis if f not in _banned]

print(f"\n  {len(all_fields)} admissible fields "
      f"({len(all_fields_asis)} before removing "
      f"{sorted(f for f in all_fields_asis if f in _banned)})")

parameters = numeric_matrix(sequences, all_fields)

masked = parameters.copy()
masked[~inside, :] = np.nan


def ordinal(values):
    """Stable integer codes for a string column, for the GBM's benefit."""
    lookup = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.array([lookup[v] for v in values], dtype=float)


categoricals = np.column_stack([
    serials.astype(float), seq_types.astype(float),
    ordinal(binaries), ordinal(orientations),
])
masked_plus = np.column_stack([masked, categoricals])

_, oracle_all, _ = heldout_group_r2(protocols[clean], durations[clean])
_, oracle_inside, _ = heldout_group_r2(protocols[clean & inside],
                                       durations[clean & inside])

print(f"\n  ALL CLEAN ROWS — {clean.sum():,}  (the population that has to hold "
      f"up)\n")
print(f"  {'feature treatment':<38} {'R2':>8} {'MAE':>9} {'vs oracle':>10} "
      f"{'vs +-15s':>9}")
rule('-')
print(f"  {'serial + protocol (oracle)':<38} {'':>8} {oracle_all:>8.1f}s "
      f"{'—':>10} {oracle_all - TARGET_MAE_S:>+8.1f}s")

variants = [
    ('parameters as-is (stale included)', parameters[clean]),
    ('parameters masked where not in-segment', masked[clean]),
    ('... masked + sequence/orientation/serial', masked_plus[clean]),
    ('as-is + sequence/orientation/serial',
     np.column_stack([parameters, categoricals])[clean]),
]
for label, matrix in variants:
    r2, mae = heldout_regressor_score(matrix, durations[clean], repeats=REPEATS)
    print(f"  {label:<38} {r2:>7.1f}% {mae:>8.1f}s {mae - oracle_all:>+9.1f}s "
          f"{mae - TARGET_MAE_S:>+8.1f}s")
rule('-')

print(f"\n  IN-SEGMENT ROWS ONLY — {(clean & inside).sum():,}  (where the "
      f"parameters are known good)\n")
print(f"  {'feature treatment':<38} {'R2':>8} {'MAE':>9} {'vs oracle':>10} "
      f"{'vs +-15s':>9}")
rule('-')
print(f"  {'serial + protocol (oracle)':<38} {'':>8} {oracle_inside:>8.1f}s "
      f"{'—':>10} {oracle_inside - TARGET_MAE_S:>+8.1f}s")
r2, mae_inside = heldout_regressor_score(parameters[clean & inside],
                                         durations[clean & inside],
                                         repeats=REPEATS)
print(f"  {'parameters (the 9.7s reference)':<38} {r2:>7.1f}% {mae_inside:>8.1f}s "
      f"{mae_inside - oracle_inside:>+9.1f}s {mae_inside - TARGET_MAE_S:>+8.1f}s")
rule('-')

print("""
  HOW TO READ THIS

    masked BEATS as-is
        A stale message is worse than no message, which means the 'before'
        fallback is costing accuracy rather than buying coverage. Drop it:
        record 'none' and let SEQPARAM_MISSING_DEFAULTS carry those rows.
    masked LOSES to as-is
        Even a neighbour's parameters are informative — plausible, since
        consecutive scans in one exam share a body region and a protocol
        family. Keep the fallback, and the coverage gap stays a data problem.
    masked+categoricals reaches the +-15s bar
        The model is quotable on the FULL population today, without recovering
        a single extra SUT message. That is the cheapest route to the headline
        and it needs no rebuild.""")

# COMMAND ----------

# =============================================================================
# C — WHICH OF THE 89 EARN THEIR PLACE, AND IS ANY OF THEM A LEAK?
#
# Scored on clean in-segment rows: the parameters are only known-good there, and
# ranking fields on rows where they belong to a different scan would rank noise.
#
# C1 is the audit. 03d caught MUID by suspecting one field and testing it; that
# only works for suspicions somebody has. `ST` and `TST` sat inside the 93-field
# vector for three consecutive reports precisely because nobody re-checked the
# selection rule. Ranking the whole vector in seconds of MAE makes a
# duration-in-disguise visible without having to guess its name first.
#
# C2 is the shipping decision. 89 fields is not a model: step 07 must synthesise
# every conditioning feature before it can generate, and SEQPARAM_CANDIDATES
# needs a hand-calibrated divisor and missing-default per field. The nested
# prefix curve says what a shorter list costs.
# =============================================================================

rule()
print(" C. WHICH OF THE 89 EARN THEIR PLACE")
rule()

score_mask = clean & inside
print(f"\n  scored on {score_mask.sum():,} clean in-segment rows")

ranking = permutation_importance_mae(
    parameters[score_mask], durations[score_mask], field_names=all_fields,
)
baseline = ranking[0]['baseline_mae']
print(f"  single-fit baseline MAE {baseline:.1f}s "
      f"(the held-out number this ranking is relative to)\n")

# Fields whose name or unit says "time" or "identity". Not a denylist — a
# reading aid, so a suspect sitting high in the ranking is noticed rather than
# admired. Everything here is currently ADMISSIBLE and in the vector.
SUSPECT_NOTES = {
    'BHD': 'breath-hold duration — a time in seconds',
    'ACQW': 'acquisition window — a time',
    'TGD': 'trigger delay — a time',
    'TEU': 'TE in microseconds — duplicate of TE',
    'SNR': 'derived from the protocol, not set by the operator',
    'TP': 'table position — geometry, but fingerprints an exam',
    'ES': 'echo spacing — a time, but a per-echo one',
}

print(f"  {'rank':>4}  {'field':<8} {'importance':>11}   note")
rule('-')
for position, row in enumerate(ranking[:25], start=1):
    note = SUSPECT_NOTES.get(row['name'], '')
    flag = '  <-- CHECK' if note else ''
    print(f"  {position:>4}  {row['name']:<8} {row['importance_s']:>+10.2f}s   "
          f"{note}{flag}")
rule('-')
_carries_nothing = sum(1 for row in ranking if row['importance_s'] < 0.05)
print(f"  {_carries_nothing} of {len(ranking)} fields move the held-out MAE by "
      f"less than 0.05s.")
print("""
  Correlated fields share credit — TR and PEL move together, so neither scores
  its full worth and both look cheaper here than they are. Use this for the
  ORDER and C2 for the COUNT.

  A field flagged CHECK sitting in the top few is the thing to stop for. The
  test is the one applied to ST: could a scheduler know this value before the
  scan runs, and is it a copy of the answer rather than a cause of it?""")

# COMMAND ----------

# =============================================================================
# C2 — HOW MANY FIELDS DO WE ACTUALLY NEED?
# =============================================================================

rule()
print(" C2. MAE vs FIELD COUNT")
rule()

ranked_names = [row['name'] for row in ranking]
print(f"\n  {'fields':>7} {'R2':>8} {'MAE':>9} {'vs oracle':>10} {'vs +-15s':>9}"
      f"   newly added")
rule('-')
previous = 0
for size in (3, 5, 8, 12, 20, 30, len(ranked_names)):
    if size > len(ranked_names):
        continue
    subset = ranked_names[:size]
    r2, mae = heldout_regressor_score(
        numeric_matrix(sequences, subset)[score_mask],
        durations[score_mask], repeats=REPEATS,
    )
    added = ", ".join(ranked_names[previous:size][:6])
    if size - previous > 6:
        added += ", ..."
    print(f"  {size:>7} {r2:>7.1f}% {mae:>8.1f}s "
          f"{mae - oracle_inside:>+9.1f}s {mae - TARGET_MAE_S:>+8.1f}s   {added}")
    previous = size
rule('-')
print("  The knee is the shipping candidate. A prefix that already clears the\n"
      "  oracle is worth more than a longer one that clears it by more: every\n"
      "  extra field is a SEQPARAM_CANDIDATES divisor to calibrate and another\n"
      "  quantity step 07 has to synthesise before it can generate.")

# COMMAND ----------

# =============================================================================
# C3 — THE NAMED SETS ON THE SAME AXES
#
# 03e already scores luke and navneet. Repeated here on identical rows so the
# hand-picked sets and the ranked prefixes can be read off one table instead of
# two notebooks — that comparison is the whole decision.
# =============================================================================

rule()
print(" C3. HAND-PICKED vs RANKED")
rule()

print(f"\n  {'predictor':<30} {'fields':>7} {'R2':>8} {'MAE':>9} "
      f"{'vs oracle':>10}")
rule('-')
print(f"  {'serial + protocol (oracle)':<30} {'—':>7} {'':>8} "
      f"{oracle_inside:>8.1f}s {'—':>10}")
for name, features in sorted(PARAM_SETS.items()):
    keys = [NAME_TO_RAW_KEY[f] for f in features]
    r2, mae = heldout_regressor_score(numeric_matrix(sequences, keys)[score_mask],
                                      durations[score_mask], repeats=REPEATS)
    star = '*' if name == PARAM_SET else ' '
    print(f" {star}{name:<30} {len(features):>7} {r2:>7.1f}% {mae:>8.1f}s "
          f"{mae - oracle_inside:>+9.1f}s")
for size in (8, 12):
    if size > len(ranked_names):
        continue
    r2, mae = heldout_regressor_score(
        numeric_matrix(sequences, ranked_names[:size])[score_mask],
        durations[score_mask], repeats=REPEATS,
    )
    print(f"  {f'ranked top {size}':<30} {size:>7} {r2:>7.1f}% {mae:>8.1f}s "
          f"{mae - oracle_inside:>+9.1f}s")
rule('-')
print(f"  * = the set PARAM_SET={PARAM_SET!r} currently selects.")

print("""
  A ranked prefix that beats both named sets at the same field count says the
  hand-picking cost accuracy, and the third PARAM_SETS entry writes itself. A
  named set that holds its own says the physics argument was right and the
  ranking is fitting noise — in which case prefer the named set, because it can
  be explained to a domain expert and a ranked list cannot.

  Promoting a field is NOT free: each one needs a SEQPARAM_CANDIDATES row with a
  divisor (use the observed p99 from 03b section 1 — NOT that table's 'suggested
  divisor' column, which rounds down to a power of ten and would let TR arrive
  at 9.0) and a missing default that means 'does not apply' (1.0 for the
  multiplicative factors, 0.0 for measurements). Check the real values pinned in
  tests/test_sut_parser.py before assuming a factor is 1-based — REP is not.""")

# COMMAND ----------

# =============================================================================
# NEXT
#
#   A2 finds a safe tolerance  ->  widen _choose_sut_row, rebuild, re-run.
#   B says masking wins        ->  drop the 'before' fallback in step 03 and let
#                                  SEQPARAM_MISSING_DEFAULTS carry those rows.
#   C names a compact set      ->  add it to SEQPARAM_CANDIDATES + PARAM_SETS,
#                                  confirm with 03e, then train it with 04.
#
# What this notebook is NOT: a gradient-boosted regressor over a flat parameter
# matrix is not the sequence model. It ranks feature sets and row populations,
# which is what is asked here; it does not predict what the trained model does
# with them.
# =============================================================================
