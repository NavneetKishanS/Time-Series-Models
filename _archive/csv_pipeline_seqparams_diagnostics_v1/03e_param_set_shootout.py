# Databricks notebook source
"""
03e_param_set_shootout.py — whose parameter set is better?

NO TRAINING, NO SPARK. Reads the same pkl 03b/03c/03d read. ~5 minutes.

WHY THIS EXISTS
---------------
config.py now carries two named parameter sets behind one PARAM_SET switch:

    luke     the acquisition-time formula, as its own multiplicands —
             TA ~= TR x PEL x AVG x CONC / (PAT x TF), plus the slice count.
    navneet  the 2026-08-04 domain-expert selection — TR plus only the things
             that change HOW MANY TRs run (slices, averages, repetitions,
             matrix size, phase/slice partial Fourier), on the argument that
             TE / flip angle / echo spacing / slice thickness are physically
             real but redundant because they surface as a floor on TR.

Settling that with two full training runs costs two GPU jobs and a day. The
same question — does this set carry more duration signal than that one? —
answers in five minutes with the estimator 03b/03c/03d already use, on the same
rows and the same split convention. If a set loses badly here it will lose
after training too, and the GPU run is wasted.

It scores against two reference points that are already on the record:

    protocol oracle   03d section 3 put it at 11.0s MAE on this population.
                      Görtler ruled the protocol name OUT as a feature and IN
                      as the benchmark — the target is to match it without it.
    every field       "throw it all in". A hand-picked set that cannot beat it
                      has not earned its hand-picking.

CAREFUL WITH THE OLD NUMBERS
----------------------------
03b section 5 (23.3s), 03c section D (13.5s) and 03d section 3 (19.6s) all
selected fields on presence and numeric-ness ALONE:

    fields = [row['key'] for row in inventory
              if row['numeric_pct'] > 90 and row['presence_pct'] > 20]

No denylist. So `ST` and `TST` were IN every one of those vectors — and ST is
the planned scanning time, exact on ~86-87% of rows per 03c section E. Those
three numbers are therefore not clean "parameters only" results; they had a
partial copy of the target in them. Section 2 below scores the all-fields
reference both ways so the size of that contamination is measured rather than
argued about, and only the denylisted column is comparable to the named sets.

WHAT IT DOES NOT NEED
---------------------
A rebuilt pkl. It reads `sut_raw` — the unfiltered per-message capture step 03
has written since d1a8f3e — and maps the PARAM_SETS names back through
SUT_FIELD_MAP to the raw keys itself. So it runs against whatever pkl you have
now, including one built before SEQPARAM_CANDIDATES existed.

Run after 03. Independent of 03b/03c/03d — it recomputes what it needs.
"""

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import pickle
import sys

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
    heldout_regressor_score, numeric_field_inventory, numeric_matrix,
)

PKL_PATH = os.environ.get('PKL_PATH', PKL_OUTPUT)

# Görtler named +-15s as the gold standard; scanner + protocol measured 15.3s
# MAE on the real exam CSVs, so the two agree.
TARGET_MAE_S = float(os.environ.get('TARGET_MAE_S', '15.0'))

# Repeats for heldout_regressor_score. 2 matches 03d section 3, so the numbers
# below sit alongside the ones already reported. Raise it if two sets land
# within noise of each other and the ranking has to be trusted.
REPEATS = int(os.environ.get('SHOOTOUT_REPEATS', '2'))


def rule(char='=', width=78):
    print(char * width)


# `sut_raw` is keyed by the RAW message mnemonic (TR, SLC, PEL, ...) while
# PARAM_SETS names the stable parameter names (TR, num_slices, ...). Invert
# SUT_FIELD_MAP to cross between them.
NAME_TO_RAW_KEY = {name: key for key, name in SUT_FIELD_MAP.items()}


def raw_keys_for(feature_names):
    """The sut_raw keys behind a list of stable parameter names."""
    return [NAME_TO_RAW_KEY[name] for name in feature_names]


with open(PKL_PATH, 'rb') as f:
    sequences = pickle.load(f)['examination']

if not any('sut_raw' in s for s in sequences):
    raise RuntimeError(
        "This pkl carries no 'sut_raw' — it predates the widened SUT capture. "
        "Re-run csv_pipeline_seqparams/03_build_preprocessed_pkl.py."
    )

durations = np.array([float(s.get('total_duration', 0.0)) for s in sequences])
keep = durations > 0
sequences = [s for s, k in zip(sequences, keep) if k]
durations = durations[keep]
print(f"{len(sequences):,} sequences with a positive duration from {PKL_PATH}")
print(f"Selected PARAM_SET is {PARAM_SET!r}; scoring all of {sorted(PARAM_SETS)}")

protocols = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])

# 03d section 3's population, rebuilt here so this notebook stands alone.
# Aborts end on MRI_MSR_34 and bypass the 10s floor, so they are a different
# population and would otherwise flatter every column equally.
is_abort = np.array([
    len(s.get('sequence', [])) > 0
    and int(s['sequence'][-1]) == SOURCEID_VOCAB['MRI_MSR_34']
    for s in sequences
])
clean = ~is_abort & (durations >= 10) & (durations <= 600)
scopes = np.array([s.get('sut_scope', '') for s in sequences])

print(f"  non-abort, 10-600s:            {clean.sum():,} rows")
if (scopes == 'inside').any():
    print(f"  ... of which in-segment join:  {(clean & (scopes == 'inside')).sum():,}")
else:
    print("  !! no 'sut_scope' on these rows — this pkl predates the join fix "
          "(dcdd381).\n     Every number below is measured over a mix of valid "
          "and stale joins, and\n     03c section A says the stale ones agree "
          "with the actual sequence 28.1% of\n     the time. Rebuild with 03 "
          "before quoting anything from here.")

# COMMAND ----------

# =============================================================================
# 1 — COVERAGE AND SCALE OF THE SELECTED FIELDS
#
# Before comparing two sets, check both are actually populated. A parameter
# present on 4% of rows cannot carry a set, however good the physics, and the
# divisor check is the same one step 03 runs at write time — repeated here
# because this notebook is the one you run BEFORE the rebuild.
# =============================================================================

rule()
print(" 1. ARE THESE FIELDS ACTUALLY THERE?")
rule()

inventory = {row['key']: row for row in numeric_field_inventory(sequences)}

print(f"\n  {'parameter':<24} {'raw':>5} {'present':>9} {'distinct':>9} "
      f"{'numeric':>8} {'p99':>10} {'p99/div':>9}")
rule('-')
for name in SEQPARAM_ALL_CANDIDATES:
    key = NAME_TO_RAW_KEY[name]
    row = inventory.get(key)
    sets = ','.join(s for s, names in PARAM_SETS.items() if name in names) or '-'
    if row is None:
        print(f"  {name:<24} {key:>5} {'ABSENT — not in any message':>48}")
        continue
    divisor = SEQPARAM_CANDIDATES[name][0]
    scaled = row['p99'] / divisor if divisor else float('inf')
    flag = '' if 0.05 <= scaled <= 20.0 else '  <-- CHECK DIVISOR'
    print(f"  {name:<24} {key:>5} {row['presence_pct']:>8.1f}% "
          f"{row['distinct']:>9,} {row['numeric_pct']:>7.1f}% "
          f"{row['p99']:>10.1f} {scaled:>9.3f}{flag}  [{sets}]")
rule('-')
print("  'present' below ~100% means the field is SEQUENCE-SCOPED. That is not\n"
      "  disqualifying — SEQPARAM_CANDIDATES gives each name a missing default\n"
      "  that means 'does not apply' (1.0 for a multiplicative factor) rather\n"
      "  than a fabricated zero — but a field near 0% is carrying nothing.")

# COMMAND ----------

# =============================================================================
# 2 — THE SHOOTOUT
#
# Same estimator (HistGradientBoostingRegressor via heldout_regressor_score),
# same split convention and the same population as 03d section 3, so these
# numbers are directly comparable to the ones already reported.
#
# NaN is passed through rather than filled: the regressor learns a split
# direction for missing values, which is the correct treatment for a parameter
# that does not apply to a sequence family. Note this differs on purpose from
# the MODEL path, which cannot represent NaN and gets SEQPARAM_CANDIDATES'
# neutral defaults instead. The gap between the two is a known and deliberate
# limitation of reading a training result off this notebook.
# =============================================================================

rule()
print(" 2. PARAMETER SETS, HEAD TO HEAD")
rule()

# Exactly 03b/03c/03d's selection rule, so the "as 03d measured it" row below
# reproduces their number...
all_fields_asis = [
    row['key'] for row in numeric_field_inventory(sequences)
    if row['numeric_pct'] > 90 and row['presence_pct'] > 20
]
# ...and the same list with both denylists applied, which is the only version
# comparable to the named sets. The difference between the two rows IS the
# contamination the old numbers carried.
_banned = set(SUT_IDENTIFIER_DENYLIST) | set(SUT_LEAKAGE_DENYLIST)
all_fields = [f for f in all_fields_asis if f not in _banned]
_removed = [f for f in all_fields_asis if f in _banned]

print(f"\n  All-field reference: {len(all_fields_asis)} fields as 03b/03c/03d "
      f"selected them,\n  {len(all_fields)} after removing {sorted(_removed)}.")

contenders = [(name, list(features)) for name, features in sorted(PARAM_SETS.items())]

subsets = [('all rows', clean)]
if (scopes == 'inside').any():
    subsets.append(('in-segment join only', clean & (scopes == 'inside')))

for label, mask in subsets:
    n = int(mask.sum())
    print(f"\n  {label.upper()}  —  {n:,} rows")
    if n < 1000:
        print("    too few rows to score")
        continue

    _, oracle_mae, _ = heldout_group_r2(protocols[mask], durations[mask])

    print(f"\n  {'predictor':<28} {'fields':>7} {'R2':>8} {'MAE':>9} "
          f"{'vs oracle':>10} {'vs +-15s':>9}")
    rule('-')
    print(f"  {'serial + protocol (oracle)':<28} {'—':>7} {'':>8} "
          f"{oracle_mae:>8.1f}s {'—':>10} {oracle_mae - TARGET_MAE_S:>+8.1f}s")

    results = []
    for name, features in contenders:
        matrix = numeric_matrix(sequences, raw_keys_for(features))
        r2, mae = heldout_regressor_score(matrix[mask], durations[mask],
                                          repeats=REPEATS)
        results.append((name, mae))
        star = '*' if name == PARAM_SET else ' '
        print(f" {star}{name:<28} {len(features):>7} {r2:>7.1f}% {mae:>8.1f}s "
              f"{mae - oracle_mae:>+9.1f}s {mae - TARGET_MAE_S:>+8.1f}s")

    # The "throw it all in" reference, both ways round.
    matrix = numeric_matrix(sequences, all_fields)
    r2, admissible_mae = heldout_regressor_score(matrix[mask], durations[mask],
                                                 repeats=REPEATS)
    print(f"  {'every admissible field':<28} {len(all_fields):>7} {r2:>7.1f}% "
          f"{admissible_mae:>8.1f}s {admissible_mae - oracle_mae:>+9.1f}s "
          f"{admissible_mae - TARGET_MAE_S:>+8.1f}s")

    matrix = numeric_matrix(sequences, all_fields_asis)
    r2, asis_mae = heldout_regressor_score(matrix[mask], durations[mask],
                                           repeats=REPEATS)
    print(f"  {'  ... as 03b/03c/03d had it':<28} {len(all_fields_asis):>7} "
          f"{r2:>7.1f}% {asis_mae:>8.1f}s {asis_mae - oracle_mae:>+9.1f}s "
          f"{asis_mae - TARGET_MAE_S:>+8.1f}s")
    rule('-')
    if _removed:
        print(f"  Removing {sorted(_removed)} costs "
              f"{admissible_mae - asis_mae:+.1f}s — that is how much of the "
              f"earlier\n  parameter result was ST/TST and identifiers rather "
              f"than physics.")

    if results:
        best, best_mae = min(results, key=lambda kv: kv[1])
        margin = sorted(m for _, m in results)
        spread = (margin[1] - margin[0]) if len(margin) > 1 else 0.0
        print(f"  Best named set on this subset: {best!r} at {best_mae:.1f}s "
              f"(next is {spread:+.1f}s behind).")
        if spread < 0.5:
            print("  Margin is inside the noise of a 2-repeat GBM — re-run with "
                  "SHOOTOUT_REPEATS=5\n  before calling a winner.")

print(f"\n  * = the set PARAM_SET={PARAM_SET!r} currently selects.")

# COMMAND ----------

# =============================================================================
# 3 — WHICH PARAMETERS EARN THEIR PLACE
#
# The set-level number says which list wins. This says why, and it is the part
# that answers a domain expert rather than a metric: drop each field in turn
# and see what the set loses. A field whose removal costs nothing is a field
# the expert can be told to stop worrying about.
#
# Leave-one-out rather than permutation importance because the sets are small
# and correlated — TR and PEL move together — and a drop test measures what
# actually happens if you delete the column, which is the decision on the table.
# =============================================================================

rule()
print(" 3. LEAVE-ONE-OUT WITHIN EACH SET")
rule()

loo_mask = clean
if (scopes == 'inside').any() and (clean & (scopes == 'inside')).sum() >= 1000:
    loo_mask = clean & (scopes == 'inside')
    print("\n  (scored on in-segment rows — the population where the SUT join "
          "is trustworthy)")
else:
    print("\n  (scored on all clean rows)")

for name, features in contenders:
    matrix = numeric_matrix(sequences, raw_keys_for(features))
    _, full_mae = heldout_regressor_score(matrix[loo_mask], durations[loo_mask],
                                          repeats=REPEATS)
    print(f"\n  {name.upper()}  —  {len(features)} fields, {full_mae:.1f}s with "
          f"all of them")
    print(f"    {'dropped field':<26} {'MAE without':>12} {'cost of dropping':>18}")
    rule('-')
    rows = []
    for dropped in features:
        kept = [f for f in features if f != dropped]
        sub = numeric_matrix(sequences, raw_keys_for(kept))
        _, mae = heldout_regressor_score(sub[loo_mask], durations[loo_mask],
                                         repeats=REPEATS)
        rows.append((dropped, mae, mae - full_mae))
    for dropped, mae, cost in sorted(rows, key=lambda r: -r[2]):
        note = '' if cost > 0.2 else '   <- carries nothing on its own'
        print(f"    {dropped:<26} {mae:>11.1f}s {cost:>+17.1f}s{note}")
    rule('-')

print("""
  HOW TO READ THIS

    A large positive cost means the set genuinely depends on that field.
    A cost near zero means either the field is uninformative OR another field
    in the same set already carries it — which is exactly Görtler's redundancy
    argument, and the reason TE / flip angle / echo spacing were dropped. The
    drop test cannot tell those two apart on its own; section 1's presence
    column usually can.""")

# COMMAND ----------

# =============================================================================
# NEXT STEP
#
# If a set wins here by a clear margin, train it: set PARAM_SET and run 04.
# Models and analysis are namespaced by PARAM_SET, so both can be trained and
# kept side by side without either overwriting the other.
#
# Remember what this notebook is NOT: a gradient-boosted regressor over a flat
# parameter matrix is not the sequence model. It ranks feature sets, which is
# the question asked here; it does not predict what the trained model will do
# with them.
# =============================================================================
