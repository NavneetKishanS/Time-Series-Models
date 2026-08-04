# Databricks notebook source
"""
03d_identifier_leakage_check.py — is the parameter result real, or is MUID
smuggling the protocol name back in?

NO TRAINING, NO SPARK. Reads the same pkl 03b and 03c read. ~5 minutes.

WHY THIS EXISTS
---------------
03b section 5 scored the full 93-field parameter vector at 23.3s MAE overall,
and 03c section D put it at 13.5s on join-valid rows against the protocol
oracle's 13.3s. Parameters matching the oracle that closely is either the
result we wanted or a leak, and from the outside those look identical.

`MUID` is in that 93-field vector, is present on 100% of rows, and has 3,188
distinct values against 3,051 distinct protocol names — a ratio of 1.04. If it
identifies the protocol, then a model given MUID has been told the protocol
after all, which is precisely what Görtler ruled out:

    "a protocol is a saved state ... customers copy Siemens presets, rename
    them and edit them ... a protocol-keyed model learns one site and
    transfers to none"

This is not the same objection as the `scanning_time` denylist. That one bans a
field because it is ~equal to the TARGET. This one bans a field because it is
~equal to a FEATURE we agreed not to use. Both are leaks; only the first was
guarded.

WHAT IT ANSWERS
---------------
  1. Does MUID determine the protocol? (purity, both directions)
  2. Does MUID predict duration as well as the protocol does?
  3. Does the parameter result survive removing it?

Section 3 is the decision. Sections 1 and 2 explain whichever way it goes, so
the answer comes with a mechanism rather than just a number.

Run after 03. Independent of 03b/03c — it recomputes what it needs.
"""

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import os
import pickle
import sys
from collections import Counter, defaultdict

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

# Fields suspected of carrying identity rather than physics.
#   MUID — the suspect. 3,188 values against 3,051 protocol names.
#   VER  — scanner software version. Not a scan property at all; it splits the
#          corpus by install, which lets a tree memorise per-site behaviour.
# Override to test a different set, e.g. IDENTIFIER_FIELDS=MUID
IDENTIFIER_FIELDS = [
    f.strip() for f in os.environ.get('IDENTIFIER_FIELDS', 'MUID,VER').split(',')
    if f.strip()
]


def rule(char='=', width=78):
    print(char * width)


with open(PKL_PATH, 'rb') as f:
    sequences = pickle.load(f)['examination']

durations = np.array([float(s.get('total_duration', 0.0)) for s in sequences])
keep = durations > 0
sequences = [s for s, k in zip(sequences, keep) if k]
durations = durations[keep]
print(f"{len(sequences):,} sequences with a positive duration from {PKL_PATH}")
print(f"Testing identifier fields: {IDENTIFIER_FIELDS}")

serials = np.array([int(s.get('serial_idx', 0)) for s in sequences])
protocols = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])
muids = np.array([
    f"{s.get('serial_idx', 0)}|{(s.get('sut_raw') or {}).get('MUID', '')}"
    for s in sequences
])

# COMMAND ----------

# =============================================================================
# 1 — DOES MUID DETERMINE THE PROTOCOL?
#
# Purity is measured BOTH ways, because they mean different things:
#
#   MUID -> protocol   high = knowing MUID tells you the protocol. THIS is the
#                      leak: the parameter vector is carrying protocol identity.
#   protocol -> MUID   high = the mapping is 1:1 rather than many-to-one. Tells
#                      you whether MUID is the protocol id itself or a coarser
#                      grouping (e.g. a Siemens preset family, which would be
#                      customer-agnostic and therefore ALLOWED).
#
# The second distinction matters for what we do next. A field that maps many
# customer protocols onto one Siemens preset is exactly the "generated protocol
# name" Görtler said we SHOULD use.
# =============================================================================

rule()
print(" 1. IS MUID A PROTOCOL IDENTIFIER?")
rule()


def purity(keys, targets):
    """Row-weighted share of rows whose target is the modal target of its key."""
    members = defaultdict(list)
    for key, target in zip(keys, targets):
        members[key].append(target)
    matched = sum(Counter(group).most_common(1)[0][1] for group in members.values())
    return len(members), 100.0 * matched / max(1, len(keys))


muid_groups, muid_to_protocol = purity(muids, protocols)
protocol_groups, protocol_to_muid = purity(protocols, muids)

print(f"\n  distinct MUIDs (per scanner)      {muid_groups:>8,}")
print(f"  distinct protocols (per scanner)  {protocol_groups:>8,}")
print(f"  ratio                             {muid_groups / max(1, protocol_groups):>8.2f}")

print(f"\n  MUID -> protocol purity           {muid_to_protocol:>7.1f}%   "
      f"<- the leak test")
print(f"  protocol -> MUID purity           {protocol_to_muid:>7.1f}%")

print("""
  HOW TO READ THIS

    MUID -> protocol above ~90%
        Knowing MUID tells you the protocol. It is protocol identity wearing a
        numeric costume, and it must come out of the parameter vector.
    MUID -> protocol below ~60%
        MUID is a coarser grouping than the protocol. It may still be useful
        AND admissible — a Siemens-standard family id is exactly the
        customer-agnostic descriptor Görtler pointed us at.
    protocol -> MUID also high
        The mapping is 1:1 in both directions, i.e. the same thing twice.""")

# COMMAND ----------

# =============================================================================
# 2 — DOES MUID PREDICT DURATION AS WELL AS THE PROTOCOL DOES?
#
# Purity says whether MUID *could* stand in for the protocol. This says whether
# it actually does, on the only axis we care about. Same estimator and split
# convention as every other number in 03b/03c, so the rows are comparable.
# =============================================================================

rule()
print(" 2. MUID AS A PREDICTOR IN ITS OWN RIGHT")
rule()

print(f"\n  {'predictor':<34} {'distinct':>9} {'R2':>8} {'MAE':>9}")
rule('-')
for label, keys in (
    ('serial + protocol  (the oracle)', protocols),
    ('serial + MUID', muids),
    ('serial alone', serials.astype(str)),
):
    r2, mae, _ = heldout_group_r2(keys, durations)
    print(f"  {label:<34} {len(set(keys.tolist())):>9,} {r2:>7.1f}% {mae:>8.1f}s")
rule('-')
print("  If MUID scores like the protocol, it IS the protocol for our purposes.")

# COMMAND ----------

# =============================================================================
# 3 — THE DECISION: DOES THE PARAMETER RESULT SURVIVE WITHOUT IT?
#
# Same field selection, same estimator and same row filters as 03c section D,
# so the "with" column reproduces the number already reported and the "without"
# column is the only thing that changes.
# =============================================================================

rule()
print(" 3. THE PARAMETER MODEL WITH AND WITHOUT THE IDENTIFIER FIELDS")
rule()

inventory = numeric_field_inventory(sequences)
fields = [
    row['key'] for row in inventory
    if row['numeric_pct'] > 90 and row['presence_pct'] > 20
]
clean_fields = [f for f in fields if f not in IDENTIFIER_FIELDS]
dropped = [f for f in fields if f in IDENTIFIER_FIELDS]

print(f"\n  {len(fields)} numeric fields, dropping {dropped} -> {len(clean_fields)}")
if not dropped:
    print("  !! None of the identifier fields are in the vector — nothing to test.")

# 03c section D's population, rebuilt here so this notebook stands alone.
# Aborts end on MRI_MSR_34 and bypass the 10s floor, so they are a different
# population and would otherwise flatter both columns equally.
is_abort = np.array([
    len(s.get('sequence', [])) > 0
    and int(s['sequence'][-1]) == SOURCEID_VOCAB['MRI_MSR_34']
    for s in sequences
])
clean = ~is_abort & (durations >= 10) & (durations <= 600)

scopes = np.array([s.get('sut_scope', '') for s in sequences])
subsets = [('all rows', clean)]
if scopes.any():
    subsets.append(('in-segment join only', clean & (scopes == 'inside')))

with_ids = numeric_matrix(sequences, fields)
without_ids = numeric_matrix(sequences, clean_fields)

print(f"\n  {'row subset':<26} {'n':>8} {'protocol':>10} {'with ids':>10} "
      f"{'WITHOUT':>10} {'delta':>8}")
rule('-')
for label, mask in subsets:
    if mask.sum() < 1000:
        print(f"  {label:<26} {mask.sum():>8,}   too few rows to score")
        continue
    _, p_mae, _ = heldout_group_r2(protocols[mask], durations[mask])
    _, with_mae = heldout_regressor_score(with_ids[mask], durations[mask], repeats=2)
    _, without_mae = heldout_regressor_score(without_ids[mask], durations[mask],
                                             repeats=2)
    print(f"  {label:<26} {mask.sum():>8,} {p_mae:>9.1f}s {with_mae:>9.1f}s "
          f"{without_mae:>9.1f}s {without_mae - with_mae:>+7.1f}s")
rule('-')

print("""
  THE VERDICT

    delta under ~1.5s
        MUID was not doing the work. The parameter result is clean and the
        number is quotable as it stands. Add the identifier fields to a
        permanent denylist anyway — a field that can leak should not be one
        retrain away from leaking.

    delta of several seconds
        MUID was carrying the protocol. The honest headline is the WITHOUT
        number, and every parameter MAE reported so far needs restating.
        Not a failure of the approach: it means the physics fields have not
        been given their fair test yet, because a shortcut was available.

  Either way the identifier fields must leave EXAMINATION_SEQPARAM_FEATURES'
  candidate pool. The whole premise is a model that transfers between
  customers, and an id does not transfer.""")
