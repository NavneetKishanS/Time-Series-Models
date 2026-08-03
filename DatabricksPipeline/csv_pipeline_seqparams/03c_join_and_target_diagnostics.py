# Databricks notebook source
"""
03c_join_and_target_diagnostics.py — why 03b's numbers cannot be read yet.

NO TRAINING, NO SPARK. Reads the same pkl 03b read.

03b returned two results that contradict each other, and both have to be
resolved before anything is reported to Görtler or spent on a retrain.

1. THE SUT PARAMETERS MAY BELONG TO A DIFFERENT SCAN.
   `sequence_binary` (DLL, 46 values, straight from the Siemens sequence
   binary) scored R2 3.6% / MAE 88.6s. `sequence_type` (12 values, from the
   MRI_MSR_100 message) scored 25.4% / 80.2s. Both describe the SAME scan, and
   the finer one cannot be seven times worse unless it is describing a
   different scan. `ST` agrees: correlation 0.205 with duration, and longer
   than the measurement containing it on 37.7% of rows.

   The mechanism is in 03_build_preprocessed_pkl.py:672 — the join takes the
   most recent MRI_SUT_1005 strictly BEFORE the segment start, and SUT fires at
   only ~0.25-0.31 the rate of MRI_MSR_100. So one SUT event is shared by 3-4
   consecutive measurements. Two readings:

     (a) SUT fires only when parameters CHANGE -> carrying it forward is
         correct, and the join is fine.
     (b) SUT is periodic/sampled -> ~70% of scans carry a neighbour's
         parameters, and every SUT-derived number in 03b is noise.

   Section A discriminates. Under (a) the sequence named in the SUT message
   must match the sequence named in the MSR_100 message; under (b) it will not.
   The test uses the repo's own classify_sequence_type on the DLL string, so no
   hand-made mapping is involved.

2. THE TARGET IS NOT THE QUANTITY WE BENCHMARKED ON.
   pkl total_duration: mean 115.7s, median 73s, sd 214.7s over 56,812 rows.
   Step 02's `duration`: mean 105.2s, median 89s, sd 91.0s over 40,921 rows.
   The median FELL while the mean rose and the sd more than doubled. Protocol
   scores 82.0% / 17.2s on the second and 63.3% / 31.2s on the first, so every
   MAE 03b printed is inflated by whatever this is.

   Measured locally 2026-08-03 by matching the 07-24 pkl against the step-02
   exam CSVs scan-by-scan (same serial, start within 90s): the two definitions
   AGREE. 73.0% of matched pairs have byte-identical durations and the pair
   correlation is 0.82. So this is a POPULATION difference, not a different
   quantity — the pkl carries ~15,800 measurements the CSVs never had, and they
   are both shorter on average (median 73 vs 88) and far more tailed (467 rows
   above 1000s against 1 in the CSVs). Two mechanisms are already identified:

     (a) OVERLAPPING SEGMENTS. Step 03 binds every MRI_MSR_100 to the next
         MRI_MSR_104/34, so two MSR_100 events before one terminator yield two
         segments ending at the same instant. 17.0% of pkl rows sit in such a
         cluster. Step 02 instead emits one row per finish event, dated from the
         most recent MSR_100. Keeping one segment per terminator locally moved
         the pkl to mean 94.1s / sd 125.5s, from 115.6s / 214.6s.
     (b) STEP 02 DROPS ROWS THE PKL KEEPS. `get_seq`
         (02_exam_preprocessing.py:100) requires BOTH `Sequence: '...'` and
         `Protocol: '...',` to parse; on failure `start` goes False and the
         whole measurement is discarded. Step 03's `_seq_type_from_msg` falls
         back to 'other' and keeps it. Step 02 also drops duration > 4000s and
         accepts MSR_26/22/24/25/40 as terminators, which step 03 does not —
         an error-terminated measurement therefore runs on to the next MSR_104.

   Sections B, B2 and C quantify what is left on the real pkl.

Run after 03b. Output decides whether Stage C happens, whether step 03's join
needs fixing first, or both.
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
    build_protocol_vocab, heldout_group_r2, normalize_protocol_name,
    protocol_id,
)
from AlternatingPipeline.data.parameter_analysis import (
    heldout_regressor_score, numeric_field_inventory, numeric_matrix,
    terminator_clusters,
)

PKL_PATH = os.environ.get('PKL_PATH', PKL_OUTPUT)


def rule(char='=', width=78):
    print(char * width)


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
protocol_keys = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])

# COMMAND ----------

# =============================================================================
# A — IS THE SUT MESSAGE ATTACHED TO THE RIGHT SCAN?
#
# classify_sequence_type() is the pipeline's own mapper from a raw sequence
# string to a SEQUENCE_TYPE_VOCAB id. Applied to the DLL leaf from the SUT
# message it must reproduce the sequence_type derived from the MRI_MSR_100
# message — IF both messages describe the same scan.
#
# Restricted to DLL values that classify to something other than 'other', since
# 'other' would agree with anything unclassifiable and inflate the result.
# =============================================================================

rule()
print(" A. SUT-TO-SEGMENT JOIN VALIDITY")
rule()

dll_classified = np.array([classify_sequence_type(b) for b in binaries])
other_id = SEQUENCE_TYPE_VOCAB['other']
testable = (binaries != '') & (dll_classified != other_id)
agrees = dll_classified == seq_types

print(f"\n  DLL present on            {100.0 * (binaries != '').mean():>6.1f}% of rows")
print(f"  DLL classifies to a named type on {100.0 * testable.mean():>6.1f}% "
      f"of rows  ({testable.sum():,} testable)")
print(f"\n  AGREEMENT between the SUT message's sequence and the MSR_100 "
      f"message's:\n    {100.0 * agrees[testable].mean():>6.1f}%   "
      f"({agrees[testable].sum():,} of {testable.sum():,})")

print(f"\n  Per-DLL breakdown (testable values, most common first):")
print(f"    {'DLL':<16} {'n':>7} {'expected':<12} {'agree':>7}  most common actual")
rule('-')
for binary, count in Counter(binaries[testable]).most_common(15):
    member = testable & (binaries == binary)
    expected = ID_TO_SEQUENCE_TYPE[classify_sequence_type(binary)]
    actual = Counter(seq_types[member]).most_common(2)
    actual_str = ', '.join(
        f"{ID_TO_SEQUENCE_TYPE[t]} {100.0 * n / member.sum():.0f}%"
        for t, n in actual
    )
    print(f"    {binary[:15]:<16} {member.sum():>7,} {expected:<12} "
          f"{100.0 * agrees[member].mean():>6.1f}%  {actual_str}")
rule('-')

print("""
  HOW TO READ THIS

    HIGH (>90%)   SUT fires when parameters CHANGE and carrying it forward is
                  correct. The join is sound, 03b's SUT numbers stand, and the
                  parameters genuinely do not carry the duration.
    LOW  (<60%)   SUT is periodic/sampled and the join at
                  03_build_preprocessed_pkl.py:672 attaches a NEIGHBOURING
                  scan's parameters to most rows. Every SUT-derived number in
                  03b — sections 3, 5, 6 and 7 — is then measuring noise, and
                  step 03 must be fixed and re-run before any of it means
                  anything.

  A low number here also re-explains the old TR/num_slices result. Those scored
  0.0% permutation importance and it was attributed to TR being one of six
  multiplicands in the acquisition-time formula. If the join is wrong, the
  simpler explanation is that TR belonged to a different scan on most rows.""")

# COMMAND ----------

# =============================================================================
# B — WHAT IS IN THE TARGET THAT WAS NOT IN THE BENCHMARK?
#
# Step 02's `duration` is endTime - startTime with its own outlier filter.
# The pkl's total_duration is timediffs[-1] - timediffs[0] over the segment,
# capped at MAX_EXAMINATION_DURATION (3000s) and floored at
# MIN_EXAMINATION_DURATION (10s) — EXCEPT for aborts, which bypass the floor
# (03_build_preprocessed_pkl.py:698-700).
# =============================================================================

rule()
print(" B. THE DURATION TARGET")
rule()

is_abort = np.array([
    len(s.get('sequence', [])) > 0
    and int(s['sequence'][-1]) == SOURCEID_VOCAB['MRI_MSR_34']
    for s in sequences
])
tokens = np.array([len(s.get('sequence', [])) for s in sequences])

print(f"\n  pkl total_duration   n {len(durations):>7,}  mean {durations.mean():>6.1f}s  "
      f"median {np.median(durations):>5.0f}s  sd {durations.std():>6.1f}s")
print(f"  step 02 `duration`   n  40,921  mean  105.2s  median    89s  sd   91.0s"
      f"   <- the benchmark")

print(f"\n  percentiles: " + "  ".join(
    f"p{p}={np.percentile(durations, p):.0f}s" for p in (1, 5, 25, 50, 75, 95, 99)))
print(f"  share below 10s: {100.0 * (durations < 10).mean():>5.1f}%   "
      f"above 600s: {100.0 * (durations > 600).mean():>5.1f}%   "
      f"above 1200s: {100.0 * (durations > 1200).mean():>5.1f}%")

print(f"\n  aborts (segment ends MRI_MSR_34): {100.0 * is_abort.mean():.1f}% "
      f"of rows — these BYPASS the {MIN_EXAMINATION_DURATION}s floor")
for label, mask in [('abort', is_abort), ('completed', ~is_abort)]:
    member = durations[mask]
    if member.size:
        print(f"    {label:<10} n {member.size:>7,}  mean {member.mean():>6.1f}s  "
              f"median {np.median(member):>5.0f}s  sd {member.std():>6.1f}s  "
              f"below 10s {100.0 * (member < 10).mean():>5.1f}%")

print(f"\n  per serial:")
print(f"    {'serial':<10} {'n':>7} {'mean':>8} {'median':>8} {'sd':>8} {'>600s':>7}")
rule('-')
for serial in np.unique(serials):
    member = durations[serials == serial]
    print(f"    {serial:<10} {member.size:>7,} {member.mean():>7.1f}s "
          f"{np.median(member):>7.0f}s {member.std():>7.1f}s "
          f"{100.0 * (member > 600).mean():>6.1f}%")
rule('-')

print(f"\n  segment length (events per measurement): "
      f"median {np.median(tokens):.0f}, p99 {np.percentile(tokens, 99):.0f}, "
      f"max {tokens.max()}")
_long = tokens > np.percentile(tokens, 99)
if _long.any():
    print(f"  the longest 1% of segments average {durations[_long].mean():.0f}s "
          f"against {durations[~_long].mean():.0f}s for the rest")

# COMMAND ----------

# =============================================================================
# B2 — HOW MANY SEGMENTS ARE THE SAME MEASUREMENT COUNTED TWICE?
#
# 03_build_preprocessed_pkl.py:676 walks every MRI_MSR_100 and ends its segment
# at the next MRI_MSR_104/34. Two MSR_100 events before one terminator therefore
# produce two segments that END AT THE SAME INSTANT — the earlier one having
# swallowed the gap and whatever ran in it.
#
# csv_pipeline/02_exam_preprocessing.py:220 does not do this: it emits one row
# per finish event and dates it from the MOST RECENT MSR_100. So the shorter
# member of each cluster is the measurement step 02 kept, and the longer member
# has no counterpart in the CSVs the ±15s benchmark was measured on.
#
# Measured on the local 07-24 pkl: 17.0% of rows sit in a cluster, and keeping
# one segment per terminator moved mean 115.6 -> 94.1s and sd 214.6 -> 125.5s.
# =============================================================================

rule()
print(" B2. OVERLAPPING SEGMENTS (same terminator, counted more than once)")
rule()

clusters = terminator_clusters(sequences)
cluster_size = clusters['size']
is_primary = clusters['is_primary']
shared = cluster_size > 1

print(f"\n  segments                          {len(sequences):>8,}")
print(f"  distinct terminator events        {int(is_primary.sum()):>8,}")
print(f"  segments sharing a terminator     {int(shared.sum()):>8,}   "
      f"({100.0 * shared.mean():.1f}% of rows)")

print(f"\n  {'segments per terminator':<26} {'clusters':>9} {'rows':>9} "
      f"{'mean dur':>10} {'median':>9}")
rule('-')
for size in sorted(set(cluster_size.tolist())):
    member = cluster_size == size
    if not member.any():
        continue
    print(f"    {size:<24} {int(member.sum() // size):>9,} {int(member.sum()):>9,} "
          f"{durations[member].mean():>9.1f}s {np.median(durations[member]):>8.0f}s")
rule('-')

print(f"\n  the two views of the target:")
print(f"    {'view':<34} {'n':>8} {'mean':>9} {'median':>8} {'sd':>9} {'>600s':>7}")
rule('-')
for label, mask in [
    ('as-is (what 03b used)', np.ones(len(durations), dtype=bool)),
    ('one segment per terminator', is_primary),
]:
    member = durations[mask]
    print(f"    {label:<34} {member.size:>8,} {member.mean():>8.1f}s "
          f"{np.median(member):>7.0f}s {member.std():>8.1f}s "
          f"{100.0 * (member > 600).mean():>6.1f}%")
print(f"    {'step 02 `duration` <- the benchmark':<34} {40921:>8,} "
      f"{105.2:>8.1f}s {89:>7.0f}s {91.0:>8.1f}s {0.1:>6.1f}%")
rule('-')

print("""
  If dedup alone moves the sd most of the way from 214.7s to 91.0s, the heavy
  tail is a segmentation artefact rather than a property of the scans, and
  step 03 should keep one segment per terminator before anything is retrained.
  What dedup does NOT close is the remaining row count: step 02 additionally
  discards every measurement whose MSR_100 message fails `get_seq`, which is a
  separate filter and needs its own decision.""")

# COMMAND ----------

# =============================================================================
# C — HOW MUCH OF THE 24.0s vs 15.3s GAP IS THE TARGET?
#
# Recompute the protocol benchmark on progressively cleaner targets. If it
# climbs back toward the 15.3s measured on step 02's `duration`, the gap is the
# target population and not the protocol signal — which also means every MAE in
# 03b should be re-read on the cleaned target.
# =============================================================================

rule()
print(" C. THE BENCHMARK ON PROGRESSIVELY CLEANER TARGETS")
rule()

names = [s.get('protocol_name', '') for s in sequences]
vocab = build_protocol_vocab(names, min_count=3)
protocol_ids = np.array([protocol_id(n, vocab) for n in names])
serial_protocol = np.array(
    [f"{a}|{b}" for a, b in zip(serials, protocol_ids)])

variants = [
    ('as-is (what 03b used)', np.ones(len(durations), dtype=bool)),
    ('one segment per terminator', is_primary),
    ('excluding aborts', ~is_abort),
    ('excluding aborts, >=10s', ~is_abort & (durations >= 10)),
    ('excluding aborts, 10-600s', ~is_abort & (durations >= 10) & (durations <= 600)),
    ('deduped + non-abort, 10-600s',
     is_primary & ~is_abort & (durations >= 10) & (durations <= 600)),
]
print(f"\n  {'target':<32} {'n':>8} {'sd':>8} {'protocol':>10} "
      f"{'serial+proto':>13}")
rule('-')
for label, mask in variants:
    if mask.sum() < 1000:
        continue
    _, p_mae, _ = heldout_group_r2(protocol_ids[mask], durations[mask])
    _, sp_mae, _ = heldout_group_r2(serial_protocol[mask], durations[mask])
    print(f"  {label:<32} {mask.sum():>8,} {durations[mask].std():>7.1f}s "
          f"{p_mae:>9.1f}s {sp_mae:>12.1f}s")
rule('-')
print("  step 02 `duration`, for reference      40,921    91.0s      17.2s"
      "         15.3s")

# COMMAND ----------

# =============================================================================
# D — WHAT DO THE PARAMETERS DO WHERE THE JOIN LOOKS VALID?
#
# If section A shows a broken join, this estimates what the parameters would be
# worth once it is fixed: restrict to rows where the SUT message and the
# MSR_100 message name the same sequence, on the cleanest target from C. Not a
# substitute for fixing the join — the surviving rows are a biased sample — but
# it is the difference between "parameters do not carry duration" and "we have
# never actually measured them".
# =============================================================================

rule()
print(" D. PARAMETERS ON JOIN-VALID ROWS ONLY")
rule()

clean = is_primary & ~is_abort & (durations >= 10) & (durations <= 600)
inventory = numeric_field_inventory(sequences)
fields = [
    row['key'] for row in inventory
    if row['numeric_pct'] > 90 and row['presence_pct'] > 20
]
parameters = numeric_matrix(sequences, fields)

print(f"\n  {len(fields)} numeric fields, target restricted to 10-600s "
      f"non-abort rows\n")
print(f"  {'row subset':<34} {'n':>8} {'protocol':>10} {'params GBM':>12}")
rule('-')
for label, mask in [
    ('all rows', clean),
    ('join-valid only (A agrees)', clean & testable & agrees),
    ('join-INVALID only (A disagrees)', clean & testable & ~agrees),
]:
    if mask.sum() < 1000:
        print(f"  {label:<34} {mask.sum():>8,}   too few rows to score")
        continue
    _, p_mae, _ = heldout_group_r2(serial_protocol[mask], durations[mask])
    _, g_mae = heldout_regressor_score(parameters[mask], durations[mask],
                                       repeats=2)
    print(f"  {label:<34} {mask.sum():>8,} {p_mae:>9.1f}s {g_mae:>11.1f}s")
rule('-')
print("""
  A large gap between the join-valid and join-invalid rows is direct confirmation
  that the join is the problem: the same parameters, the same model, the same
  target, differing only in whether the SUT message belongs to the scan.""")

# COMMAND ----------

rule()
print(" WHAT TO DO WITH THIS")
rule()
print("""
  A low  + C recovers  ->  BOTH are broken. Fix the join in step 03 (join the
                           SUT event INSIDE the segment, and record how far it
                           sits from the start so the next run can prove it),
                           reconcile the target population against step 02's,
                           re-run 03 and 03b. Do not report 03b's section 5 to
                           anyone in the meantime.
  A high + C recovers  ->  the join is fine and the target was the whole story.
                           Re-read 03b on the cleaned target; Stage C proceeds.
  A low  + C flat      ->  fix the join first, then re-examine the target — the
                           heavy tail is then a second, independent defect.
  A high + C flat      ->  the parameters really do not carry the duration, and
                           that is the honest finding to take back to Görtler.
""")
rule()
