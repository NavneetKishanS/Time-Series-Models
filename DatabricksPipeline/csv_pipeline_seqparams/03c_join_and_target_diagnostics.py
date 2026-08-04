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
from datetime import datetime

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
    _to_float, heldout_regressor_score, numeric_field_inventory, numeric_matrix,
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
# A2 — DID THE JOIN FIX WORK?
#
# The 2026-08-03 run of section A returned 45.7%, so step 03 was changed to
# prefer the MRI_SUT_1005 event emitted INSIDE the segment over the most recent
# one before it (`_choose_sut_row`). Each row now records which rule fired
# ('sut_scope'), how far the event sat from the segment start ('sut_offset_s'),
# and — where the two rules disagree — what the old rule would have said
# ('sut_binary_before').
#
# So section A can be split by scope. 'inside' is the fix; 'before' is the old
# behaviour still running as a fallback. If 'inside' does not score materially
# higher than 'before', the fix is wrong and the in-segment event is not the
# one describing the measurement either.
# =============================================================================

rule()
print(" A2. DID THE JOIN FIX WORK?")
rule()

scopes = np.array([s.get('sut_scope', '') for s in sequences])

if not scopes.any():
    print("\n  This pkl predates the join fix — no 'sut_scope' field. Re-run "
          "step 03, then this section reports itself.")
else:
    print(f"\n  {'scope':<10} {'rows':>8} {'share':>7} {'testable':>9} {'agree':>7}")
    rule('-')
    for scope in ('inside', 'before', 'none'):
        member = scopes == scope
        if not member.any():
            continue
        scored = member & testable
        agree_str = f"{100.0 * agrees[scored].mean():>6.1f}%" if scored.any() else "     --"
        print(f"  {scope:<10} {member.sum():>8,} {100.0 * member.mean():>6.1f}% "
              f"{scored.sum():>9,} {agree_str}")
    rule('-')

    offsets = np.array([float(s.get('sut_offset_s', np.nan)) for s in sequences])
    for scope in ('inside', 'before'):
        member = (scopes == scope) & np.isfinite(offsets)
        if member.any():
            values = np.abs(offsets[member])
            print(f"  {scope:<10} distance from segment start: "
                  f"p50 {np.percentile(values, 50):>6.1f}s   "
                  f"p90 {np.percentile(values, 90):>7.1f}s   "
                  f"max {values.max():>8.1f}s")

    # The direct before/after comparison, on the rows where the rules disagree.
    changed = np.array([bool(s.get('sut_binary_before'))
                        and s.get('sut_binary_before') != s.get('sequence_binary')
                        for s in sequences])
    moved = changed & testable
    if moved.any():
        old_ids = np.array([classify_sequence_type(s.get('sut_binary_before') or '')
                            for s in sequences])
        print(f"\n  On the {moved.sum():,} testable rows where the new rule picked a "
              f"DIFFERENT sequence than the old one:")
        print(f"    old most-recent-before rule agreed  "
              f"{100.0 * (old_ids[moved] == seq_types[moved]).mean():>6.1f}%")
        print(f"    new in-segment rule agrees          "
              f"{100.0 * agrees[moved].mean():>6.1f}%")
        print("    ^ this is the fix, isolated. If the second number is not "
              "clearly higher,\n      the in-segment event is not the right one "
              "either and section A stands.")

print("""
  WHAT TO DO WITH THIS

    inside is the large majority AND scores >90%
        The join is fixed. 03b's SUT sections can be re-run and believed.
    inside scores high but covers a minority of rows
        Fix landed, coverage did not. Either train on 'inside' rows only, or
        treat 'before' rows as missing parameters rather than wrong ones.
    inside and before score the same
        The fix is wrong. Neither event describes the measurement; the SUT
        message is periodic and the parameter route needs a different source.""")

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
if shared.mean() < 0.01:
    print("\n  ~0% — this pkl was built with DEDUPE_SHARED_TERMINATOR on, so the "
          "\n  overlap is already gone at build time and the comparison below is "
          "\n  a no-op. That is the confirmation, not a missing result.")

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

# =============================================================================
# E — WHY IS ST LONGER THAN THE MEASUREMENT CONTAINING IT?
#
# 03b section 7 (2026-08-04) has ST correlating 0.764 with duration — up from
# 0.205 under the old join, so the join fix clearly reached ST too. But ST
# still EXCEEDS the measured duration on 12.7% of rows, and its median (77s)
# is above the duration median (57s). A planned acquisition time cannot be
# longer than the span containing it, so one of two things is true:
#
#   (a) a units or per-family scaling problem in ST, or
#   (b) ST describes MORE than one of our segments.
#
# (b) has a concrete mechanism. `CONC` (concatenations) reaches 36 at p99 and
# `REP` (repetitions) reaches 89, so one protocol can run as several
# back-to-back measurements, each with its own MSR_100 -> MSR_104 pair, all
# carrying the SAME SUT message and therefore the SAME whole-protocol ST.
#
# `MUID` is 100% present with 3,188 distinct values against 3,051 distinct
# protocol names — so it identifies the protocol, not the individual run. A
# maximal run of consecutive segments on one scanner sharing a MUID is
# therefore exactly one protocol's worth of measurement. If ST fits that RUN
# rather than the single segment, (b) is the answer and ST is usable once
# summed to the right unit.
# =============================================================================

rule()
print(" E. ST vs THE SEGMENT, AND ST vs THE PROTOCOL RUN")
rule()

st_values = np.array([_to_float((s.get('sut_raw') or {}).get('ST')) for s in sequences])
muids = np.array([str((s.get('sut_raw') or {}).get('MUID', '')) for s in sequences])
starts = [s.get('start_datetime') for s in sequences]
has_st = np.isfinite(st_values) & (st_values > 0) & (durations > 0)

if not has_st.any():
    print("\n  No usable ST in this pkl — nothing to reconcile.")
else:
    print(f"\n  ST usable on {100.0 * has_st.mean():.1f}% of rows\n")
    print(f"  {'row subset':<28} {'n':>8} {'ST>dur':>8} {'median ST/dur':>14}")
    rule('-')
    for label, mask in (
        ('all rows', has_st),
        ('in-segment join only', has_st & (scopes == 'inside')),
        ('before-join only', has_st & (scopes == 'before')),
    ):
        if not mask.any():
            continue
        ratio = st_values[mask] / durations[mask]
        print(f"  {label:<28} {mask.sum():>8,} "
              f"{100.0 * (ratio > 1.0).mean():>7.1f}% {np.median(ratio):>14.2f}")
    rule('-')
    print("  If 'before' is much worse than 'in-segment', the residual ST "
          "problem is\n  just the rows we already know carry the wrong "
          "message.")

    # Maximal runs of consecutive segments on one scanner sharing a MUID.
    order = sorted(range(len(sequences)),
                   key=lambda i: (serials[i], starts[i] is None,
                                  starts[i] or datetime.min))
    run_id = np.full(len(sequences), -1, dtype=int)
    runs, current = [], []
    previous = None
    for position in order:
        key = (serials[position], muids[position])
        if previous is not None and key != previous:
            runs.append(current)
            current = []
        current.append(position)
        previous = key
    if current:
        runs.append(current)
    for index, members in enumerate(runs):
        run_id[members] = index

    run_duration = np.zeros(len(sequences))
    run_size = np.zeros(len(sequences), dtype=int)
    for members in runs:
        run_duration[members] = durations[members].sum()
        run_size[members] = len(members)

    multi = has_st & (run_size > 1)
    print(f"\n  Consecutive same-MUID runs: {len(runs):,} runs over "
          f"{len(sequences):,} segments "
          f"({100.0 * (run_size > 1).mean():.1f}% of rows sit in a run of 2+)")

    if multi.any():
        print(f"\n  {'comparison':<34} {'n':>8} {'ST>span':>9} {'median ST/span':>15}")
        rule('-')
        single = st_values[multi] / durations[multi]
        summed = st_values[multi] / run_duration[multi]
        print(f"  {'ST vs its own segment':<34} {multi.sum():>8,} "
              f"{100.0 * (single > 1.0).mean():>8.1f}% {np.median(single):>15.2f}")
        print(f"  {'ST vs the whole MUID run':<34} {multi.sum():>8,} "
              f"{100.0 * (summed > 1.0).mean():>8.1f}% {np.median(summed):>15.2f}")
        rule('-')
        print("""
  READ IT LIKE THIS

    'ST vs the whole MUID run' lands near 1.0 and its ST>span share collapses
        ST is the PROTOCOL's acquisition time, not the segment's. It is usable
        — but as a per-run feature, and the model's unit of prediction should
        arguably be the run too. This also explains 03b section 7 completely.
    Both rows look the same
        The run grouping is not the explanation. Suspect units or a per-family
        scaling in ST, and keep ST out of the feature set until it is settled.""")
    else:
        print("\n  No multi-segment MUID runs — (b) is ruled out, so ST's "
              "excess is a units\n  or per-family scaling problem. Keep it out "
              "of the feature set for now.")

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
