# Databricks notebook source
# Databricks notebook — MRI_SUT_1005 parameter discovery (data-discovery stage)
#
# Purpose: reverse-engineer the raw MRI_SUT_1005 message format so the real
# SUT_SLOT_MAP / EXAMINATION_SEQPARAM_FEATURES in csv_pipeline_seqparams/config.py
# can be filled in with confirmed values instead of placeholders.
#
# Background: on the call, Görtler described a scanner screen with ~59-60
# numbered "SD<n>" sequence-protocol parameters (TR, number of slices,
# trigger/gating mode, ...), naming SD58 ("the second-to-last one") as
# scanning time. Navneet confirmed this unprompted, so he likely has more of
# the slot map memorized — ask him directly if this notebook's output doesn't
# make the mapping self-evident (see the fallback ladder at the bottom).
#
# MRI_SUT_1005 is ALREADY being queried out of hive_metastore.eventlog.common_eventlog
# today (it's in EXAM_EXTRA_SOURCE_TYPES in csv_pipeline/config.py, and in
# Patient_Exam_updated_with_new_columns_March2026.ipynb's own Spark filter) —
# but no code anywhere branches on it, so its Message text is read then
# silently discarded every time. This notebook is the first place that
# actually looks at that text.
#
# This is a scratch/exploratory notebook, not a pipeline stage — it produces
# a slot map by hand-inspection, which then gets copied into
# csv_pipeline_seqparams/config.py's SUT_SLOT_MAP / EXAMINATION_SEQPARAM_FEATURES
# / EXAMINATION_SEQPARAM_SCALE once confirmed.

# COMMAND ----------

import re
from collections import Counter

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

EVENTLOG_TABLE = "hive_metastore.eventlog.common_eventlog"

# Start with 2-3 of the 10 target serials — no need to query all 10 for
# format discovery. Widen if the format turns out to vary by scanner model.
DISCOVERY_SERIALS = [183242, 176148, 176227]
DATE_START = "2024-01-01"
DATE_END = "2024-01-31"

# COMMAND ----------

# =============================================================================
# STEP 1 — Query raw MRI_SUT_1005 rows and inspect the Message text format
# =============================================================================

raw_sut_spark = (
    spark.table(EVENTLOG_TABLE)
    .filter(F.col("SerialNumber").cast("long").isin(DISCOVERY_SERIALS))
    .filter(F.col("MessageIdentification") == "MRI_SUT_1005")
    .filter(F.to_date(F.col("EventDateTime")).between(DATE_START, DATE_END))
    .select(
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
        "EventDateTime",
        F.col("Message").cast("string").alias("Message"),
        "Line",
    )
    .orderBy("SerialNumber", "EventDateTime", "Line")
)
print(f"MRI_SUT_1005 rows found: {raw_sut_spark.count():,}")
raw_sut_pd = raw_sut_spark.toPandas()
display(raw_sut_pd.head(20))

# COMMAND ----------

# Print a handful of full, untruncated raw message strings — this is the
# main thing to eyeball. Look for: labeled "SD<n>: value" pairs (matching
# the established convention in this codebase — MRI_MSR_100 messages carry
# labeled `Sequence: '...'` / `Protocol: '...'` fields, MRI_CCS_1008 carries
# labeled `CoilID '...'` fields — so labeled fields are the more likely
# hypothesis for this scanner's message format too), OR an unlabeled
# positional/delimited list where "position 58" must be counted directly.
for _, row in raw_sut_pd.head(5).iterrows():
    print("=" * 100)
    print(f"Serial {row['SerialNumber']}  {row['EventDateTime']}")
    print(row['Message'])

# COMMAND ----------

# =============================================================================
# STEP 2 — Temporal relationship to MRI_MSR_100 (does SUT fire once per
# measurement like coil events, or periodically/forward-filled like PTAB?)
# =============================================================================

msr100_spark = (
    spark.table(EVENTLOG_TABLE)
    .filter(F.col("SerialNumber").cast("long").isin(DISCOVERY_SERIALS))
    .filter(F.col("MessageIdentification") == "MRI_MSR_100")
    .filter(F.to_date(F.col("EventDateTime")).between(DATE_START, DATE_END))
    .select(
        F.col("SerialNumber").cast("long").alias("SerialNumber"),
        "EventDateTime",
        F.col("Message").cast("string").alias("Message"),
    )
)
msr100_pd = msr100_spark.toPandas()

for serial in DISCOVERY_SERIALS:
    n_sut = (raw_sut_pd['SerialNumber'] == serial).sum()
    n_msr100 = (msr100_pd['SerialNumber'] == serial).sum()
    ratio = n_sut / n_msr100 if n_msr100 else float('nan')
    print(f"Serial {serial}: {n_sut} SUT events, {n_msr100} MSR_100 (measurement-start) "
          f"events, ratio={ratio:.2f}")
    # ratio near 1.0 => SUT fires once per measurement (like coil events,
    #   bisect "most-recent-before-segment-start" lookup — see
    #   03_build_preprocessed_pkl.py's coil_rows / _pc pattern).
    # ratio << 1.0 => SUT fires periodically/rarely (like PTAB — forward-fill
    #   across the whole timeline instead).

# COMMAND ----------

# =============================================================================
# STEP 3 — Attempt 1: labeled "SD<n>: value" (or similar) key-value parsing
# =============================================================================
# This regex is a FIRST GUESS based on this codebase's established message
# conventions (labeled fields elsewhere). Inspect the printed matches against
# the raw text above — if they don't look right, the format hypothesis in
# STEP 4 (positional) is the fallback.

_SD_LABELED_RE = re.compile(r"SD(\d+)\s*[:=]\s*'?([^,'\n]+)'?")


def parse_sut_labeled(message):
    """First-guess parser: labeled SD<n>: value pairs. Returns {slot_int: raw_value_str}."""
    if not isinstance(message, str):
        return {}
    return {int(n): v.strip() for n, v in _SD_LABELED_RE.findall(message)}


sample_parsed = raw_sut_pd['Message'].head(20).apply(parse_sut_labeled)
n_matched = sum(1 for d in sample_parsed if d)
print(f"Labeled-pair parse matched {n_matched}/{len(sample_parsed)} sampled messages")
if n_matched:
    print("Example parse:", sample_parsed.iloc[sample_parsed.apply(len).idxmax()])
    slot_counts = Counter(k for d in sample_parsed for k in d)
    print(f"Distinct slot numbers seen: {sorted(slot_counts)}")
    print(f"Slot count range across samples: "
          f"{min(len(d) for d in sample_parsed if d)}-{max(len(d) for d in sample_parsed if d)} "
          f"(Görtler implied ~58-60 total)")

# COMMAND ----------

# =============================================================================
# STEP 3b — Attempt 2: positional / delimited list (fallback hypothesis)
# =============================================================================
# If STEP 3 found few/no labeled matches, the format may be an unlabeled,
# comma- or pipe-delimited positional list where "SD58" means literally the
# 58th value in the list, not a "SD58:" label. Try a couple of common
# delimiters and see if a stable field count near ~58-60 emerges.

for delim in [',', '|', ';']:
    lengths = raw_sut_pd['Message'].dropna().apply(lambda m: len(m.split(delim)))
    if len(lengths):
        print(f"delimiter={delim!r}: length mean={lengths.mean():.1f} "
              f"median={lengths.median():.0f} mode={lengths.mode().tolist()}")

# COMMAND ----------

# =============================================================================
# STEP 4 — VALIDATE against the already-trusted anchor: real measured duration
# -----------------------------------------------------------------------------
# This is the decisive check, independent of which parse attempt above looks
# right. For a sample of measurements, compute the parsed SD58 candidate
# value and compare it against the ALREADY-COMPUTED real duration
# (endTime - startTime, from 02_exam_preprocessing.py's measurement_time()).
# Near-1:1 correlation confirms BOTH the slot-alignment hypothesis AND
# Görtler's leakage claim empirically, rather than by assumption.
#
# This cell is intentionally left for hands-on iteration once STEP 3 or 3b
# identifies a working parse — join the parsed SD58 (or whichever slot looks
# like scanning time) to the nearest-following MRI_MSR_104/34 finish event's
# (endTime - startTime) for the same measurement, and check the ratio.
# A ratio near 1.0 across many samples is the go/no-go signal.
# =============================================================================

print("Fill in once STEP 3/3b identifies a candidate parse function and slot number.")

# COMMAND ----------

# =============================================================================
# FALLBACK LADDER if the format resists reverse-engineering, in order:
#   1. Ask Navneet directly — he named "SD58, second-to-last" unprompted on
#      the call, so he likely has more of the slot map memorized already.
#   2. Look for Siemens protocol-dump documentation outside this repo.
#   3. Accept partial coverage: TR + num_slices + trigger_mode + confirmed
#      SD58 is sufficient to proceed — full ~59-slot coverage is not
#      required by the original ask.
#   4. Last resort: fall back to whatever partial signal already exists in
#      the Protocol/Sequence name strings (get_seq() / classify_sequence_type
#      in 02_exam_preprocessing.py / 03_build_preprocessed_pkl.py).
#
# OUTPUT OF THIS NOTEBOOK: once slots are confirmed, copy them into
# DatabricksPipeline/csv_pipeline_seqparams/config.py:
#   SUT_SLOT_MAP = {58: 'scanning_time', <n>: 'TR', <n>: 'num_slices',
#                   <n>: 'trigger_mode', ...}
#   EXAMINATION_SEQPARAM_FEATURES = ['TR', 'num_slices']  # numeric only —
#       trigger_mode is categorical, goes through TRIGGER_MODE_VOCAB instead
#   EXAMINATION_SEQPARAM_SCALE = [<TR divisor>, <num_slices divisor>]  # see
#       that file's LayerNorm-erasure warning — REQUIRED, not optional
# =============================================================================
