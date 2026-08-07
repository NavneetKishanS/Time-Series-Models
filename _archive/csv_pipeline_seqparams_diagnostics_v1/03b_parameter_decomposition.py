# Databricks notebook source
"""
03b_parameter_decomposition.py — can sequence + parameters replace the protocol?

NO TRAINING. Reads the pkl step 03 wrote and answers Görtler's action item from
the 2026-07-31 call, before anyone spends a retrain on it.

His position: the customer-authored protocol name must not be a feature in the
general model (3.8M names across 15k customers; copying a Siemens preset
destroys the link to its id, so a protocol-keyed model learns one site and
transfers to none). The gold standard is reaching the same ~±15s from
**sequence + sequence parameters**. Our held-out measurement puts scanner +
protocol at 15.3s MAE, which confirms his estimate and makes protocol identity
the benchmark rather than an input.

His sharper claim is that ±15s is not a real ceiling: if protocols ran fixed the
jitter would be ~1s, and it is 15s only because operators adjust parameters
after loading the protocol. That claim has a precondition this notebook
measures directly — section 6 — namely that the SUT message records the
EXECUTED parameters and not the protocol's stored defaults.

Run after 03_build_preprocessed_pkl.py, before 04_train_models.py.
Requires a pkl carrying 'sut_raw' (step 03's widened capture).
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
    build_protocol_vocab, heldout_group_r2, normalize_protocol_name,
    protocol_id,
)
from AlternatingPipeline.data.parameter_analysis import (
    generated_protocol_name, heldout_regressor_score, numeric_field_inventory,
    numeric_matrix, within_group_variation,
)

PKL_PATH = os.environ.get('PKL_PATH', PKL_OUTPUT)
MIN_GROUP_SIZE = int(os.environ.get('MIN_GROUP_SIZE', '10'))
PROTOCOL_MIN_COUNT = int(os.environ.get('PROTOCOL_MIN_COUNT', '3'))

# The bar. Görtler named ±15s as the gold standard; we measured scanner +
# protocol at 15.3s MAE on the real exam CSVs, so the two agree.
TARGET_MAE_S = float(os.environ.get('TARGET_MAE_S', '15.0'))


def rule(char='=', width=78):
    print(char * width)


with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

sequences = data['examination']
print(f"Loaded {len(sequences):,} examination sequences from {PKL_PATH}")

if not any('sut_raw' in s for s in sequences):
    raise RuntimeError(
        "This pkl carries no 'sut_raw' — it predates the widened SUT capture. "
        "Re-run csv_pipeline_seqparams/03_build_preprocessed_pkl.py. Without "
        "the full message there is nothing to decompose: SUT_FIELD_MAP has "
        "never covered more than a fraction of the ~60 fields per message."
    )

durations = np.array([float(s.get('total_duration', 0.0)) for s in sequences])
keep = durations > 0
sequences = [s for s, k in zip(sequences, keep) if k]
durations = durations[keep]
print(f"{len(sequences):,} with a positive duration.  "
      f"mean {durations.mean():.1f}s  median {np.median(durations):.0f}s  "
      f"sd {durations.std():.1f}s")

# COMMAND ----------

# =============================================================================
# 1 — FIELD INVENTORY
#
# Sets the EXAMINATION_SEQPARAM_SCALE divisors. An entry there is REQUIRED
# before a name is added to EXAMINATION_SEQPARAM_FEATURES: unscaled
# large-magnitude numerics silently erase categorical conditioning through
# LayerNorm, which has cost this project three multi-week flat-duration
# incidents. Divisors come from the observed p99 here, not from sampled
# messages.
# =============================================================================

rule()
print(" 1. SUT FIELD INVENTORY — every key in the message")
rule()

inventory = numeric_field_inventory(sequences)
print(f"\n  {len(inventory)} distinct keys across {len(sequences):,} sequences\n")
print(f"  {'key':<8} {'present':>8} {'distinct':>9} {'numeric':>8} "
      f"{'p50':>10} {'p99':>10} {'suggested':>10}")
print(f"  {'':<8} {'':>8} {'':>9} {'':>8} {'':>10} {'':>10} {'divisor':>10}")
rule('-')
for row in inventory:
    divisor = ''
    if np.isfinite(row['p99']) and row['numeric_pct'] > 90:
        magnitude = max(1.0, 10 ** np.floor(np.log10(max(row['p99'], 1.0))))
        divisor = f"{magnitude:.0f}"
    print(f"  {row['key']:<8} {row['presence_pct']:>7.1f}% {row['distinct']:>9,} "
          f"{row['numeric_pct']:>7.1f}% {row['p50']:>10.1f} {row['p99']:>10.1f} "
          f"{divisor:>10}")
rule('-')
print("\n  'present' below 100% means the field is SEQUENCE-SCOPED — it does not\n"
      "  apply to every sequence family (TF is on TSE-family messages, absent\n"
      "  from ep2d_diff which uses an echo factor). Promoting one of those into\n"
      "  EXAMINATION_SEQPARAM_FEATURES needs a presence flag or scoping by\n"
      "  family: _safe_float defaults a missing key to 0.0, and 0 is a real\n"
      "  value for most of these.")

# COMMAND ----------

# =============================================================================
# 2 — BASELINES: what the model uses today, and the bar it has to clear
# =============================================================================

rule()
print(" 2. BASELINES (held out: group means fitted on 80%, scored on 20%)")
rule()

names = [s.get('protocol_name', '') for s in sequences]
vocab = build_protocol_vocab(names, min_count=PROTOCOL_MIN_COUNT)
protocol_ids = np.array([protocol_id(n, vocab) for n in names])
serials = np.array([int(s.get('serial_idx', 0)) for s in sequences])
seq_types = np.array([int(s.get('sequence_type', 0)) for s in sequences])
serial_protocol = np.array([f"{a}|{b}" for a, b in zip(serials, protocol_ids)])

rows = [
    ('sequence_type', seq_types, 'what the model uses today'),
    ('serial (scanner)', serials, ''),
    ('protocol', protocol_ids, 'BENCHMARK — not a model input'),
    ('serial + protocol', serial_protocol, 'BENCHMARK — the ±15s claim'),
]
print(f"\n  {'predictor':<22} {'distinct':>9} {'R2':>8} {'MAE':>9}  note")
rule('-')
protocol_mae = None
for label, labels_arr, note in rows:
    r2, mae, _ = heldout_group_r2(labels_arr, durations)
    if label == 'serial + protocol':
        protocol_mae = mae
    print(f"  {label:<22} {len(np.unique(labels_arr)):>9,} {r2:>7.1f}% "
          f"{mae:>8.1f}s  {note}")
rule('-')
print(f"\n  Görtler's gold standard: {TARGET_MAE_S:.0f}s. Measured here for "
      f"serial+protocol: {protocol_mae:.1f}s.\n"
      f"  Everything below has to be read against that number.")

# COMMAND ----------

# =============================================================================
# 3 — CUSTOMER-AGNOSTIC CATEGORICALS
#
# DLL (Siemens sequence binary) and OR (orientation) are Siemens-standard, so
# unlike the protocol name they mean the same thing at every customer.
# =============================================================================

rule()
print(" 3. CUSTOMER-AGNOSTIC CATEGORICALS")
rule()

binaries = np.array([s.get('sequence_binary') or '?' for s in sequences])
orientations = np.array([s.get('orientation') or '?' for s in sequences])
binary_orientation = np.array(
    [f"{a}|{b}" for a, b in zip(binaries, orientations)])

print(f"\n  sequence_binary coverage: "
      f"{100.0 * (binaries != '?').mean():.1f}%  "
      f"({len(np.unique(binaries))} distinct)")
print(f"  orientation coverage:     "
      f"{100.0 * (orientations != '?').mean():.1f}%  "
      f"({len(np.unique(orientations))} distinct)\n")

print(f"  {'predictor':<34} {'distinct':>9} {'R2':>8} {'MAE':>9}")
rule('-')
for label, labels_arr in [
    ('sequence_binary (DLL)', binaries),
    ('sequence_binary + orientation', binary_orientation),
    ('serial + sequence_binary + orient.', np.array(
        [f"{a}|{b}" for a, b in zip(serials, binary_orientation)])),
]:
    r2, mae, _ = heldout_group_r2(labels_arr, durations)
    print(f"  {label:<34} {len(np.unique(labels_arr)):>9,} {r2:>7.1f}% "
          f"{mae:>8.1f}s")
rule('-')

# COMMAND ----------

# =============================================================================
# 4 — RECONSTRUCTED "GENERATED PROTOCOL NAME"
#
# Görtler: Siemens already ships a parameter-derived protocol descriptor in the
# Cubes ("transversal, T1-weighted, short TR"), built from 3-4 parameters, as
# the customer-agnostic substitute for the customer's name. We cannot read that
# field, but every input is in the SUT message. This is our reconstruction —
# and it does not depend on getting Cube access.
# =============================================================================

rule()
print(" 4. RECONSTRUCTED PARAMETER-DERIVED PROTOCOL DESCRIPTOR")
rule()

descriptors = np.array([generated_protocol_name(s) for s in sequences])
serial_descriptors = np.array(
    [f"{a}|{b}" for a, b in zip(serials, descriptors)])

print(f"\n  {'predictor':<34} {'distinct':>9} {'R2':>8} {'MAE':>9}")
rule('-')
for label, labels_arr in [
    ('generated descriptor', descriptors),
    ('serial + generated descriptor', serial_descriptors),
]:
    r2, mae, _ = heldout_group_r2(labels_arr, durations)
    print(f"  {label:<34} {len(np.unique(labels_arr)):>9,} {r2:>7.1f}% "
          f"{mae:>8.1f}s")
rule('-')

_counts = {}
for descriptor in descriptors:
    _counts[descriptor] = _counts.get(descriptor, 0) + 1
print("\n  most common descriptors:")
for descriptor, count in sorted(_counts.items(), key=lambda kv: -kv[1])[:12]:
    member = durations[descriptors == descriptor]
    print(f"    {descriptor:<44} {count:>6,}  {member.mean():>6.0f}s "
          f"+/- {member.std():>5.0f}s")

# COMMAND ----------

# =============================================================================
# 5 — THE REAL TEST: full parameter vector, gradient boosted
#
# Group means are not a valid estimator for continuous features, so this uses a
# GBM on the SAME split convention. It is the honest ceiling for what any model
# can extract from these parameters — the transformer should be judged against
# it, not against its own previous checkpoint.
# =============================================================================

rule()
print(" 5. FULL PARAMETER VECTOR (gradient boosted, held out)")
rule()

# Numeric fields present on a reasonable share of sequences. Rare fields add
# columns that are almost entirely NaN and cost more than they carry.
numeric_fields = [
    row['key'] for row in inventory
    if row['numeric_pct'] > 90 and row['presence_pct'] > 20
]
print(f"\n  {len(numeric_fields)} numeric fields: {numeric_fields}\n")

parameters = numeric_matrix(sequences, numeric_fields)
categorical_codes = np.column_stack([
    np.unique(binaries, return_inverse=True)[1],
    np.unique(orientations, return_inverse=True)[1],
    serials,
]).astype(float)
combined = np.hstack([categorical_codes, parameters])

print(f"  {'feature set':<40} {'R2':>8} {'MAE':>9}")
rule('-')
for label, features in [
    ('sequence_binary + orientation + serial', categorical_codes),
    ('parameters only', parameters),
    ('sequence + orientation + parameters', combined),
]:
    r2, mae = heldout_regressor_score(features, durations)
    verdict = '  <-- clears the bar' if mae <= TARGET_MAE_S else ''
    print(f"  {label:<40} {r2:>7.1f}% {mae:>8.1f}s{verdict}")
rule('-')

# Görtler's ~0.25 ST/duration ratio says ST covers the acquisition and the rest
# is prep/positioning/adjustment overhead. Predicting the OVERHEAD and adding a
# known acquisition time back is a different, better-posed problem than
# predicting the whole span with ST as one input among many. Costs nothing to
# check here.
st_values = numeric_matrix(sequences, ['ST']).ravel()
has_st = np.isfinite(st_values)
if has_st.sum() > 100:
    overhead = durations[has_st] - st_values[has_st]
    negative_pct = 100.0 * (overhead < 0).mean()
    print(f"\n  Alternative framing — predict (duration - ST), add ST back:")
    if negative_pct > 5:
        # A nominal acquisition time cannot exceed the measured span it sits
        # inside. More than a rounding-level share of negatives means ST is not
        # in seconds, is not the acquisition time, or is joined to the wrong
        # sequence — any of which invalidates the framing rather than the fit.
        print(f"    SKIPPED — ST exceeds the measured duration on "
              f"{negative_pct:.1f}% of rows. A planned acquisition time cannot\n"
              f"    be longer than the span containing it; check units and the "
              f"SUT-to-segment join\n    before reading anything into ST.")
    else:
        r2, mae = heldout_regressor_score(combined[has_st], overhead)
        print(f"    overhead model MAE {mae:.1f}s on {has_st.sum():,} rows "
              f"(ST known); the same MAE\n    applies to the full duration, "
              f"since ST is added back exactly.")
        print(f"    overhead is "
              f"{100.0 * overhead.mean() / durations[has_st].mean():.0f}% of "
              f"the total span — mean {overhead.mean():.1f}s of "
              f"{durations[has_st].mean():.1f}s")

# COMMAND ----------

# =============================================================================
# 6 — THE DECISION GATE: executed parameters, or protocol defaults?
#
# Görtler's whole argument that parameters can beat the protocol lookup rests
# on operators adjusting parameters after loading the protocol — slice count
# 15->17 after the localizer, SAR conflicts forcing fewer slices. If the SUT
# message instead records the protocol's stored defaults, that adjustment is
# invisible to us and no amount of parameter modelling reaches below the
# protocol lookup's own error.
# =============================================================================

rule()
print(" 6. DOES THE MESSAGE CARRY EXECUTED VALUES OR PROTOCOL DEFAULTS?")
rule()

# Keyed on the NORMALIZED PROTOCOL NAME, not the vocab id. Every protocol below
# the frequency floor shares RARE_PROTOCOL_ID, so an id-keyed grouping would
# pool hundreds of unrelated protocols into one bucket that varies trivially
# and would read as false evidence for the executed-values case.
protocol_keys = np.array([
    f"{s.get('serial_idx', 0)}|{normalize_protocol_name(s.get('protocol_name'))}"
    for s in sequences
])
print(f"\n  Within (serial, protocol) groups of at least {MIN_GROUP_SIZE} rows"
      f" — {len(np.unique(protocol_keys)):,} groups before the size filter:\n")
print(f"  {'parameter':<20} {'groups':>8} {'varying':>9} {'mean sd':>10} "
      f"{'coverage':>10}")
rule('-')
for field in ['SLC', 'SLT', 'AVG', 'TR', 'ST', 'PEL', 'FOV']:
    values = numeric_matrix(sequences, [field]).ravel()
    stats = within_group_variation(protocol_keys, values,
                                   min_group_size=MIN_GROUP_SIZE)
    print(f"  {field:<20} {stats['groups']:>8,} {stats['varying_pct']:>8.1f}% "
          f"{stats['mean_within_sd']:>10.2f} {stats['coverage_pct']:>9.1f}%")
rule('-')
print("""
  HOW TO READ THIS — 'varying' is the share of protocols in which the parameter
  is not constant across runs.

    HIGH (say >50% for SLC)  the message records what was EXECUTED. Görtler is
                             right, the post-load adjustment is visible, and
                             the parameter route can go below the protocol
                             lookup's error. Proceed to Stage C.
    LOW  (near 0%)           the message records the protocol's DEFAULTS. The
                             adjustment never reaches us, parameters cannot
                             beat a protocol lookup, and the honest finding is
                             that this data source is exhausted. Report back
                             instead of spending a retrain.""")

# COMMAND ----------

# =============================================================================
# 7 — ST BEHAVIOUR (characterisation, not a gate)
#
# Leakage is settled: ST is confirmed to be decided BEFORE the measurement, so
# it is a planned value and admissible. What is still worth knowing is how much
# of the span it covers.
# =============================================================================

rule()
print(" 7. ST vs MEASURED DURATION")
rule()

if has_st.sum() > 100:
    st_known, duration_known = st_values[has_st], durations[has_st]
    correlation = float(np.corrcoef(st_known, duration_known)[0, 1])
    ratio = float((st_known / np.maximum(duration_known, 1e-6)).mean())
    print(f"\n  ST present on {100.0 * has_st.mean():.1f}% of sequences")
    print(f"  correlation with total_duration: {correlation:.3f}")
    print(f"  mean ST / duration ratio:        {ratio:.3f}")
    print(f"  mean ST {st_known.mean():.1f}s vs mean duration "
          f"{duration_known.mean():.1f}s")
    if ratio > 1.1:
        print(f"\n  !! ST is LONGER than the span it should sit inside. A "
              f"planned acquisition\n     time cannot exceed the measured "
              f"duration — this is a units or join\n     problem, not a "
              f"feature. Do not use ST until it is resolved.")
    elif 0.9 <= ratio <= 1.1 and correlation > 0.95:
        print(f"\n  !! ST is ~equal to the measured duration. That contradicts "
              f"the confirmation\n     that it is decided before the "
              f"measurement — stop and re-check the field\n     before using "
              f"it as a feature.")
    else:
        print(f"\n  ST covers the acquisition; the remaining "
              f"{100.0 * (1 - ratio):.0f}% is prep, positioning and\n"
              f"  adjustment. That is the part the model actually has to "
              f"learn — see the\n  overhead framing in section 5.")
else:
    print("\n  ST is present on too few sequences to characterise.")

# COMMAND ----------

rule()
print(" NEXT")
rule()
print("""
  Section 6 decides. If parameters vary within protocol and section 5 lands
  near the bar, proceed to Stage C: promote the validated fields into
  EXAMINATION_SEQPARAM_FEATURES with a divisor each from section 1, give
  sequence_binary/orientation embeddings AND per-position duration biases, and
  leave num_protocols unset so the protocol path never constructs.

  A per-position path is not optional. TR/num_slices were correctly wired as
  scaled numerics on the flat conditioning tensor, trained cleanly, and scored
  0.0% permutation importance — that tensor reaches the duration head only
  through the single prepended conditioning token. sequence_type scores 57.8%
  because it also has duration_seq_type_bias.
""")
rule()
