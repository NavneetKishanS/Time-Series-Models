# Databricks notebook — Assemble and Save preprocessed_data.pkl
#
# Loads the three intermediate pickles produced by notebooks 01-03 and
# assembles them into a single preprocessed_data.pkl that is byte-for-byte
# compatible with AlternatingPipeline training code.
#
# Final output: /dbfs/FileStore/time_series_models/preprocessed_data.pkl
#
# Schema (v3):
#   {
#     'version': 3,
#     'exchange':          [list of exchange sequence dicts],
#     'examination':       [list of examination sequence dicts],
#     'daily_summaries':   [list of daily summary dicts],
#     'customer_schedules': { str(serial): { 'YYYY-MM-DD': [...] } },
#   }

# COMMAND ----------
%run ./config

# COMMAND ----------

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

os.makedirs(DBFS_OUTPUT_BASE, exist_ok=True)

# COMMAND ----------
# =============================================================================
# CELL 2: Load intermediate pickles
# =============================================================================

with open(EXCHANGE_OUTPUT, 'rb') as f:
    exchange_sequences = pickle.load(f)

with open(EXAMINATION_OUTPUT, 'rb') as f:
    examination_sequences = pickle.load(f)

with open(ORCH_OUTPUT, 'rb') as f:
    orch_data = pickle.load(f)

customer_schedules = orch_data['customer_schedules']
daily_summaries    = orch_data['daily_summaries']
orch_samples       = orch_data['orch_samples']

print(f"Exchange sequences:      {len(exchange_sequences):>6,}")
print(f"Examination sequences:   {len(examination_sequences):>6,}")
print(f"Daily summaries:         {len(daily_summaries):>6,}")
print(f"Customer schedules:      {len(customer_schedules):>6} scanners")
print(f"Orchestration samples:   {len(orch_samples):>6,}")

# COMMAND ----------
# =============================================================================
# CELL 3: Assemble final preprocessed_data dict
# =============================================================================

preprocessed_data = {
    'version':            3,   # v3: phase_type + shutdown sequences
    'exchange':           exchange_sequences,
    'examination':        examination_sequences,
    'daily_summaries':    daily_summaries,
    'customer_schedules': customer_schedules,
}

print("Assembled preprocessed_data dict.")

# COMMAND ----------
# =============================================================================
# CELL 4: Schema validation
# =============================================================================

print("Validating schema...")

# --- Top-level keys ---
required_top_keys = {'version', 'exchange', 'examination',
                     'daily_summaries', 'customer_schedules'}
assert required_top_keys.issubset(preprocessed_data.keys()), \
    f"Missing top-level keys: {required_top_keys - set(preprocessed_data.keys())}"
assert preprocessed_data['version'] == 3, "version must be 3"

# --- Exchange sequences ---
EX_KEYS = {'sequence', 'durations', 'conditioning', 'body_from',
           'body_to', 'phase_type', 'total_duration', 'start_datetime'}
COND_KEYS = {'Age', 'Weight', 'Height', 'PTAB', 'Direction_encoded',
             'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_morning'}

for i, seq in enumerate(preprocessed_data['exchange']):
    assert EX_KEYS.issubset(seq.keys()), \
        f"Exchange seq {i}: missing keys {EX_KEYS - set(seq.keys())}"
    assert len(seq['sequence']) == len(seq['durations']), \
        f"Exchange seq {i}: length mismatch"
    assert seq['phase_type'] in {0, 1, 2}, \
        f"Exchange seq {i}: bad phase_type"
    assert 0 <= seq['body_from'] <= END_REGION_ID, \
        f"Exchange seq {i}: body_from out of range"
    assert 0 <= seq['body_to'] <= END_REGION_ID, \
        f"Exchange seq {i}: body_to out of range"
    assert COND_KEYS.issubset(seq['conditioning'].keys()), \
        f"Exchange seq {i}: missing conditioning keys"
    assert hasattr(seq['start_datetime'], 'date'), \
        f"Exchange seq {i}: start_datetime not a datetime"

print(f"  Exchange: {len(preprocessed_data['exchange'])} sequences OK")

# --- Examination sequences ---
EXM_KEYS = {'sequence', 'durations', 'conditioning', 'body_region',
            'coil_config', 'total_duration', 'start_datetime'}

for i, seq in enumerate(preprocessed_data['examination']):
    assert EXM_KEYS.issubset(seq.keys()), \
        f"Examination seq {i}: missing keys {EXM_KEYS - set(seq.keys())}"
    assert len(seq['sequence']) == len(seq['durations']), \
        f"Examination seq {i}: length mismatch"
    assert 0 <= seq['body_region'] <= 10, \
        f"Examination seq {i}: body_region out of range"
    assert set(seq['coil_config'].keys()) == set(COIL_COLUMNS), \
        f"Examination seq {i}: coil_config wrong keys"
    assert COND_KEYS.issubset(seq['conditioning'].keys()), \
        f"Examination seq {i}: missing conditioning keys"
    assert hasattr(seq['start_datetime'], 'date'), \
        f"Examination seq {i}: start_datetime not a datetime"

print(f"  Examination: {len(preprocessed_data['examination'])} sequences OK")

# --- Customer schedules ---
for cid, daily in preprocessed_data['customer_schedules'].items():
    assert isinstance(cid, str), f"customer_schedules: key '{cid}' is not str"
    for date_str, patients in daily.items():
        assert isinstance(date_str, str) and len(date_str) == 10, \
            f"  {cid}: date key '{date_str}' must be 'YYYY-MM-DD'"
        for j, p in enumerate(patients):
            for pk in ('patient_id', 'body_region', 'body_region_id',
                       'age', 'weight', 'height', 'direction',
                       'hour_of_day', 'day_of_week'):
                assert pk in p, f"  {cid}/{date_str} patient {j}: missing '{pk}'"

print(f"  Customer schedules: {len(preprocessed_data['customer_schedules'])} scanners OK")

# --- Daily summaries (light check) ---
assert isinstance(preprocessed_data['daily_summaries'], list), \
    "daily_summaries must be a list"
print(f"  Daily summaries: {len(preprocessed_data['daily_summaries'])} entries OK")

print("\nAll schema validations passed!")

# COMMAND ----------
# =============================================================================
# CELL 5: Save preprocessed_data.pkl
# =============================================================================

with open(FINAL_OUTPUT, 'wb') as f:
    pickle.dump(preprocessed_data, f)

print(f"Saved preprocessed_data.pkl to {FINAL_OUTPUT}")
print(f"File size: {os.path.getsize(FINAL_OUTPUT) / 1024:.1f} KB")

# COMMAND ----------
# =============================================================================
# CELL 6: Summary statistics
# =============================================================================

print("=" * 60)
print("PREPROCESSED DATA SUMMARY")
print("=" * 60)

# --- Exchange ---
ex = preprocessed_data['exchange']
phase_counts = {0: 0, 1: 0, 2: 0}
for s in ex:
    phase_counts[s['phase_type']] = phase_counts.get(s['phase_type'], 0) + 1
print(f"\nExchange sequences: {len(ex):,}")
print(f"  startup  (phase 0): {phase_counts[0]:,}")
print(f"  between  (phase 1): {phase_counts[1]:,}")
print(f"  shutdown (phase 2): {phase_counts[2]:,}")
if ex:
    durs = [s['total_duration'] for s in ex]
    lens = [len(s['sequence']) for s in ex]
    print(f"  Duration: mean={np.mean(durs):.0f}s, max={np.max(durs):.0f}s")
    print(f"  Seq len:  mean={np.mean(lens):.1f}, max={np.max(lens)}")

# --- Examination ---
exm = preprocessed_data['examination']
region_counts = {}
for s in exm:
    r = s['body_region']
    region_counts[r] = region_counts.get(r, 0) + 1
print(f"\nExamination sequences: {len(exm):,}")
for rid in sorted(region_counts):
    region_name = ID_TO_BODY_REGION.get(rid, str(rid))
    print(f"  {region_name:10s}: {region_counts[rid]:,}")
if exm:
    durs = [s['total_duration'] for s in exm]
    lens = [len(s['sequence']) for s in exm]
    print(f"  Duration: mean={np.mean(durs):.0f}s, max={np.max(durs):.0f}s")
    print(f"  Seq len:  mean={np.mean(lens):.1f}, max={np.max(lens)}")

# --- Customer schedules ---
cs = preprocessed_data['customer_schedules']
print(f"\nCustomers: {len(cs)}")
for cid, daily in sorted(cs.items()):
    total_pats = sum(len(p) for p in daily.values())
    print(f"  {cid}: {len(daily)} days, {total_pats} patients")

# --- Sample spot-check ---
print("\n--- Spot-check: 3 exchange sequences ---")
for s in preprocessed_data['exchange'][:3]:
    print(f"  tokens={s['sequence'][:5]}...  phase={s['phase_type']}  "
          f"body={s['body_from']}→{s['body_to']}  dur={s['total_duration']:.0f}s")

print("\n--- Spot-check: 3 examination sequences ---")
for s in preprocessed_data['examination'][:3]:
    region_name = ID_TO_BODY_REGION.get(s['body_region'], str(s['body_region']))
    active_coils = [c for c, v in s['coil_config'].items() if v == 1]
    print(f"  tokens={s['sequence'][:5]}...  region={region_name}  "
          f"coils={active_coils[:3]}  dur={s['total_duration']:.0f}s")

# --- Token range check ---
all_ex_tokens  = [t for s in preprocessed_data['exchange'] for t in s['sequence']]
all_exm_tokens = [t for s in preprocessed_data['examination'] for t in s['sequence']]
if all_ex_tokens:
    print(f"\nExchange token range: [{min(all_ex_tokens)}, {max(all_ex_tokens)}] "
          f"(vocab size = {len(SOURCEID_VOCAB)})")
if all_exm_tokens:
    print(f"Examination token range: [{min(all_exm_tokens)}, {max(all_exm_tokens)}]")

print("\nDone.")
