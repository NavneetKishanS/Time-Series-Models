# Databricks notebook source
# Databricks notebook — Generate synthetic data in CSV format
#
# Loads trained models and preprocessed_data.pkl, simulates complete days
# for each scanner, and writes CSVs in exactly the same column format as
# the csv_pipeline inputs:
#
#   Exchange CSV  →  /dbfs/FileStore/csv_pipeline/synthetic/exchange/DATA_{serial}.csv
#   Exam CSV      →  /dbfs/FileStore/csv_pipeline/synthetic/exam/DATA_{serial}.csv
#
# Column formats match 01_exchange_preprocessing.py and 02_exam_preprocessing.py
# output exactly so synthetic and real data are interchangeable.
#
# Prerequisites: run 03_build_preprocessed_pkl.py and 04_train_models.py first.

# COMMAND ----------

import sys, os, re, json, pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# ── CONFIGURE THIS PATH to your Databricks Repos clone ─────────────────────
REPO_ROOT = "/Workspace/Repos/luke-schumacher/Time-Series-Models"
# ───────────────────────────────────────────────────────────────────────────

sys.path.insert(0, REPO_ROOT)

PKL_PATH       = "/dbfs/FileStore/csv_pipeline/preprocessed_data.pkl"
MODELS_DIR     = "/dbfs/FileStore/csv_pipeline/models"
SYNTH_EXCHANGE = "/dbfs/FileStore/csv_pipeline/synthetic/exchange"
SYNTH_EXAM     = "/dbfs/FileStore/csv_pipeline/synthetic/exam"

os.makedirs(SYNTH_EXCHANGE, exist_ok=True)
os.makedirs(SYNTH_EXAM,     exist_ok=True)

# Synthetic date range — must be OUTSIDE the training window (2024-01-01 → 2024-01-31)
# to avoid data leakage into the orchestration model.
SYNTH_DATE_START     = "2024-02-01"
SYNTH_DATE_END       = "2024-02-28"
WEEKDAYS_ONLY        = True          # skip Saturdays/Sundays (MRI scanners are rarely used then)
NUM_DAYS_PER_SCANNER = 30            # cap if the date range produces more days than needed

# COMMAND ----------
# =============================================================================
# Load data and models
# =============================================================================

from AlternatingPipeline.config import (
    EXCHANGE_MODEL_CONFIG, EXAMINATION_MODEL_CONFIG, ORCHESTRATION_MODEL_CONFIG,
    ID_TO_SOURCEID, SOURCEID_VOCAB, BODY_REGIONS, BODY_REGION_TO_ID,
    START_REGION_ID, END_REGION_ID, PHASE_TYPES, GENERATION_CONFIG,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID, BREAK_TOKEN_ID,
    NUM_BODY_REGIONS, ORCH_PAD_TOKEN_ID,
)
from AlternatingPipeline.models.exchange_model    import create_exchange_model
from AlternatingPipeline.models.examination_model import create_examination_model
from AlternatingPipeline.models.orchestration_model import create_orchestration_model
from AlternatingPipeline.data.orchestration_preprocessing import (
    extract_orchestration_samples, build_demographic_distributions
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- preprocessed data ---
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

customer_schedules = data['customer_schedules']
print(f"Customers: {list(customer_schedules.keys())}")

# --- load exchange model ---
exchange_model = create_exchange_model(EXCHANGE_MODEL_CONFIG).to(device)
exchange_model.load_state_dict(
    torch.load(f"{MODELS_DIR}/exchange/exchange_model_best.pt", map_location=device)
)
exchange_model.eval()
print("Exchange model loaded.")

# --- load examination model ---
examination_model = create_examination_model(EXAMINATION_MODEL_CONFIG).to(device)
examination_model.load_state_dict(
    torch.load(f"{MODELS_DIR}/examination/examination_model_best.pt", map_location=device)
)
examination_model.eval()
print("Examination model loaded.")

# --- load orchestration model ---
with open(f"{MODELS_DIR}/orchestration/scanner_to_idx.json") as f:
    scanner_to_idx = json.load(f)

orch_ckpt = torch.load(f"{MODELS_DIR}/orchestration/orchestration_model_best.pt", map_location=device)
orch_config = dict(ORCHESTRATION_MODEL_CONFIG)
orch_config['num_scanners'] = orch_ckpt['scanner_embedding.weight'].shape[0]
orchestration_model = create_orchestration_model(orch_config).to(device)
orchestration_model.load_state_dict(orch_ckpt)
orchestration_model.eval()
print("Orchestration model loaded.")

# --- demographic distributions per body region (sampled from real data) ---
orch_samples, _ = extract_orchestration_samples(data)
demographic_distributions = build_demographic_distributions(data)

# COMMAND ----------
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_cond_tensor(patient_info, current_time, day_start):
    """10-dim conditioning tensor matching build_conditioning_tensor in day_simulator."""
    current_dt  = day_start + timedelta(seconds=float(current_time))
    hour        = current_dt.hour + current_dt.minute / 60.0
    dow         = current_dt.weekday()
    direction   = patient_info.get('direction', 'Head First')
    dir_enc     = 0.0 if str(direction).strip().lower() == 'head first' else 1.0
    return torch.tensor([
        float(patient_info.get('age',    50)),
        float(patient_info.get('weight', 75)),
        float(patient_info.get('height', 1.75)),
        float(patient_info.get('ptab',   0)),
        dir_enc,
        float(np.sin(2 * np.pi * hour / 24)),
        float(np.cos(2 * np.pi * hour / 24)),
        float(np.sin(2 * np.pi * dow  / 7)),
        float(np.cos(2 * np.pi * dow  / 7)),
        1.0 if hour < 12 else 0.0,
    ], dtype=torch.float32).to(device)


def _sample_demographics(body_region_id, demographic_distributions):
    """Sample patient demographics for a given body region."""
    stats = demographic_distributions.get(body_region_id, {
        'age_mean': 50.0, 'age_std': 15.0,
        'weight_mean': 75.0, 'weight_std': 15.0,
        'height_mean': 1.75, 'height_std': 0.1,
        'direction_prob': 0.8,
    })
    return {
        'age':       float(np.clip(np.random.normal(stats['age_mean'],    stats['age_std']),    1, 100)),
        'weight':    float(np.clip(np.random.normal(stats['weight_mean'], stats['weight_std']), 20, 200)),
        'height':    float(np.clip(np.random.normal(stats['height_mean'], stats['height_std']), 0.5, 2.5)),
        'ptab':      0.0,
        'direction': 'Head First' if np.random.random() < stats['direction_prob'] else 'Feet First',
    }


def _region_name(region_id):
    if region_id == START_REGION_ID: return 'START'
    if region_id == END_REGION_ID:   return 'END'
    if 0 <= region_id < len(BODY_REGIONS): return BODY_REGIONS[region_id]
    return 'UNKNOWN'


# Exchange-specific coil columns (same as step 01 output)
_EXCHANGE_COIL_COLS = [
    'S1','S2','S3','S4','S5','S6','S7','S8',
    'HC1','HC2','HC3','HC4','NC1','NC2',
    'BC','SHL','FA','TO','FS','15K',
]

# Exam coil binary columns (same as step 02 output, #0_ prefix format)
_EXAM_COIL_COLS = [
    '#0_HC1','#0_HC2','#0_HC3','#0_HC4',
    '#0_NC1','#0_NC2',
    '#0_BC','#0_SHL','#0_FA','#0_TO','#0_FS','#0_15K',
]

# Synthetic sequence name pool (used for exam Sequence/Protocol columns)
_SEQUENCES  = ['tse', 'gre', 'tfl', 'ep2d', 'tirm', 'vibe']
_PROTOCOLS  = ['t1_tse_sag', 't2_tse_tra', 'gre_field_map', 'dti_FA',
               'localizer', 't1_mprage', 'bold_rest', 't2_flair']


def _generate_exchange_rows(tokens, durations, day_start, t_offset,
                             patient_id_from, patient_id_to,
                             body_from, body_to, patient_info, serial):
    """
    Convert a generated exchange token sequence into rows matching the
    exchange CSV format from 01_exchange_preprocessing.py.
    """
    rows = []
    tokens_list    = tokens.cpu().tolist()
    durations_list = durations.cpu().tolist()
    t = t_offset
    block_start    = t_offset

    body_from_name = _region_name(body_from)
    body_to_name   = _region_name(body_to)

    for i, tok in enumerate(tokens_list):
        if tok in (START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID):
            continue

        source_id = ID_TO_SOURCEID.get(tok, 'UNK')
        dur       = durations_list[i] if i < len(durations_list) else 0.0
        dt        = day_start + timedelta(seconds=t)

        row = {
            'datetime':          dt.strftime('%Y-%m-%d %H:%M:%S'),
            'sourceID':          source_id,
            'text':              '',             # not generated by model
            'timediff':          round(t - block_start, 2),
            'PatientID_from':    patient_id_from,
            'PatientID_to':      patient_id_to,
            'BodyPart_from':     body_from_name,
            'BodyPart_to':       body_to_name,
            'BodyGroup_from':    body_from_name,
            'BodyGroup_to':      body_to_name,
            'PTAB':              patient_info.get('ptab', 0),
            # Axis flags — not generated by model
            'ZAxisInPossible':   np.nan,
            'ZAxisOutPossible':  np.nan,
            'YAxisDownPossible': np.nan,
            'YAxisUpPossible':   np.nan,
        }
        # Coil columns — default 0 (not generated by exchange model)
        for c in _EXCHANGE_COIL_COLS:
            row[c] = 0

        row['SN']        = str(serial)
        row['Age']       = round(patient_info.get('age',    50), 1)
        row['Weight']    = round(patient_info.get('weight', 75), 1)
        row['Height']    = round(patient_info.get('height', 1.75), 2)
        row['Direction'] = patient_info.get('direction', 'Head First')

        rows.append(row)
        t += max(0.0, dur)

    return rows, t


def _generate_exam_rows(tokens, durations, day_start, t_offset,
                         patient_id, body_region_id, patient_info,
                         serial, step_counter):
    """
    Convert a generated examination token sequence into measurement-level rows
    matching the exam CSV format from 02_exam_preprocessing.py.

    Each MRI_MSR_100 → MRI_MSR_104/34 boundary = one row.
    """
    rows = []
    tokens_list    = tokens.cpu().tolist()
    durations_list = durations.cpu().tolist()

    body_name  = _region_name(body_region_id)
    ptab       = patient_info.get('ptab', 0)
    age        = round(patient_info.get('age',    50), 1)
    weight     = round(patient_info.get('weight', 75), 1)
    height     = round(patient_info.get('height', 1.75), 2)
    direction  = patient_info.get('direction', 'Head First')

    FINISH_MAP = {
        SOURCEID_VOCAB.get('MRI_MSR_104', 12): 'Successful',
        SOURCEID_VOCAB.get('MRI_MSR_34',  15): 'Stopped by User',
    }
    MSR_100 = SOURCEID_VOCAB.get('MRI_MSR_100', 10)

    t          = t_offset
    msr_start  = None
    msr_start_t = None

    for i, tok in enumerate(tokens_list):
        if tok in (START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID):
            continue

        dur = durations_list[i] if i < len(durations_list) else 0.0

        if tok == MSR_100:
            msr_start   = day_start + timedelta(seconds=t)
            msr_start_t = t

        elif tok in FINISH_MAP and msr_start is not None:
            msr_end      = day_start + timedelta(seconds=t + dur)
            duration_sec = (msr_end - msr_start).total_seconds()

            if duration_sec <= 0 or duration_sec > 4000:
                msr_start = None
                t += max(0.0, dur)
                continue

            step_counter[patient_id] = step_counter.get(patient_id, 0) + 1

            row = {
                'startTime':    msr_start.strftime('%Y-%m-%d %H:%M:%S'),
                'endTime':      msr_end.strftime('%Y-%m-%d %H:%M:%S'),
                'duration':     int(duration_sec),
                'sourceID':     ID_TO_SOURCEID.get(tok, 'UNK'),
                'Sequence':     np.random.choice(_SEQUENCES),
                'Protocol':     np.random.choice(_PROTOCOLS),
                'PatientID':    patient_id,
                'BodyPart':     body_name,
                'BodyGroup':    body_name,
                'ConnectedCoils': '',           # not generated by model
                'DOT':          False,
                'PTAB':         ptab,
                'FinishEvent':  FINISH_MAP[tok],
                'pauseTime':    0.0,            # filled in post-loop
                'StepCount':    step_counter[patient_id],
                'Age':          age,
                'Weight':       weight,
                'Height':       height,
                'Direction':    direction,
                'SN':           str(serial),
            }
            # Exam coil binary columns — default False
            for c in _EXAM_COIL_COLS:
                row[c] = False

            rows.append(row)
            msr_start = None

        t += max(0.0, dur)

    return rows, t


def _fill_pause_times(rows):
    """Compute pauseTime = gap between this measurement's endTime and next's startTime."""
    for i in range(len(rows) - 1):
        end_i   = pd.to_datetime(rows[i]['endTime'])
        start_j = pd.to_datetime(rows[i + 1]['startTime'])
        rows[i]['pauseTime'] = max(0.0, (start_j - end_i).total_seconds())
    if rows:
        rows[-1]['pauseTime'] = 0.0
    return rows


def _generate_orch_tokens(scanner_idx, date, demographic_distributions):
    """
    Use orchestration model to predict the body region sequence for one day.
    Returns a list of body region token IDs (breaks already filtered out).
    """
    from AlternatingPipeline.data.orchestration_preprocessing import _build_orchestration_conditioning
    import math

    dt = datetime.strptime(str(date), '%Y-%m-%d')
    dow = dt.weekday()
    stats = {
        'avg_patients_per_day': 8.0,
        'region_distribution':  np.ones(NUM_BODY_REGIONS) / NUM_BODY_REGIONS,
    }
    cond = _build_orchestration_conditioning(str(date), dow, stats)
    cond_t = torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(device)
    scanner_t = torch.tensor([scanner_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        tokens = orchestration_model.generate(
            cond_t, scanner_t,
            max_length=ORCHESTRATION_MODEL_CONFIG['max_seq_len'],
            temperature=GENERATION_CONFIG['temperature'],
            top_k=GENERATION_CONFIG['top_k'],
        )
    token_list = tokens[0].cpu().tolist()
    # Keep only valid body region IDs
    return [t for t in token_list
            if 0 <= t < NUM_BODY_REGIONS and t not in (START_REGION_ID, END_REGION_ID)]

# COMMAND ----------
# =============================================================================
# MAIN GENERATION LOOP — one synthetic day per customer per date
# =============================================================================

gen_config = GENERATION_CONFIG

for serial_str, daily_schedules in customer_schedules.items():
    print(f"\n{'='*60}\nSerial: {serial_str}\n{'='*60}")

    scanner_idx = scanner_to_idx.get(serial_str, 0)

    all_exchange_rows = []
    all_exam_rows     = []

    # Generate dates from the synthetic range (outside training window — no leakage)
    _start = datetime.strptime(SYNTH_DATE_START, '%Y-%m-%d')
    _end   = datetime.strptime(SYNTH_DATE_END,   '%Y-%m-%d')
    dates  = []
    _d = _start
    while _d <= _end and len(dates) < NUM_DAYS_PER_SCANNER:
        if not WEEKDAYS_ONLY or _d.weekday() < 5:
            dates.append(_d.strftime('%Y-%m-%d'))
        _d += timedelta(days=1)
    if not dates:
        print(f"  No dates in synthetic range — skipping.")
        continue

    step_counter = {}   # tracks StepCount per patient_id across the day

    for date_str in dates:
        print(f"  Day: {date_str}")
        day_start = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=7, minute=0)

        # --- Use orchestration model to decide patient body region sequence ---
        orch_tokens = _generate_orch_tokens(scanner_idx, date_str, demographic_distributions)
        if not orch_tokens:
            print(f"    Orchestration returned no tokens — skipping day.")
            continue

        # Build patient list from orchestration output
        patients = []
        for tok in orch_tokens:
            demographics = _sample_demographics(tok, demographic_distributions)
            patients.append({
                'patient_id':     f'SYNTH_{date_str}_{len(patients):03d}',
                'body_region_id': tok,
                'body_region':    _region_name(tok),
                **demographics,
            })

        print(f"    {len(patients)} patients from orchestration")

        current_t       = 0.0
        prev_region     = START_REGION_ID
        prev_patient_id = 'START'
        step_counter    = {}

        for p_idx, patient in enumerate(patients):
            pat_id     = patient['patient_id']
            region_id  = patient['body_region_id']
            phase_type = PHASE_TYPES['startup'] if p_idx == 0 else PHASE_TYPES['between']

            cond = _build_cond_tensor(patient, current_t, day_start)

            # ── EXCHANGE ──
            with torch.no_grad():
                ex_tokens, ex_durations = exchange_model.generate(
                    cond,
                    {'body_from': prev_region, 'body_to': region_id},
                    phase_type=phase_type,
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                )

            ex_rows, current_t = _generate_exchange_rows(
                ex_tokens[0], ex_durations[0],
                day_start, current_t,
                prev_patient_id, pat_id,
                prev_region, region_id,
                patient, serial_str,
            )
            all_exchange_rows.extend(ex_rows)

            # ── EXAMINATION ──
            cond = _build_cond_tensor(patient, current_t, day_start)
            with torch.no_grad():
                exam_tokens, exam_durations = examination_model.generate(
                    cond,
                    {'body_region': region_id},
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                )

            exam_rows, current_t = _generate_exam_rows(
                exam_tokens[0], exam_durations[0],
                day_start, current_t,
                pat_id, region_id, patient,
                serial_str, step_counter,
            )
            all_exam_rows.extend(exam_rows)

            prev_region     = region_id
            prev_patient_id = pat_id

        # ── SHUTDOWN EXCHANGE ──
        if patients:
            last_patient = patients[-1]
            cond = _build_cond_tensor(last_patient, current_t, day_start)
            with torch.no_grad():
                sd_tokens, sd_durations = exchange_model.generate(
                    cond,
                    {'body_from': prev_region, 'body_to': END_REGION_ID},
                    phase_type=PHASE_TYPES['shutdown'],
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    top_p=gen_config['top_p'],
                )
            sd_rows, _ = _generate_exchange_rows(
                sd_tokens[0], sd_durations[0],
                day_start, current_t,
                prev_patient_id, 'END',
                prev_region, END_REGION_ID,
                last_patient, serial_str,
            )
            all_exchange_rows.extend(sd_rows)

    # ── Fill pause times for exam rows ──
    all_exam_rows = _fill_pause_times(all_exam_rows)

    # ── Save exchange CSV ──
    if all_exchange_rows:
        df_ex = pd.DataFrame(all_exchange_rows)
        ex_path = f"{SYNTH_EXCHANGE}/DATA_{serial_str}.csv"
        df_ex.to_csv(ex_path, index=False)
        print(f"  Exchange CSV: {len(df_ex):,} rows → {ex_path}")

    # ── Save exam CSV ──
    if all_exam_rows:
        df_exam = pd.DataFrame(all_exam_rows)
        exam_path = f"{SYNTH_EXAM}/DATA_{serial_str}.csv"
        df_exam.to_csv(exam_path, index=False)
        print(f"  Exam CSV:     {len(df_exam):,} rows → {exam_path}")

# COMMAND ----------
# =============================================================================
# Download links
# =============================================================================

links = ""
for serial_str in customer_schedules.keys():
    links += f'<li><a href="/files/csv_pipeline/synthetic/exchange/DATA_{serial_str}.csv">Exchange {serial_str}</a> &nbsp;|&nbsp; '
    links += f'<a href="/files/csv_pipeline/synthetic/exam/DATA_{serial_str}.csv">Exam {serial_str}</a></li>\n'

displayHTML(f'<h3>Synthetic CSVs</h3><ul>{links}</ul>')

# COMMAND ----------
# =============================================================================
# Visualizations — synthetic data quality
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob as _glob

# ── Load all synthetic CSVs ──────────────────────────────────────────────────
ex_files   = _glob.glob(f"{SYNTH_EXCHANGE}/DATA_*.csv")
exam_files = _glob.glob(f"{SYNTH_EXAM}/DATA_*.csv")

df_ex_all   = pd.concat([pd.read_csv(f) for f in ex_files],   ignore_index=True) if ex_files   else pd.DataFrame()
df_exam_all = pd.concat([pd.read_csv(f) for f in exam_files], ignore_index=True) if exam_files else pd.DataFrame()

print(f"Loaded {len(df_ex_all):,} exchange rows and {len(df_exam_all):,} exam rows across {len(ex_files)} scanners.")

# ── Figure 1: Exam overview (3 panels) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Synthetic Examination Data', fontsize=14, fontweight='bold')

if not df_exam_all.empty:
    # Body region distribution
    region_counts = df_exam_all['BodyPart'].value_counts()
    axes[0].bar(region_counts.index, region_counts.values, color='steelblue', edgecolor='white')
    axes[0].set_title('Body Region Distribution')
    axes[0].set_xlabel('Body Region')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Examination duration histogram
    durations = df_exam_all['duration'].dropna()
    durations = durations[(durations > 0) & (durations < 4000)]
    axes[1].hist(durations / 60, bins=40, color='steelblue', edgecolor='white')
    axes[1].set_title('Examination Duration')
    axes[1].set_xlabel('Duration (minutes)')
    axes[1].set_ylabel('Count')
    axes[1].axvline(durations.mean() / 60, color='red', linestyle='--', linewidth=1.5, label=f'Mean {durations.mean()/60:.1f} min')
    axes[1].legend()

    # Rows per scanner
    scanner_counts = df_exam_all['SN'].value_counts().sort_index()
    axes[2].bar(scanner_counts.index.astype(str), scanner_counts.values, color='steelblue', edgecolor='white')
    axes[2].set_title('Exam Rows per Scanner')
    axes[2].set_xlabel('Scanner SN')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
display(fig)
plt.close(fig)

# ── Figure 2: Duration by body region (box plot) ────────────────────────────
if not df_exam_all.empty:
    regions = df_exam_all['BodyPart'].dropna().unique()
    region_data = [
        df_exam_all.loc[df_exam_all['BodyPart'] == r, 'duration'].dropna().values / 60
        for r in regions
    ]
    region_data = [d[(d > 0) & (d < 4000 / 60)] for d in region_data]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.boxplot(region_data, labels=regions, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Examination Duration by Body Region', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Body Region')
    ax2.set_ylabel('Duration (minutes)')
    ax2.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    display(fig2)
    plt.close(fig2)

# ── Figure 3: Exchange event type distribution ───────────────────────────────
if not df_ex_all.empty:
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Synthetic Exchange Data', fontsize=14, fontweight='bold')

    event_counts = df_ex_all['sourceID'].value_counts()
    axes3[0].barh(event_counts.index, event_counts.values, color='teal', edgecolor='white')
    axes3[0].set_title('Event Type Distribution')
    axes3[0].set_xlabel('Count')
    axes3[0].invert_yaxis()

    timediff = df_ex_all['timediff'].dropna()
    timediff = timediff[(timediff >= 0) & (timediff < 3600)]
    axes3[1].hist(timediff, bins=40, color='teal', edgecolor='white')
    axes3[1].set_title('Time Between Events')
    axes3[1].set_xlabel('Seconds')
    axes3[1].set_ylabel('Count')

    plt.tight_layout()
    display(fig3)
    plt.close(fig3)

# ── Figure 4: Daily patient count per scanner ───────────────────────────────
if not df_exam_all.empty and 'startTime' in df_exam_all.columns:
    df_exam_all['date'] = pd.to_datetime(df_exam_all['startTime']).dt.date
    daily = df_exam_all.groupby(['SN', 'date'])['PatientID'].nunique().reset_index()
    daily.columns = ['SN', 'date', 'patients']

    scanners = daily['SN'].unique()
    fig4, ax4 = plt.subplots(figsize=(16, 5))
    for sn in scanners:
        d = daily[daily['SN'] == sn].sort_values('date')
        ax4.plot(d['date'], d['patients'], marker='o', markersize=3, linewidth=1, label=str(sn), alpha=0.7)
    ax4.set_title('Unique Patients per Day per Scanner', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Patients')
    ax4.legend(title='Scanner', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax4.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    display(fig4)
    plt.close(fig4)
