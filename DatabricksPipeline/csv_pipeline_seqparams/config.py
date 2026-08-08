# Databricks notebook source
"""
csv_pipeline_seqparams/config.py — feature-rich examination-duration pipeline.

Forked from DatabricksPipeline/csv_pipeline/config.py to add MRI_SUT_1005
sequence-protocol parameters (TR, number of slices, trigger/gating mode) as
extra conditioning features for the examination duration model — per the
call with Görtler (domain expert): "scanning time" itself must NEVER be used
as an input feature, since it is ~equal to the duration target (leakage).

Scoped to the SAME 10 serial numbers / date range as csv_pipeline/, so the
old and new models are directly comparable on identical held-out data. Does
NOT modify csv_pipeline/ or AlternatingPipeline/config.py — the old model
and its checkpoint are completely untouched by this fork. Constants below
are duplicated (not cross-imported) from csv_pipeline/config.py, matching
that file's own established convention for this codebase.

*** STATUS ***
sut_parameter_discovery.py has been run against real MRI_SUT_1005 data. The
originally-hypothesized labeled "SD<n>: value" format was FALSIFIED (0/20
real messages matched) — the real format is whitespace-delimited, self-named
"KEY:VALUE" tokens (e.g. "TR:866 TE:101 SLC:9 ... MUID:17"). TR and num_slices
(mapped from the "SLC" key) are CONFIRMED via SUT_FIELD_MAP below.

WHICH parameters reach the model is now a single switch, PARAM_SET, defined
immediately below this docstring: two named sets (`luke`, the acquisition-time
formula; `navneet`, the 2026-08-04 domain-expert selection) drawn from one
SEQPARAM_CANDIDATES table. trigger_mode remains a PLACEHOLDER —
no field in the sampled messages showed variation consistent with a
gating/trigger mode (candidates CMM/CDM/RCM/PHYS/PAC were constant across all
samples), so TRIGGER_MODE_VOCAB is unused for now and every row defaults to
'none' (safe, non-leaking default — see _trigger_mode_id below).
"""

import json
import math
import os

# ============================================================================
# PARAM_SET — WHICH PARAMETERS GO INTO THE MODEL. The one switch.
#
# Görtler's action item from the 2026-08-04 call: "there isn't really a clear
# part of the code where you can select which parameters are put into the
# model, it's mostly they are split up ... that we can run two different model
# results, based on two parameter selections."
#
# This file is the only one all seven notebooks %run, and 03/04/05/06/07 all
# read EXAMINATION_SEQPARAM_FEATURES / EXAMINATION_SEQPARAM_SCALE / MODELS_DIR
# / ANALYSIS_DIR from here. Resolving those four from PARAM_SET reaches every
# call site without editing any of them.
#
# The named sets are defined in PARAM_SETS, further down next to the candidate
# table they draw from. Override without editing this file:
#
#     PARAM_SET=navneet   (env var, same convention as PKL_PATH / TARGET_MAE_S)
#
# Models and analysis output are namespaced by this value — see MODELS_DIR /
# ANALYSIS_DIR below — so two runs never overwrite each other's checkpoint.
#
# THE DEFAULT IS 'all' as of the 2026-08-07 Görtler meeting. His position is
# that a digital twin aiming at the extraordinary 1% cannot hide fields from
# itself: rare cases are rare BECAUSE they turn on fields that are constant on
# the other 99%, so hand-picking on aggregate MAE deletes the tail signal. 03f
# had already found the same thing — its ranked top-12 beat both hand-picked
# sets. `luke` and `navneet` are kept as the control group, because "more is
# better" is a claim to be measured rather than assumed.
# ============================================================================

PARAM_SET = os.environ.get('PARAM_SET', 'all').strip().lower()

# ============================================================================
# TARGET SCANNERS & DATE RANGE — identical to csv_pipeline/, for a like-for-
# like comparison against the old model.
# ============================================================================

SERIAL_NUMBERS = [183242, 176148, 176227, 175912, 175670,
                  183776, 182625, 176615, 176240, 175828]

DATE_START = "2024-01-01"
DATE_END   = "2024-01-31"

TIMEZONE_OFFSET_HOURS = 1   # UTC+1 (CET); adjust per site

# ============================================================================
# DATABRICKS TABLE PATHS
# ============================================================================

EVENTLOG_TABLE    = "hive_metastore.eventlog.common_eventlog"
EXAMINATION_TABLE = "hive_metastore.examination.examination_workflow"

BODY_GROUP_MAPPING_PATH = "/dbfs/FileStore/tables/bodyupdated.xlsx"

# ONE pkl serves every parameter set: step 03 writes the whole
# SEQPARAM_ALL_CANDIDATES union into 'conditioning', and training reads only
# the names in EXAMINATION_SEQPARAM_FEATURES. So switching sets costs a
# training run, not another Spark rebuild.
PKL_OUTPUT   = "/dbfs/FileStore/csv_pipeline_seqparams/preprocessed_data.pkl"

# Written by step 03 alongside the pkl: one p99-derived scale divisor per SUT
# field it actually observed, so PARAM_SET='all' does not require somebody to
# hand-write ~89 divisors. Read back by this file at import time (see
# _load_divisor_table below). Absent on a fresh clone and in the test suite,
# which is why the fallback is the hand-curated SEQPARAM_CANDIDATES table.
SEQPARAM_DIVISOR_TABLE = os.environ.get(
    'SEQPARAM_DIVISOR_TABLE',
    "/dbfs/FileStore/csv_pipeline_seqparams/seqparam_divisors.json",
)

# ----------------------------------------------------------------------------
# THE OTHER PIPELINE — what this one still borrows, and why it is a constant.
#
# This pipeline's pkl holds ONLY 'examination' (see 03_build_preprocessed_pkl.py),
# so 07_generate_synthetic_data.py reads 'exchange' and 'customer_schedules'
# from csv_pipeline's pkl and its exchange/orchestration checkpoints from
# csv_pipeline's models directory. That is the reason onboarding a new customer
# currently means running BOTH pipelines and keeping them in sync.
#
# The fix is for step 03 to produce all four keys, which absorbs
# 01_exchange_preprocessing.py and 02_exam_preprocessing.py. That is deliberately
# NOT done yet: Navneet is reworking csv_pipeline for a GPU cluster, and forking
# those two now means forking code that is being rewritten and then re-forking it
# after his PR lands.
#
# What IS done now is making the dependency a NAMED CONSTANT instead of two
# string literals buried at 07:181-183, so the day the merge happens it is a
# config change rather than a code change — and so `grep BASE_` finds every
# place this pipeline still leans on the other one.
# ----------------------------------------------------------------------------
BASE_PKL = os.environ.get(
    'BASE_PKL', "/dbfs/FileStore/csv_pipeline/preprocessed_data.pkl")
BASE_MODELS_DIR = os.environ.get(
    'BASE_MODELS_DIR', "/dbfs/FileStore/csv_pipeline/models")

# Models and analysis ARE namespaced by PARAM_SET. Two sets can widen
# base_conditioning_dim to the same number while meaning different things per
# column, so a shared directory would silently overwrite one checkpoint with
# the other and leave the comparison unreadable — with nothing in the manifest
# to say which set produced it.
#
# Env-overridable so the report can be produced somewhere other than /dbfs. That
# is not a convenience: 03b_parameter_report.py is the notebook whose OUTPUT has
# to be readable by someone who cannot start a cluster, and being able to run it
# against a downloaded pkl is what makes that true.
MODELS_DIR   = os.environ.get(
    'MODELS_DIR', f"/dbfs/FileStore/csv_pipeline_seqparams/models/{PARAM_SET}")
ANALYSIS_DIR = os.environ.get(
    'ANALYSIS_DIR', f"/dbfs/FileStore/csv_pipeline_seqparams/analysis/{PARAM_SET}")

# Note: no EXAM_OUTPUT_DIR / step-02 CSV export in this pipeline. Unlike
# csv_pipeline/, this fork has no 02_exam_preprocessing.py — the examination-
# sequence path (03_build_preprocessed_pkl.py Section 2, the direct ancestor
# of this fork) queries Spark directly and never reads step 02's CSV; that
# CSV only ever fed csv_pipeline/03's Section 3 (customer_schedules), which
# this pipeline correctly drops as out of scope (examination-only, decision
# #1). So there is nothing for a step-02 fork to produce that this pipeline
# needs.

# ============================================================================
# SOURCE ID FILTER
#
# GAP FOUND while planning this fork: csv_pipeline/config.py's
# SOURCE_ID_FILTER (aliased REAL_EVENT_TYPES there, used by step 03's Spark
# query that actually builds the training pkl) does NOT include
# MRI_SUT_1005 — only the narrower EXAM_EXTRA_SOURCE_TYPES (used by
# csv_pipeline/02's CSV export, which this fork has no equivalent of — see
# the note above) does. Without widening this filter, SUT rows would never
# reach the training pkl even once parsing exists. REAL_EVENT_TYPES below is
# already the widened list.
# ============================================================================

SOURCE_ID_FILTER = [
    "MRI_CCS_11", "MRI_EXU_95", "MRI_FRR_18", "MRI_FRR_257", "MRI_FRR_264",
    "MRI_FRR_2", "MRI_FRR_3", "MRI_FRR_34", "MRI_MPT_1005", "MRI_MSR_100",
    "MRI_MSR_104", "MRI_MSR_21", "MRI_MSR_34", "MRI_FRR_256",
]
SOURCE_ID_FILTER_SEQPARAMS = SOURCE_ID_FILTER + ["MRI_SUT_1005"]
REAL_EVENT_TYPES = SOURCE_ID_FILTER_SEQPARAMS  # step 03 Spark query filter

# ============================================================================
# COIL / TOKEN / BODY-REGION VOCAB — byte-identical copies of
# csv_pipeline/config.py (that file's own comment: these MUST stay
# byte-identical across pipeline variants, since step 03 writes the IDs and
# training reads them back).
# ============================================================================

COIL_ABBREV_MAP = {
    'HC1': 'HE1', 'HC2': 'HE2', 'HC3': 'HE3', 'HC4': 'HE4',
    'NC1': 'NE1', 'NC2': 'NE2',
    'BC':  'BC',  'SHL': 'SHL', 'FA':  'FA',  'TO':  'TO',
    'FS':  'FS',  '15K': '15K', 'SN':  'SN',
    'SP1': 'SP1', 'SP2': 'SP2', 'SP3': 'SP3', 'SP4': 'SP4',
    'SP5': 'SP5', 'SP6': 'SP6', 'SP7': 'SP7', 'SP8': 'SP8',
    'HW1': 'HW1', 'HW2': 'HW2', 'HW3': 'HW3',
    'HE1': 'HE1', 'HE2': 'HE2', 'HE3': 'HE3', 'HE4': 'HE4',
    'NE1': 'NE1', 'NE2': 'NE2',
    'BO1': 'BO1', 'BO2': 'BO2', 'BO3': 'BO3',
    'PA1': 'PA1', 'PA2': 'PA2', 'PA3': 'PA3',
    'PA4': 'PA4', 'PA5': 'PA5', 'PA6': 'PA6',
}

SOURCEID_VOCAB = {
    'PAD': 0, 'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3,
    'MRI_FRR_257': 4, 'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7,
    'MRI_FRR_34': 8, 'MRI_MPT_1005': 9, 'MRI_MSR_100': 10, 'START': 11,
    'MRI_MSR_104': 12, 'MRI_MSR_21': 13, 'END': 14, 'MRI_MSR_34': 15,
    'MRI_FRR_256': 16, 'UNK': 17,
}

BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']
BODY_REGION_TO_ID = {r: i for i, r in enumerate(BODY_REGIONS)}

SEQUENCE_TYPE_VOCAB = {
    'other': 0, 'scout': 1, 'localizer': 2, 'tse': 3, 'space': 4,
    'haste': 5, 'gre': 6, 'flash': 7, 'epi': 8, 'tfl': 9, 'tirm': 10,
    'vibe': 11, 'dixon': 12, 'swi': 13, 'medic': 14,
}
NUM_SEQUENCE_TYPES = len(SEQUENCE_TYPE_VOCAB)
ID_TO_SEQUENCE_TYPE = {v: k for k, v in SEQUENCE_TYPE_VOCAB.items()}
_SEQUENCE_TYPE_KEYS = [
    'localizer', 'scout', 'haste', 'space', 'tirm', 'vibe', 'dixon',
    'medic', 'swi', 'tfl', 'flash', 'tse', 'gre',
]


def classify_sequence_type(raw):
    """Map a raw `Sequence` string to a SEQUENCE_TYPE_VOCAB id."""
    s = str(raw or '').lower()
    if not s:
        return SEQUENCE_TYPE_VOCAB['other']
    for key in _SEQUENCE_TYPE_KEYS:
        if key in s:
            return SEQUENCE_TYPE_VOCAB[key]
    if 'ep2d' in s or 'epi' in s or 'bold' in s or 'diff' in s or 'dwi' in s:
        return SEQUENCE_TYPE_VOCAB['epi']
    return SEQUENCE_TYPE_VOCAB['other']


MAX_EXCHANGE_DURATION    = 7200
MAX_EXAMINATION_DURATION = 3000
MIN_EXAMINATION_DURATION = 10
MAX_PER_TOKEN_DURATION   = 600

# Keep ONE segment per MRI_MSR_104/34 terminator, dated from the most recent
# MRI_MSR_100 — the same rule csv_pipeline/02_exam_preprocessing.py:220 uses.
#
# Binding every MSR_100 to the next terminator made two MSR_100 events before
# one terminator produce two OVERLAPPING segments that both claimed the same
# measurement. 03c section B2 measured that on 16.9% of rows and showed it was
# the whole reason the pkl's durations ran fat against the exam CSVs the ±15s
# benchmark came from (mean 115.7s → 94.1s, sd 214.7s → 125.5s once deduped).
# Section C scored the consequence on held-out data:
#
#     as-is        protocol 31.2s   serial+protocol 24.0s
#     deduped      protocol 18.3s   serial+protocol 15.4s   <- recovers the 15.3s benchmark
#     + non-abort, 10-600s          serial+protocol 11.4s   <- beats it
#
# Set False to rebuild the old overlapping population for comparison. It drops
# ~17% of rows, so it is a population decision, not a bug fix — hence a flag.
# Measured 2026-08-04: 56,873 -> 51,321 rows (9.8%), and the pkl's sd landed at
# 91.4s against step 02's 91.0s — the heavy tail was a segmentation artefact.
DEDUPE_SHARED_TERMINATOR = True

# How stale a most-recent-before MRI_SUT_1005 event may be before we call the
# parameters MISSING instead of wrong.
#
# 03c A2 (2026-08-04) measured the 'before' fallback at p50 106s, p90 312s and
# max 320,227s — 3.7 DAYS. Those extremes are segments with no SUT event
# anywhere earlier in the serial's log window, so the join reaches back to
# whatever ran last. 'before' rows agree with the actual sequence only 28.1% of
# the time (against 94.8% for in-segment), so a stale one carries no
# information; recording it as 'none' lets training mask the row instead of
# learning from another day's scan.
#
# 600s is deliberately generous — it keeps everything inside p90 and removes
# only the absurd tail. It does NOT fix the 'before' rows, which are wrong for
# a different reason (no SUT event was emitted inside the measurement at all).
SUT_MAX_BEFORE_DISTANCE_S = 600.0

COIL_COLUMNS = [
    'BC', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8', '15K',
    'HW1', 'HW2', 'HW3', 'HE1', 'HE2', 'HE3', 'HE4', 'NE1', 'NE2', 'SHL',
    'BO1', 'BO2', 'BO3', 'FA', 'TO', 'FS',
    'PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'SN',
]

# ============================================================================
# SUT SEQUENCE-PARAMETER VOCAB & FEATURE LIST
#
# Confirmed via DatabricksPipeline/sut_parameter_discovery.py against real
# MRI_SUT_1005 data (Jan 2024, serials 183242/176148/176227). The message is
# whitespace-delimited "KEY:VALUE" tokens, self-named per field (e.g. TR, TE,
# SLC, SLT, FA, ...) — NOT the originally-hypothesized numbered "SD<n>: value"
# scheme, which matched 0/20 real messages. Field sets differ by sequence
# type (haste vs. ep2d_diff carry different key sets entirely), so there is
# no fixed slot count to index into — SUT_FIELD_MAP looks fields up BY NAME.
#
# 'SLC' -> 'num_slices': confirmed by inspection (values 9/15/18 match
# plausible slice counts; a separate 'SLT' key already carries slice
# *thickness*, ruling out the alternative reading). 'TR' needs no renaming —
# it's the standard MR repetition-time mnemonic already.
#
# trigger_mode is NOT yet mapped: none of the candidate fields (CMM, CDM,
# RCM, PHYS, PAC) varied across the sampled messages, so there's no evidence
# which (if any) encodes gating/trigger mode. Follow up with Navneet. Until
# then every row defaults to 'none' via _trigger_mode_id's fallback — a
# constant, uninformative embedding, not a bug.
#
# scanning_time (the original SD58 leakage concern) does not correspond to
# any literal field in this KEY:VALUE format — there's no "SD" label at all.
# SUT_LEAKAGE_DENYLIST is kept below as defense-in-depth regardless.
# ============================================================================

# raw message key -> stable parameter name.
#
# Fields below TR/num_slices are PARSED but do NOT reach the model: only names
# listed in EXAMINATION_SEQPARAM_FEATURES do. They land in each sequence's
# audit-only `sut_debug` so the within-protocol residual (sd 25.7s after
# conditioning on protocol) can be analysed without another Spark rebuild.
#
# Why this set: acquisition time is roughly TR x PEL x AVG x CONC /
# (PAT x TF), so TR is one of six multiplicands and the one that varies least
# within a protocol. That is the physical reason TR alone scored 0.0%
# permutation importance — not a model defect. TF (turbo factor) was the last
# missing multiplicand and is now mapped.
#
# ST/TST are PARSED but DENYLISTED as features — see SUT_LEAKAGE_DENYLIST.
#
# The 2026-07-31 reading was that ST is decided BEFORE the measurement, so a
# planned value rather than an observed outcome, and therefore admissible. The
# 2026-08-04 call overrides that: 03c section E measured ST as exact on ~86-87%
# of rows, and a duration model handed a field that IS the answer 86% of the
# time learns identity instead of physics. Being computed pre-scan does not
# make it a legal input; it makes it a *second* copy of the target. 07 would
# also have to synthesise ST before it could generate with it, which is the
# same problem one step later.
#
# It stays in SUT_FIELD_MAP on purpose — 03c section E's ST-vs-duration
# diagnosis reads it out of sut_debug, and that work continues.
#
# (Note SLC x TR reproduces ST for haste (9 x 866ms -> ST:8; 15 x 1000ms ->
# ST:15) but NOT for ep2d_diff (18 x 4300ms = 77s vs ST:400) — a genuine
# per-family computation, which is precisely why it carries target information
# the individual fields do not.) Names stay as the raw message mnemonics.
#
# NOT here on purpose: DLL (sequence binary) and OR (orientation) are STRINGS.
# They are extracted separately in 03_build_preprocessed_pkl.py — a category id
# mixed into a scaled numeric vector is meaningless, and they need embeddings
# like sequence_type, not a divisor.
SUT_FIELD_MAP = {
    'TR':   'TR',
    'SLC':  'num_slices',
    'ST':   'ST',        # acquisition time, seconds — planned, see above
    'TST':  'TST',       # total scan time, seconds (ST + 1-2s)
    'SLT':  'slice_thickness',   # mm. Görtler: set by gradient strength, so
                                 # calculated rather than measured. THIS is the
                                 # field he described when asked about "ST".
    'AVG':  'averages',       # NEX/NSA — how many times the protocol acquires
    'CONC': 'concatenations',
    'PEL':  'phase_encoding_lines',
    'PAT':  'parallel_imaging_factor',
    'PATP': 'parallel_imaging_phase',
    'ACC':  'acceleration_factor',
    'FOV':  'field_of_view',
    # SEQUENCE-SCOPED, not universal: TF is on the haste (TSE-family) messages
    # and ABSENT on ep2d_diff, which uses an echo factor (EF) instead. That was
    # a blocker while every missing feature defaulted to 0.0 — a turbo factor
    # of 0 is not a real value, it means "does not apply to this sequence".
    # RESOLVED by the per-name missing default in SEQPARAM_CANDIDATES: TF
    # defaults to 1.0, which is not a fudge but the physically correct neutral
    # (one echo per TR is exactly what a non-turbo sequence does), and it is
    # the identity element of the TA formula's divisor. Any DIFF/BV0/BVM/EF
    # field added later needs the same treatment — a default that means
    # "does not apply", not a zero.
    'TF':   'turbo_factor',      # last missing multiplicand in the TA formula
    'TE':   'TE',                # echo time, ms — with TR/FA gives the
    'FA':   'flip_angle',        # weighting (T1/T2/PD) behind Görtler's
                                 # parameter-derived "generated protocol name"
    'BR':   'base_resolution',
    'REP':  'repetitions',
    'PPF':  'phase_partial_fourier',
    'SPF':  'slice_partial_fourier',
    'ES':   'echo_spacing',      # microseconds
    'BW':   'bandwidth',         # Hz/px
}

# String-valued SUT keys, extracted separately from the numeric map above.
# Both are Siemens-standard and customer-agnostic — unlike the protocol name,
# which is customer-authored (3.8M names across 15k customers) and is why
# Görtler ruled protocol identity out as a feature for the general model.
SUT_CATEGORICAL_FIELD_MAP = {
    'DLL': 'sequence_binary',   # e.g. '%SiemensSeq%\\haste' -> 'haste'
    'OR':  'orientation',       # e.g. 'CT' / 'SCT' / 'T'
}

# ============================================================================
# WHAT MAY NEVER BE A FEATURE — three denylists, three distinct objections.
#
#   SUT_LEAKAGE_DENYLIST          ~equal to the TARGET.
#   SUT_IDENTIFIER_DENYLIST       an IDENTITY that cannot transfer to another
#                                 customer or another day — the objection that
#                                 ruled the protocol name out in the first place.
#   SUT_PLANNER_DERIVED_DENYLIST  a quantity the scanner COMPUTED FROM the
#                                 timing it was about to run.
#
# These sit ABOVE the candidate table on purpose: what is admissible has to be
# decided before what is available is enumerated, because PARAM_SET='all'
# filters the discovered field table through them (SEQPARAM_ADMISSIBLE_
# DISCOVERED below). The three GATES that enforce them, and assert_no_leakage
# itself, stay further down next to the resolved feature list they check.
# ============================================================================

# ST/TST join scanning_time here as of the 2026-08-04 call — see the long note
# above SUT_FIELD_MAP. 03c section E measured ST as exact on ~86-87% of rows.
SUT_LEAKAGE_DENYLIST = {'scanning_time', 'ST', 'TST'}

# 03d_identifier_leakage_check.py, run 2026-08-04, ACQUITTED MUID as a protocol
# proxy: purity MUID -> protocol 29.8% (the leak threshold was ~90%), 11,409
# MUIDs against 3,357 protocols — finer than the protocol, not a coarser
# Siemens family id — and dropping MUID+VER moved the parameter model by -0.0s
# (19.6s either way). As a predictor in its own right it scored R2 -19.6% /
# 72.5s MAE, WORSE than the serial-alone baseline (5.8% / 66.9s): the signature
# of a high-cardinality nuisance id, not a hidden protocol.
#
# Görtler described the mechanism unprompted in the same week's call, and it
# matches the measurement exactly: MUID is a per-day measurement counter that
# resets nightly and skips the ids consumed by adjustment scans, so it carries
# ordering information and nothing else.
#
# Denylisted anyway, per 03d's own verdict: a field that CAN leak should not be
# one retrain away from leaking. VER (scanner software version) is not a scan
# property at all — it splits the corpus by install, which lets a tree memorise
# per-site behaviour.
SUT_IDENTIFIER_DENYLIST = {'MUID', 'VER'}

# ----------------------------------------------------------------------------
# THE THIRD OBJECTION, added 2026-08-07 after the Görtler meeting.
#
# His position is to pass every parameter and let the model work out which
# matter, because a model that hides fields from itself cannot reach the
# extraordinary 1%. That is right, and 03f agreed with it before he said it: the
# ranked top-12 beat both hand-picked sets. The reason a denylist survives that
# position is that "don't hide things from the model" is a claim about SCAN
# PHYSICS, and these fields are not physics.
#
#   SUT_LEAKAGE_DENYLIST        ~equal to the TARGET.
#   SUT_IDENTIFIER_DENYLIST     an IDENTITY that cannot transfer.
#   SUT_PLANNER_DERIVED_DENYLIST  a quantity the scanner COMPUTED FROM the
#                               timing it was about to run. Not the answer, and
#                               not an id — a re-expression of the answer.
#
# SNR is the seed and currently the only entry. The three real messages pinned
# in tests/test_sut_parser.py move it monotonically AGAINST scan time:
#
#     haste       SNR 87   ST   8
#     haste       SNR 69   ST  15
#     ep2d_diff   SNR 39   ST 400
#
# which is what MR physics predicts, since predicted SNR carries an acquisition-
# time term. n=3 is not proof, and there is a real counter-argument: if SNR is a
# deterministic function of parameters we already supply, it is REDUNDANT rather
# than leaky and costs nothing to include. The distinction turns on whether the
# console computes it before or after the operator's adjustments, which nobody
# in the 08-07 meeting could answer.
#
# So it is denied by DEFAULT and its exclusion carries a PRICE TAG: gate 4 of
# the parameter report scores the vector with and without this denylist on the
# grouped split and prints the difference. A denylist entry that cannot show
# what it costs is an assertion; one that can is a decision. If the measured
# cost is large and the grouped split shows no divergence, this entry should be
# removed — and the report is what tells us that.
SUT_PLANNER_DERIVED_DENYLIST = {'SNR'}

# NOT a denylist — a reading aid, migrated out of 03f_coverage_and_masking.py so
# it survives that notebook's archiving. These fields are ADMISSIBLE and stay in
# the vector; the report flags them wherever they rank high, so a suspect is
# noticed rather than admired.
#
# The test to apply, the same one that convicted ST: is this a re-expression of
# the answer, or a cause of it? A cause stays in, however strongly it correlates.
#
#   BHD/ACQW/TGD are absent from all three pinned messages (they ride gated and
#   breath-hold sequences), so nothing here has been measured on our corpus yet.
#   BHD is the closest call — a breath-hold duration on a breath-hold sequence is
#   very nearly the measurement duration — and it is left admissible rather than
#   denied on a guess, because gate 5 can settle it with numbers.
SUT_SUSPECT_WATCHLIST = {
    'BHD':  'breath-hold duration — a time in seconds, and on a breath-hold '
            'sequence it is close to the measurement itself. Strongest denial '
            'candidate; unverified because no sampled message carries it.',
    'ACQW': 'acquisition window — a time, but per cardiac cycle. Reads as a '
            'cause rather than a re-expression.',
    'TGD':  'trigger delay — a time, but it precedes the acquisition rather '
            'than measuring it.',
    'TEU':  'TE in microseconds — an exact duplicate of TE (TE:101/TEU:101000). '
            'Redundant, not leaky; harmless under "use all".',
    'TP':   'table position — geometry, but it fingerprints an exam. An '
            'IDENTITY concern, not a planner-derived one: gate 5 catches it by '
            'the random-vs-grouped importance gap, not by this note.',
    'ES':   'echo spacing — a time, but a per-echo one, and a genuine driver '
            'of how long a turbo train takes.',
    'UT':   'unidentified, 3-digit, varies between messages on one scanner '
            '(655/683). Named here so it is not mistaken for something known.',
}

# The union every consumer wants. 03f built this by hand at its line 400; having
# it here means adding a fourth objection later reaches every caller at once.
SUT_ALL_DENYLISTS = (
    SUT_LEAKAGE_DENYLIST | SUT_IDENTIFIER_DENYLIST | SUT_PLANNER_DERIVED_DENYLIST
)

TRIGGER_MODE_VOCAB = {
    'none': 0, 'ecg': 1, 'peripheral_pulse': 2, 'respiratory': 3,
    'external': 4, 'unknown': 5,
}  # placeholder categories — trigger_mode field itself is still unidentified
NUM_TRIGGER_MODES = len(TRIGGER_MODE_VOCAB)

# ============================================================================
# THE CANDIDATE TABLE — every numeric parameter any PARAM_SET may draw on.
#
#   name -> (scale divisor, value to write when the field is absent)
#
# ONE keyed table rather than two positionally-aligned lists, because the
# aligned-list arrangement it replaces could desync silently and needed an
# assert at the bottom of this file to catch it. Here a name cannot have a
# divisor without a default, or land at the wrong index.
#
# THE DIVISOR is not cosmetic. An unscaled large-magnitude numeric silently
# erases the categorical conditioning through LayerNorm — see the
# conditioning_scale warning in AlternatingPipeline/models/sequence_generator.py.
# That exact bug caused three separate multi-week flat-duration incidents in
# this project. Divisors come from the observed p99 (03b section 1's
# 'suggested divisor' column); step 03 re-checks each one against real data at
# write time and warns when p99/divisor lands outside [0.05, 20].
#
# THE MISSING DEFAULT matters just as much, and is why this table exists at
# all. _safe_float in step 03 and safe_float in
# AlternatingPipeline/training/utils.py both default an absent key to 0.0. For
# the multiplicative factors in the acquisition-time formula
#
#     TA ~= TR x PEL x AVG x CONC / (PAT x TF)
#
# zero is WRONG: those fields are sequence-scoped, and absent means "does not
# apply to this sequence family", not "zero of it". 1.0 is the identity element
# of the formula and the physically correct reading — one average, one
# concatenation, no parallel-imaging acceleration, one echo per TR. A genuine
# measurement (TR, slice count, matrix lines) has no neutral value, so those
# keep 0.0 and the model learns the missing-ness.
# ============================================================================

# Divisors and defaults are calibrated against the real sampled messages
# pinned in tests/test_sut_parser.py (haste + ep2d_diff, serial 176148),
# not guessed — the observed values are in the comments.
SEQPARAM_CANDIDATES = {
    # MEASUREMENTS — no neutral value. 0.0 reads as "not recorded", which is
    # a distinction the model can learn.
    'TR':                      (1000.0, 0.0),   # ms;  observed 866 / 1000 / 4300
    'num_slices':              (  30.0, 0.0),   # SLC; observed 9 / 15 / 18
    'phase_encoding_lines':    ( 256.0, 0.0),   # PEL; observed 333 / 80
    'base_resolution':         ( 256.0, 0.0),   # BR;  observed 320 / 130
    # MULTIPLICATIVE FACTORS — 1-based, so 1.0 is the identity element of the
    # TA formula and the correct reading of "does not apply to this sequence".
    #
    # NOTE the divisors below are NOT all 1.0 any more, and the reason is worth
    # keeping. They were, because all three sampled messages carry AVG:1,
    # CONC:1, PAT:2, REP:0 — a 1.0 divisor is exactly right for those. The
    # 2026-08-08 build over 51,321 rows measured p99 of CONC 23, PAT 256 and
    # REP 89, so those three entered the conditioning vector at 23x, 256x and
    # 89x and would have erased the categorical conditioning through LayerNorm.
    # Three samples cannot see a p99. The values here now match the corpus; the
    # band-enforcement pass further down repairs any that drift again.
    'averages':                (   1.0, 1.0),   # AVG; observed 1, corpus p99 3
    'concatenations':          (  50.0, 1.0),   # CONC; observed 1, corpus p99 23
    'parallel_imaging_factor': ( 500.0, 1.0),   # PAT; observed 2, corpus p99 256
    'turbo_factor':            ( 256.0, 1.0),   # TF;  observed 256 on haste,
                                                #      ABSENT on ep2d_diff (EF)
    # REP is 0-BASED — it counts ADDITIONAL measurements, and all three sampled
    # messages carry REP:0 for a single-measurement scan. So the neutral value
    # here is 0, not 1, and an absent REP means the same thing the common
    # present value does. (Watch this one in 03e's leave-one-out: a field that
    # is 0 on nearly every row cannot carry a set, whatever the physics says.)
    # ...and the corpus answered that parenthesis: REP p99 is 89, not 0. It is
    # a real feature on some sequence families and absent-meaning-zero on most.
    'repetitions':             ( 100.0, 0.0),   # REP; observed 0, corpus p99 89
    # PPF/SPF are Siemens ENUM CODES, not fractions — observed PPF 1 and 8,
    # SPF 16. Scaled by the apparent ceiling of the shared enum so both land
    # at O(1) and stay on one scale, since they are the same kind of quantity.
    # Being an enum, the model reads them as ordered categories rather than
    # ratios; step 03's divisor check will flag it if a value ever exceeds 16.
    'phase_partial_fourier':   (  16.0, 1.0),   # PPF; observed 1 / 8
    'slice_partial_fourier':   (  16.0, 1.0),   # SPF; observed 16
}

# ============================================================================
# PRESENCE FLAGS — what makes "pass every parameter" actually work.
#
# Görtler (2026-08-07): pass everything and let the model sort it out, because a
# model that hides fields from itself cannot reach the extraordinary 1%. He is
# right, and the pipeline could not execute it, for a reason that is invisible
# from outside the code:
#
#   Every number in 03b-03f came from HistGradientBoostingRegressor, which has a
#   NATIVE THIRD STATE for a missing value and learns a split direction for it.
#   The model path has no such state. build_conditioning_tensor runs every value
#   through safe_float (AlternatingPipeline/training/utils.py:107), which turns
#   NaN into a real number. With 7 fields that is tolerable. With ~89 — most of
#   them sequence-family-scoped and therefore absent on most rows — the majority
#   of the vector becomes fabricated measurements, and LayerNorm then makes the
#   near-constant columns compete with the categorical embeddings for amplitude.
#   That is the exact mechanism behind this project's three multi-week
#   flat-duration incidents.
#
# A presence flag restores the third state in the one place it was missing. It
# also removes the hardest manual bottleneck: with a flag saying "this field was
# not in the message", the VALUE written for an absent field stops carrying
# meaning, so a newly discovered field needs a divisor but not a hand-classified
# missing default. Hand-classifying ~89 fields as multiplicative-or-measurement
# was the thing that made "use all" a week of guessing.
#
# The hand-curated defaults in SEQPARAM_CANDIDATES stay: they are correct, and
# they keep the vector sane if flags are ever switched off for an ablation.
# ============================================================================

SEQPARAM_PRESENCE_SUFFIX = '__present'

# Env-overridable so the flags themselves can be ablated. Gate 4 of the
# parameter report scores with and without, which is the only way to show the
# masking result (03f section B: 9.0s masked vs 24.6s as-is) survives the move
# from the GBM to the conditioning tensor.
SEQPARAM_USE_PRESENCE_FLAGS = os.environ.get(
    'SEQPARAM_USE_PRESENCE_FLAGS', '1').strip().lower() not in ('0', 'false', 'no')


def presence_name(name):
    """The flag key that accompanies a parameter: 1.0 present, 0.0 absent."""
    return f"{name}{SEQPARAM_PRESENCE_SUFFIX}"


def is_presence_name(name):
    return name.endswith(SEQPARAM_PRESENCE_SUFFIX)


# ============================================================================
# DERIVED conditioning features — computed by step 03 rather than parsed out of
# the message, and therefore exempt from the SUT_FIELD_MAP check below.
#
# `sut_in_segment` is the row-level counterpart to the per-field flags, and it
# matters more than any single parameter. 03f section B: the in-segment join
# fires on 80.3% of segments; on the rest, step 03 falls back to the most recent
# SUT event BEFORE the segment, and 71.5% of those name a DIFFERENT sequence
# than the one that actually ran. Those rows are not noisy, they are wrong, and
# no additional field fixes them. Masking them scored 9.0s MAE against 24.6s for
# using them as-is.
#
# One flag lets the model learn that distinction instead of us having to choose
# between dropping a fifth of the corpus and poisoning it.
# ============================================================================

SEQPARAM_DERIVED = {
    # name: (divisor, value when the information is unavailable)
    'sut_in_segment': (1.0, 0.0),
}

# ============================================================================
# THE DISCOVERED FIELD TABLE — how PARAM_SET='all' gets ~89 fields without
# anybody hand-writing ~89 divisors.
#
# THE DIVISOR IS NOT COSMETIC (see the long note above SEQPARAM_CANDIDATES): an
# unscaled large-magnitude numeric erases the categorical conditioning through
# LayerNorm. So every field needs one, and picking them by hand was the
# bottleneck that kept the selected set at 7. Step 03 already computes p99 per
# field for its own divisor sanity check; it now WRITES that out, and this reads
# it back.
#
# Safe against desync by construction: `conditioning_scale` is a persistent
# buffer registered on the model (sequence_generator.py:172-178) and travels
# INSIDE the checkpoint, so a served model always receives the divisors it was
# trained with even if this table has moved on since. What a changed table CAN
# do is make two training runs incomparable, which is why
# SEQPARAM_DIVISOR_FINGERPRINT below goes into the model manifest.
# ============================================================================

# A field on 1% of rows is on ~500 rows of the current corpus — thin, but thin is
# exactly where the extraordinary 1% lives, and a presence flag means a mostly-
# absent field costs the model nothing to ignore. This floor is the single knob
# that trades tail coverage against vector width; the report prints what it
# excludes so the trade is visible rather than assumed.
SEQPARAM_MIN_PRESENCE_PCT = float(os.environ.get('SEQPARAM_MIN_PRESENCE_PCT', '1.0'))
# Below this, a "numeric" field is mostly strings and belongs in an embedding,
# not in a scaled numeric column. Same threshold 03b-03f used.
SEQPARAM_MIN_NUMERIC_PCT = float(os.environ.get('SEQPARAM_MIN_NUMERIC_PCT', '90.0'))

# The band a scaled feature has to land in. Above it, the feature erases the
# categorical conditioning through LayerNorm; below it, the feature is invisible
# to the model. Neither raises anything at training time, which is why it is
# enforced at config-load time instead.
SEQPARAM_SCALE_BAND = (0.05, 20.0)


def seqparam_stable_name(raw_key):
    """The one name a SUT field is known by everywhere downstream.

    A field has two names: the raw message mnemonic (`SLC`) and the mapped
    stable name (`num_slices`). Mixing them is not cosmetic — `SLC` and
    `num_slices` would both survive into PARAM_SET='all' as separate columns
    holding identical values, doubling a field's weight in the conditioning
    vector for no reason and making its permutation importance read half its
    real worth in the report.

    So there is exactly one rule, here, and both step 03 and the divisor loader
    apply it: mapped name where SUT_FIELD_MAP has one, raw key otherwise.
    """
    return SUT_FIELD_MAP.get(raw_key, raw_key)


def suggest_divisor(magnitude):
    """A scale divisor putting a field's typical MAGNITUDE at O(1), snapped to a
    1/2/5 ladder.

    SNAPPED, not exact, and that is the whole point. An exact divisor moves a
    little on every rebuild, which would change every column's scale, change the
    fingerprint, and make two training runs incomparable for no reason. The
    1/2/5 x 10^k ladder is coarse enough that ordinary month-to-month drift
    leaves it untouched, and fine enough to keep every field inside
    SEQPARAM_SCALE_BAND.

    MAGNITUDE, NOT p99. Passing a raw p99 was wrong for a field that is entirely
    NEGATIVE: `TP` (table position) runs about -1900 to -989, so its p99 is
    -989, the old `p99 > 0` guard returned an identity divisor, and the field
    entered the conditioning vector at ~1500x scale — the exact LayerNorm-
    erasure mode this ladder exists to prevent. Callers pass
    max(|p01|, |p99|); the abs() here is a second line of defence for a caller
    that forgets.

    Returns 1.0 for a non-finite or all-zero field: nothing to rescale, and
    dividing by zero would be worse.
    """
    try:
        magnitude = abs(float(magnitude))
    except (TypeError, ValueError):
        return 1.0
    if not (magnitude > 0) or magnitude != magnitude or magnitude == float('inf'):
        return 1.0

    exponent = math.floor(math.log10(magnitude))
    mantissa = magnitude / (10.0 ** exponent)
    for step in (1.0, 2.0, 5.0):
        if mantissa <= step:
            return step * (10.0 ** exponent)
    return 10.0 ** (exponent + 1)


def _load_divisor_table(path):
    """Read step 03's emitted divisor table, or return ({}, None) if absent.

    Fails SOFT on purpose. This module is %run by every notebook and imported by
    the test suite, neither of which can require a /dbfs artefact to exist — a
    fresh clone must still be able to load the config and run the guards. When
    the table is missing, PARAM_SET='all' falls back to the hand-curated
    candidates and says so loudly at import time.

    Keys are normalised through seqparam_stable_name on the way in, so a table
    written under either convention (or an older one written before the rule
    existed) resolves to the same field set.
    """
    try:
        with open(path, 'r') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}, None

    fields = payload.get('fields') or {}
    table = {}
    for raw_name, spec in fields.items():
        try:
            divisor = float(spec['divisor'])
        except (KeyError, TypeError, ValueError):
            continue
        if not divisor or divisor != divisor:      # 0.0 or NaN
            continue

        name = seqparam_stable_name(raw_name)
        # `magnitude` is max(|p01|, |p99|) and is what the band pass below
        # checks against. Carrying it through is load-bearing, not bookkeeping:
        # without it the pass has nothing to compare a divisor to and skips
        # every field, which is how an out-of-band divisor reaches training.
        # Older tables predate the key, so fall back to |p99|.
        _p99 = float(spec.get('p99', 0.0) or 0.0)
        entry = {
            'divisor': divisor,
            # Absent fields are carried by the presence flag, so 0.0 is a safe
            # default for anything not hand-classified. A curated entry still
            # wins — see _missing_default_for below.
            'missing_default': float(spec.get('missing_default', 0.0)),
            'presence_pct': float(spec.get('presence_pct', 0.0)),
            'numeric_pct': float(spec.get('numeric_pct', 100.0)),
            'p99': _p99,
            'magnitude': float(spec.get('magnitude', 0.0) or 0.0) or abs(_p99),
            'raw_key': raw_name,
        }
        # A table carrying BOTH `SLC` and `num_slices` collapses to one entry.
        # Keep whichever is present on more rows rather than whichever the dict
        # happened to yield last, so the choice does not depend on JSON order.
        if name in table and table[name]['presence_pct'] >= entry['presence_pct']:
            continue
        table[name] = entry

    return table, payload.get('fingerprint')


SEQPARAM_DISCOVERED, SEQPARAM_DIVISOR_FINGERPRINT = _load_divisor_table(
    SEQPARAM_DIVISOR_TABLE)

# ----------------------------------------------------------------------------
# BAND ENFORCEMENT — the corpus is allowed to overrule a hand-calibration.
#
# The 2026-08-08 build caught three CURATED divisors badly out of band:
#
#     parallel_imaging_factor   p99 256 / 1.0  = 256x
#     repetitions               p99  89 / 1.0  =  89x
#     concatenations            p99  23 / 1.0  =  23x
#
# All three were calibrated against the three real messages pinned in
# tests/test_sut_parser.py, which carry PAT:2, REP:0 and CONC:1. Those readings
# were correct and the corpus is 51,321 rows wide: PAT reaches 256, REP 89,
# CONC 23. Three samples cannot see that.
#
# "Curated wins" was written to protect a real thing — PPF and SPF are Siemens
# enum codes that must share ONE scale, which a per-field percentile would
# split. It was not meant to protect a number the data has since falsified. So
# the rule now has an exception with a stated test: a curated divisor wins
# UNLESS the corpus puts the field outside SEQPARAM_SCALE_BAND, in which case
# the measured magnitude wins and the override is recorded for the report.
#
# This also repairs a stale or malformed table without a rebuild, which matters
# because the alternative is a silently-erased conditioning vector.
# ----------------------------------------------------------------------------

SEQPARAM_DIVISOR_OVERRIDES = []

for _name, _spec in SEQPARAM_DISCOVERED.items():
    _magnitude = _spec.get('magnitude') or abs(_spec.get('p99') or 0.0)
    if _magnitude <= 0:
        continue
    _preferred = (SEQPARAM_CANDIDATES[_name][0] if _name in SEQPARAM_CANDIDATES
                  else _spec['divisor'])
    _ratio = _magnitude / _preferred if _preferred else float('inf')
    _low, _high = SEQPARAM_SCALE_BAND
    if _low <= _ratio <= _high:
        _spec['divisor'] = _preferred
        continue
    _repaired = suggest_divisor(_magnitude)
    _spec['divisor'] = _repaired
    SEQPARAM_DIVISOR_OVERRIDES.append({
        'name': _name,
        'was': _preferred,
        'now': _repaired,
        'magnitude': _magnitude,
        'was_ratio': _ratio,
        'source': 'curated' if _name in SEQPARAM_CANDIDATES else 'table',
    })

if SEQPARAM_DIVISOR_OVERRIDES:
    print(f"[config] {len(SEQPARAM_DIVISOR_OVERRIDES)} divisor(s) re-derived from "
          f"the corpus because the configured value put the feature outside "
          f"{SEQPARAM_SCALE_BAND}:")
    for _o in SEQPARAM_DIVISOR_OVERRIDES:
        print(f"           {_o['name']:<24} {_o['source']:>8} {_o['was']:>10.1f} "
              f"-> {_o['now']:>10.1f}   (|magnitude| {_o['magnitude']:.1f}, "
              f"was {_o['was_ratio']:.1f}x)")


def classify_seqparam_field(name, stats):
    """Why a discovered field is or is not admissible. One rule, one place.

    `stats` is {'presence_pct', 'numeric_pct'} for the field, under its STABLE
    name (see seqparam_stable_name).

    Returns (admissible: bool, reason: str). The reason is not decoration — it
    is what the human-readable half of the parameter report prints, so "how many
    parameters are we using and why" is answered from the same code that made
    the decision rather than from a comment that can drift away from it.
    """
    if name in SUT_LEAKAGE_DENYLIST:
        return False, 'denied: ~equal to the duration target'
    if name in SUT_IDENTIFIER_DENYLIST:
        return False, 'denied: an identity that cannot transfer to a new customer'
    if name in SUT_PLANNER_DERIVED_DENYLIST:
        return False, 'denied: derived by the scanner from the timing it was about to run'
    if stats.get('numeric_pct', 100.0) < SEQPARAM_MIN_NUMERIC_PCT:
        return False, (f"not numeric on {stats.get('numeric_pct', 0.0):.0f}% of rows "
                       f"— belongs in an embedding, not a scaled column")
    if stats.get('presence_pct', 0.0) < SEQPARAM_MIN_PRESENCE_PCT:
        return False, (f"present on only {stats.get('presence_pct', 0.0):.2f}% of rows "
                       f"(floor {SEQPARAM_MIN_PRESENCE_PCT:.1f}%)")
    if name in SEQPARAM_CANDIDATES:
        return True, 'admissible: hand-calibrated candidate'
    return True, 'admissible: discovered, divisor from p99'

# Everything the discovered table offers that is admissible under all three
# objections and clears both floors. Sorted for a stable column order — the
# order IS the meaning of a trained checkpoint's conditioning columns, so it
# must not depend on dict iteration or on which run wrote the table.
SEQPARAM_ADMISSIBLE_DISCOVERED = sorted(
    name for name, spec in SEQPARAM_DISCOVERED.items()
    if classify_seqparam_field(name, spec)[0]
)

# ============================================================================
# THE NAMED SETS. Select with PARAM_SET at the top of this file.
# ============================================================================

PARAM_SETS = {
    # The acquisition-time formula, as its own multiplicands. Every term of
    # TA ~= TR x PEL x AVG x CONC / (PAT x TF), plus the slice count. The bet
    # is that handing the model the factors it would need to reconstruct the
    # formula beats handing it a shorter list of stronger correlates.
    'luke': [
        'TR', 'num_slices', 'phase_encoding_lines', 'averages',
        'concatenations', 'parallel_imaging_factor', 'turbo_factor',
    ],
    # Görtler, 2026-08-04. Deliberately NOT the formula: he argued most timing
    # parameters are physically real but redundant, because echo spacing, TE
    # and slice thickness all impose constraints that surface as a floor on TR,
    # and TR is repeated hundreds of times per scan. So the set is TR plus only
    # the things that change HOW MANY TRs run.
    #   base_resolution — the gap he flagged: "512 will take two times longer"
    #                     than 256. Already parsed as BR, never promoted.
    #   partial Fourier — the genuine exception to the redundancy argument. A
    #                     256-line acquisition drops to ~136 lines (128 + a few
    #                     to sample the matrix centre twice), roughly halving
    #                     the time. Not a longer TR — fewer of them.
    # Ruled out by name in that call: flip angle, echo spacing, TE, slice
    # thickness, field of view, MUID, ST.
    'navneet': [
        'TR', 'num_slices', 'averages', 'repetitions',
        'base_resolution', 'phase_partial_fourier', 'slice_partial_fourier',
    ],
    # Görtler, 2026-08-07, and now the DEFAULT. Every admissible field the
    # corpus actually contains — no hand-picking at all.
    #
    # His argument, which the data had already made: rare cases are rare
    # BECAUSE they are characterised by unusual values of fields that are
    # constant on the other 99%. Aggregate MAE cannot see those fields, so
    # selecting features on aggregate MAE systematically deletes exactly the
    # signal the extraordinary 1% needs. 03f agreed before he said it — its
    # ranked top-12 beat both hand-picked sets below.
    #
    # `luke` and `navneet` are KEPT, as the control group. They cost two lines
    # each and they are the only way to later PROVE that more was better rather
    # than assert it. Run 03b's gate 4 to score all three on one split.
    #
    # This resolves from the discovered table when step 03 has written one, and
    # falls back to the hand-curated candidates otherwise (with a warning at
    # import). Sorted for a stable column order — the order IS the meaning of a
    # trained checkpoint's conditioning columns.
    'all': (SEQPARAM_ADMISSIBLE_DISCOVERED
            or sorted(SEQPARAM_CANDIDATES)),
}

if PARAM_SET == 'all' and not SEQPARAM_ADMISSIBLE_DISCOVERED:
    print(
        f"[config] PARAM_SET='all' but no divisor table at "
        f"{SEQPARAM_DIVISOR_TABLE} — falling back to the "
        f"{len(SEQPARAM_CANDIDATES)} hand-curated candidates instead of every "
        f"discovered field. Run 03_build_preprocessed_pkl.py to emit the table, "
        f"or point SEQPARAM_DIVISOR_TABLE at an existing one."
    )

if PARAM_SET not in PARAM_SETS:
    raise ValueError(
        f"PARAM_SET={PARAM_SET!r} is not a known parameter set. "
        f"Choose one of {sorted(PARAM_SETS)}, or add it to PARAM_SETS in "
        f"csv_pipeline_seqparams/config.py."
    )

# ============================================================================
# RESOLUTION — PARAM_SET to the actual feature list, in three blocks.
#
#   [ values ][ presence flags ][ derived ]
#
# Appended as BLOCKS rather than interleaved so an analysis can slice the
# conditioning vector by meaning: vec[:n] is what the scanner reported, vec[n:]
# is what we know about whether it reported it. The order is fixed and sorted
# within each block, because the order IS the meaning of a trained checkpoint's
# columns — a reordering silently repoints every learned weight.
#
# WHY NOTHING ELSE NEEDS TO CHANGE: build_seqparams_model_config() at the bottom
# of this file derives BOTH base_conditioning_dim AND conditioning_scale from
# these two lists, and build_conditioning_tensor reads names from a list. So
# widening the vector here propagates to 04/05/06/07 and into the model with no
# change to any of them, and no change to the model code at all.
# ============================================================================

_selected_values = list(PARAM_SETS[PARAM_SET])


def _divisor_for(name):
    """Scale divisor for a value feature. The resolved table wins.

    SEQPARAM_DISCOVERED is checked FIRST, and that ordering is the fix for the
    2026-08-08 finding rather than an accident. The band-enforcement pass above
    has already written the CURATED divisor into that table wherever the
    curated value is sound, and a corpus-derived one wherever it is not. So the
    table is the single resolved answer, and reading SEQPARAM_CANDIDATES first
    would reinstate exactly the three out-of-band divisors the pass repaired.

    The curated entries still win in the ordinary case — including the one they
    exist for, PPF/SPF, which are Siemens enum codes sharing one scale that a
    per-field percentile would split.
    """
    if name in SEQPARAM_DISCOVERED:
        return SEQPARAM_DISCOVERED[name]['divisor']
    if name in SEQPARAM_CANDIDATES:
        return SEQPARAM_CANDIDATES[name][0]
    if name in SEQPARAM_DERIVED:
        return SEQPARAM_DERIVED[name][0]
    raise ValueError(
        f"{name!r} has no scale divisor. Every feature needs one before it can "
        f"reach the model — an unscaled large-magnitude numeric erases the "
        f"categorical conditioning through LayerNorm (see the note above "
        f"SEQPARAM_CANDIDATES). Add it to SEQPARAM_CANDIDATES, or rebuild the "
        f"divisor table with 03_build_preprocessed_pkl.py."
    )


def _missing_default_for(name):
    """Value written when a field is absent from the message.

    Curated defaults win: 1.0 for the 1-based multiplicands of the TA formula
    ("does not apply to this sequence"), 0.0 for a measurement ("not recorded").
    Anything discovered gets 0.0, which is safe precisely BECAUSE the presence
    flag carries the meaning — that is what removes the need to hand-classify
    every newly discovered field.
    """
    if name in SEQPARAM_CANDIDATES:
        return SEQPARAM_CANDIDATES[name][1]
    if name in SEQPARAM_DERIVED:
        return SEQPARAM_DERIVED[name][1]
    return 0.0


if SEQPARAM_USE_PRESENCE_FLAGS:
    _selected_flags = [presence_name(n) for n in _selected_values]
    _selected_derived = sorted(SEQPARAM_DERIVED)
else:
    # The ablation arm. Without flags the vector is exactly what it was before
    # 2026-08-07, so `luke` reproduces the historical 7-feature checkpoint and
    # gate 4 can price the flags rather than assume them.
    _selected_flags = []
    _selected_derived = []

# Numeric SUT parameter feature names that ride the flat conditioning tensor
# (see AlternatingPipeline/training/utils.py::build_conditioning_tensor
# extra_feature_names). Resolved from PARAM_SET — this is the name every
# downstream notebook already reads, which is why the switch reaches all of
# them without touching any.
EXAMINATION_SEQPARAM_FEATURES = (
    _selected_values + _selected_flags + _selected_derived
)
EXAMINATION_SEQPARAM_SCALE = [
    # Flags are already 0/1, so their divisor is the identity. Giving them a
    # p99-derived one would rescale a boolean and make "absent" a nonzero value.
    1.0 if is_presence_name(name) else _divisor_for(name)
    for name in EXAMINATION_SEQPARAM_FEATURES
]

# What step 03 writes into every row's 'conditioning' — the union of every set,
# not just the selected one, plus a flag per value and the derived features.
# build_conditioning_tensor reads only the names in extra_feature_names, so the
# surplus keys are inert at training time, and one Spark rebuild then serves
# every parameter set.
#
# NOTE this is the union of what CONFIG knows about. Step 03 additionally writes
# every admissible field it discovers in the messages, which is how PARAM_SET=
# 'all' has values to read on the run AFTER the one that emitted the table.
_all_values = sorted(set(SEQPARAM_CANDIDATES) | set(SEQPARAM_ADMISSIBLE_DISCOVERED))
SEQPARAM_ALL_CANDIDATES = (
    _all_values
    + [presence_name(n) for n in _all_values]
    + sorted(SEQPARAM_DERIVED)
)

SEQPARAM_MISSING_DEFAULTS = {
    name: (0.0 if is_presence_name(name) else _missing_default_for(name))
    for name in SEQPARAM_ALL_CANDIDATES
}

# ============================================================================
# LEAKAGE GUARD — three independent gates:
#   gate #1 (this file, config-import-time, static feature list — the
#            assert_no_leakage() call at the bottom of this file)
#   gate #2 (step 03, preprocessing-write-time, actual runtime dict keys —
#            catches a filtering-logic bug even if the static list is clean)
#   gate #3 (AlternatingPipeline/training/utils.py::build_conditioning_tensor,
#            training-tensor-build-time — enforced at the actual point of
#            tensor construction via its optional `denylist` parameter, which
#            04/05/06 set to SUT_ALL_DENYLISTS. This is independent of gates
#            #1/#2: it fires regardless of how a caller assembled
#            extra_conditioning_features, not just the one path gate #1 already
#            validates.)
#
# Görtler's transcript: "scanning time" is ~equal to the duration target and
# must never be an input feature. The originally-named "SD58" slot doesn't
# correspond to any literal field in the confirmed KEY:VALUE message format
# (no "SD" label exists in it at all) — kept as defense-in-depth in case a
# future field is found to be duration-equivalent.
#
# The three denylists these gates enforce are defined ABOVE the candidate table,
# because PARAM_SET='all' has to filter the discovered fields through them
# before it can enumerate anything.
# ============================================================================


def assert_no_leakage(feature_names, denylist=SUT_LEAKAGE_DENYLIST,
                      identifier_denylist=SUT_IDENTIFIER_DENYLIST,
                      planner_denylist=SUT_PLANNER_DERIVED_DENYLIST):
    """Raise if any banned name is present, naming which objection applies.

    feature_names: any iterable of feature-name strings (a static config
        list, or the actual runtime keys of a conditioning dict).
    """
    feature_names = list(feature_names)

    overlap = denylist.intersection(feature_names)
    if overlap:
        raise ValueError(
            f"Leakage guard tripped: {sorted(overlap)} must never be used as "
            f"an input feature — it is ~equal to the duration target."
        )

    identity = (identifier_denylist or set()).intersection(feature_names)
    if identity:
        raise ValueError(
            f"Identifier guard tripped: {sorted(identity)} must never be used "
            f"as an input feature — it identifies a measurement, a day or an "
            f"install rather than describing the scan, so it cannot transfer "
            f"to another customer."
        )

    derived = (planner_denylist or set()).intersection(feature_names)
    if derived:
        raise ValueError(
            f"Planner-derived guard tripped: {sorted(derived)} must never be "
            f"used as an input feature — the scanner computed it FROM the "
            f"timing it was about to run, so it is a re-expression of the "
            f"target rather than a cause of it. If the parameter report's "
            f"price tag says this exclusion is too expensive, remove it from "
            f"SUT_PLANNER_DERIVED_DENYLIST deliberately — do not route around "
            f"the guard."
        )


# Gate #1 — fires the moment this config module is loaded, for EVERY named set
# rather than only the selected one. A typo in a set nobody has run yet is a
# bug you want to hear about now, not on the day you switch to it.
_KNOWN_FEATURE_SOURCES = (
    set(SEQPARAM_CANDIDATES) | set(SEQPARAM_DERIVED) | set(SEQPARAM_DISCOVERED)
)

for _set_name, _set_features in PARAM_SETS.items():
    assert_no_leakage(_set_features)

    _unknown = [n for n in _set_features if n not in _KNOWN_FEATURE_SOURCES]
    if _unknown:
        raise ValueError(
            f"PARAM_SETS[{_set_name!r}] names {_unknown}, which have no scale "
            f"divisor — not in SEQPARAM_CANDIDATES, not in SEQPARAM_DERIVED, "
            f"and not in the discovered table at {SEQPARAM_DIVISOR_TABLE}. "
            f"Every feature needs an explicit divisor before it can reach the "
            f"model (see the LayerNorm-erasure warning above)."
        )

# Every CURATED candidate must be something the SUT parser actually produces, or
# step 03 would write its default into every row forever and the feature would
# look dead rather than misspelled.
#
# Scoped to SEQPARAM_CANDIDATES on purpose. Discovered fields come FROM the
# parsed messages, so they cannot fail this check by construction, and derived
# features are computed rather than parsed — requiring either to appear in
# SUT_FIELD_MAP would make the map a second place to maintain the field list.
_unmapped = [n for n in SEQPARAM_CANDIDATES if n not in set(SUT_FIELD_MAP.values())]
if _unmapped:
    raise ValueError(
        f"SEQPARAM_CANDIDATES names {_unmapped}, which SUT_FIELD_MAP never "
        f"produces. Add the raw message key to SUT_FIELD_MAP first, or the "
        f"feature will be a constant."
    )

assert len(EXAMINATION_SEQPARAM_FEATURES) == len(EXAMINATION_SEQPARAM_SCALE), (
    "EXAMINATION_SEQPARAM_FEATURES and EXAMINATION_SEQPARAM_SCALE must be "
    "the same length — every new numeric feature needs an explicit scale "
    "divisor (see the LayerNorm-erasure warning above)."
)

# Every selected feature must be something step 03 actually writes, or it reads
# as 0.0 on every row: build_conditioning_tensor falls back to 0.0 for a key
# that is not in the conditioning dict, which is silent and looks exactly like a
# dead feature. This is the check that catches a PARAM_SET='all' resolved from a
# divisor table NEWER than the pkl beside it.
_unwritten = [n for n in EXAMINATION_SEQPARAM_FEATURES
              if n not in set(SEQPARAM_ALL_CANDIDATES)]
if _unwritten:
    raise ValueError(
        f"PARAM_SET={PARAM_SET!r} selects {_unwritten}, which step 03 does not "
        f"write into 'conditioning'. Those would read as 0.0 on every row "
        f"rather than raising. Rebuild the pkl with 03_build_preprocessed_pkl.py "
        f"so the divisor table and the pkl come from the same run."
    )

# ============================================================================
# STALE-SOURCE GUARD for the notebooks that IMPORT AlternatingPipeline from a
# /tmp copy rather than copying it themselves.
#
# 04_train_models.py copies the repo to TMP_ROOT via _api_copy_py.
# 05_feature_analysis.py and 06_compare_models.py do NOT — they just
# sys.path.insert that directory and import. /tmp survives for the entire life
# of the cluster, so a copy made before a `git pull` stays stale indefinitely:
# any module ADDED to the repo since that copy fails to import, with a bare
# ModuleNotFoundError that points at the module rather than at the stale copy.
# (Observed 2026-07-27: checkpoint_compat.py, added after the 07-24 training
# run, was missing from a TMP_ROOT copied during that run on a cluster that had
# been up ever since.)
#
# Lives here because config.py is `%run`-loaded straight from the Workspace by
# all three notebooks, so it is always current — unlike anything under
# TMP_ROOT, which is exactly what cannot be trusted at this point.
# ============================================================================

def assert_pipeline_source_fresh(
    tmp_root,
    required_modules=(),
    purge=True,
    stale_prefixes=("AlternatingPipeline", "csv_pipeline_seqparams"),
):
    """Verify TMP_ROOT actually has the modules this notebook needs.

    Also evicts stale imports of them from the long-lived kernel, by NAME
    PREFIX — these are namespace packages whose top-level module object has
    `__file__ is None`, so a `__file__`-based purge silently misses it (see
    04_train_models.py for the full story).

    Raises RuntimeError naming the missing modules and the exact steps to fix
    it, instead of letting the import fail with an unrelated-looking error.
    """
    import importlib
    import os as _os
    import sys as _sys

    if purge:
        # BOTH conditions are required — each catches what the other misses.
        # By NAME PREFIX: namespace packages (no __init__.py) have
        # __file__ is None on the top-level module object, so a __file__ check
        # never evicts them. By __file__ UNDER tmp_root:
        # AlternatingPipeline/models/examination_model.py does
        # `from models.sequence_generator import ...` — a legacy TOP-LEVEL name
        # the prefix list cannot match. Missing that second case let a stale
        # model class survive a re-run on 2026-07-27 while the pre-flight
        # (which reads files directly) reported the new sha.
        _root = tmp_root.replace("\\", "/").rstrip("/")
        for _name, _mod in list(_sys.modules.items()):
            _mod_file = (getattr(_mod, "__file__", None) or "").replace("\\", "/")
            if (any(_name == p or _name.startswith(p + ".") for p in stale_prefixes)
                    or (_root and _mod_file.startswith(_root))):
                del _sys.modules[_name]
        # The directory listing cached by Python's FileFinder predates the
        # re-copy; without this a freshly-copied file can still be invisible.
        importlib.invalidate_caches()

    missing = [
        dotted for dotted in required_modules
        if not _os.path.isfile(_os.path.join(tmp_root, *dotted.split(".")) + ".py")
    ]
    if missing:
        raise RuntimeError(
            f"Stale source copy at {tmp_root} — missing: {', '.join(missing)}.\n"
            f"This directory was copied by 04_train_models.py before those modules "
            f"existed, and /tmp persists for the whole cluster lifetime, so it never "
            f"refreshes on its own.\n"
            f"Fix:\n"
            f"  1. Pull the Databricks Repos clone so it is on the latest commit.\n"
            f"  2. Re-run 04_train_models.py CELLS 1-2 ONLY (the _api_copy_py cell) — "
            f"do NOT run the training cell.\n"
            f"  3. Re-run this notebook."
        )
    return True


# ============================================================================
# MODEL CONFIG ASSEMBLY — the single source of truth for combining
# AlternatingPipeline.config.EXAMINATION_MODEL_CONFIG with this pipeline's
# SUT additions. 04_train_models.py, 05_feature_analysis.py, and
# 06_compare_models.py all call this instead of each re-deriving the same
# dict, so a future config-key addition can't be forgotten in one of the
# three and silently desync training from analysis/comparison.
#
# Takes the base config as a PARAMETER rather than importing it directly —
# this module is %run-loaded by 02/03 before AlternatingPipeline necessarily
# becomes import-able (see the module docstring), so the cross-package
# import has to happen in the caller, which then passes the result in here.
# ============================================================================

def build_seqparams_model_config(base_examination_config):
    """Combine base_examination_config with use_sut_conditioning + the
    widened base_conditioning_dim / conditioning_scale for this pipeline's
    extra numeric features. Does not mutate base_examination_config."""
    return {
        **base_examination_config,
        'use_sut_conditioning': True,
        'num_trigger_modes': NUM_TRIGGER_MODES,
        'base_conditioning_dim': (
            base_examination_config['base_conditioning_dim']
            + len(EXAMINATION_SEQPARAM_FEATURES)
        ),
        'conditioning_scale': (
            list(base_examination_config['conditioning_scale'])
            + list(EXAMINATION_SEQPARAM_SCALE)
        ),
    }
