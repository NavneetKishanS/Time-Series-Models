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
# ============================================================================

PARAM_SET = os.environ.get('PARAM_SET', 'luke').strip().lower()

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

# Models and analysis ARE namespaced by PARAM_SET. Two sets can widen
# base_conditioning_dim to the same number while meaning different things per
# column, so a shared directory would silently overwrite one checkpoint with
# the other and leave the comparison unreadable — with nothing in the manifest
# to say which set produced it.
MODELS_DIR   = f"/dbfs/FileStore/csv_pipeline_seqparams/models/{PARAM_SET}"
ANALYSIS_DIR = f"/dbfs/FileStore/csv_pipeline_seqparams/analysis/{PARAM_SET}"

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
    'averages':                (   1.0, 1.0),   # AVG; observed 1
    'concatenations':          (   1.0, 1.0),   # CONC; observed 1
    'parallel_imaging_factor': (   1.0, 1.0),   # PAT; observed 2
    'turbo_factor':            ( 256.0, 1.0),   # TF;  observed 256 on haste,
                                                #      ABSENT on ep2d_diff (EF)
    # REP is 0-BASED — it counts ADDITIONAL measurements, and all three sampled
    # messages carry REP:0 for a single-measurement scan. So the neutral value
    # here is 0, not 1, and an absent REP means the same thing the common
    # present value does. (Watch this one in 03e's leave-one-out: a field that
    # is 0 on nearly every row cannot carry a set, whatever the physics says.)
    'repetitions':             (   1.0, 0.0),   # REP; observed 0 on all three
    # PPF/SPF are Siemens ENUM CODES, not fractions — observed PPF 1 and 8,
    # SPF 16. Scaled by the apparent ceiling of the shared enum so both land
    # at O(1) and stay on one scale, since they are the same kind of quantity.
    # Being an enum, the model reads them as ordered categories rather than
    # ratios; step 03's divisor check will flag it if a value ever exceeds 16.
    'phase_partial_fourier':   (  16.0, 1.0),   # PPF; observed 1 / 8
    'slice_partial_fourier':   (  16.0, 1.0),   # SPF; observed 16
}

# ============================================================================
# THE TWO NAMED SETS. Select with PARAM_SET at the top of this file.
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
}

if PARAM_SET not in PARAM_SETS:
    raise ValueError(
        f"PARAM_SET={PARAM_SET!r} is not a known parameter set. "
        f"Choose one of {sorted(PARAM_SETS)}, or add it to PARAM_SETS in "
        f"csv_pipeline_seqparams/config.py."
    )

# Numeric SUT parameter feature names that ride the flat conditioning tensor
# (see AlternatingPipeline/training/utils.py::build_conditioning_tensor
# extra_feature_names). Resolved from PARAM_SET — this is the name every
# downstream notebook already reads, which is why the switch reaches all of
# them without touching any.
EXAMINATION_SEQPARAM_FEATURES = list(PARAM_SETS[PARAM_SET])
EXAMINATION_SEQPARAM_SCALE = [
    SEQPARAM_CANDIDATES[name][0] for name in EXAMINATION_SEQPARAM_FEATURES
]

# What step 03 writes into every row's 'conditioning' — the union of every set,
# not just the selected one. build_conditioning_tensor reads only the names in
# extra_feature_names, so the surplus keys are inert at training time, and one
# Spark rebuild then serves every parameter set.
SEQPARAM_ALL_CANDIDATES = list(SEQPARAM_CANDIDATES)

SEQPARAM_MISSING_DEFAULTS = {
    name: default for name, (_, default) in SEQPARAM_CANDIDATES.items()
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
#            04_train_models.py sets to SUT_LEAKAGE_DENYLIST. This is
#            independent of gates #1/#2: it fires regardless of how a caller
#            assembled extra_conditioning_features, not just the one path
#            gate #1 already validates.)
# Görtler's transcript: "scanning time" is ~equal to the duration target and
# must never be an input feature. The originally-named "SD58" slot doesn't
# correspond to any literal field in the confirmed KEY:VALUE message format
# (no "SD" label exists in it at all) — kept here as defense-in-depth in case
# a future field is found to be duration-equivalent.
#
# TWO denylists, because there are two distinct objections and only the first
# was ever guarded:
#
#   SUT_LEAKAGE_DENYLIST     bans a field for being ~equal to the TARGET.
#   SUT_IDENTIFIER_DENYLIST  bans a field for being an IDENTITY — an id that
#                            cannot transfer to another customer or another
#                            day, which is the objection that ruled the
#                            protocol name out as a feature in the first place.
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


def assert_no_leakage(feature_names, denylist=SUT_LEAKAGE_DENYLIST,
                      identifier_denylist=SUT_IDENTIFIER_DENYLIST):
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


# Gate #1 — fires the moment this config module is loaded, for EVERY named set
# rather than only the selected one. A typo in a set nobody has run yet is a
# bug you want to hear about now, not on the day you switch to it.
for _set_name, _set_features in PARAM_SETS.items():
    assert_no_leakage(_set_features)

    _unknown = [n for n in _set_features if n not in SEQPARAM_CANDIDATES]
    if _unknown:
        raise ValueError(
            f"PARAM_SETS[{_set_name!r}] names {_unknown}, which have no entry "
            f"in SEQPARAM_CANDIDATES. Every feature needs an explicit scale "
            f"divisor and missing-value default before it can reach the model "
            f"(see the LayerNorm-erasure warning above)."
        )

# Every candidate must be something the SUT parser actually produces, or step
# 03 would write its default into every row forever and the feature would look
# dead rather than misspelled.
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
