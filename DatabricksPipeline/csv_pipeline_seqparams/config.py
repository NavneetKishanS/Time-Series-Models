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
SUT_SLOT_MAP, TRIGGER_MODE_VOCAB, EXAMINATION_SEQPARAM_FEATURES and
EXAMINATION_SEQPARAM_SCALE below are PLACEHOLDERS pending the data-discovery
stage. Run DatabricksPipeline/sut_parameter_discovery.py first and replace
the placeholder values with confirmed slot numbers / ranges before trusting
any training run built from EXAMINATION_MODEL_CONFIG_SEQPARAMS (assembled in
04_train_models.py). Until then EXAMINATION_SEQPARAM_FEATURES is empty,
which is a safe default (equivalent to the old 10-dim conditioning), not a
silent no-op bug.
"""

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

PKL_OUTPUT   = "/dbfs/FileStore/csv_pipeline_seqparams/preprocessed_data.pkl"
MODELS_DIR   = "/dbfs/FileStore/csv_pipeline_seqparams/models"
ANALYSIS_DIR = "/dbfs/FileStore/csv_pipeline_seqparams/analysis"

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

COIL_COLUMNS = [
    'BC', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8', '15K',
    'HW1', 'HW2', 'HW3', 'HE1', 'HE2', 'HE3', 'HE4', 'NE1', 'NE2', 'SHL',
    'BO1', 'BO2', 'BO3', 'FA', 'TO', 'FS',
    'PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'SN',
]

# ============================================================================
# SUT SEQUENCE-PARAMETER VOCAB & FEATURE LIST  ***  PLACEHOLDERS  ***
#
# Filled in by DatabricksPipeline/sut_parameter_discovery.py (data-discovery
# stage). Only SD58 = scanning_time is CONFIRMED (Navneet named it directly
# on the call, unprompted). Everything else below is a structural
# placeholder, not a real finding.
# ============================================================================

# slot number (within the MRI_SUT_1005 message) -> stable parameter name.
SUT_SLOT_MAP = {
    58: 'scanning_time',   # CONFIRMED — Navneet, call transcript. LEAKAGE — see denylist below.
    # 'TR':           <slot>,  # TODO: confirm via sut_parameter_discovery.py
    # 'num_slices':   <slot>,  # TODO: confirm via sut_parameter_discovery.py
    # 'trigger_mode': <slot>,  # TODO: confirm via sut_parameter_discovery.py
}

TRIGGER_MODE_VOCAB = {
    'none': 0, 'ecg': 1, 'peripheral_pulse': 2, 'respiratory': 3,
    'external': 4, 'unknown': 5,
}  # placeholder categories — finalize against Phase 1 findings
NUM_TRIGGER_MODES = len(TRIGGER_MODE_VOCAB)

# Numeric SD-parameter feature names that will ride the flat conditioning
# tensor (see AlternatingPipeline/training/utils.py::build_conditioning_tensor
# extra_feature_names). Empty until Phase 1 confirms real slot numbers —
# training against an empty list is equivalent to the OLD 10-dim
# conditioning (a safe default), not a silent no-op bug.
EXAMINATION_SEQPARAM_FEATURES = []  # e.g. ['TR', 'num_slices'] once confirmed

# Per-feature O(1) scale divisors, aligned 1:1 with
# EXAMINATION_SEQPARAM_FEATURES. CRITICAL: an entry here is REQUIRED before
# adding the matching feature name above — see the conditioning_scale /
# LayerNorm-erasure warning in AlternatingPipeline/models/sequence_generator.py
# (raw large-magnitude numeric features silently erase categorical
# conditioning if left unscaled — this exact bug caused three separate
# multi-week flat-duration incidents in this project). Placeholder guesses
# once features are added: TR ~300-3000ms -> divide by 1000;
# num_slices ~1-60 -> divide by 30.
EXAMINATION_SEQPARAM_SCALE = []  # e.g. [1000.0, 30.0] once confirmed — MUST
                                  # match EXAMINATION_SEQPARAM_FEATURES length

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
# Görtler's transcript: "scanning time" (SD58) is ~equal to the duration
# target and must never be an input feature.
# ============================================================================

SUT_LEAKAGE_DENYLIST = {'scanning_time'}


def assert_no_leakage(feature_names, denylist=SUT_LEAKAGE_DENYLIST):
    """Raise if any denylisted (duration-equivalent) name is present.

    feature_names: any iterable of feature-name strings (a static config
        list, or the actual runtime keys of a conditioning dict).
    """
    overlap = denylist.intersection(feature_names)
    if overlap:
        raise ValueError(
            f"Leakage guard tripped: {overlap} must never be used as an "
            f"input feature — it is ~equal to the duration target."
        )


# Gate #1 — fires the moment this config module is loaded.
assert_no_leakage(EXAMINATION_SEQPARAM_FEATURES)
assert len(EXAMINATION_SEQPARAM_FEATURES) == len(EXAMINATION_SEQPARAM_SCALE), (
    "EXAMINATION_SEQPARAM_FEATURES and EXAMINATION_SEQPARAM_SCALE must be "
    "the same length — every new numeric feature needs an explicit scale "
    "divisor (see the LayerNorm-erasure warning above)."
)

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
