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
(mapped from the "SLC" key) are now CONFIRMED via SUT_FIELD_MAP below and
active in EXAMINATION_SEQPARAM_FEATURES. trigger_mode remains a PLACEHOLDER —
no field in the sampled messages showed variation consistent with a
gating/trigger mode (candidates CMM/CDM/RCM/PHYS/PAC were constant across all
samples), so TRIGGER_MODE_VOCAB is unused for now and every row defaults to
'none' (safe, non-leaking default — see _trigger_mode_id below).
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
# ST is CONFIRMED (2026-07-31) to be decided BEFORE the measurement — a planned
# value, not an observed outcome, so it is not the SD58 leak and is admissible
# as a model feature. Note SLC x TR reproduces ST for haste (9 x 866ms -> ST:8;
# 15 x 1000ms -> ST:15) but NOT for ep2d_diff (18 x 4300ms = 77s vs ST:400), so
# it is a genuine per-family computation rather than a redundant product of two
# fields already held. Names stay as the raw message mnemonics.
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
    # WARNING — sequence-scoped, NOT universal: TF is present on the haste
    # (TSE-family) messages and ABSENT on ep2d_diff, which uses an echo factor
    # (EF) instead. _safe_float(...) defaults a missing key to 0.0, and a turbo
    # factor of 0 is not a real value — it means "does not apply to this
    # sequence". Safe while it stays out of EXAMINATION_SEQPARAM_FEATURES;
    # promoting it needs an explicit presence flag or scoping by sequence
    # family. Same applies to any DIFF/BV0/BVM/EF field added later.
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

# Numeric SUT parameter feature names that ride the flat conditioning tensor
# (see AlternatingPipeline/training/utils.py::build_conditioning_tensor
# extra_feature_names).
EXAMINATION_SEQPARAM_FEATURES = ['TR', 'num_slices']

# Per-feature O(1) scale divisors, aligned 1:1 with
# EXAMINATION_SEQPARAM_FEATURES. CRITICAL: an entry here is REQUIRED before
# adding the matching feature name above — see the conditioning_scale /
# LayerNorm-erasure warning in AlternatingPipeline/models/sequence_generator.py
# (raw large-magnitude numeric features silently erase categorical
# conditioning if left unscaled — this exact bug caused three separate
# multi-week flat-duration incidents in this project). Real observed ranges
# from discovery (TR 866-4300ms, SLC/num_slices 9-18) confirm these divisors
# land features in a sane order of magnitude: TR / 1000; num_slices / 30.
EXAMINATION_SEQPARAM_SCALE = [1000.0, 30.0]  # MUST match
                                              # EXAMINATION_SEQPARAM_FEATURES length

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
