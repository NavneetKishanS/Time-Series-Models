# Findings — the sequence-parameter investigation, 2026-07-31 to 2026-08-07

Five notebooks (`03b`–`03f`, 2,321 lines) were written between 2026-07-31 and
2026-08-05, each to resolve a contradiction the previous one raised. They did
their job. This file is what they concluded; the notebooks themselves are in
`_archive/csv_pipeline_seqparams_diagnostics_v1/` if a number needs re-deriving.

The checks that still have to run on every build were kept and consolidated into
`03b_parameter_report.py`. Everything below is settled and does not need re-running.

---

## The question

Görtler, 2026-07-31: the customer-authored protocol name must not be a feature in
the general model. A protocol is a saved state — one sequence plus ~50 parameter
settings — and customers copy Siemens presets, rename them, and edit them,
destroying the link to the original preset id. The database holds ~3.8M protocol
names across ~15,000 customers against ~3,800 Siemens presets, so a
protocol-keyed model learns one site and transfers to none.

What he wanted instead: **sequence + sequence parameters**, reaching the same
±15s. Our held-out measurement put scanner + protocol at 15.3s MAE, which both
confirmed his estimate and made protocol identity the **benchmark** rather than
an input.

His sharper claim: ±15s is not a real ceiling. If protocols ran fixed the jitter
would be ~1s; it is 15s because operators adjust parameters after loading the
protocol. So executed parameters should *beat* the protocol lookup — provided
the SUT message records executed values rather than stored defaults.

---

## What was settled

### 1. The message format was not what we thought (03b)

The hypothesised labelled `SD<n>: value` format was **falsified** — 0 of 20 real
messages matched. `MRI_SUT_1005` is whitespace-delimited, self-named `KEY:VALUE`
tokens: `TR:866 TE:101 SLC:9 ... MUID:17`. Real messages carry ~60–80 keys, and
the field set differs by sequence family. Three real messages are pinned in
`tests/test_sut_parser.py`.

### 2. The join was wrong, and then it was right (03c)

03b returned two results that contradicted each other. `sequence_binary` (46
values, straight from the Siemens sequence binary) scored R² 3.6% / 88.6s while
the coarser `sequence_type` (12 values) scored 25.4% / 80.2s. Both describe the
same scan; the finer one cannot be seven times worse unless it is describing a
**different** scan.

Cause: the join took the most recent SUT event strictly *before* the segment
start, and SUT fires at ~0.25–0.31 the rate of `MRI_MSR_100`, so one event was
shared by 3–4 consecutive measurements.

**Fixed** (`dcdd381`, then the AdjGreSeq skip). The in-segment rule now agrees
with the actually-scanned sequence on **100.0%** of testable rows, up from 94.8%
and, on the rows where old and new disagree, from 0.5%. There is nothing left to
fix in the rule itself — which is why gate 2 of the report is a regression test
rather than an investigation.

### 3. MUID was acquitted, and denylisted anyway (03d)

`MUID` is on 100% of rows with 3,188 distinct values against 3,051 protocol
names — a ratio of 1.04, which looks exactly like the protocol name in numeric
costume.

It is not. Purity MUID → protocol was **29.8%** (the leak threshold was ~90%),
with 11,409 MUIDs against 3,357 protocols — *finer* than the protocol, not a
coarser Siemens family id. Dropping MUID+VER moved the parameter model by −0.0s
(19.6s either way). As a predictor in its own right it scored R² −19.6% / 72.5s,
**worse** than the serial-alone baseline (5.8% / 66.9s): the signature of a
high-cardinality nuisance id.

Görtler described the mechanism unprompted in the same week and it matches
exactly: MUID is a per-day measurement counter that resets nightly and skips ids
consumed by adjustment scans.

Denylisted regardless, per 03d's own verdict: *a field that CAN leak should not
be one retrain away from leaking.*

### 4. ST and TST are the target (03c section E)

`ST` was read as a *planned* value in the 07-31 call and therefore admissible.
The 08-04 call and 03c section E overrode that: ST is **exact on ~86–87% of
rows**. A duration model handed ST learns identity, not physics. Both joined
`scanning_time` in `SUT_LEAKAGE_DENYLIST`.

### 5. Parameters beat the protocol — where the join fires (03e, 03f)

On clean in-segment rows, the 89-field admissible parameter vector scored **9.7s
MAE against the protocol oracle's 13.2s**. That is Görtler's thesis confirmed:
executed parameters beat the customer protocol name, using nothing
customer-specific, clearing the ±15s bar by 5.3s.

### 6. Coverage, not field count, is the bottleneck (03f)

The in-segment rule fires on **80.3%** of segments. Across *all* 49,514 clean
rows the same vector scores 24.6s against an oracle of 11.0s. So the gap between
"we beat the oracle" and "we lose to it by 13.6s" is entirely **which rows have a
parameter message**, and no additional field closes it.

Of the rows without an in-segment join, ~29% had one that was an *adjustment* —
those segments are probably the adjustment itself and belong in the population
filter rather than counted against the join.

### 7. Stale parameters are worse than none (03f section B)

On the rows the join misses, the fallback carries the most recent preceding SUT
event, and **71.5% of those name a different sequence than the one that ran**.
Those rows are not noisy, they are *wrong*.

| treatment | MAE |
|---|---:|
| parameters as-is (stale included) | 24.6s |
| masked where not in-segment | 9.0s |
| masked + sequence/orientation/serial | 8.6s |

**This result did not transfer to the model as measured.** It came from
`HistGradientBoostingRegressor`, which has a native third state for a missing
value. `build_conditioning_tensor` runs every value through `safe_float`, which
turns NaN into a real number, so the tensor path had no way to express "masked".
That is what the `sut_in_segment` flag added on 2026-08-07 is for.

### 8. Hand-picked sets lost to "throw it all in" (03e, 03f section C)

Two hand-picked sets were built and scored:

- **`luke`** — the acquisition-time formula as its own multiplicands:
  `TA ≈ TR × PEL × AVG × CONC / (PAT × TF)`, plus the slice count.
- **`navneet`** — Görtler, 2026-08-04. Deliberately *not* the formula: he argued
  most timing parameters are physically real but redundant, because echo
  spacing, TE and slice thickness all impose constraints that surface as a floor
  on TR, and TR repeats hundreds of times per scan. So: TR plus only the things
  that change *how many* TRs run — slices, averages, repetitions, matrix size,
  phase/slice partial Fourier.

Both lost to the ranked top-12. That result, more than any argument, is why
`PARAM_SET='all'` became the default on 2026-08-07. Both sets are **kept** as the
control group — they are the only way to later prove more was better rather than
assert it.

---

## What was NOT settled, and became the 2026-08-07 work

1. **Every number above came from a random 80/20 row split.** Sequences from one
   exam landed on both sides and every scanner in test was also in train. That
   inflates a score in the direction that matters least. All of these figures
   should be read as optimistic; `data.holdout` and the grouped splits now in
   `heldout_regressor_score` / `heldout_group_r2` are the fix.

2. **The masking result (finding 7) was not representable in the model.** Fixed
   by per-field presence flags plus the row-level `sut_in_segment` flag.

3. **`SNR` was load-bearing in the only configuration that beat the oracle**, and
   03f flagged it as "derived from the protocol, not set by the operator"
   without acting on it. It is now in `SUT_PLANNER_DERIVED_DENYLIST`, with the
   report measuring what that exclusion costs rather than asserting it.

4. **Four of the ranked top-12 raw keys were unmapped** — used but not
   understood. Under "pass everything" that is acceptable for training and still
   blocks step 07, which must synthesise every field it conditions on. Gate 3 of
   the report lists them every run.

---

## Fields flagged and never resolved

03f carried a `SUSPECT_NOTES` reading aid. It now lives in
`config.SUT_SUSPECT_WATCHLIST` so it survives this archiving, and gate 5 of the
report settles each entry with a number — the random-vs-grouped importance gap —
rather than a judgement.

`BHD` (breath-hold duration) is the strongest remaining denial candidate: on a
breath-hold sequence it is very nearly the measurement duration. It is absent
from all three pinned messages, so nothing about it has been measured on our
corpus, and it is left admissible rather than denied on a guess.
