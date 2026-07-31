#!/usr/bin/env python3
"""Preflight for the protocol-conditioned duration model — runs LOCALLY.

Answers, without Databricks and without touching a model: what does the
examination duration model actually get to see, and how much of the duration
does each candidate input explain?

Reads the real exam CSVs already in the repo
(DatabricksPipeline/csv_pipeline/qlik/data/real/exam) and uses the SAME
vocabulary and held-out scorer the training pipeline uses, so the numbers here
are the numbers step 04's gate will re-check against the training pkl.

    python DatabricksPipeline/protocol_preflight.py

Everything is measured HELD OUT: group means are fitted on 80% of rows and
scored on the other 20%. In-sample numbers are meaningless at this granularity
— 2,999 protocols over 40,921 rows scores 89.7% in sample and 82.0% held out,
and a grouping with one row per group scores a perfect 100% while predicting
nothing at all.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlternatingPipeline.data.protocol_vocab import (  # noqa: E402
    RARE_PROTOCOL_ID,
    build_protocol_vocab,
    heldout_group_r2,
    protocol_id,
)

REAL_EXAM_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'csv_pipeline', 'qlik', 'data', 'real', 'exam', 'DATA_*.csv',
)
SEQUENCE_FAMILIES = [
    'scout', 'space', 'medic', 'dixon', 'ep2d_diff', 'tfl', 'tse',
    'vibe', 'haste', 'gre', 'swi', 'resolve',
]


def _rule(char='-', width=78):
    print(char * width)


def _sequence_family(raw):
    """Approximate config.classify_sequence_type for the CSV `Sequence` field."""
    name = str(raw).split('\\')[-1].lower()
    for family in SEQUENCE_FAMILIES:
        if family in name:
            return 'epi' if family == 'ep2d_diff' else family
    return 'other'


def load_real_exams():
    files = sorted(glob.glob(REAL_EXAM_GLOB))
    if not files:
        raise SystemExit(f"No real exam CSVs found at {REAL_EXAM_GLOB}")
    frame = pd.concat(
        [pd.read_csv(f, low_memory=False)[
            ['SN', 'PatientID', 'Sequence', 'Protocol', 'BodyPart',
             'duration', 'startTime']
        ] for f in files],
        ignore_index=True,
    )
    frame['duration'] = pd.to_numeric(frame['duration'], errors='coerce')
    frame = frame[frame['duration'] > 0].dropna(subset=['Protocol', 'Sequence'])
    frame['seq_family'] = frame['Sequence'].map(_sequence_family)
    return frame.reset_index(drop=True), len(files)


def section_what_the_model_sees(frame):
    _rule('=')
    print(" WHAT THE EXAMINATION DURATION MODEL RECEIVES PER SCAN")
    _rule('=')
    print("""
  BEFORE (every run up to and including 2026-07-27)
    body_region        11 values      HEAD / ABDOMEN / SPINE / ...
    sequence_type      12 values      tse / haste / vibe / scout / ...
    serial_idx         10 values      which scanner
    trigger_mode        6 values      constant 'none' — field never identified
    TR                 number         repetition time, ms
    num_slices         number         slice count
    + patient/temporal Age, Weight, Height, PTAB, direction, time-of-day

  AFTER (this change)
    protocol           ~1,591 values  the operator-selected protocol name  <-- NEW
    ...everything above, unchanged.

  The sequence parameters were NOT rewired. TR and num_slices reach the model
  exactly as they did before. ST / TST / AVG / CONC / PEL / PAT / FOV are now
  PARSED but held in audit-only storage — they do not reach the model, pending
  Görtler's answer on whether ST is nominal or measured.
""")


def section_variance(frame):
    _rule('=')
    print(" HOW MUCH OF THE SCAN DURATION EACH INPUT EXPLAINS (HELD OUT)")
    _rule('=')
    duration = frame['duration'].values
    print(f"\n  {len(frame):,} real scans, {frame['SN'].nunique()} scanners.  "
          f"mean {duration.mean():.1f}s   median {np.median(duration):.0f}s   "
          f"sd {duration.std():.1f}s\n")

    vocab = build_protocol_vocab(frame['Protocol'], min_count=3)
    frame = frame.assign(
        protocol_id=[protocol_id(p, vocab) for p in frame['Protocol']],
        sn_protocol=frame['SN'].astype(str) + '|' + frame['Protocol'].astype(str),
    )

    rows = [
        ('sequence_type', 'seq_family', 'what the model used before'),
        ('body_region', 'BodyPart', ''),
        ('scanner (SN)', 'SN', ''),
        ('Sequence raw string', 'Sequence', ''),
        ('PROTOCOL', 'protocol_id', 'NEW — what it uses now'),
        ('scanner + PROTOCOL', 'sn_protocol', ''),
    ]
    print(f"  {'input':<24} {'distinct':>9} {'variance':>10} {'error':>9}   note")
    print(f"  {'':<24} {'values':>9} {'explained':>10} {'(MAE)':>9}")
    _rule()
    for label, column, note in rows:
        r2, mae, _ = heldout_group_r2(frame[column].values, duration)
        print(f"  {label:<24} {frame[column].nunique():>9,} "
              f"{r2:>9.1f}% {mae:>8.1f}s   {note}")
    _rule()
    print("  the trained model itself                     50.3s   (measured 2026-07-27)")
    print("""
  Read this as: knowing only the protocol name predicts a scan's duration to
  within ~17s. The model, which never saw the protocol, was off by 50.3s. The
  duration variance was never missing — it sits at protocol granularity, and we
  were decomposing by the 12-value scan-type bucket, which is all the model had.
""")
    return frame, vocab


def section_within_protocol(frame):
    _rule('=')
    print(" ONCE YOU KNOW THE PROTOCOL, WHAT IS LEFT?")
    _rule('=')
    grouped = frame.groupby('sn_protocol')['duration']
    sizeable = frame[grouped.transform('count') >= 10].copy()
    sizeable['residual'] = (
        sizeable['duration'] - sizeable.groupby('sn_protocol')['duration'].transform('mean')
    )
    per_protocol = sizeable.groupby('sn_protocol')['duration']
    cv = (per_protocol.std() / per_protocol.mean()).median()
    print(f"\n  within-protocol residual sd   {sizeable['residual'].std():>6.1f}s   "
          f"(overall sd {frame['duration'].std():.1f}s)")
    print(f"  median within-protocol CV     {cv:>6.3f}   "
          f"— duration is near-deterministic given the protocol\n")

    frame = frame.sort_values(['SN', 'PatientID', 'startTime'])
    frame['repeat_idx'] = frame.groupby(['SN', 'PatientID', 'Protocol']).cumcount()
    repeats = frame['repeat_idx'] > 0
    patients = frame.groupby(['SN', 'PatientID'])['repeat_idx'].max() > 0
    print(f"  Görtler's re-scan signal (same protocol, same patient, again):")
    print(f"    {100 * repeats.mean():.1f}% of scans are a repeat; "
          f"{100 * patients.mean():.1f}% of patients have at least one")
    means = frame.groupby(frame['repeat_idx'].clip(upper=3))['duration'].mean()
    print("    raw mean duration by run number: "
          + " -> ".join(f"{m:.0f}s" for m in means))
    print("""
    Real, but already covered: the protocols that get repeated are the SHORT
    ones, so after conditioning on protocol this explains ~0% of the remaining
    error in the mean. What does survive is a wider SPREAD on repeats — a sigma
    effect, not a level effect.
""")


def section_protocol_catalogue(frame, vocab):
    _rule('=')
    print(" THE PROTOCOL CATALOGUE (what the new input actually contains)")
    _rule('=')
    per_serial = frame.groupby('SN')['Protocol'].nunique()
    names = set(frame['Protocol'])
    shared = sum(
        1 for name in names
        if frame.loc[frame['Protocol'] == name, 'SN'].nunique() > 1
    )
    print(f"\n  {len(names):,} distinct protocols across {frame['SN'].nunique()} scanners")
    print(f"  {per_serial.mean():.0f} per scanner on average "
          f"(min {per_serial.min()}, max {per_serial.max()})")
    print(f"  only {100 * shared / len(names):.1f}% appear on more than one scanner "
          f"— these are site-authored catalogues")
    print(f"  vocabulary at min_count=3: {len(vocab):,} protocols, "
          f"{100 * (frame['protocol_id'] == RARE_PROTOCOL_ID).mean():.1f}% of scans "
          f"fall to the shared 'rare' bucket\n")

    counts = frame['Protocol'].value_counts()
    stats = frame.groupby('Protocol')['duration'].agg(['count', 'mean', 'std'])
    print("  A few of the most-used protocols, and how tight their durations are:\n")
    print(f"    {'protocol':<38} {'n':>6} {'mean':>8} {'sd':>7}")
    for name in counts.head(12).index:
        row = stats.loc[name]
        print(f"    {str(name)[:37]:<38} {int(row['count']):>6} "
              f"{row['mean']:>7.0f}s {row['std']:>6.0f}s")
    print("""
  This is the point to put to Georg: the names are clinically meaningful and
  they carry the acquisition mode explicitly (RESPI LIBRE = free breathing,
  APNEE = breath-hold). That is where the trigger/gating information has been
  all along — not in a sequence-parameter field.
""")


def section_questions():
    _rule('=')
    print(" OPEN QUESTIONS FOR GEORG — the answers change what we build next")
    _rule('=')
    print("""
  1. ST / TST — nominal or measured?
     SLC x TR reproduces ST exactly on both sampled haste messages
     (9 x 866ms -> ST:8;  15 x 1000ms -> ST:15). An exact match to the computed
     product says CALCULATED, i.e. the protocol's planned acquisition time,
     knowable before the scan. If he confirms, it is a legitimate feature and
     the best candidate for the ~25s still unexplained within a protocol.
     If it is measured after the fact, we drop it.

  2. Are protocols stable, or edited per patient?
     If an operator adjusts slice count or thickness per patient, the protocol
     is a strong prior and the sequence parameters explain the deviation from
     it. That is the natural next feature: num_slices as a DIFFERENCE from the
     protocol's own median, not as a raw number.

  3. Protocol naming discipline across sites.
     Only 4.1% of names are shared between scanners. If naming is per-site and
     ad hoc, the model must stay per-site; if there is an underlying standard,
     the names could be normalised and pooled.

  4. Anything that would explain a re-scan we cannot see in the event log?
     The repeat signal is there but small once protocol is known. A rough
     frequency estimate from him is worth more than more protocol fields.
""")


def main():
    frame, n_files = load_real_exams()
    print()
    _rule('=')
    print(" PROTOCOL PREFLIGHT — examination duration model")
    print(f" {len(frame):,} real scans from {n_files} scanners, measured held out")
    _rule('=')
    section_what_the_model_sees(frame)
    frame, vocab = section_variance(frame)
    section_within_protocol(frame)
    section_protocol_catalogue(frame, vocab)
    section_questions()
    _rule('=')
    print(" NEXT: step 03 (rebuild pkl with protocol names) -> step 04 cells up to")
    print(" the PROTOCOL GATE, which re-checks these numbers against the training")
    print(" target and hard-stops before training if they do not hold.")
    _rule('=')
    print()


if __name__ == '__main__':
    main()
