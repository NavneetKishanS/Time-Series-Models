#!/usr/bin/env python3
"""
Stage-1 validation harness for the examination-model retrain.

Background: the examination model collapsed (class-weight bug) to ~26 rows total
across 10 serials — every row an isolated MSR_100->MSR_104 span with StepCount=1
and predicted_mu pinned ~0.00186. Stage 1 removed the class weighting and added
targeted abort oversampling. This script reads the NEW step-05 examination output
(the "(3)"-schema CSVs: sourceID, Sequence, BodyGroup, StepCount, predicted_mu)
and prints a PASS / FAIL / REVIEW verdict against the Stage-1 success criteria.

Each output row = one completed MRI_MSR_100 -> MRI_MSR_104/34 span, so `sourceID`
is ALWAYS the finish token by construction (not a health signal). The real
signals are: row count, StepCount spread, FinishEvent containing "Stopped by
User", and predicted_mu variance (overall + across Sequence types).

Usage:
    python validate_stage1_examination.py NEW_EXAM*.csv
    python validate_stage1_examination.py "new_run/DATA_*.csv" --baseline "old/DATA_*(3).csv"

Body-group diversity is intentionally NOT a Stage-1 gate — body_region is UNKNOWN
in training until Stage 2, so expect limited groups (NECK/FOOT/HEAD) for now.
"""
import argparse
import glob
import sys
import pandas as pd
import numpy as np

EXAM_MARKERS = {'Sequence', 'BodyGroup', 'StepCount', 'predicted_mu'}
# Known broken-run baseline (the "(3)" files) for reference in the report.
BROKEN = dict(total_rows=26, per_serial_max=5, stepcount_max=1, mu_std=4.6e-4, mu_mean=0.00186)

GREEN, RED, YELLOW, RESET = '\033[92m', '\033[91m', '\033[93m', '\033[0m'

DEFAULT_EXAM_GLOB = '/dbfs/FileStore/csv_pipeline/synthetic/exam/DATA_*.csv'


def _expand(patterns):
    files = []
    for p in patterns:
        hits = glob.glob(p)
        files.extend(hits if hits else ([p] if p else []))
    return sorted(set(files))


def _load(patterns, label):
    files = _expand(patterns)
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  ! skip {f}: {e}")
            continue
        if not EXAM_MARKERS.issubset(df.columns):
            print(f"  ! skip {f}: not an examination-schema CSV "
                  f"(missing {EXAM_MARKERS - set(df.columns)})")
            continue
        df['__file'] = f
        frames.append(df)
    if not frames:
        return None, files
    return pd.concat(frames, ignore_index=True), files


def _tag(ok):
    return f"{GREEN}PASS{RESET}" if ok is True else (
        f"{RED}FAIL{RESET}" if ok is False else f"{YELLOW}REVIEW{RESET}")


def validate(df):
    results = []
    n_serials = df['SN'].nunique() if 'SN' in df.columns else 1
    per_serial = (df.groupby('SN').size() if 'SN' in df.columns
                  else pd.Series([len(df)]))

    # 1. Row count — the headline collapse signal.
    total = len(df)
    ps_min, ps_med = int(per_serial.min()), int(per_serial.median())
    ok = (total > 500) and (ps_min > 10)
    results.append((
        "Row count (collapse undone)", ok,
        f"total={total} across {n_serials} serials | per-serial min={ps_min} "
        f"median={ps_med}  [broken baseline: {BROKEN['total_rows']} total, max {BROKEN['per_serial_max']}/serial]"))

    # 2. StepCount spread — multi-scan patients, not all StepCount=1.
    if 'StepCount' in df.columns:
        sc = pd.to_numeric(df['StepCount'], errors='coerce').dropna()
        frac_gt1 = (sc > 1).mean() if len(sc) else 0.0
        ok = sc.max() > 1 and frac_gt1 > 0.05
        results.append((
            "StepCount spread", ok,
            f"max={int(sc.max())} mean={sc.mean():.2f} | {frac_gt1*100:.1f}% of rows have StepCount>1  "
            f"[broken: all =1]"))

    # 3. Abort surfacing — 'Stopped by User' should now appear.
    if 'FinishEvent' in df.columns:
        fe = df['FinishEvent'].astype(str).str.strip()
        n_abort = (fe == 'Stopped by User').sum()
        rate = n_abort / len(df) * 100
        ok = n_abort > 0
        results.append((
            "Abort 'Stopped by User' appears", ok,
            f"{n_abort} abort rows ({rate:.2f}%) | FinishEvent mix: "
            f"{fe.value_counts().to_dict()}  [broken: 0 aborts]"))

    # 4. Duration head un-collapsed — predicted_mu varies (not pinned ~0.00186).
    mu = pd.to_numeric(df['predicted_mu'], errors='coerce').dropna()
    mu_std, mu_mean = mu.std(), mu.mean()
    ok = mu_std > 5 * BROKEN['mu_std'] and not (0.0015 < mu_mean < 0.0025 and mu_std < 1e-3)
    results.append((
        "predicted_mu variability", ok,
        f"mean={mu_mean:.5f} std={mu_std:.5f} range=[{mu.min():.5f},{mu.max():.5f}]  "
        f"[broken: mean~{BROKEN['mu_mean']}, std~{BROKEN['mu_std']:.1e}]"))

    # 5. Duration varies across scan types (scout != tse != ...).
    if 'Sequence' in df.columns and 'duration' in df.columns:
        dur = pd.to_numeric(df['duration'], errors='coerce')
        by_seq = df.assign(_d=dur).groupby('Sequence')['_d'].agg(['count', 'mean', 'median'])
        by_seq = by_seq[by_seq['count'] >= 5].sort_values('mean')
        spread = (by_seq['mean'].max() / by_seq['mean'].min()) if len(by_seq) >= 2 and by_seq['mean'].min() > 0 else 1.0
        ok = None if len(by_seq) < 2 else spread > 1.3
        tbl = " | ".join(f"{s}:{r['mean']:.0f}s(n{int(r['count'])})" for s, r in by_seq.iterrows())
        results.append((
            "Duration differs by scan type", ok,
            f"mean-duration spread {spread:.2f}x across types -> {tbl}"))

    # 6. Body-group distribution — INFORMATIONAL only (Stage 2 territory).
    if 'BodyGroup' in df.columns:
        bg = df['BodyGroup'].astype(str).str.strip().value_counts().to_dict()
        results.append((
            "Body groups present (INFO — Stage 2)", None,
            f"{bg}  [expected limited until step-03 body_region fix]"))

    # Normalise verdicts: numpy booleans fail Python `is True/False` identity
    # checks, which would silently downgrade real FAILs to REVIEW and skip them
    # in the gate count. Coerce to plain bool (or None for informational rows).
    return [(name, (None if ok is None else bool(ok)), detail) for name, ok, detail in results]


def _is_databricks_interactive():
    """Detect if running inside Databricks interactive kernel (not CLI)."""
    return any('ipykernel' in arg or 'db_ipykernel' in arg for arg in sys.argv)


def main():
    interactive = _is_databricks_interactive()

    # In Databricks interactive mode, sys.argv contains kernel launcher args
    # that argparse cannot parse. Bypass argument parsing and use defaults.
    if interactive:
        input_patterns = [DEFAULT_EXAM_GLOB]
    else:
        ap = argparse.ArgumentParser(description="Validate Stage-1 examination retrain output.")
        ap.add_argument('inputs', nargs='+', help="New step-05 examination CSV file(s) or glob(s).")
        ap.add_argument('--baseline', nargs='*', default=[], help="Optional old/broken (3) CSV(s) for comparison.")
        args = ap.parse_args()
        input_patterns = args.inputs

    print("=" * 78)
    print("STAGE-1 EXAMINATION RETRAIN VALIDATION")
    print("=" * 78)

    df, files = _load(input_patterns, 'new')
    if df is None:
        print(f"{RED}No examination-schema CSVs found in: {input_patterns}{RESET}")
        if not interactive:
            sys.exit(2)
        return
    print(f"Loaded {len(files)} file(s), {len(df)} rows.\n")

    results = validate(df)
    gates = [ok for _, ok, _ in results if ok is not None]
    for name, ok, detail in results:
        print(f"[{_tag(ok)}] {name}")
        print(f"         {detail}\n")

    n_fail = sum(1 for ok in gates if ok is False)
    n_pass = sum(1 for ok in gates if ok is True)
    print("=" * 78)
    if n_fail == 0:
        print(f"{GREEN}VERDICT: collapse fixed — {n_pass}/{len(gates)} gates passed.{RESET}")
    else:
        print(f"{RED}VERDICT: {n_fail} gate(s) FAILED — collapse not resolved. Inspect above.{RESET}")
    print("=" * 78)
    if not interactive:
        sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
