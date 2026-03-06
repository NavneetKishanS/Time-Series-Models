"""
Archive Duration Priors

Loads pre-aggregated examination records from the archive CSVs (one row = one
complete MRI scan, with a pre-computed `duration` column grouped by BodyGroup).
Provides per-body-group duration statistics to anchor the Examination model's
duration head during training.
"""
import os
import pandas as pd

ARCHIVE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '_archive',
    'SeqofSeq_Pipeline_v1', 'data'
)
ARCHIVE_FILES = ['175832.csv', '176625.csv']


def load_examination_priors() -> dict:
    """
    Returns {body_group_str: {'mean': float, 'std': float, 'count': int}}
    by reading archive examination CSVs and grouping on BodyGroup + duration.
    Falls back to empty dict if files are missing.
    """
    dfs = []
    for fname in ARCHIVE_FILES:
        path = os.path.join(ARCHIVE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'BodyGroup' not in df.columns or 'duration' not in df.columns:
                continue
            df = df[['BodyGroup', 'duration']].dropna()
            df = df[df['duration'] > 0]
            dfs.append(df)
    if not dfs:
        return {}
    combined = pd.concat(dfs, ignore_index=True)
    stats = combined.groupby('BodyGroup')['duration'].agg(['mean', 'std', 'count'])
    stats['std'] = stats['std'].fillna(60.0).clip(lower=30.0)
    return stats.to_dict('index')


if __name__ == '__main__':
    priors = load_examination_priors()
    if priors:
        print(f"Loaded priors for {len(priors)} body groups:")
        for group, s in sorted(priors.items()):
            print(f"  {group}: mean={s['mean']:.1f}s, std={s['std']:.1f}s, n={s['count']}")
    else:
        print("No archive files found — returning empty priors.")
