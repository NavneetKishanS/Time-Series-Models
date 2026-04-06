# Databricks notebook source
# Databricks notebook — Download training CSVs
#
# Provides download links for all exchange and exam CSVs produced by steps 01
# and 02.  Run this after 01_exchange_preprocessing.py and
# 02_exam_preprocessing.py to pull the raw training data down locally.
# This step is for data access only — it is not required to run 03 or 04.
#
# Files served:
#   /dbfs/FileStore/csv_pipeline/exchange/DATA_{serial}.csv  (one per scanner)
#   /dbfs/FileStore/csv_pipeline/exam/DATA_{serial}.csv      (one per scanner)

# COMMAND ----------
%run ./config

# COMMAND ----------

import os
import pandas as pd

def _dbfs_to_url(dbfs_path):
    """Convert a /dbfs/FileStore/... path to the Databricks /files/... download URL."""
    return dbfs_path.replace("/dbfs/FileStore", "/files")

def _file_info(path):
    """Return (exists, size_kb, row_count) for a CSV on DBFS."""
    if not os.path.exists(path):
        return False, 0, 0
    size_kb = os.path.getsize(path) / 1024
    try:
        row_count = sum(1 for _ in open(path)) - 1  # subtract header
    except Exception:
        row_count = -1
    return True, size_kb, row_count

# COMMAND ----------
# =============================================================================
# Collect file info
# =============================================================================

print(f"Serials:    {SERIAL_NUMBERS}")
print(f"Date range: {DATE_START} → {DATE_END}")
print()

rows = []

for serial in SERIAL_NUMBERS:
    for label, base_dir in [("exchange", EXCHANGE_OUTPUT_DIR), ("exam", EXAM_OUTPUT_DIR)]:
        path = f"{base_dir}/DATA_{serial}.csv"
        exists, size_kb, row_count = _file_info(path)
        rows.append({
            "serial":    serial,
            "type":      label,
            "path":      path,
            "url":       _dbfs_to_url(path),
            "exists":    exists,
            "size_kb":   round(size_kb, 1),
            "row_count": row_count,
        })

df_files = pd.DataFrame(rows)

# Print text summary
for _, r in df_files.iterrows():
    status = f"{r['size_kb']:>8.1f} KB  {r['row_count']:>6} rows" if r["exists"] else "  NOT FOUND"
    print(f"  [{r['type']:8s}]  {r['serial']}  {status}")

# COMMAND ----------
# =============================================================================
# Download links
# =============================================================================

_found    = df_files[df_files["exists"]]
_missing  = df_files[~df_files["exists"]]

if not _missing.empty:
    print(f"\nWarning: {len(_missing)} file(s) not found — run steps 01 and 02 first.")

# Build HTML table with one download link per file
_rows_html = ""
for _, r in _found.iterrows():
    _rows_html += (
        f"<tr>"
        f"<td>{r['serial']}</td>"
        f"<td>{r['type']}</td>"
        f"<td>{r['size_kb']:,.1f} KB</td>"
        f"<td>{r['row_count']:,}</td>"
        f"<td><a href='{r['url']}' download>Download</a></td>"
        f"</tr>\n"
    )

displayHTML(f"""
<h3>Training CSV Downloads</h3>
<p>Date range: <b>{DATE_START}</b> → <b>{DATE_END}</b> &nbsp;|&nbsp; {len(_found)} file(s) ready</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; font-family:monospace;">
  <thead style="background:#f0f0f0;">
    <tr>
      <th>Serial</th><th>Type</th><th>Size</th><th>Rows</th><th>Link</th>
    </tr>
  </thead>
  <tbody>
{_rows_html}
  </tbody>
</table>
<p style="font-size:0.85em; color:#666;">
  Files are saved under <code>/dbfs/FileStore/csv_pipeline/</code>.
  These CSVs are the raw training data — not needed by the model directly (that uses the .pkl from step 03).
</p>
""")
