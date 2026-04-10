# Fetching CSVs from Databricks into `data/`

The Qlik load script reads from `qlik/data/{real,synthetic}/{exchange,exam}/DATA_*.csv`.
This document covers three ways to populate that folder.

**Source paths in DBFS:**

| Source | DBFS path | Produced by |
|---|---|---|
| Real exchange | `/dbfs/FileStore/csv_pipeline/exchange/DATA_*.csv` | `01_exchange_preprocessing.py` |
| Real exam | `/dbfs/FileStore/csv_pipeline/exam/DATA_*.csv` | `02_exam_preprocessing.py` |
| Synthetic exchange | `/dbfs/FileStore/csv_pipeline/synthetic/exchange/DATA_*.csv` | `05_generate_synthetic_data.py` |
| Synthetic exam | `/dbfs/FileStore/csv_pipeline/synthetic/exam/DATA_*.csv` | `05_generate_synthetic_data.py` |

**Target layout (this folder):**

```
qlik/data/
├── real/
│   ├── exchange/DATA_*.csv
│   └── exam/DATA_*.csv
└── synthetic/
    ├── exchange/DATA_*.csv
    └── exam/DATA_*.csv
```

---

## Method 1 — Databricks CLI (recommended)

The fastest, most repeatable option. Requires the Databricks CLI
installed and authenticated once against your workspace.

### One-time setup

```bash
pip install databricks-cli
databricks configure --token
# paste your workspace URL and a personal access token when prompted
```

### Pull the files

Run from the repo root:

```bash
# Real data (from steps 01 and 02)
databricks fs cp -r \
    dbfs:/FileStore/csv_pipeline/exchange/ \
    DatabricksPipeline/csv_pipeline/qlik/data/real/exchange/ \
    --overwrite

databricks fs cp -r \
    dbfs:/FileStore/csv_pipeline/exam/ \
    DatabricksPipeline/csv_pipeline/qlik/data/real/exam/ \
    --overwrite

# Synthetic data (from step 05)
databricks fs cp -r \
    dbfs:/FileStore/csv_pipeline/synthetic/exchange/ \
    DatabricksPipeline/csv_pipeline/qlik/data/synthetic/exchange/ \
    --overwrite

databricks fs cp -r \
    dbfs:/FileStore/csv_pipeline/synthetic/exam/ \
    DatabricksPipeline/csv_pipeline/qlik/data/synthetic/exam/ \
    --overwrite
```

### Verify

```bash
ls DatabricksPipeline/csv_pipeline/qlik/data/real/exchange/
ls DatabricksPipeline/csv_pipeline/qlik/data/real/exam/
ls DatabricksPipeline/csv_pipeline/qlik/data/synthetic/exchange/
ls DatabricksPipeline/csv_pipeline/qlik/data/synthetic/exam/
```

Each folder should contain `DATA_{serial}.csv` for every scanner in
`csv_pipeline/config.py` `SERIAL_NUMBERS`.

---

## Method 2 — Step 03b browser download (no CLI needed)

If you don't have the Databricks CLI set up, use the existing download
notebook.

1. In Databricks, open and run
   `DatabricksPipeline/csv_pipeline/03b_download_training_csvs.py`. It
   prints one download link per file — click each to save locally.
2. The browser will save them as `DATA_{serial}.csv` for the first file
   and `DATA_{serial}(1).csv` if two files share a name (because exam
   and exchange both use `DATA_{serial}.csv`).
3. Unpack into the target layout:
   - Files without `(1)` → `qlik/data/real/exchange/`
   - Files with `(1)` → `qlik/data/real/exam/` (rename to strip the `(1)`)

**For synthetic data**, step 05 prints a similar block of download links
at the bottom of the notebook. Repeat the process:
   - Synthetic exchange files → `qlik/data/synthetic/exchange/`
   - Synthetic exam files → `qlik/data/synthetic/exam/`

This is slower and more error-prone than Method 1, but requires
nothing installed locally beyond a browser.

---

## Method 3 — Databricks Workspace REST API (scriptable, no CLI)

For automation without installing the Databricks CLI.

### Environment variables

```bash
export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<personal-access-token>"
```

### Pull one file

```bash
curl -s -o DATA_183242.csv \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    "$DATABRICKS_HOST/api/2.0/dbfs/read?path=/FileStore/csv_pipeline/exchange/DATA_183242.csv"
```

DBFS files above 1 MB must be read in chunks. For larger files use the
DBFS `list` + `read` pattern with offset handling — or just use Method
1, which handles this transparently.

---

## After the files are in place

1. Run the Qlik load script (`load_script.qvs`) in the Qlik Data Load
   Editor.
2. Check the Data Model Viewer — you should see `FACT_Exchange`,
   `FACT_Exam`, `DIM_Scanner`, `DIM_Date`, `DIM_DataSource`.
3. Build Sheet 1 from `dashboard_spec.md`. The sanity-check text box
   should show row counts approximately equal to:

   ```
   Exchange: ~27k real + ~97k synthetic
   Exam:     ~7k real  + ~3k  synthetic
   ```

   (Exact numbers depend on how many scanners and days are in scope.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Empty `FACT_Exchange` after reload | CSVs not in the expected folder | Re-check paths in `load_script.qvs` and the `data/` tree |
| Only real rows, no synthetic | Step 05 failed or synthetic dir empty | Rerun step 05, recheck DBFS `synthetic/` folder |
| `Direction` column missing on synthetic | Old step 05 (before commit `08663b9`) | Rerun step 05 with the fix pulled |
| Qlik complains "Field not found: Age" on synthetic exchange | Same — old step 05 output | Rerun step 05 |
| Row counts hugely mismatched between real and synthetic | Different date windows — real is 1 month, synthetic is intentionally a different 1 month (outside training) | Expected; compare proportions, not absolute counts |
