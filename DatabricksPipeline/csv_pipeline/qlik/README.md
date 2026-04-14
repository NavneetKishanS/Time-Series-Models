# Qlik Validation Dashboard — How-To

**Goal:** Compare the synthetic MRI pipeline output against real training
data in a Qlik Sense dashboard, side-by-side, by manually uploading two
consolidated CSV files.

Everything here is **local**. No Databricks pipeline changes, no Qlik
load scripts to maintain — just a small Python helper that merges your
per-scanner CSVs into two files, plus a few drag-and-drops in Qlik.

---

## What you'll do (5 steps, ~30 minutes)

1. **Fetch** the per-scanner CSVs from Databricks into `data/`
2. **Consolidate** them with `python consolidate.py`
3. **Upload** the two combined files to Qlik Sense
4. **Build** six comparison charts on one sheet
5. **Read** the numbers to see if synthetic matches real

> **Tip:** you can do steps 1–4 using only real data first to test the
> workflow. Add synthetic later (after step 05 reruns) and re-run the
> consolidation to refresh.

---

## Prerequisites

- **Python 3** with pandas installed (`pip install pandas`)
- **Qlik Sense Desktop** or **Qlik Sense SaaS** account — both work identically
- CSVs from steps 01, 02, and 05 of the Databricks pipeline (see step 1 below)

---

## Step 1 — Fetch the CSVs

You need four sets of files:

| Type | DBFS source | Target folder |
|---|---|---|
| Real exchange | `/dbfs/FileStore/csv_pipeline/exchange/` | `data/real/exchange/` |
| Real exam | `/dbfs/FileStore/csv_pipeline/exam/` | `data/real/exam/` |
| Synthetic exchange | `/dbfs/FileStore/csv_pipeline/synthetic/exchange/` | `data/synthetic/exchange/` |
| Synthetic exam | `/dbfs/FileStore/csv_pipeline/synthetic/exam/` | `data/synthetic/exam/` |

See [`fetch_from_dbfs.md`](fetch_from_dbfs.md) for three ways to do this
(Databricks CLI is the fastest; browser download via step 03b is the
no-install option).

**After this step**, the `data/` tree should look like this:

```
data/
├── real/
│   ├── exchange/   DATA_175670.csv, DATA_175828.csv, ...
│   └── exam/       DATA_175670.csv, DATA_175828.csv, ...
└── synthetic/
    ├── exchange/   DATA_175670.csv, DATA_175828.csv, ...
    └── exam/       DATA_175670.csv, DATA_175828.csv, ...
```

File names match the scanner serial numbers configured in
`csv_pipeline/config.py`. Missing synthetic files are OK for a first
pass — you can still run steps 2–4 with real data only.

---

## Step 2 — Consolidate into two flat files

From this folder:

```bash
cd DatabricksPipeline/csv_pipeline/qlik
python consolidate.py
```

The script:

- reads every `DATA_*.csv` under `data/real/` and `data/synthetic/`
- inserts a `DataSource` column (`Real` or `Synthetic`) as the first column
- concatenates all scanners for each kind into one DataFrame
- renames the `sample_idx` column to `ExchangeBlockID` (exchange file)
  and `PatientVisitID` (exam file) so Qlik doesn't mistakenly auto-link
  them across tables
- **prefixes every non-key column** with `Exch_` (exchange file) or
  `Exam_` (exam file). The only fields left unprefixed are the four
  intentional association keys: `DataSource`, `SN`, `ExchangeBlockID`,
  `PatientVisitID`. Everything else — `Age`, `Weight`, `Height`,
  `duration`, `PatientID`, `datetime`, `token_name`, `FinishEvent`, …
  — would otherwise collide by name between the two files and cause
  Qlik's associative engine to auto-join them, silently inflating
  every aggregate. Prefixing keeps the link graph to exactly two edges.
- sorts rows by `DataSource`, `SN`, then timestamp
- writes two files under `data/combined/`

**Expected output** (approximate, with real data only):

```
================================================================
Consolidating EXCHANGE
================================================================
  Real       175670:   10,668 rows,  24 cols
  Real       175828:    9,284 rows,  24 cols
  ... (10 scanners) ...
  Synthetic  (none yet)

  → data/combined/exchange_combined.csv
     120,735 rows, 24 cols, 22.7 MB
    Real:       120,735 rows
    Synthetic:        0 rows

================================================================
Consolidating EXAM
================================================================
  ... similar ...
```

**After this step** you should have exactly two files ready to upload:

```
data/combined/
├── exchange_combined.csv   (~23 MB)
└── exam_combined.csv       (~23 MB)
```

These files are **local-only** (`.gitignore`'d) — they are never
committed to git.

---

## Step 3 — Upload both files to Qlik Sense

### 3a. Create a new app

1. Open Qlik Sense (Desktop or SaaS)
2. Click **Create new app** and name it e.g. `MRI Pipeline Validation`
3. Click **Open app**

### 3b. Add the exchange file

1. Click **Add data from files and other sources** (or the big **+** button)
2. Drag `data/combined/exchange_combined.csv` into the drop zone
3. On the table preview screen:
   - Leave the auto-detected field types
   - Click the table name at the top and rename it from `exchange_combined` to `Exchange` *(shorter = cleaner chart expressions later)*
   - Click **Add data**

### 3c. Add the exam file

Repeat 3b for `data/combined/exam_combined.csv`, renaming the table to `Exam`.

### 3d. Let Qlik associate the two tables

Qlik auto-links tables on any field that appears in both. Because
`consolidate.py` prefixes all non-key columns, our two files share
exactly **two** fields: `SN` (scanner serial) and `DataSource`
(`Real`/`Synthetic`). Qlik will draw two green association lines
between `Exchange` and `Exam` in the data manager — that's all you
want. If you see more than two lines, either the consolidation step
didn't run or the table names were already populated from an older
(pre-prefix) upload — delete both tables and re-add them.

Click **Load data** and wait a few seconds. The status should say
*"Data loaded"*.

### 3e. Sanity check before building charts

Go to a blank sheet, add a **Text & image** object, and paste:

```qlik
='Exchange: ' & Count({<DataSource={'Real'}>} ExchangeBlockID) & ' real + '
  & Count({<DataSource={'Synthetic'}>} ExchangeBlockID) & ' synthetic'
  & Chr(10) & 'Exam: '
  & Count({<DataSource={'Real'}>} PatientVisitID) & ' real + '
  & Count({<DataSource={'Synthetic'}>} PatientVisitID) & ' synthetic'
```

You should see something like:

```
Exchange: 120735 real + 0 synthetic
Exam: 41106 real + 0 synthetic
```

If both numbers are 0, the data didn't load — check step 3b/3c table
names. If you only see real numbers, step 05 hasn't been rerun yet
(that's fine for initial testing).

---

## Step 4 — Build the six comparison charts

Create a new sheet titled **Synthetic vs Real**. Drop these six charts
onto it. Every chart uses `DataSource` as a color or grouping so real
and synthetic appear side by side.

### Chart 1 — Scans per patient (mean)

The #1 validation chart. Real patients get ~9–10 scans; the broken
pre-fix synthetic version produced exactly 1.

- Type: **Bar chart**
- Dimension: `DataSource`
- Measure (rename "Average scans"):
  ```
  =Avg(Aggr(Max(Exam_StepCount), Exam_PatientID, DataSource))
  ```
- Title: **Scans per patient (mean)**

**Pass:** both bars land in 7–12. **Fail:** synthetic near 1.

### Chart 2 — Exam duration distribution

- Type: **Histogram** (or bar chart)
- Dimension: `Class(Exam_duration/60, 0.5)`  *(half-minute bins)*
- Measure: `Count(PatientVisitID)`
- Color by: `DataSource`
- Title: **Exam duration (minutes)**

**Pass:** two overlapping distributions centered near 1.7 min, most
mass under 5 min.

### Chart 3 — Finish event breakdown

- Type: **100% stacked bar chart**
- Dimension 1: `DataSource`
- Dimension 2: `Exam_FinishEvent`
- Measure: `Count(PatientVisitID)`
- Title: **Finish event distribution**

**Pass:** both bars show ~96% Successful, ~3–4% Stopped by User.
**Fail:** synthetic shows 100% Successful (no stop events learned).

### Chart 4 — Body region distribution

- Type: **Bar chart** grouped by DataSource
- Dimension: `Exam_BodyPart`
- Measure:
  ```
  =Count(PatientVisitID) / Count(TOTAL <DataSource> PatientVisitID)
  ```
- Color by: `DataSource`
- Sort: descending by real %
- Title: **Body part share of exams**

**Pass:** same top-3 rank for both (BRAIN, ABDOMEN, LIVER) and low
UNKNOWN share on the synthetic side.

### Chart 5 — Exchange event type distribution

- Type: **Horizontal bar chart**
- Dimension: `Exch_token_name`
- Measure:
  ```
  =Count(Exch_token_name) / Count(TOTAL <DataSource> Exch_token_name)
  ```
- Color by: `DataSource`
- Sort: descending by real %
- Title: **Exchange event type share**

**Pass:** top 5 match the real ordering
(`MRI_FRR_264`, `MRI_FRR_257`, `MRI_FRR_256`, `MRI_FRR_2`, `MRI_CCS_11`).
This is already the metric the exchange model passes.

### Chart 6 — Fidelity Score (headline KPI)

A single number summarising how close synthetic is to real.

- Type: **KPI card**
- Measure:
  ```
  =Round(100 * (1 - (
      (
          Fabs(
              Avg({<DataSource={'Real'}>}      Aggr(Max(Exam_StepCount), Exam_PatientID))
            - Avg({<DataSource={'Synthetic'}>} Aggr(Max(Exam_StepCount), Exam_PatientID))
          )
          / Avg({<DataSource={'Real'}>} Aggr(Max(Exam_StepCount), Exam_PatientID))
        +
          Fabs(
              Avg({<DataSource={'Real'}>}      Exam_duration/60)
            - Avg({<DataSource={'Synthetic'}>} Exam_duration/60)
          )
          / Avg({<DataSource={'Real'}>} Exam_duration/60)
        +
          Fabs(
              Count({<DataSource={'Real'},      Exam_FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)
            - Count({<DataSource={'Synthetic'}, Exam_FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Synthetic'}>} PatientVisitID)
          )
      ) / 3
  )), 1) & ' / 100'
  ```
- Title: **Fidelity Score**

**Interpretation:** 100 means synthetic matches real on all three key
metrics. Ship threshold: **≥ 80**. (Real-only data shows N/A because
there's no synthetic to compare yet.)

### Optional — global filter pane

Add a filter pane at the top of the sheet with these fields:

- `DataSource`
- `SN` *(as "Scanner")*
- `Exam_BodyPart`
- A date field — click on `Exch_datetime` in the fields panel to add

Clicking any value filters every chart on the sheet simultaneously.
Because `DataSource` and `SN` are unprefixed (the only two association
keys), selecting a value in either of these filter panes cross-filters
both Exchange and Exam charts at once. All other filter fields are
table-local, which is what you want.

---

## Step 5 — Read the results

Use this grade sheet as your meeting talking points:

| Chart | Real baseline | Pass threshold for synthetic |
|---|---|---|
| 1. Scans per patient | **9.7** mean | 7.5 – 12 |
| 2. Exam duration | **1.75** min mean | 1.4 – 2.1 min |
| 2. Exam duration (% <1 min) | ~10% | < 20% |
| 3. Stopped-by-User rate | **3.6%** | 1 – 6% |
| 4. Top body parts (rank) | BRAIN, ABDOMEN, LIVER | top 3 match |
| 5. Top exchange tokens (rank) | FRR_264, FRR_257, FRR_256, FRR_2, CCS_11 | top 5 match |
| 6. Fidelity Score | n/a | ≥ 75 good, ≥ 80 ship |

If most rows fall in the pass range, the model is producing plausible
data. The broken April 9 run failed Charts 1, 2 (<1 min rate), and 3 —
those three are the smoking guns for examination-model health.

---

## Refreshing when a new pipeline run completes

```bash
# 1. Pull new CSVs into data/synthetic/ (and data/real/ if that changed too)
#    See fetch_from_dbfs.md
# 2. Rerun the consolidation
cd DatabricksPipeline/csv_pipeline/qlik
python consolidate.py
# 3. In Qlik, right-click the data source → Refresh data
#    (or open Data manager and click the reload icon next to each table)
```

That's the entire refresh loop. No script editing.

---

## FAQ

**Why combine per-scanner CSVs into one file?**
Manual uploading in Qlik is simplest with one file per table. With
per-scanner files you would drag-and-drop 40 files per refresh, which
is error-prone. Two files = two drag-and-drops. The `SN` column in each
file still preserves the scanner identity for filtering.

**Where does the Qlik app live?**
Your choice — Qlik Sense Desktop (everything local, no sharing) or Qlik
Sense SaaS (cloud, shareable with the team). This manual workflow works
identically in both.

**Who owns the refresh cadence?**
Whoever runs `consolidate.py`. Pipeline runs are infrequent (weekly at
most), so manual refresh is fine. If the cadence picks up later, use
the automated `load_script.qvs` in this folder instead.

**Can we automate this later?**
Yes. `load_script.qvs` is a paste-ready Qlik load script that reads
per-scanner files directly and builds the same data model. Switch to
that when you want scheduled refreshes or live Databricks connections.

**The exam file has 100+ columns — why?**
Different scanners emit different coil column sets (27 cols on scanner
182625 vs 92 cols on scanner 176227). Pandas takes the union during
consolidation and pads missing cells with NaN. Qlik handles this
cleanly and you can ignore the coil columns for validation work —
they're there if you ever want coil-level drill-down later.

**What if synthetic and real have different column counts?**
That's expected during the transition period — older synthetic CSVs
(pre-commit `08663b9`) don't have `Age/Weight/Height/Direction/PTAB`.
Pandas fills missing cells with NaN so the concatenation still works;
demographic charts will just show blanks on the synthetic side until
step 05 is rerun with the fix.

**Why not just rename columns inside Qlik's load dialog?**
You can, but it's a per-app manual step that nobody will remember on
the next refresh. Doing it once in `consolidate.py` means every
downstream person gets the right names for free, and the README's
chart expressions copy-paste cleanly. Also, renaming inside Qlik
breaks the "upload the same two files and click Load" refresh loop,
which is the whole point of the manual workflow.

**The fidelity score is blank — why?**
You only have real data loaded (no synthetic yet). The formula divides
by synthetic values; with zero synthetic rows it returns NaN. Run step
05, refetch, rerun `consolidate.py`, refresh Qlik, and it will populate.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `consolidate.py` says "No DATA_*.csv files" | Populate `data/real/` and/or `data/synthetic/` first — see [`fetch_from_dbfs.md`](fetch_from_dbfs.md) |
| `ModuleNotFoundError: pandas` | `pip install pandas` |
| Qlik can't find a column like `Exam_StepCount` or `Exch_token_name` when you paste an expression | You're on an old combined CSV from before the prefix pass. Rerun `consolidate.py`, re-upload both tables in Qlik (delete the old data sources first, then add fresh) |
| Chart expressions reference bare `StepCount`, `duration`, `FinishEvent`, `BodyPart`, `PatientID`, `token_name` | Those are the pre-prefix column names. Every exam-side field is now `Exam_<name>` and every exchange-side field is now `Exch_<name>`. Only `DataSource`, `SN`, `ExchangeBlockID`, `PatientVisitID` are unprefixed |
| Qlik draws more than two association lines between `Exchange` and `Exam` in Data Manager | You uploaded pre-prefix CSVs or mixed an old file with a new one. Delete both tables, rerun `consolidate.py`, re-add |
| Qlik shows "synthetic key" or "circular reference" warning | Same as above — prefixing keeps the link graph to exactly two edges (`SN` and `DataSource`). If you still see this, open Data Manager and check whether some field other than those two is creating a green link |
| Chart 3 shows only "Successful" on the synthetic side | Your exam model is still producing no stop events — rerun step 04 and step 05 with commit `08663b9` or later |
| Chart 1 shows exactly 1.0 for synthetic | Same — you're looking at pre-fix synthetic output |
| Fidelity Score is `NaN / 100` | One side has zero rows — check the sanity-check text from step 3e |

---

## Column naming reference

Every non-key column is prefixed per kind so Qlik's associative engine
only joins the two tables on the fields you actually want. Use this
table as a lookup when writing your own chart expressions.

| Qlik field name | Lives in | What it is |
|---|---|---|
| `DataSource` | **both** (link) | `'Real'` or `'Synthetic'` — drives every comparison |
| `SN` | **both** (link) | Scanner serial number — cross-filter key |
| `ExchangeBlockID` | Exchange | Per-row block id (was `sample_idx`) |
| `PatientVisitID` | Exam | Per-visit id (was `sample_idx`) |
| `Exch_token_name` | Exchange | Event name, e.g. `MRI_FRR_264` |
| `Exch_token_id` | Exchange | Event id (integer) |
| `Exch_datetime` | Exchange | Event timestamp |
| `Exch_timediff` | Exchange | Seconds since previous event |
| `Exch_PatientID_from` / `Exch_PatientID_to` | Exchange | Patient handoff direction |
| `Exch_BodyGroup_from` / `Exch_BodyGroup_to` | Exchange | Body-region handoff |
| `Exch_predicted_mu` / `Exch_predicted_sigma` / `Exch_sampled_duration` | Exchange | Exchange-model outputs |
| `Exch_Age` / `Exch_Weight` / `Exch_Height` / `Exch_Direction` / `Exch_PTAB` | Exchange | Patient demographics as seen by exchange rows |
| `Exam_PatientID` | Exam | Patient id (note: independent of `Exch_PatientID_*`) |
| `Exam_BodyPart` / `Exam_BodyGroup` | Exam | Anatomical region |
| `Exam_Sequence` / `Exam_Protocol` | Exam | MRI sequence identifiers |
| `Exam_ConnectedCoils` | Exam | Comma-separated coil list |
| `Exam_FinishEvent` | Exam | `Successful`, `Stopped by User`, etc. |
| `Exam_duration` / `Exam_startTime` / `Exam_endTime` / `Exam_pauseTime` | Exam | Timing fields |
| `Exam_StepCount` | Exam | Scans per patient visit |
| `Exam_predicted_mu` / `Exam_predicted_sigma` / `Exam_sampled_duration` | Exam | Examination-model outputs |
| `Exam_Age` / `Exam_Weight` / `Exam_Height` / `Exam_Direction` / `Exam_PTAB` | Exam | Patient demographics as seen by exam rows |
| `Exam_#0_BC`, `Exam_#0_SP1`, … | Exam | Coil columns (pre-fixed verbatim; not meant to be linked) |

**Rule of thumb for your own charts:** if the field exists in exactly
one file, use the prefixed name. If it exists in both files and you
want them joined, use one of the four unprefixed keys. If you find
yourself wanting to join on `PatientID` across tables, stop — those
are `Exch_PatientID_*` and `Exam_PatientID`, and they mean different
things even in real data (and are completely independent in synthetic).

---

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | You are here — step-by-step how-to |
| `consolidate.py` | Local Python script that merges per-scanner CSVs into two flat files |
| `fetch_from_dbfs.md` | Three ways to download CSVs from Databricks |
| `load_script.qvs` | Alternative: automated Qlik load script (for later, when you want scheduled refreshes) |
| `dashboard_spec.md` | Advanced reference: full 4-sheet dashboard with pivot tables and extras (uses the automated load_script.qvs model) |
| `data/` | Landing zone for CSVs. Content is git-ignored; folder structure preserved via `.gitkeep` files |
| `data/combined/` | Output of `consolidate.py` — the two files you upload to Qlik |
