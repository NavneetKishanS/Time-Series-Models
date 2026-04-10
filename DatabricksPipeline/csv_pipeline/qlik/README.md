# Qlik Validation Dashboard — Synthetic vs Real

This folder contains everything needed to set up a Qlik Sense dashboard
that compares the synthetic MRI pipeline output against the real
training data, side-by-side.

```
qlik/
├── README.md               ← you are here — meeting overview + workflow
├── load_script.qvs         ← paste-ready Qlik Data Load script
├── dashboard_spec.md       ← 6 comparison charts, expressions, sheet layout
├── fetch_from_dbfs.md      ← how to populate data/ from Databricks DBFS
└── data/                   ← drop CSVs here (gitignored content, structure tracked)
    ├── real/
    │   ├── exchange/       ← copy step 01 output here
    │   └── exam/           ← copy step 02 output here
    └── synthetic/
        ├── exchange/       ← copy step 05 exchange output here
        └── exam/           ← copy step 05 exam output here
```

---

## 1. Why we're building this

The pipeline already produces a text evaluation report at the end of
step 05. It's useful, but has four limits:

| Limit | Impact |
|---|---|
| Text-only, printed once | Can't share outside Databricks |
| Only shows synthetic stats | No side-by-side comparison with real |
| Frozen at that run | Disappears on the next rerun |
| Not drillable | You can't click "what about scanner X on Feb 12?" |

A Qlik dashboard solves all four: **persistent, side-by-side,
shareable, drillable.**

---

## 2. The approach in one sentence

**Load real CSVs and synthetic CSVs into the same Qlik fact tables,
tag each row with `DataSource = 'Real' | 'Synthetic'`, and let Qlik's
associative engine do the comparison work automatically.**

Every chart becomes a side-by-side comparison just by adding
`DataSource` as a series/colour. No joins, no unions, no schema
gymnastics.

---

## 3. Data model — why we keep exchange and exam separate

Exchange and exam data live at different **grains**:

| | Grain | Rows per scanner-month | What each row means |
|---|---|---|---|
| **Exchange** | event-level | ~9k–19k | One MRI log event during a patient changeover |
| **Exam** | measurement-level | ~2.5k–5k | One MRI scan (start → finish) |

Smashing them into a single table would either force one to lose
detail or pad the other with empty columns. Worse, it would fight
Qlik's **associative model** — Qlik's strongest feature is that
separate tables linked by shared keys auto-filter across each other.
Two clean tables beat one messy one.

**The final data model:**

```
                ┌──────────────────┐
                │   DIM_DataSource │  Real | Synthetic
                └────────┬─────────┘
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
│ DIM_Scanner │   │ DIM_Date    │   │  (patient   │
│             │   │  (calendar) │   │  dimensions │
└──────┬──────┘   └──────┬──────┘   │  derived)   │
       │                 │          └──────┬──────┘
       └────────┬────────┴─────────────────┘
                │
     ┌──────────┴───────────┐
     │                      │
┌────┴──────────┐   ┌──────┴────────┐
│ FACT_Exchange │   │   FACT_Exam   │
│ (events)      │   │ (measurements)│
└───────────────┘   └───────────────┘
```

Every fact row carries `DataSource`, `ScannerID`, `PatientID`, and
`EventDate`, so clicking any of those filters across both tables and
both real/synthetic sides at once.

---

## 4. The six metrics we'll compare

The dashboard validates the four findings from the step 05 evaluation
report plus two supporting signals. Targets are derived from the real
training data.

| # | Metric | Real baseline | Synthetic target | Why it matters |
|---|---|---|---|---|
| 1 | **Scans per patient** (mean) | ~8 | 6–10 | Caught the "1 scan per patient" bug |
| 2 | **Exam duration** (mean, minutes) | 1.6–2.8 | within ±20% | Caught the duration-scale issue |
| 3 | **Finish event mix** | 96–98% Successful, 2–4% Stopped | similar spread | Model should learn stop tokens |
| 4 | **Body region distribution** | HEAD/PELVIS/SPINE dominant, low UNKNOWN | low UNKNOWN share, similar rank | Orchestration model health |
| 5 | **Exchange event type distribution** | FRR_264 / FRR_256 / FRR_257 / CCS_11 top | top 5 match real | Exchange model health (passing today) |
| 6 | **Patients per scanner-day** | ~14–30 | within ±30% | Throughput realism |

See `dashboard_spec.md` for the exact chart expressions.

---

## 5. How new data gets into the dashboard (the refresh loop)

```
Databricks                            Local / Qlik server
───────────                           ───────────────────
step 01 ─┐
step 02 ─┤
         ├──> DBFS: /FileStore/csv_pipeline/{exchange,exam}/DATA_*.csv
step 05 ─┘                                        │
                                                  │  fetch_from_dbfs.md
                                                  ▼
                                     qlik/data/{real,synthetic}/...
                                                  │
                                                  │  Qlik Data Load Editor
                                                  ▼
                                     load_script.qvs  →  reload app
                                                  │
                                                  ▼
                                     Dashboard refreshed
```

**Three triggers for a refresh:**

1. **New model run** — someone retrains (step 04) and regenerates
   (step 05). Fetch new synthetic CSVs → reload Qlik app.
2. **New real data** — a fresh month of scanner logs lands. Fetch new
   real CSVs → reload.
3. **New scanners** — SERIAL_NUMBERS in `csv_pipeline/config.py`
   changes. Rerun the full pipeline → fetch both → reload.

The refresh is idempotent: the Qlik load script reads every CSV under
`data/{real,synthetic}/{exchange,exam}/DATA_*.csv`, so dropping new
files in and reloading is the entire workflow.

---

## 6. What to present at the meeting

Suggested talking-point order (~5 minutes):

1. **Problem** — "We have a text report that tells us synthetic data is
   mostly good but hard to share and impossible to drill into."
2. **Direction** — "Build a Qlik dashboard that shows synthetic
   side-by-side with real, for the six metrics that matter."
3. **Design decision** — "Keep exchange and exam as separate fact
   tables. Tag rows with DataSource. Let Qlik's associative model do
   the comparison."
4. **What's ready today** — folder structure, load script, chart
   specs, refresh workflow (point at this folder).
5. **What's pending** — step 04 retraining finishes tonight; step 05
   rerun tomorrow morning; first dashboard by end of week.
6. **Ask** — who owns the Qlik app, where does it live, who has
   access, what's the refresh cadence?

---

## 7. Immediate next steps

- [ ] **Step 04 retrain finishes** (in-flight, ~5 h remaining as of
      meeting time)
- [ ] **Rerun step 05** to pick up the multi-scan fix + new model
      weights (commit `08663b9`)
- [ ] **Fetch CSVs** from DBFS into `data/` (see `fetch_from_dbfs.md`)
- [ ] **Run the load script** in a new Qlik Sense app
      (`load_script.qvs`)
- [ ] **Build Sheet 1** from `dashboard_spec.md` (Overview + headline
      comparison KPIs)
- [ ] **Review with team** — sanity-check chart targets against real
      data
- [ ] **Schedule refresh cadence** — align with pipeline rerun cadence

---

## 8. Files in this folder

| File | Purpose |
|---|---|
| `README.md` | This file — overview, data model, workflow, meeting notes |
| `load_script.qvs` | Paste-ready Qlik Data Load Editor script |
| `dashboard_spec.md` | 6 comparison charts with Qlik expressions + sheet layout |
| `fetch_from_dbfs.md` | How to populate `data/` from Databricks |
| `data/` | Landing zone for CSVs (content ignored by git, folders tracked) |

---

## 9. Open questions for the meeting

- Where will the Qlik app live? (Qlik Sense SaaS, on-prem, Cloud Hub?)
- Who owns the load/refresh cadence?
- Do we want a Databricks JDBC/ODBC connector for live data later, or
  stick with file-based refresh for now?
- Do we need role-based access (PHI considerations for real patient
  IDs)?
- What's the promotion path from "dev dashboard" to "team dashboard"?
