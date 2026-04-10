# Dashboard Spec — Synthetic vs Real Comparison

This document defines the minimum viable Qlik dashboard for validating
synthetic pipeline output against real training data. Four sheets, six
core comparison charts, one headline score.

**Prerequisite:** `load_script.qvs` must be loaded and all five tables
(`FACT_Exchange`, `FACT_Exam`, `DIM_Scanner`, `DIM_Date`,
`DIM_DataSource`) visible in the Data Model Viewer.

---

## Global filter bar (every sheet)

Put these as filter panes across the top of every sheet. Clicking any
of them narrows every chart on the page simultaneously.

| Filter | Field | Notes |
|---|---|---|
| Data Source | `DataSource` | Default: both selected |
| Scanner | `ScannerLabel` | Multi-select |
| Date | `EventDate` | Slider |
| Body Region | `BodyRegion` | Multi-select |

---

## Sheet 1 — Overview (the meeting chart)

The single sheet an exec would look at. All KPIs show **real** and
**synthetic** as two big numbers side-by-side, with a delta in between.

### KPI 1 — Scans per patient

| Side | Expression |
|---|---|
| Real | `Avg({<DataSource={'Real'}>} Aggr(Max(ScanIndexInVisit), PatientVisitID))` |
| Synthetic | `Avg({<DataSource={'Synthetic'}>} Aggr(Max(ScanIndexInVisit), PatientVisitID))` |
| Delta % | `([Synth mean] - [Real mean]) / [Real mean]` |

**Pass criterion:** both values in the 6–10 range, delta within ±25%.

### KPI 2 — Mean exam duration (minutes)

| Side | Expression |
|---|---|
| Real | `Avg({<DataSource={'Real'}>} ExamDurationMin)` |
| Synthetic | `Avg({<DataSource={'Synthetic'}>} ExamDurationMin)` |

**Pass criterion:** synthetic within ±20% of real.

### KPI 3 — Short exam rate (% of exams < 1 minute)

| Side | Expression |
|---|---|
| Real | `Count({<DataSource={'Real'}, ExamDurationMin={'<1'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)` |
| Synthetic | `Count({<DataSource={'Synthetic'}, ExamDurationMin={'<1'}>} PatientVisitID) / Count({<DataSource={'Synthetic'}>} PatientVisitID)` |

**Pass criterion:** synthetic < 10%. (The broken run hit 38.8% — this
is the smoking gun.)

### KPI 4 — "Stopped by User" rate

| Side | Expression |
|---|---|
| Real | `Count({<DataSource={'Real'}, FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)` |
| Synthetic | same with `Synthetic` in the set |

**Pass criterion:** synthetic ≥ 1%. (The broken run hit 0% — the model
never learned to stop.)

### Headline score — "Fidelity Score" (0–100)

A single number that summarizes how close synthetic is to real across
the four metrics above. Put it top-left as the biggest number on the
sheet.

```qlik
// Weighted mean of four relative errors, inverted so 100 = perfect.
= Round(
    100 * (1 - (
        (Fabs(
            Avg({<DataSource={'Real'}>}      Aggr(Max(ScanIndexInVisit), PatientVisitID))
          - Avg({<DataSource={'Synthetic'}>} Aggr(Max(ScanIndexInVisit), PatientVisitID))
        ) / Avg({<DataSource={'Real'}>} Aggr(Max(ScanIndexInVisit), PatientVisitID))
         +
         Fabs(
            Avg({<DataSource={'Real'}>}      ExamDurationMin)
          - Avg({<DataSource={'Synthetic'}>} ExamDurationMin)
        ) / Avg({<DataSource={'Real'}>} ExamDurationMin)
         +
         Fabs(
            Count({<DataSource={'Real'},      ExamDurationMin={'<1'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)
          - Count({<DataSource={'Synthetic'}, ExamDurationMin={'<1'}>} PatientVisitID) / Count({<DataSource={'Synthetic'}>} PatientVisitID)
        )
         +
         Fabs(
            Count({<DataSource={'Real'},      FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Real'}>} PatientVisitID)
          - Count({<DataSource={'Synthetic'}, FinishEvent={'Stopped by User'}>} PatientVisitID) / Count({<DataSource={'Synthetic'}>} PatientVisitID)
        )
        ) / 4
    )),
    1
)
```

**Pass criterion:** score ≥ 75.

---

## Sheet 2 — Distributions (the "look at the shape" sheet)

A 2×3 grid of distribution charts, each with `DataSource` as a colour
series so real and synthetic overlay or sit side-by-side.

### Chart 1 — Scans per patient histogram

- Type: **bar chart**
- Dimension: `Aggr(Max(ScanIndexInVisit), PatientVisitID, DataSource)`
- Measure: `Count(DISTINCT PatientVisitID)`
- Series (colour): `DataSource`
- Stacking: grouped (side-by-side bars)

### Chart 2 — Exam duration histogram

- Type: **bar chart**
- Dimension: `Class(ExamDurationMin, 0.5)` (half-minute bins)
- Measure: `Count(PatientVisitID)`
- Series: `DataSource`

### Chart 3 — Finish event breakdown

- Type: **stacked bar, 100% mode**
- Dimension 1: `DataSource`
- Dimension 2: `FinishEvent`
- Measure: `Count(PatientVisitID)`

### Chart 4 — Body region distribution

- Type: **bar chart**
- Dimension: `BodyRegion`
- Measure:
  `Count(PatientVisitID) / Count(TOTAL <DataSource> PatientVisitID)`
  (percentage per DataSource)
- Series: `DataSource`

### Chart 5 — Exchange event type distribution

- Type: **horizontal bar chart**
- Dimension: `EventTypeName`
- Measure:
  `Count(ExchangeBlockID) / Count(TOTAL <DataSource> ExchangeBlockID)`
- Series: `DataSource`
- Sort descending by real rate

### Chart 6 — Demographics (optional)

- Type: **three small histograms** (Age, Weight, Height)
- Dimension: `Class(Age, 5)`, `Class(Weight, 5)`, `Class(Height, 0.05)`
- Measure: `Count(DISTINCT PatientID)`
- Series: `DataSource`

---

## Sheet 3 — Per-scanner drill-down

A pivot table that rolls up every metric by scanner × data source. Use
when someone asks "is scanner X broken, or is the whole model broken?"

- Type: **pivot table**
- Rows: `ScannerLabel`
- Columns: `DataSource`
- Measures:
  - Patients: `Count(DISTINCT PatientID)`
  - Scans per patient: `Avg(Aggr(Max(ScanIndexInVisit), PatientVisitID))`
  - Exam duration mean: `Avg(ExamDurationMin)`
  - Stopped rate: `Count({<FinishEvent={'Stopped by User'}>} PatientVisitID) / Count(PatientVisitID)`

Add a **bar chart** underneath showing `Count(DISTINCT PatientID)` per
scanner per datasource, to visualize throughput differences.

---

## Sheet 4 — Patient timeline (drill-through)

For debugging individual cases. Pick a patient, see all their events
in order.

- **Filter pane**: `PatientID`, `EventDate`, `DataSource` (single-select)
- **Gantt chart** (or stacked timeline):
  - Dimension: `EventTimestamp`
  - Sub-dimension: `EventTypeName` (for exchange) OR `ScanSequence`
    (for exam)
  - Bar length: `TimeDiffSec` or `ExamDurationSec`

This sheet answers "what does a typical synthetic patient visit look
like compared to a real one?"

---

## Sanity check text box (put on Sheet 1)

Paste this into a text object to confirm the load succeeded:

```
Rows loaded:
  Exchange: =Count({<DataSource={'Real'}>} ExchangeBlockID) & ' real + '
          & Count({<DataSource={'Synthetic'}>} ExchangeBlockID) & ' synthetic'
  Exam:     =Count({<DataSource={'Real'}>} PatientVisitID) & ' real + '
          & Count({<DataSource={'Synthetic'}>} PatientVisitID) & ' synthetic'

Scanners: =Count(DISTINCT ScannerID)
Date range: =Min(EventDate) & ' → ' & Max(EventDate)
```

---

## What each chart caught in the first pipeline run (Apr 9 report)

Use this table as the "before/after" story for the meeting.

| Chart | Apr 9 finding | Expected post-fix |
|---|---|---|
| KPI 1 (Scans/patient) | 1.0 synth vs ~8 real → **FAIL** | ~8 both → PASS |
| KPI 2 (Exam duration) | 1.5 min synth — within range | no change expected |
| KPI 3 (Short exam rate) | 38.8% synth → **FAIL** | <10% → PASS |
| KPI 4 (Stopped rate) | 0.0% synth → **FAIL** | 2–4% → PASS |
| Chart 4 (Body region) | 26% UNKNOWN → **WARN** | unchanged (not yet fixed) |
| Chart 5 (Exchange events) | top 5 match real → PASS | unchanged |

After the next rerun (commits `08663b9` + `930587c`), KPI 1, 3, 4
should flip to PASS. Chart 4 (UNKNOWN share) remains an open item for
a follow-up orchestration-model fix.
