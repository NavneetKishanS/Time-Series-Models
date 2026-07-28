"""Regression tests for the simulated-day length bound.

The 2026-07-28 step 07 run on serial 176148 died with

    GenerationIntegrityError: Synthetic output integrity check failed (2 issue(s)):
      - exam row 935 overlaps the preceding examination
      - exam row 1526 overlaps the preceding examination

Both rows were the FIRST examination of a day, and both followed the two
busiest days in the plan (29 and 31 patients). The day simulator lays patients
end to end from a fixed 07:00 start with no idle time and nothing bounded the
result, so those two days generated more than 24 h of content and ran past the
next day's 07:00 start. The next day's opening scan then began before the
previous day's closing scan had ended — two examinations at once on one
scanner.
"""

import unittest
from datetime import datetime, timedelta

from AlternatingPipeline.generation.day_budget import DayBudget
from AlternatingPipeline.generation.output_integrity import (
    GenerationIntegrityError,
    validate_rendered_output,
)


BODY_REGIONS = ['HEAD', 'ABDOMEN']


def _exam_row(patient_id, step, start, duration_sec):
    return {
        'PatientID':   patient_id,
        'StepCount':   step,
        'duration':    duration_sec,
        'startTime':   start.strftime('%Y-%m-%d %H:%M:%S'),
        'endTime':     (start + timedelta(seconds=duration_sec)).strftime('%Y-%m-%d %H:%M:%S'),
        'FinishEvent': 'Successful',
        'sourceID':    'MRI_MSR_104',
        'BodyPart':    'HEAD',
    }


def _simulate(dates, patients_per_day, scans_per_patient, scan_sec,
              exchange_sec, budget=None):
    """Mirror the day loop in 07_generate_synthetic_data.py.

    Returns (exam_rows, expected_exam_counts) — the two arguments
    validate_rendered_output compares against each other.
    """
    rows = []
    expected = {}
    for date_str in dates:
        day_start = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=7, minute=0)
        current_t = 0.0
        served = 0
        dropped_scans = 0
        for p_idx in range(patients_per_day):
            if budget is not None and not budget.accepts_patient(current_t, served):
                break
            patient_id = f'SYNTH_{date_str}_{p_idx:03d}'
            current_t += exchange_sec
            rendered = 0
            for _ in range(scans_per_patient):
                if budget is not None and not budget.accepts_scan(current_t, rendered):
                    dropped_scans += scans_per_patient - rendered
                    break
                rows.append(_exam_row(
                    patient_id, rendered + 1,
                    day_start + timedelta(seconds=current_t), scan_sec,
                ))
                current_t += scan_sec
                rendered += 1
            expected[patient_id] = rendered
            served += 1
        if budget is not None:
            budget.record_day(patients_per_day - served, dropped_scans)
    return rows, expected


class DayBudgetTests(unittest.TestCase):
    def test_first_patient_and_first_scan_always_run(self):
        """A budget must never produce an empty day or a patient with no scan."""
        budget = DayBudget(window_sec=1.0)
        self.assertTrue(budget.accepts_patient(elapsed_sec=99999.0, patients_served=0))
        self.assertTrue(budget.accepts_scan(elapsed_sec=99999.0, scans_rendered=0))

    def test_further_work_is_refused_once_the_window_is_spent(self):
        budget = DayBudget(window_sec=100.0)
        self.assertTrue(budget.accepts_patient(elapsed_sec=99.0, patients_served=3))
        self.assertFalse(budget.accepts_patient(elapsed_sec=100.0, patients_served=3))
        self.assertTrue(budget.accepts_scan(elapsed_sec=99.0, scans_rendered=2))
        self.assertFalse(budget.accepts_scan(elapsed_sec=100.0, scans_rendered=2))

    def test_window_must_be_positive(self):
        for bad in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                DayBudget(window_sec=bad)

    def test_summary_is_silent_until_something_is_dropped(self):
        budget = DayBudget(window_sec=100.0)
        budget.record_day(patients_dropped=0, scans_dropped=0)
        self.assertIsNone(budget.summary())
        budget.record_day(patients_dropped=4, scans_dropped=9)
        self.assertEqual(budget.days_truncated, 1)
        self.assertEqual(budget.patients_dropped, 4)
        self.assertEqual(budget.scans_dropped, 9)
        self.assertIn('4', budget.summary())


class DayOverrunTests(unittest.TestCase):
    # The shape the 2026-07-28 run generated: 31 patients (176148's busiest
    # day) x 8 scans x 300 s + 31 x 560 s of exchange = 25.5 h, just past the
    # 24 h at which a day starts colliding with the next one.
    DATES = ['2024-02-01', '2024-02-02']
    PATIENTS = 31
    SCANS = 8
    SCAN_SEC = 300.0
    EXCHANGE_SEC = 560.0

    def test_unbounded_day_spills_into_the_next_day(self):
        """Reproduce the reported failure: no bound, so the day runs past 24 h."""
        rows, expected = _simulate(
            self.DATES, self.PATIENTS, self.SCANS,
            self.SCAN_SEC, self.EXCHANGE_SEC, budget=None,
        )
        with self.assertRaises(GenerationIntegrityError) as ctx:
            validate_rendered_output(
                [], rows, expected, 0, BODY_REGIONS,
            )
        self.assertIn('overlaps the preceding examination', str(ctx.exception))

    def test_budgeted_day_stays_inside_its_own_day(self):
        budget = DayBudget(window_sec=13 * 3600.0)
        rows, expected = _simulate(
            self.DATES, self.PATIENTS, self.SCANS,
            self.SCAN_SEC, self.EXCHANGE_SEC, budget=budget,
        )
        # No overlap anywhere, and the planned/rendered counts still agree.
        report = validate_rendered_output([], rows, expected, 0, BODY_REGIONS)
        self.assertEqual(report['exam_rows'], len(rows))

        by_day = {}
        for row in rows:
            start = datetime.fromisoformat(row['startTime'])
            end = datetime.fromisoformat(row['endTime'])
            day = start.date() if start.hour >= 7 else (start - timedelta(hours=7)).date()
            first, last = by_day.get(day, (start, end))
            by_day[day] = (min(first, start), max(last, end))
        for day, (first, last) in by_day.items():
            span_h = (last - first).total_seconds() / 3600.0
            self.assertLess(span_h, 15.0, f"{day} spans {span_h:.1f} h")
        self.assertEqual(len(by_day), len(self.DATES))
        self.assertGreater(budget.patients_dropped, 0)

    def test_a_realistic_day_is_never_truncated(self):
        """With real-scale durations the budget must not bind at all.

        Real serials run 12-29 patients/day at ~105 s mean scan duration; the
        budget exists to catch over-generation, not to reshape normal output.
        """
        budget = DayBudget(window_sec=13 * 3600.0)
        rows, expected = _simulate(
            self.DATES, self.PATIENTS, self.SCANS,
            scan_sec=105.0, exchange_sec=self.EXCHANGE_SEC, budget=budget,
        )
        validate_rendered_output([], rows, expected, 0, BODY_REGIONS)
        self.assertEqual(budget.patients_dropped, 0)
        self.assertEqual(budget.scans_dropped, 0)
        self.assertEqual(budget.days_truncated, 0)
        self.assertEqual(len(rows), len(self.DATES) * self.PATIENTS * self.SCANS)


if __name__ == '__main__':
    unittest.main()
