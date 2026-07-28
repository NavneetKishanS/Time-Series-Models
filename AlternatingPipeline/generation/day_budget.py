"""Keep one simulated scanner day inside a physically realisable window."""

import math


class DayBudget:
    """Decides how much more work still fits in the day being generated.

    The day simulator lays patients end to end from a fixed 07:00 start with no
    idle time, so a day's wall-clock length is simply whatever the models
    happen to generate — and nothing bounded it. A day whose content exceeds
    24 h runs past the NEXT day's 07:00 start, and that day's opening
    examination then begins before the previous day's closing one ended: two
    scans at once on a single scanner, which ``validate_rendered_output``
    correctly rejects with "exam row N overlaps the preceding examination".

    That is exactly how the 2026-07-28 run on serial 176148 failed, on the two
    busiest days of its plan (29 and 31 patients) and nowhere else. Serial
    183242 survived the same run only because its longest day came to 22.9 h.

    Real days on these serials span at most 12.4 h (mean 8.7-10.0 h), so a day
    that wants more than the configured window is over-generated, not unusually
    busy. Stopping at a patient or scan boundary and counting what was dropped
    reports that over-generation instead of silently shipping a 22-hour day.

    The first patient of a day and the first scan of a patient always run, so
    the budget can never produce an empty day or a patient with no scan — it
    bounds the day at ``window_sec`` plus one patient's opening exchange and
    scan, which is still far inside the next day's start.
    """

    def __init__(self, window_sec):
        window_sec = float(window_sec)
        if not math.isfinite(window_sec) or window_sec <= 0:
            raise ValueError(f"window_sec must be positive and finite, got {window_sec!r}")
        self.window_sec = window_sec
        self.days_truncated = 0
        self.patients_dropped = 0
        self.scans_dropped = 0

    def accepts_patient(self, elapsed_sec, patients_served):
        """True if another patient may be scheduled into the day."""
        return int(patients_served) <= 0 or float(elapsed_sec) < self.window_sec

    def accepts_scan(self, elapsed_sec, scans_rendered):
        """True if another scan may be added to the patient being generated."""
        return int(scans_rendered) <= 0 or float(elapsed_sec) < self.window_sec

    def record_day(self, patients_dropped=0, scans_dropped=0):
        """Fold one finished day's drops into the run totals."""
        patients_dropped = max(0, int(patients_dropped))
        scans_dropped = max(0, int(scans_dropped))
        if not patients_dropped and not scans_dropped:
            return
        self.days_truncated += 1
        self.patients_dropped += patients_dropped
        self.scans_dropped += scans_dropped

    def summary(self):
        """One-line report of what the budget cut, or None if it never bound."""
        if not self.days_truncated:
            return None
        return (
            f"day budget bound on {self.days_truncated} day(s): dropped "
            f"{self.patients_dropped} patient(s) and {self.scans_dropped} "
            f"scan(s) that did not fit a {self.window_sec / 3600:.1f}h day — "
            f"generated durations are longer than real ones"
        )
