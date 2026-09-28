"""
incident_detect.py — Feature 19: incident detection.
===========================================================================
"Incident detection: expected arrivals at the next camera go missing while
the queue grows." Tracks, per camera-to-camera link, which vehicles were
seen departing the upstream camera and are now overdue at the downstream
one, and combines that with a queue-growth signal (e.g.
green_wave.QueuePredictor.trend()) to flag a likely incident rather than
just an unusually slow but still-flowing link.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

from datetime import datetime
from typing import Dict, List


class IncidentDetector:
    """One instance per camera-to-camera link (e.g. CAM_03 -> CAM_04)."""

    def __init__(self, typical_travel_minutes: float, tolerance_minutes: float = 2.0):
        self.typical_travel_minutes = typical_travel_minutes
        self.tolerance_minutes = tolerance_minutes
        self._departures: Dict[str, datetime] = {}   # vehicle_id -> departure time from upstream camera

    def record_departure(self, vehicle_id: str, departure_time_iso: str) -> None:
        self._departures[vehicle_id] = datetime.fromisoformat(departure_time_iso)

    def record_arrival(self, vehicle_id: str) -> None:
        """Vehicle showed up at the downstream camera — stop tracking it as expected."""
        self._departures.pop(vehicle_id, None)

    def overdue_vehicles(self, now_iso: str) -> List[str]:
        """Vehicle ids that departed upstream but haven't arrived downstream within the expected window."""
        now = datetime.fromisoformat(now_iso)
        deadline_minutes = self.typical_travel_minutes + self.tolerance_minutes
        overdue = []
        for vehicle_id, departed_at in self._departures.items():
            elapsed_minutes = (now - departed_at).total_seconds() / 60.0
            if elapsed_minutes > deadline_minutes:
                overdue.append(vehicle_id)
        return overdue

    def pending_count(self) -> int:
        return len(self._departures)

    def clear(self) -> None:
        self._departures.clear()


def detect_incident(
    overdue_count: int, queue_trend: str, missing_threshold: int = 3
) -> Dict:
    """
    Combines "expected arrivals missing" (overdue_count, from
    IncidentDetector.overdue_vehicles) with "queue growing" (queue_trend,
    e.g. green_wave.QueuePredictor.trend() == "RISING"). Either signal
    alone is common and often benign (a slow link, a temporary lull in
    arrivals) — both together is the actual incident signature.
    """
    likely_incident = overdue_count >= missing_threshold and queue_trend == "RISING"
    return {
        "incident_likely": likely_incident,
        "overdue_count": overdue_count,
        "queue_trend": queue_trend,
        "missing_threshold": missing_threshold,
    }
