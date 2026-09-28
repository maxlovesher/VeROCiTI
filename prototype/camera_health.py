"""
camera_health.py — Feature 20: camera health monitoring.
===========================================================================
"Camera health monitoring: automatic switch to the fallback level when a
camera fails." Tracks a heartbeat per camera; once a camera goes quiet
past its timeout it's marked unhealthy, and the fleet's overall health
ratio maps to a coverage fallback level other code (analytics, dashboards)
can react to — e.g. leaning more on historical OD patterns instead of live
counts once enough cameras are down.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

from datetime import datetime
from typing import Dict, Iterable, List

# Ordered worst-to-best is not needed here; ordered best-to-worst by
# ascending health-ratio threshold, matching fallback_level_for()'s scan.
COVERAGE_LEVELS = ["FULL_COVERAGE", "PARTIAL_COVERAGE", "MINIMAL_COVERAGE"]


class CameraHealthMonitor:
    def __init__(self, timeout_s: float = 30.0):
        self.timeout_s = timeout_s
        self._last_seen: Dict[str, datetime] = {}

    def heartbeat(self, camera_id: str, timestamp_iso: str) -> None:
        """Call on every detection/keepalive ping received from a camera."""
        self._last_seen[camera_id] = datetime.fromisoformat(timestamp_iso)

    def is_healthy(self, camera_id: str, now_iso: str) -> bool:
        last = self._last_seen.get(camera_id)
        if last is None:
            return False   # never reported in -> can't call it healthy
        now = datetime.fromisoformat(now_iso)
        return (now - last).total_seconds() <= self.timeout_s

    def unhealthy_cameras(self, all_camera_ids: Iterable[str], now_iso: str) -> List[str]:
        return [c for c in all_camera_ids if not self.is_healthy(c, now_iso)]

    def fleet_health_ratio(self, all_camera_ids: Iterable[str], now_iso: str) -> float:
        ids = list(all_camera_ids)
        if not ids:
            return 1.0
        healthy = sum(1 for c in ids if self.is_healthy(c, now_iso))
        return round(healthy / len(ids), 3)

    def forget(self, camera_id: str) -> None:
        self._last_seen.pop(camera_id, None)


def fallback_level_for(health_ratio: float, full_threshold: float = 0.9, partial_threshold: float = 0.5) -> str:
    """
    Maps a fleet health ratio (0..1) to a coverage fallback level:
      >= full_threshold    -> FULL_COVERAGE     (trust live per-camera counts)
      >= partial_threshold -> PARTIAL_COVERAGE  (blend live data with historical patterns for gaps)
      otherwise             -> MINIMAL_COVERAGE  (lean on historical/OD patterns, flag live data as unreliable)
    """
    if health_ratio >= full_threshold:
        return "FULL_COVERAGE"
    if health_ratio >= partial_threshold:
        return "PARTIAL_COVERAGE"
    return "MINIMAL_COVERAGE"
