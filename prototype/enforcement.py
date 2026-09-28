"""
enforcement.py — Features 12–17: automated enforcement checks.
===========================================================================
  12. Section (average) speed between two cameras.
  13. Red-light violation detection.
  14. Wrong-way detection.
  15. Heavy-vehicle time-window enforcement.
  16. School-zone timed speed limits.
  17. Junction-mouth parking detection.

Each check is a small, independent, pure function returning a structured
record: {"violation": bool, "type": str, "details": {...}}. Kept
independent of Flask/database so each rule is trivially unit-testable and
can be wired into whichever ingestion path needs it (live camera pipeline,
a batch job over stored detections, or a future API endpoint).
"""

from datetime import datetime, time as _time
from typing import Dict, List, Optional, Tuple

from analytics import haversine_km


# ---------------------------------------------------------------------------
# 12. Section (average) speed between two cameras.
# ---------------------------------------------------------------------------

def section_speed_kmph(lat1: float, lon1: float, t1_iso: str, lat2: float, lon2: float, t2_iso: str) -> Optional[float]:
    """Average speed over the straight-line distance between two camera sightings. None if the timestamps don't parse or are non-positive apart."""
    try:
        t1 = datetime.fromisoformat(t1_iso)
        t2 = datetime.fromisoformat(t2_iso)
    except (ValueError, TypeError):
        return None
    hours = (t2 - t1).total_seconds() / 3600.0
    if hours <= 0:
        return None
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return round(dist_km / hours, 1)


def check_section_speed(lat1, lon1, t1_iso, lat2, lon2, t2_iso, limit_kmph: float) -> Dict:
    speed = section_speed_kmph(lat1, lon1, t1_iso, lat2, lon2, t2_iso)
    violated = speed is not None and speed > limit_kmph
    return {
        "violation": violated,
        "type": "SECTION_SPEED" if violated else "NONE",
        "details": {"section_speed_kmph": speed, "limit_kmph": limit_kmph},
    }


# ---------------------------------------------------------------------------
# 13. Red-light violation detection.
# ---------------------------------------------------------------------------

def check_red_light_violation(crossing_time_iso: str, red_start_time_iso: str, grace_period_s: float = 1.0) -> Dict:
    """
    crossing_time_iso: when the vehicle was detected crossing the stop line.
    red_start_time_iso: when the signal for that approach turned red.
    grace_period_s: a short allowance for vehicles already committed to the
    intersection when the light changed (standard practice — the very
    instant of red isn't a hard cutoff for a vehicle already mid-crossing).
    """
    try:
        crossing = datetime.fromisoformat(crossing_time_iso)
        red_start = datetime.fromisoformat(red_start_time_iso)
    except (ValueError, TypeError):
        return {"violation": False, "type": "NONE", "details": {"error": "invalid timestamp"}}

    elapsed = (crossing - red_start).total_seconds()
    violated = elapsed > grace_period_s
    return {
        "violation": violated,
        "type": "RED_LIGHT_VIOLATION" if violated else "NONE",
        "details": {"seconds_after_red": round(elapsed, 2), "grace_period_s": grace_period_s},
    }


# ---------------------------------------------------------------------------
# 14. Wrong-way detection.
# ---------------------------------------------------------------------------

def check_wrong_way(observed_direction: str, allowed_directions: List[str]) -> Dict:
    """observed_direction / allowed_directions use compass points ("N","S","E","W") or lane-tagged equivalents."""
    allowed_set = {d.strip().upper() for d in allowed_directions}
    observed = (observed_direction or "").strip().upper()
    violated = bool(observed) and observed not in allowed_set
    return {
        "violation": violated,
        "type": "WRONG_WAY" if violated else "NONE",
        "details": {"observed_direction": observed, "allowed_directions": sorted(allowed_set)},
    }


# ---------------------------------------------------------------------------
# 15. Heavy-vehicle time-window enforcement.
# ---------------------------------------------------------------------------

HEAVY_VEHICLE_CLASSES = {"truck", "heavy", "bus", "lorry", "trailer"}


def _in_time_window(t: _time, window: Tuple[_time, _time]) -> bool:
    start, end = window
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end  # window wraps past midnight, e.g. 23:00-06:00


def check_heavy_vehicle_window(
    vehicle_class: str, timestamp_iso: str, allowed_windows: List[Tuple[_time, _time]]
) -> Dict:
    """
    allowed_windows: list of (start_time, end_time) the vehicle class IS
    permitted to be on this road (e.g. [(time(23,0), time(6,0))] for a
    night-only heavy-vehicle corridor). A non-heavy vehicle_class never
    violates this check regardless of time.
    """
    is_heavy = (vehicle_class or "").strip().lower() in HEAVY_VEHICLE_CLASSES
    if not is_heavy:
        return {"violation": False, "type": "NONE", "details": {"vehicle_class": vehicle_class, "is_heavy": False}}

    try:
        t = datetime.fromisoformat(timestamp_iso).time()
    except (ValueError, TypeError):
        return {"violation": False, "type": "NONE", "details": {"error": "invalid timestamp"}}

    permitted = any(_in_time_window(t, w) for w in allowed_windows)
    violated = not permitted
    return {
        "violation": violated,
        "type": "HEAVY_VEHICLE_RESTRICTED_HOURS" if violated else "NONE",
        "details": {"vehicle_class": vehicle_class, "time": t.isoformat()},
    }


# ---------------------------------------------------------------------------
# 16. School-zone timed speed limits.
# ---------------------------------------------------------------------------

def check_school_zone_speed(
    speed_kmph: float, timestamp_iso: str, zone_schedule: List[Tuple[_time, _time, float]]
) -> Dict:
    """
    zone_schedule: list of (start_time, end_time, limit_kmph) — e.g. a
    school zone that drops the limit to 25 km/h only during
    [(time(7,30), time(9,0), 25.0), (time(14,0), time(15,30), 25.0)].
    Outside all listed windows, this check never fires (normal limits apply
    elsewhere and aren't this function's concern).
    """
    try:
        t = datetime.fromisoformat(timestamp_iso).time()
    except (ValueError, TypeError):
        return {"violation": False, "type": "NONE", "details": {"error": "invalid timestamp"}}

    for start, end, limit in zone_schedule:
        if _in_time_window(t, (start, end)) and speed_kmph > limit:
            return {
                "violation": True,
                "type": "SCHOOL_ZONE_SPEED",
                "details": {"speed_kmph": speed_kmph, "limit_kmph": limit, "time": t.isoformat()},
            }
    return {"violation": False, "type": "NONE", "details": {"speed_kmph": speed_kmph, "time": t.isoformat()}}


# ---------------------------------------------------------------------------
# 17. Junction-mouth parking detection.
# ---------------------------------------------------------------------------

def check_junction_mouth_parking(dwell_seconds: float, threshold_seconds: float = 90.0) -> Dict:
    """A vehicle stationary in a junction-mouth no-parking zone for longer than threshold_seconds."""
    violated = dwell_seconds > threshold_seconds
    return {
        "violation": violated,
        "type": "JUNCTION_MOUTH_PARKING" if violated else "NONE",
        "details": {"dwell_seconds": round(dwell_seconds, 1), "threshold_seconds": threshold_seconds},
    }
