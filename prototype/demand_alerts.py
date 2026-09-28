"""
demand_alerts.py — Feature 23: unusual-demand alerts.
===========================================================================
"Unusual-demand alerts: festivals, rallies, matches." Flags a time window
whose traffic volume is a statistical outlier against its own recent
baseline — a z-score test, deliberately simple and explainable (no ML
model to justify), which is exactly what "this doesn't look like a normal
day" needs to be.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

import statistics
from typing import Dict, List, Optional, Tuple


def rolling_baseline(history: List[float]) -> Tuple[float, float]:
    """Mean and (sample) standard deviation of a history of past period counts. std is 0.0 for <2 samples."""
    if not history:
        return 0.0, 0.0
    mean = statistics.mean(history)
    std = statistics.stdev(history) if len(history) >= 2 else 0.0
    return round(mean, 2), round(std, 2)


def detect_unusual_demand(current_count: float, history: List[float], z_threshold: float = 2.5) -> Dict:
    """
    current_count: this period's unique-vehicle count (or detections/min,
    whatever unit `history` also uses).
    history: recent past periods' counts for the same time-of-day/location,
    e.g. the last several same-weekday same-hour readings.

    A flat/near-flat baseline (std ~0) is common for a quiet camera and
    shouldn't fire on any tiny bump, so std is floored at 1.0 count before
    dividing — this keeps the z-score from exploding on essentially noise.
    """
    mean, std = rolling_baseline(history)
    effective_std = max(std, 1.0)
    z_score = round((current_count - mean) / effective_std, 2)
    unusual = z_score >= z_threshold
    return {
        "unusual_demand": unusual,
        "z_score": z_score,
        "current_count": current_count,
        "baseline_mean": mean,
        "baseline_std": std,
        "z_threshold": z_threshold,
    }


def check_unusual_demand_for_cameras(
    current_counts: Dict[str, float], histories: Dict[str, List[float]], z_threshold: float = 2.5
) -> List[Dict]:
    """Batch version over several cameras/zones; returns only the ones flagged, worst z-score first."""
    flagged = []
    for camera_id, count in current_counts.items():
        result = detect_unusual_demand(count, histories.get(camera_id, []), z_threshold)
        if result["unusual_demand"]:
            flagged.append({"camera_id": camera_id, **result})
    flagged.sort(key=lambda r: r["z_score"], reverse=True)
    return flagged
