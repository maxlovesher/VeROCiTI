"""
travel_time_analytics.py — Feature 22: travel-time reliability and
bottleneck reports.
===========================================================================
"Travel-time reliability and bottleneck reports: median and 95th-percentile
travel times per road."

Pure logic, no DB/Flask dependency, independently unit-testable. Feeds from
whatever already computes per-hop travel times — e.g.
analytics.enrich_trajectory()'s "travel_minutes"/"est_speed_kmph" per road,
or ambulance.py-style ETA links — grouped by road name.
"""

import math
from typing import Dict, List


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]), no numpy needed. 0.0 for an empty list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    fraction = rank - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * fraction, 2)


def road_travel_time_stats(travel_times_by_road: Dict[str, List[float]]) -> Dict[str, Dict]:
    """
    travel_times_by_road: {"road_name": [minutes, minutes, ...], ...}.
    Returns per road: median, p95, sample_size, and a reliability_ratio
    (p95 / median — the "planning time index" style buffer a driver would
    need to budget for; 1.0 means perfectly consistent, higher is less
    reliable). Roads with fewer than 2 samples get reliability_ratio None
    (not enough data to say anything about consistency).
    """
    stats = {}
    for road, times in travel_times_by_road.items():
        clean_times = [t for t in times if isinstance(t, (int, float)) and t >= 0]
        if not clean_times:
            stats[road] = {"median": 0.0, "p95": 0.0, "sample_size": 0, "reliability_ratio": None}
            continue
        median = percentile(clean_times, 50)
        p95 = percentile(clean_times, 95)
        reliability_ratio = round(p95 / median, 2) if len(clean_times) >= 2 and median > 0 else None
        stats[road] = {
            "median": median, "p95": p95, "sample_size": len(clean_times), "reliability_ratio": reliability_ratio,
        }
    return stats


def identify_bottlenecks(stats: Dict[str, Dict], min_samples: int = 3, reliability_threshold: float = 1.5) -> List[Dict]:
    """
    Roads whose reliability_ratio exceeds reliability_threshold (i.e. the
    95th-percentile trip takes meaningfully longer than the typical one —
    a real bottleneck, not just a naturally slow-but-consistent road),
    sorted worst-first. Roads without enough samples are excluded rather
    than flagged, since a spike from 2 samples isn't a reliable signal.
    """
    bottlenecks = [
        {"road": road, **road_stats}
        for road, road_stats in stats.items()
        if road_stats.get("sample_size", 0) >= min_samples
        and road_stats.get("reliability_ratio") is not None
        and road_stats["reliability_ratio"] >= reliability_threshold
    ]
    bottlenecks.sort(key=lambda r: r["reliability_ratio"], reverse=True)
    return bottlenecks
