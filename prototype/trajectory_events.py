"""
trajectory_events.py — Feature 2 (Foundations): "Trajectory events with
hashed IDs: {hashed_id, camera, time} events stitched into trajectories."
===========================================================================
Operates purely on hashed identities (identity.hash_id), never raw plate
text — the privacy-preserving counterpart to database.get_trajectory(),
which works on plain plates. Use build_trajectories() wherever a feature
needs "this vehicle's path through the camera network" without needing to
know which vehicle it actually is.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

from datetime import datetime
from typing import Dict, Iterable, List

import identity


def build_trajectories(events: Iterable[dict]) -> Dict[str, List[dict]]:
    """
    events: iterable of {"hashed_id", "camera", "time"} (identity.hashed_event
    shape; "time" as an ISO-8601 string). Groups by hashed_id and returns
    each group's events sorted chronologically — one trajectory per identity.
    Events missing hashed_id/camera/time are skipped rather than raising,
    since this typically runs over live, occasionally-messy sensor data.
    """
    grouped: Dict[str, List[dict]] = {}
    for event in events:
        hashed_id = event.get("hashed_id")
        camera = event.get("camera")
        time_iso = event.get("time")
        if not hashed_id or not camera or not time_iso:
            continue
        grouped.setdefault(hashed_id, []).append(dict(event))

    for hashed_id, trajectory in grouped.items():
        trajectory.sort(key=lambda e: e["time"])

    return grouped


def trajectory_for(events: Iterable[dict], hashed_id: str) -> List[dict]:
    """Convenience: just the one identity's chronological trajectory."""
    return build_trajectories(events).get(hashed_id, [])


def events_from_detections(detections: Iterable[dict], day=None, secret: str = None) -> List[dict]:
    """
    Bridges plate-based detection rows (database.get_recent_detections() /
    get_trajectory() shape: {"plate", "camera_id", "timestamp", ...}) into
    hashed {hashed_id, camera, time} events, without ever persisting the
    plate alongside the hash. This is the on-ramp real camera ingestion
    would use to build privacy-preserving trajectories from ANPR reads.
    """
    events = []
    for row in detections:
        plate = row.get("plate")
        camera_id = row.get("camera_id")
        timestamp = row.get("timestamp")
        if not plate or not camera_id or not timestamp:
            continue
        events.append(identity.hashed_event(identity.hash_id(plate, day, secret), camera_id, timestamp))
    return events


def dwell_seconds(trajectory: List[dict]) -> float:
    """Total elapsed time across one identity's trajectory, in seconds. 0 for a single-point or empty trajectory."""
    if len(trajectory) < 2:
        return 0.0
    try:
        start = datetime.fromisoformat(trajectory[0]["time"])
        end = datetime.fromisoformat(trajectory[-1]["time"])
        return max(0.0, (end - start).total_seconds())
    except (ValueError, KeyError):
        return 0.0
