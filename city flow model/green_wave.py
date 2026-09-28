"""
green_wave.py — Corridor selection, short-term queue prediction, and
green-light speed advisory (Features 5, 6, 8).
===========================================================================
Pure logic, no CityFlow/Flask dependency, independently unit-testable.
Used by agent.py (TrafficAgent for prediction/advisory, MultiAgentCoordinator
for corridor selection).
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Feature 5: automatic green waves — pick the corridor(s) to synchronise from
# origin-destination flow data.
# ---------------------------------------------------------------------------

def select_corridors(
    od_flow: Sequence[Dict],
    adjacency: Dict[str, Dict[str, str]],
    top_n: int = 1,
) -> List[Dict]:
    """
    od_flow: rows like {"origin": "J1", "destination": "J4", "trips": 42}
             (same shape as database.get_od_patterns()/analytics.get_od_flow()).
    adjacency: {junction_id: {neighbor_id: phase_name_facing_that_neighbor}},
               i.e. which of a junction's own phases points at each neighbor —
               this is exactly agent.TrafficAgent.outgoing_neighbors inverted.
    Returns up to top_n corridors, each:
        {"junctions": [...], "phase_sequence": [...], "trips": total_trips}
    A corridor is only returned if every hop in it is actually a direct
    neighbor link (so we never "invent" a green wave through junctions that
    aren't really connected in sequence).
    """
    scored = []
    for row in od_flow:
        origin, dest, trips = row.get("origin"), row.get("destination"), row.get("trips", 0)
        if not origin or not dest or origin == dest:
            continue
        path = _shortest_adjacency_path(origin, dest, adjacency)
        if path and len(path) >= 2:
            scored.append((trips, path))

    scored.sort(key=lambda t: t[0], reverse=True)

    corridors = []
    seen_paths = set()
    for trips, path in scored:
        key = tuple(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        phase_sequence = []
        for i in range(len(path) - 1):
            phase_sequence.append(adjacency[path[i]][path[i + 1]])
        corridors.append({"junctions": path, "phase_sequence": phase_sequence, "trips": trips})
        if len(corridors) >= top_n:
            break
    return corridors


def _shortest_adjacency_path(origin: str, dest: str, adjacency: Dict[str, Dict[str, str]]) -> Optional[List[str]]:
    """BFS shortest hop path from origin to dest using only direct adjacency links."""
    if origin not in adjacency:
        return None
    frontier = deque([[origin]])
    visited = {origin}
    while frontier:
        path = frontier.popleft()
        node = path[-1]
        if node == dest:
            return path
        for neighbor in adjacency.get(node, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(path + [neighbor])
    return None


def corridor_offsets(corridor: Dict, link_distances_m: Dict[Tuple[str, str], float], travel_speed_kmph: float = 40.0) -> List[float]:
    """
    Seconds each junction in the corridor should start its green after the
    first one, so a vehicle leaving the first junction on green hits every
    following one on green too (classic green-wave offset calculation).
    """
    speed_mps = max(0.1, travel_speed_kmph / 3.6)
    junctions = corridor["junctions"]
    offsets = [0.0]
    cumulative = 0.0
    for i in range(len(junctions) - 1):
        pair = (junctions[i], junctions[i + 1])
        dist = link_distances_m.get(pair) or link_distances_m.get((pair[1], pair[0])) or 0.0
        cumulative += dist / speed_mps
        offsets.append(round(cumulative, 1))
    return offsets


# ---------------------------------------------------------------------------
# Feature 6: short-term queue prediction (~5 minutes ahead).
# ---------------------------------------------------------------------------

class QueuePredictor:
    """
    Keeps a short rolling history of queue-length samples (one per call to
    `record`, expected to be spaced `sample_interval_s` apart — matching the
    coordinator's step cadence) and extrapolates a linear trend forward.
    Deliberately simple (least-squares line fit) rather than a learned model:
    predictable, explainable, and cheap enough to run every step.
    """

    def __init__(self, history_len: int = 20, sample_interval_s: float = 3.0):
        self.history_len = history_len
        self.sample_interval_s = sample_interval_s
        self._samples: deque = deque(maxlen=history_len)

    def record(self, queue_length: float) -> None:
        self._samples.append(float(queue_length))

    def predict(self, seconds_ahead: float = 300.0) -> float:
        """Predicted queue length `seconds_ahead` from now. Never negative."""
        n = len(self._samples)
        if n == 0:
            return 0.0
        if n == 1:
            return max(0.0, self._samples[0])

        # Least-squares slope/intercept over the sample index (x = 0..n-1).
        xs = list(range(n))
        ys = list(self._samples)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = (num / den) if den else 0.0

        steps_ahead = seconds_ahead / self.sample_interval_s
        predicted = ys[-1] + slope * steps_ahead
        return round(max(0.0, predicted), 2)

    def trend(self) -> str:
        """Cheap human-readable label for dashboards."""
        if len(self._samples) < 2:
            return "STABLE"
        delta = self._samples[-1] - self._samples[0]
        if delta > max(1.0, 0.15 * max(self._samples)):
            return "RISING"
        if delta < -max(1.0, 0.15 * max(self._samples)):
            return "FALLING"
        return "STABLE"

    def reset(self) -> None:
        self._samples.clear()


# ---------------------------------------------------------------------------
# Feature 8: green-light speed advisory (GLOSA), shown on the VMS.
# ---------------------------------------------------------------------------

def green_speed_advisory(
    distance_m: float,
    time_to_green_s: float,
    speed_limit_kmph: float = 50.0,
    min_kmph: float = 15.0,
) -> Dict:
    """
    Recommended approach speed so a driver `distance_m` from the stop line
    arrives right as the light turns green, without exceeding the speed
    limit or being told to crawl below `min_kmph`.

    Returns {"advisory_kmph": float|None, "arrives_on_red": bool, "message": str}.
    `advisory_kmph` is None when arriving exactly at green isn't achievable
    within the limits (already arriving on green anytime soon, or would need
    to exceed the speed limit) — the caller should then just show the limit.
    """
    if distance_m <= 0:
        return {"advisory_kmph": None, "arrives_on_red": False, "message": "At the stop line."}

    if time_to_green_s <= 0:
        return {"advisory_kmph": None, "arrives_on_red": False, "message": "Light is already green — proceed at the posted limit."}

    required_mps = distance_m / time_to_green_s
    required_kmph = required_mps * 3.6

    if required_kmph > speed_limit_kmph:
        # Can't slow down enough to *not* beat the light within the limit —
        # you'll arrive before it turns green no matter what (within legal speed).
        return {
            "advisory_kmph": round(speed_limit_kmph, 1),
            "arrives_on_red": True,
            "message": f"Maintain {round(speed_limit_kmph)} km/h — you'll still reach the signal before it turns green.",
        }

    if required_kmph < min_kmph:
        return {
            "advisory_kmph": round(min_kmph, 1),
            "arrives_on_red": False,
            "message": f"Ease to {round(min_kmph)} km/h — plenty of time before green.",
        }

    return {
        "advisory_kmph": round(required_kmph, 1),
        "arrives_on_red": False,
        "message": f"Hold {round(required_kmph)} km/h for a green light at the junction.",
    }
