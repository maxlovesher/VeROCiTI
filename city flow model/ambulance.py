"""
ambulance.py — Smarter ambulance corridor, corridor-level logic (Feature 11).
===========================================================================
Covers the two pieces of Feature 11 that operate above a single junction:
  - Hospital-aware routing: pick the nearest hospital and the route to it.
  - ETA-based triggering with an early-start allowance for draining the
    queue already sitting on the ambulance's approach, so the junction
    isn't just "green when the ambulance arrives" but "already clearing
    itself out of the way beforehand".

The other two pieces of Feature 11 — amber+all-red before any switch, and
the starved-first recovery phase once the ambulance has passed — are
per-junction signal behaviour and live in agent.TrafficAgent
(_initiate_switch's all_red option, set_emergency's recovery trigger,
compute_priority's recovery boost).

Pure logic, no CityFlow/Flask dependency, independently unit-testable.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


def shortest_path(origin: str, dest: str, adjacency: Dict[str, Dict[str, str]]) -> Optional[List[str]]:
    """BFS shortest hop path from origin to dest using only direct adjacency links."""
    if origin == dest:
        return [origin]
    if origin not in adjacency:
        return None
    frontier = deque([[origin]])
    visited = {origin}
    while frontier:
        path = frontier.popleft()
        node = path[-1]
        for neighbor in adjacency.get(node, {}):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == dest:
                return new_path
            visited.add(neighbor)
            frontier.append(new_path)
    return None


def nearest_hospital(
    origin: str, hospitals: Dict[str, str], adjacency: Dict[str, Dict[str, str]]
) -> Optional[Tuple[str, List[str]]]:
    """
    hospitals: {hospital_name: nearest_junction_id}.
    Returns (hospital_name, route_junctions) for whichever hospital is
    reachable in the fewest hops from `origin`, or None if none are reachable.
    """
    best = None
    for hospital_name, junction in hospitals.items():
        path = shortest_path(origin, junction, adjacency)
        if path and (best is None or len(path) < len(best[1])):
            best = (hospital_name, path)
    return best


def eta_seconds(distance_remaining_m: float, speed_mps: float) -> float:
    return max(0.0, distance_remaining_m) / max(0.1, speed_mps)


def should_trigger(eta_s: float, lead_time_s: float, queue_length: int = 0, discharge_headway_s: float = 2.0) -> bool:
    """
    True once it's time to preempt this junction. The required lead time
    isn't just "long enough to run the yellow/all-red sequence" — it also
    budgets `discharge_headway_s` per vehicle already queued on the
    ambulance's approach, so the junction gets an early start on clearing
    that queue out of the ambulance's path before it actually arrives.
    """
    required_lead = lead_time_s + queue_length * discharge_headway_s
    return eta_s <= required_lead
