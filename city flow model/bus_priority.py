"""
bus_priority.py — Conditional transit signal priority (Feature 7).
===========================================================================
"Conditional bus priority: only for buses running late." Unlike the
ambulance corridor (agent.py's emergency_override, which fully preempts a
junction), a late bus only gets a bounded priority *boost* toward its
approach's normal priority score — it competes more favourably, it doesn't
seize the junction outright. An on-time bus gets nothing: that's the whole
point of "conditional".

Pure logic, no CityFlow/Flask dependency, independently unit-testable.
"""

from typing import Optional


def lateness_minutes(scheduled_minutes: float, predicted_minutes: float) -> float:
    """
    scheduled_minutes / predicted_minutes: minutes-from-now the bus was
    timetabled to reach its next timing point vs. when it's actually
    predicted to (e.g. from GPS + current speed). Positive = running late.
    """
    return predicted_minutes - scheduled_minutes


def is_priority_warranted(lateness_min: float, threshold_min: float = 3.0) -> bool:
    """A bus only qualifies for priority once it's at least `threshold_min` behind schedule."""
    return lateness_min >= threshold_min


def priority_boost(lateness_min: float, threshold_min: float = 3.0, max_boost: float = 0.5, cap_lateness_min: float = 15.0) -> float:
    """
    0.0 if not late enough to qualify. Otherwise scales linearly from 0 at
    the threshold up to `max_boost` at `cap_lateness_min` late, then holds
    at max_boost (a bus that's 30 minutes late doesn't get an ever-growing
    boost — it gets the same maximum priority as one that's 15 minutes late).
    """
    if not is_priority_warranted(lateness_min, threshold_min):
        return 0.0
    span = max(0.001, cap_lateness_min - threshold_min)
    fraction = min(1.0, (lateness_min - threshold_min) / span)
    return round(max_boost * fraction, 3)


class BusPriorityRegistry:
    """
    Tracks active bus priority requests per phase for one junction agent.
    Call `update` each time a bus's schedule/prediction is refreshed (or to
    clear it once it has passed / priority is no longer needed), and
    `boost_for_phase` from compute_priority to fold the result into the
    normal priority score.
    """

    def __init__(self, threshold_min: float = 3.0, max_boost: float = 0.5, cap_lateness_min: float = 15.0):
        self.threshold_min = threshold_min
        self.max_boost = max_boost
        self.cap_lateness_min = cap_lateness_min
        self._requests = {}   # bus_id -> {"phase_name": str, "lateness_min": float}

    def update(self, bus_id: str, phase_name: str, scheduled_minutes: float, predicted_minutes: float) -> bool:
        """Returns True if this bus now qualifies for priority."""
        late = lateness_minutes(scheduled_minutes, predicted_minutes)
        warranted = is_priority_warranted(late, self.threshold_min)
        if warranted:
            self._requests[bus_id] = {"phase_name": phase_name, "lateness_min": late}
        else:
            self._requests.pop(bus_id, None)
        return warranted

    def clear(self, bus_id: str) -> None:
        self._requests.pop(bus_id, None)

    def clear_all(self) -> None:
        self._requests.clear()

    def boost_for_phase(self, phase_name: str) -> float:
        """The largest boost any currently-qualifying bus contributes to this phase."""
        best = 0.0
        for req in self._requests.values():
            if req["phase_name"] != phase_name:
                continue
            boost = priority_boost(req["lateness_min"], self.threshold_min, self.max_boost, self.cap_lateness_min)
            best = max(best, boost)
        return best

    def active_requests_for_phase(self, phase_name: str) -> int:
        return sum(1 for req in self._requests.values() if req["phase_name"] == phase_name)
