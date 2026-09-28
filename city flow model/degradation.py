"""
degradation.py — Health-driven degradation ladder controller (Feature 9).
===========================================================================
Tracks a stream of "was this step's data good?" signals and decides which
rung of the ladder (agent.TrafficAgent.CONTROL_MODES) the whole junction
network should be running at. Downgrades one rung at a time after a run of
bad steps and upgrades one rung at a time after a run of good steps
(hysteresis), so a single flaky reading can't cause a full jump straight to
FIXED_TIME and a single good reading can't snap it straight back to FULL_AI.

Pure logic, no CityFlow/Flask dependency — used by agent.MultiAgentCoordinator
and independently unit-testable.
"""

from typing import List, Optional

RUNGS: List[str] = ["FULL_AI", "HISTORICAL_PROFILE", "LOCAL_ACTUATED", "FIXED_TIME"]


class DegradationLadder:
    def __init__(self, downgrade_after: int = 3, upgrade_after: int = 10):
        """
        downgrade_after: consecutive bad steps before dropping one rung.
        upgrade_after:   consecutive good steps before climbing back one rung.
        Upgrade takes longer than downgrade on purpose — fail fast, recover cautiously.
        """
        self.downgrade_after = downgrade_after
        self.upgrade_after = upgrade_after
        self._rung_index = 0
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._forced: Optional[str] = None

    @property
    def current(self) -> str:
        if self._forced is not None:
            return self._forced
        return RUNGS[self._rung_index]

    def force(self, rung: Optional[str]) -> str:
        """Pin the ladder to a specific rung (e.g. an operator override), or
        pass None to release the pin and resume automatic health tracking."""
        if rung is not None and rung not in RUNGS:
            raise ValueError(f"Unknown rung '{rung}', expected one of {RUNGS}")
        self._forced = rung
        return self.current

    def record_health(self, ok: bool) -> str:
        """Feed in one step's health signal; returns the (possibly new) current rung."""
        if ok:
            self._consecutive_bad = 0
            self._consecutive_good += 1
            if self._consecutive_good >= self.upgrade_after and self._rung_index > 0:
                self._rung_index -= 1
                self._consecutive_good = 0
        else:
            self._consecutive_good = 0
            self._consecutive_bad += 1
            if self._consecutive_bad >= self.downgrade_after and self._rung_index < len(RUNGS) - 1:
                self._rung_index += 1
                self._consecutive_bad = 0

        return self.current

    def reset(self) -> None:
        self._rung_index = 0
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._forced = None
