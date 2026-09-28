"""
test_degradation_ladder.py — offline unit tests for Feature 9.

Covers:
  - degradation.DegradationLadder: hysteresis on the way down and up, force/release, reset.
  - agent.TrafficAgent per-rung decision logic (FIXED_TIME, HISTORICAL_PROFILE, LOCAL_ACTUATED).
  - agent.MultiAgentCoordinator.step(data_ok=...) driving the ladder end to end.

Pure-logic tests: no CityFlow engine or Flask server required. Run with:
    python "city flow model/scripts/test_degradation_ladder.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degradation import DegradationLadder, RUNGS
from agent import TrafficAgent, MultiAgentCoordinator


class FakeEngine:
    def __init__(self, lane_vehicles=None, lane_waiting=None, current_time=100):
        self._lane_vehicles = lane_vehicles or {}
        self._lane_waiting = lane_waiting or {}
        self._current_time = current_time
        self.phase_calls = []

    def get_lane_waiting_vehicle_count(self):
        return dict(self._lane_waiting)

    def get_lane_vehicle_count(self):
        return dict(self._lane_vehicles)

    def get_current_time(self):
        return self._current_time

    def set_tl_phase(self, junction_id, phase_idx):
        self.phase_calls.append((junction_id, phase_idx))

    def get_vehicles(self, include_waiting=True):
        return []

    def next_step(self):
        self._current_time += 1

    def reset(self):
        self._current_time = 0


def make_agent(engine, outgoing_neighbors=None):
    return TrafficAgent(
        agent_id="Agent-TEST",
        junction_id="JT",
        engine=engine,
        incoming_roads={"EW": ["road_A"], "NS": ["road_B"]},
        outgoing_neighbors=outgoing_neighbors or {"EW": "JDOWN", "NS": None},
    )


class TestDegradationLadderClass(unittest.TestCase):
    def test_starts_at_full_ai(self):
        self.assertEqual(DegradationLadder().current, "FULL_AI")

    def test_downgrades_one_rung_after_threshold_bad_steps(self):
        ladder = DegradationLadder(downgrade_after=3, upgrade_after=10)
        for _ in range(2):
            self.assertEqual(ladder.record_health(False), "FULL_AI")  # not yet
        self.assertEqual(ladder.record_health(False), "HISTORICAL_PROFILE")  # 3rd bad step

    def test_does_not_skip_rungs_in_one_go(self):
        ladder = DegradationLadder(downgrade_after=2, upgrade_after=10)
        for _ in range(20):
            ladder.record_health(False)
        # Even after a long run of bad steps, it must have walked the ladder
        # one rung at a time and be resting at the bottom, not skip past it.
        self.assertEqual(ladder.current, RUNGS[-1])

    def test_a_single_good_step_does_not_cancel_a_bad_streak_below_threshold(self):
        ladder = DegradationLadder(downgrade_after=3, upgrade_after=10)
        ladder.record_health(False)
        ladder.record_health(False)
        ladder.record_health(True)   # resets the bad streak
        ladder.record_health(False)
        ladder.record_health(False)
        self.assertEqual(ladder.current, "FULL_AI")  # streak was reset, only 2 in a row now

    def test_upgrades_one_rung_after_threshold_good_steps(self):
        ladder = DegradationLadder(downgrade_after=1, upgrade_after=3)
        ladder.record_health(False)  # -> HISTORICAL_PROFILE
        self.assertEqual(ladder.current, "HISTORICAL_PROFILE")
        for _ in range(2):
            self.assertEqual(ladder.record_health(True), "HISTORICAL_PROFILE")
        self.assertEqual(ladder.record_health(True), "FULL_AI")

    def test_force_pins_and_release_resumes_tracking(self):
        ladder = DegradationLadder(downgrade_after=1, upgrade_after=1)
        ladder.force("FIXED_TIME")
        self.assertEqual(ladder.current, "FIXED_TIME")
        ladder.record_health(True)  # should not move the pinned rung
        self.assertEqual(ladder.current, "FIXED_TIME")
        ladder.force(None)
        self.assertEqual(ladder.current, "FULL_AI")  # underlying state untouched by the pin

    def test_force_rejects_unknown_rung(self):
        with self.assertRaises(ValueError):
            DegradationLadder().force("NOT_A_REAL_RUNG")


class TestTrafficAgentControlModes(unittest.TestCase):
    def test_fixed_time_ignores_priority_and_cycles_on_fixed_duration(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 14, "road_B_0": 0},   # EW totally jammed, NS empty
            lane_waiting={"road_A_0": 14, "road_B_0": 0},
        )
        agent = make_agent(engine)
        agent.set_control_mode("FIXED_TIME")

        last_reason = ""
        for _ in range(TrafficAgent.FIXED_TIME_GREEN):
            agent.observe({})
            last_reason = agent.decide_and_act({})

        # Despite EW being saturated (which would normally hold priority-based
        # green), fixed-time must switch strictly on the clock.
        self.assertIn("FIXED_TIME", last_reason)
        self.assertTrue(agent.is_yellow)

    def test_historical_profile_follows_configured_split_not_live_load(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 14, "road_B_0": 0},
            lane_waiting={"road_A_0": 14, "road_B_0": 0},
        )
        agent = make_agent(engine)
        agent.set_control_mode("HISTORICAL_PROFILE")
        agent.set_historical_profile({"EW": 0.2, "NS": 0.8})  # EW should get a SHORT green historically

        ew_target = round(0.2 * TrafficAgent.HISTORICAL_CYCLE_LENGTH)
        last_reason = ""
        for _ in range(max(ew_target, TrafficAgent.MIN_GREEN)):
            agent.observe({})
            last_reason = agent.decide_and_act({})

        self.assertIn("HISTORICAL_PROFILE", last_reason)
        self.assertTrue(agent.is_yellow)  # switched away from EW on schedule, not because of its huge queue

    def test_historical_profile_still_respects_hard_starvation_cap(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 1, "road_B_0": 1}, lane_waiting={"road_A_0": 0, "road_B_0": 0})
        agent = make_agent(engine)
        agent.set_control_mode("HISTORICAL_PROFILE")
        agent.set_historical_profile({"EW": 0.9, "NS": 0.1})  # NS would normally wait a long time

        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            agent.decide_and_act({})

        # Override the *derived* observation directly: observe() would
        # otherwise recompute waiting_time_tracker from the (zero) raw
        # waiting count on its next call and stomp this override.
        agent.local_obs.setdefault("NS", {})["waiting_time"] = TrafficAgent.STARVATION_HARD_CAP + 1
        reason = agent.decide_and_act({})
        self.assertIn("Hard starvation cap", reason)
        self.assertTrue(agent.is_yellow)

    def test_local_actuated_ignores_downstream_spillback(self):
        # Same scenario that triggers spillback prevention under FULL_AI...
        engine = FakeEngine(lane_vehicles={"road_A_0": 5, "road_B_0": 1}, lane_waiting={"road_A_0": 1, "road_B_0": 0})
        agent = make_agent(engine)
        agent.set_control_mode("LOCAL_ACTUATED")
        neighbor_states = {"JDOWN": {"overall_density": 0.99}}  # exit road full

        last_reason = ""
        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            last_reason = agent.decide_and_act(neighbor_states)

        # ...must NOT trigger under LOCAL_ACTUATED, which has no downstream awareness.
        self.assertNotIn("Spillback prevention", last_reason)


class TestCoordinatorDegradationIntegration(unittest.TestCase):
    def _build_coordinator(self):
        engine = FakeEngine(lane_vehicles={}, lane_waiting={})
        coordinator = MultiAgentCoordinator(engine)
        return coordinator

    def test_repeated_bad_data_pushes_all_agents_down_the_ladder(self):
        coordinator = self._build_coordinator()
        for _ in range(coordinator.degradation.downgrade_after):
            coordinator.step({}, data_ok=False)

        self.assertEqual(coordinator.degradation.current, "HISTORICAL_PROFILE")
        for agent in coordinator.agents.values():
            self.assertEqual(agent.control_mode, "HISTORICAL_PROFILE")

    def test_reset_restores_full_ai(self):
        coordinator = self._build_coordinator()
        for _ in range(coordinator.degradation.downgrade_after):
            coordinator.step({}, data_ok=False)
        self.assertNotEqual(coordinator.degradation.current, "FULL_AI")

        coordinator.reset()
        self.assertEqual(coordinator.degradation.current, "FULL_AI")
        for agent in coordinator.agents.values():
            self.assertEqual(agent.control_mode, "FULL_AI")

    def test_default_step_call_keeps_full_ai(self):
        coordinator = self._build_coordinator()
        for _ in range(20):
            coordinator.step({})   # data_ok defaults to True
        self.assertEqual(coordinator.degradation.current, "FULL_AI")


if __name__ == "__main__":
    unittest.main(verbosity=2)
