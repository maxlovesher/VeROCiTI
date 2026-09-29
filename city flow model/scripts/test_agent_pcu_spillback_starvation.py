"""
test_agent_pcu_spillback_starvation.py — offline unit tests for agent.py

Covers:
  Feature 1  — PCU-weighted queue measurement (pcu.py + TrafficAgent.observe)
  Feature 3  — Spillback / box-blocking prevention
  Feature 4  — Starvation hard cap

Pure-logic tests: no CityFlow engine or Flask server required. Run with:
    python "city flow model/scripts/test_agent_pcu_spillback_starvation.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pcu
from agent import TrafficAgent


class FakeEngine:
    """Minimal stand-in for the CityFlow engine, just what TrafficAgent touches."""

    def __init__(self, lane_vehicles=None, lane_waiting=None, lane_vehicle_ids=None, current_time=100):
        self._lane_vehicles = lane_vehicles or {}
        self._lane_waiting = lane_waiting or {}
        self._lane_vehicle_ids = lane_vehicle_ids  # None => engine doesn't support get_lane_vehicles
        self._current_time = current_time
        self.phase_calls = []

        if lane_vehicle_ids is not None:
            # Only expose get_lane_vehicles when the test wants it available,
            # so the "engine without this API" fallback path is exercised too.
            self.get_lane_vehicles = self._get_lane_vehicles

    def _get_lane_vehicles(self):
        return self._lane_vehicle_ids

    def get_lane_waiting_vehicle_count(self):
        return dict(self._lane_waiting)

    def get_lane_vehicle_count(self):
        return dict(self._lane_vehicles)

    def get_current_time(self):
        return self._current_time

    def set_tl_phase(self, junction_id, phase_idx):
        self.phase_calls.append((junction_id, phase_idx))


def make_agent(engine, outgoing_neighbors=None):
    return TrafficAgent(
        agent_id="Agent-TEST",
        junction_id="JT",
        engine=engine,
        incoming_roads={"EW": ["road_A"], "NS": ["road_B"]},
        outgoing_neighbors=outgoing_neighbors or {"EW": "JDOWN", "NS": None},
    )


class TestPcuWeighting(unittest.TestCase):
    def test_pcu_weight_lookup(self):
        self.assertEqual(pcu.pcu_weight("car"), 1.0)
        self.assertEqual(pcu.pcu_weight("Bus"), 3.0)
        self.assertEqual(pcu.pcu_weight("motorbike"), 0.5)
        self.assertEqual(pcu.pcu_weight(None), pcu.DEFAULT_PCU)
        self.assertEqual(pcu.pcu_weight("nonsense_type"), pcu.DEFAULT_PCU)

    def test_vehicle_type_from_flow_index(self):
        pcu.register_flow_types({0: "bus", 1: "motorbike"})
        try:
            self.assertEqual(pcu.vehicle_type_from_id("flow_0_5"), "bus")
            self.assertEqual(pcu.vehicle_type_from_id("flow_1_2"), "motorbike")
            self.assertEqual(pcu.vehicle_type_from_id("flow_9_2"), "car")   # unregistered index
            self.assertEqual(pcu.vehicle_type_from_id("garbage_id"), "car")  # unparsable id
        finally:
            pcu.register_flow_types({})

    def test_weighted_count_mixed_fleet(self):
        pcu.register_flow_types({0: "bus"})
        try:
            raw, weighted = pcu.weighted_count(["flow_0_1", "flow_0_2", "flow_2_1", "flow_2_2"])
            self.assertEqual(raw, 4)
            self.assertAlmostEqual(weighted, 3.0 + 3.0 + 1.0 + 1.0)
        finally:
            pcu.register_flow_types({})

    def test_observe_uses_pcu_when_engine_supports_lane_vehicle_ids(self):
        pcu.register_flow_types({0: "bus"})
        try:
            engine = FakeEngine(
                lane_vehicles={"road_A_0": 2, "road_B_0": 2},
                lane_waiting={"road_A_0": 2, "road_B_0": 2},
                lane_vehicle_ids={
                    "road_A_0": ["flow_0_1", "flow_0_2"],   # two buses -> 3.0 PCU each
                    "road_B_0": ["flow_2_1", "flow_2_2"],   # two cars -> 1.0 PCU each
                },
            )
            agent = make_agent(engine)
            obs = agent.observe({})
            self.assertEqual(obs["EW"]["vehicle_count"], 2)          # raw count unchanged (API compat)
            self.assertAlmostEqual(obs["EW"]["vehicle_count_pcu"], 6.0)  # 2 buses = 6 PCU
            self.assertAlmostEqual(obs["NS"]["vehicle_count_pcu"], 2.0)  # 2 cars = 2 PCU
            # Same raw vehicle_count on both approaches, but EW (buses) is denser in PCU terms.
            self.assertGreater(obs["EW"]["density"], obs["NS"]["density"])
        finally:
            pcu.register_flow_types({})

    def test_observe_falls_back_to_raw_counts_without_lane_vehicle_api(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 3, "road_B_0": 3},
            lane_waiting={"road_A_0": 1, "road_B_0": 1},
            lane_vehicle_ids=None,   # simulate an engine without get_lane_vehicles
        )
        agent = make_agent(engine)
        obs = agent.observe({})
        self.assertEqual(obs["EW"]["vehicle_count_pcu"], 3.0)
        self.assertEqual(obs["NS"]["vehicle_count_pcu"], 3.0)
        self.assertEqual(obs["EW"]["density"], obs["NS"]["density"])

    def test_waiting_pcu_uses_each_lane_own_ratio(self):
        # Regression: the waiting-PCU estimate used the running total across
        # the approach's roads, so every lane after the first was over-counted.
        pcu.register_flow_types({0: "bus"})
        try:
            engine = FakeEngine(
                lane_vehicles={"road_A1_0": 2, "road_A2_0": 2, "road_B_0": 0},
                lane_waiting={"road_A1_0": 2, "road_A2_0": 2, "road_B_0": 0},
                lane_vehicle_ids={
                    "road_A1_0": ["flow_0_1", "flow_0_2"],   # 2 buses = 6 PCU
                    "road_A2_0": ["flow_2_1", "flow_2_2"],   # 2 cars  = 2 PCU
                    "road_B_0": [],
                },
            )
            agent = TrafficAgent(
                agent_id="Agent-TEST", junction_id="JT", engine=engine,
                incoming_roads={"EW": ["road_A1", "road_A2"], "NS": ["road_B"]},
                outgoing_neighbors={"EW": None, "NS": None},
            )
            obs = agent.observe({})
            self.assertAlmostEqual(obs["EW"]["vehicle_count_pcu"], 8.0)
            # Everyone is waiting, so waiting PCU must equal total PCU (was 14.0 before the fix).
            self.assertAlmostEqual(obs["EW"]["queue_length_pcu"], 8.0)
        finally:
            pcu.register_flow_types({})


class TestSpillbackPrevention(unittest.TestCase):
    def _advance_to_min_green_boundary(self, agent, neighbor_states):
        # steps_on_phase increments at the top of decide_and_act, so
        # MIN_GREEN - 1 calls land steps_on_phase at MIN_GREEN - 1 (still
        # under the floor); the caller makes the MIN_GREEN-th call itself so
        # it can inspect that exact decision's reason.
        for _ in range(TrafficAgent.MIN_GREEN - 1):
            agent.observe({})
            agent.decide_and_act(neighbor_states)

    def test_switches_away_when_exit_road_is_full(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 5, "road_B_0": 1},
            lane_waiting={"road_A_0": 1, "road_B_0": 0},
        )
        agent = make_agent(engine)  # EW's outgoing neighbor is "JDOWN"
        neighbor_states = {"JDOWN": {"overall_density": 0.97}}  # exit road nearly full

        self._advance_to_min_green_boundary(agent, neighbor_states)
        agent.observe({})
        reason = agent.decide_and_act(neighbor_states)

        self.assertIn("Spillback prevention", reason)
        self.assertTrue(agent.is_yellow)

    def test_does_not_trigger_when_exit_road_has_room(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 5, "road_B_0": 1},
            lane_waiting={"road_A_0": 1, "road_B_0": 0},
        )
        agent = make_agent(engine)
        neighbor_states = {"JDOWN": {"overall_density": 0.40}}  # plenty of room

        self._advance_to_min_green_boundary(agent, neighbor_states)
        agent.observe({})
        reason = agent.decide_and_act(neighbor_states)

        self.assertNotIn("Spillback prevention", reason)


class TestStarvationHardCap(unittest.TestCase):
    def test_forces_switch_past_hard_cap_even_with_low_relative_priority(self):
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 10, "road_B_0": 10},
            lane_waiting={"road_A_0": 10, "road_B_0": 10},
        )
        agent = make_agent(engine, outgoing_neighbors={"EW": None, "NS": None})
        neighbor_states = {}

        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            agent.decide_and_act(neighbor_states)

        # Force NS's tracked wait time past the hard cap directly rather than
        # simulating dozens of steps.
        agent.waiting_time_tracker["NS"] = TrafficAgent.STARVATION_HARD_CAP + 1
        agent.observe({})
        reason = agent.decide_and_act(neighbor_states)

        self.assertIn("Hard starvation cap", reason)
        self.assertTrue(agent.is_yellow)
        self.assertEqual(agent.target_phase, agent.phase_names.index("NS"))

    def test_soft_starvation_limit_still_requires_higher_priority(self):
        # Below the hard cap, a merely-waiting phase with lower priority than
        # the currently-served phase should NOT be force-switched.
        engine = FakeEngine(
            lane_vehicles={"road_A_0": 12, "road_B_0": 0},
            lane_waiting={"road_A_0": 12, "road_B_0": 0},
        )
        agent = make_agent(engine, outgoing_neighbors={"EW": None, "NS": None})
        neighbor_states = {}

        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            agent.decide_and_act(neighbor_states)

        agent.waiting_time_tracker["NS"] = TrafficAgent.STARVATION_LIMIT + 1  # over soft limit, under hard cap
        agent.observe({})
        reason = agent.decide_and_act(neighbor_states)

        self.assertNotIn("Hard starvation cap", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
