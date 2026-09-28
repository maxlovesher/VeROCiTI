"""
test_bus_priority.py — offline unit tests for Feature 7
(conditional bus priority: only for buses running late).

Run with: python "city flow model/scripts/test_bus_priority.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus_priority import lateness_minutes, is_priority_warranted, priority_boost, BusPriorityRegistry
from agent import TrafficAgent


class FakeEngine:
    def __init__(self, lane_vehicles=None, lane_waiting=None):
        self._lane_vehicles = lane_vehicles or {}
        self._lane_waiting = lane_waiting or {}

    def get_lane_waiting_vehicle_count(self):
        return dict(self._lane_waiting)

    def get_lane_vehicle_count(self):
        return dict(self._lane_vehicles)

    def get_current_time(self):
        return 0

    def set_tl_phase(self, junction_id, phase_idx):
        pass


def make_agent(engine):
    return TrafficAgent(
        agent_id="Agent-TEST", junction_id="JT", engine=engine,
        incoming_roads={"EW": ["road_A"], "NS": ["road_B"]},
        outgoing_neighbors={"EW": None, "NS": None},
    )


class TestLatenessMath(unittest.TestCase):
    def test_on_time_bus_is_not_warranted(self):
        self.assertFalse(is_priority_warranted(lateness_minutes(10, 10)))

    def test_early_bus_is_not_warranted(self):
        self.assertFalse(is_priority_warranted(lateness_minutes(10, 8)))

    def test_slightly_late_below_threshold_not_warranted(self):
        late = lateness_minutes(10, 12)  # 2 min late, default threshold 3
        self.assertFalse(is_priority_warranted(late, threshold_min=3.0))

    def test_late_enough_is_warranted(self):
        late = lateness_minutes(10, 14)  # 4 min late
        self.assertTrue(is_priority_warranted(late, threshold_min=3.0))

    def test_boost_is_zero_below_threshold(self):
        self.assertEqual(priority_boost(2.0, threshold_min=3.0), 0.0)

    def test_boost_scales_up_with_lateness(self):
        b1 = priority_boost(4.0, threshold_min=3.0, max_boost=0.5, cap_lateness_min=15.0)
        b2 = priority_boost(10.0, threshold_min=3.0, max_boost=0.5, cap_lateness_min=15.0)
        self.assertGreater(b2, b1)
        self.assertGreater(b1, 0.0)

    def test_boost_caps_at_max(self):
        b_at_cap = priority_boost(15.0, threshold_min=3.0, max_boost=0.5, cap_lateness_min=15.0)
        b_way_over = priority_boost(60.0, threshold_min=3.0, max_boost=0.5, cap_lateness_min=15.0)
        self.assertAlmostEqual(b_at_cap, 0.5)
        self.assertAlmostEqual(b_way_over, 0.5)  # doesn't keep growing past the cap


class TestBusPriorityRegistry(unittest.TestCase):
    def test_update_returns_warranted_flag(self):
        reg = BusPriorityRegistry(threshold_min=3.0)
        self.assertFalse(reg.update("BUS1", "EW", scheduled_minutes=5, predicted_minutes=6))   # 1 min late
        self.assertTrue(reg.update("BUS1", "EW", scheduled_minutes=5, predicted_minutes=10))    # 5 min late

    def test_boost_for_phase_only_counts_that_phase(self):
        reg = BusPriorityRegistry(threshold_min=3.0)
        reg.update("BUS1", "EW", scheduled_minutes=5, predicted_minutes=12)
        self.assertGreater(reg.boost_for_phase("EW"), 0.0)
        self.assertEqual(reg.boost_for_phase("NS"), 0.0)

    def test_on_time_bus_never_registers(self):
        reg = BusPriorityRegistry(threshold_min=3.0)
        reg.update("BUS1", "EW", scheduled_minutes=5, predicted_minutes=5.5)
        self.assertEqual(reg.boost_for_phase("EW"), 0.0)
        self.assertEqual(reg.active_requests_for_phase("EW"), 0)

    def test_clear_removes_request(self):
        reg = BusPriorityRegistry(threshold_min=3.0)
        reg.update("BUS1", "EW", scheduled_minutes=5, predicted_minutes=12)
        reg.clear("BUS1")
        self.assertEqual(reg.boost_for_phase("EW"), 0.0)

    def test_multiple_buses_take_the_max_boost(self):
        reg = BusPriorityRegistry(threshold_min=3.0, max_boost=0.5, cap_lateness_min=15.0)
        reg.update("BUS1", "EW", scheduled_minutes=0, predicted_minutes=4)    # 4 min late -> small boost
        reg.update("BUS2", "EW", scheduled_minutes=0, predicted_minutes=20)  # 20 min late -> capped boost
        self.assertAlmostEqual(reg.boost_for_phase("EW"), 0.5)


class TestTrafficAgentIntegration(unittest.TestCase):
    def test_late_bus_raises_priority_over_on_time_baseline(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 4, "road_B_0": 4}, lane_waiting={"road_A_0": 1, "road_B_0": 1})
        agent = make_agent(engine)
        agent.observe({})
        baseline = agent.compute_priority("EW", {})

        agent.register_bus("BUS1", "EW", scheduled_minutes=0, predicted_minutes=10)  # 10 min late
        boosted = agent.compute_priority("EW", {})

        self.assertGreater(boosted, baseline)

    def test_on_time_bus_does_not_change_priority(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 4, "road_B_0": 4}, lane_waiting={"road_A_0": 1, "road_B_0": 1})
        agent = make_agent(engine)
        agent.observe({})
        baseline = agent.compute_priority("EW", {})

        agent.register_bus("BUS1", "EW", scheduled_minutes=0, predicted_minutes=0.5)  # on time
        unchanged = agent.compute_priority("EW", {})

        self.assertEqual(unchanged, baseline)

    def test_clear_bus_removes_the_boost(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 4, "road_B_0": 4}, lane_waiting={"road_A_0": 1, "road_B_0": 1})
        agent = make_agent(engine)
        agent.observe({})
        agent.register_bus("BUS1", "EW", scheduled_minutes=0, predicted_minutes=10)
        boosted = agent.compute_priority("EW", {})
        agent.clear_bus("BUS1")
        after_clear = agent.compute_priority("EW", {})
        self.assertLess(after_clear, boosted)

    def test_reset_clears_bus_registry(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 4, "road_B_0": 4}, lane_waiting={"road_A_0": 1, "road_B_0": 1})
        agent = make_agent(engine)
        agent.observe({})
        agent.register_bus("BUS1", "EW", scheduled_minutes=0, predicted_minutes=10)
        agent.reset()
        agent.observe({})
        self.assertEqual(agent.bus_priority.boost_for_phase("EW"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
