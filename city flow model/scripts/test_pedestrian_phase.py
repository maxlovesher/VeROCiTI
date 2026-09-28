"""
test_pedestrian_phase.py — offline unit tests for Feature 10
(pedestrian-demand walk phase).

Pure-logic tests: no CityFlow engine or Flask server required. Run with:
    python "city flow model/scripts/test_pedestrian_phase.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import TrafficAgent


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


def make_agent(engine, outgoing_neighbors=None):
    return TrafficAgent(
        agent_id="Agent-TEST",
        junction_id="JT",
        engine=engine,
        incoming_roads={"EW": ["road_A"], "NS": ["road_B"]},
        outgoing_neighbors=outgoing_neighbors or {"EW": None, "NS": None},
    )


# Low, roughly-balanced traffic so vehicle priority alone never forces the
# switches under test — only the pedestrian logic should.
QUIET_TRAFFIC = dict(
    lane_vehicles={"road_A_0": 2, "road_B_0": 2},
    lane_waiting={"road_A_0": 0, "road_B_0": 0},
)


class TestPedestrianDemandTracking(unittest.TestCase):
    def test_no_demand_means_no_wait_accrual(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        for _ in range(5):
            agent.observe({})
        self.assertEqual(agent.pedestrian_wait_tracker["EW"], 0)

    def test_wait_accrues_only_while_demanded_phase_is_green(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.set_pedestrian_demand("EW", True)   # EW is current_phase (index 0) => green
        for _ in range(5):
            agent.observe({})
        self.assertEqual(agent.pedestrian_wait_tracker["EW"], 5)
        self.assertTrue(agent.local_obs["EW"]["pedestrian_waiting"])

    def test_clearing_demand_resets_wait(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.set_pedestrian_demand("EW", True)
        for _ in range(5):
            agent.observe({})
        agent.set_pedestrian_demand("EW", False)
        self.assertEqual(agent.pedestrian_wait_tracker["EW"], 0)


class TestPedestrianForcedSwitch(unittest.TestCase):
    def test_forces_switch_after_max_wait_despite_balanced_traffic(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.set_pedestrian_demand("EW", True)

        last_reason = ""
        for _ in range(TrafficAgent.PED_MAX_WAIT + TrafficAgent.MIN_GREEN + 1):
            agent.observe({})
            last_reason = agent.decide_and_act({})
            if agent.is_yellow:
                break

        self.assertIn("Pedestrian call", last_reason)
        self.assertTrue(agent.is_yellow)
        self.assertEqual(agent.target_phase, agent.phase_names.index("NS"))

    def test_no_forced_switch_when_wait_is_below_max(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.set_pedestrian_demand("EW", True)

        for _ in range(TrafficAgent.PED_MAX_WAIT - 5):
            agent.observe({})
            reason = agent.decide_and_act({})
            self.assertNotIn("Pedestrian call", reason)


class TestPedestrianWalkProtection(unittest.TestCase):
    def test_protection_window_blocks_switching_back_early(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.set_pedestrian_demand("EW", True)

        # Drive the forced switch away from EW.
        for _ in range(TrafficAgent.PED_MAX_WAIT + TrafficAgent.MIN_GREEN + 1):
            agent.observe({})
            agent.decide_and_act({})
            if agent.is_yellow:
                break
        self.assertTrue(agent.is_yellow)
        self.assertEqual(agent.pedestrian_protection_remaining["EW"], TrafficAgent.PED_MIN_WALK)

        # Finish the yellow transition -> now on NS, EW is protected.
        for _ in range(TrafficAgent.YELLOW_TIME):
            agent.observe({})
            agent.decide_and_act({})
        self.assertEqual(agent.phase_names[agent.current_phase], "NS")

        # Even once NS clears MIN_GREEN, an ordinary adaptive/starvation pull
        # back toward EW must be refused while its walk window is protected.
        # Make EW look attractive to a normal adaptive controller (heavy
        # queue) to prove the block is specifically the ped protection, not
        # just "EW isn't busy yet".
        engine._lane_vehicles["road_A_0"] = 14
        engine._lane_waiting["road_A_0"] = 14
        checked_while_protected = False
        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            still_protected = agent.pedestrian_protection_remaining["EW"] > 0
            reason = agent.decide_and_act({})
            if still_protected:
                # The protection was live going into this decision, so it
                # must not have switched back to EW regardless of load.
                self.assertNotIn("Switching NS", reason)
                checked_while_protected = True
        self.assertTrue(checked_while_protected, "test setup never exercised a still-protected decision")

    def test_protection_expires_after_min_walk(self):
        engine = FakeEngine(**QUIET_TRAFFIC)
        agent = make_agent(engine)
        agent.pedestrian_protection_remaining["EW"] = 2
        agent.current_phase = agent.phase_names.index("NS")  # EW is currently red (protection ticking)

        for _ in range(3):
            agent.observe({})
        self.assertEqual(agent.pedestrian_protection_remaining["EW"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
