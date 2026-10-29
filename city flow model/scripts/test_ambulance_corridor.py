"""
test_ambulance_corridor.py — offline unit tests for Feature 11
(smarter ambulance corridor: ETA-based triggering, early-start queue
clearance, amber+all-red, hospital-aware routing, starved-first recovery).

Run with: python "city flow model/scripts/test_ambulance_corridor.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ambulance
from agent import TrafficAgent, MultiAgentCoordinator

ADJACENCY = {
    "J1": {"J3": "EW"},
    "J2": {"J3": "NS"},
    "J3": {"J4": "EW", "J5": "NS", "J1": "EW", "J2": "NS"},
    "J4": {"J3": "EW"},
    "J5": {"J3": "NS"},
}


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


def make_agent(engine, outgoing_neighbors=None):
    return TrafficAgent(
        agent_id="Agent-TEST", junction_id="JT", engine=engine,
        incoming_roads={"EW": ["road_A"], "NS": ["road_B"]},
        outgoing_neighbors=outgoing_neighbors or {"EW": None, "NS": None},
    )


# ---------------------------------------------------------------------------
# ambulance.py pure-function tests
# ---------------------------------------------------------------------------

class TestShortestPathAndHospital(unittest.TestCase):
    def test_shortest_path_same_node(self):
        self.assertEqual(ambulance.shortest_path("J1", "J1", ADJACENCY), ["J1"])

    def test_shortest_path_multi_hop(self):
        self.assertEqual(ambulance.shortest_path("J1", "J4", ADJACENCY), ["J1", "J3", "J4"])

    def test_shortest_path_unreachable(self):
        self.assertIsNone(ambulance.shortest_path("J1", "NOWHERE", ADJACENCY))

    def test_nearest_hospital_picks_fewer_hops(self):
        hospitals = {"Far": "J4", "SameAsOrigin": "J1"}
        name, path = ambulance.nearest_hospital("J1", hospitals, ADJACENCY)
        self.assertEqual(name, "SameAsOrigin")
        self.assertEqual(path, ["J1"])

    def test_nearest_hospital_none_reachable(self):
        self.assertIsNone(ambulance.nearest_hospital("J1", {"Ghost": "NOWHERE"}, ADJACENCY))


class TestEtaAndTrigger(unittest.TestCase):
    def test_eta_seconds_basic(self):
        self.assertAlmostEqual(ambulance.eta_seconds(160.0, 16.0), 10.0)

    def test_eta_seconds_never_negative_distance(self):
        self.assertEqual(ambulance.eta_seconds(-50.0, 16.0), 0.0)

    def test_should_trigger_within_base_lead_time(self):
        self.assertTrue(ambulance.should_trigger(eta_s=10.0, lead_time_s=15.0, queue_length=0))
        self.assertFalse(ambulance.should_trigger(eta_s=20.0, lead_time_s=15.0, queue_length=0))

    def test_should_trigger_early_start_for_queued_traffic(self):
        # 20s ETA is outside the bare 15s lead time...
        self.assertFalse(ambulance.should_trigger(eta_s=20.0, lead_time_s=15.0, queue_length=0))
        # ...but with 5 vehicles queued at 2s/vehicle discharge, required lead grows to 25s, so it should trigger.
        self.assertTrue(ambulance.should_trigger(eta_s=20.0, lead_time_s=15.0, queue_length=5, discharge_headway_s=2.0))


# ---------------------------------------------------------------------------
# TrafficAgent: amber + all-red sequencing
# ---------------------------------------------------------------------------

class TestAmberAllRedSequencing(unittest.TestCase):
    def test_emergency_switch_goes_through_yellow_then_all_red_then_green(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 2, "road_B_0": 2}, lane_waiting={"road_A_0": 0, "road_B_0": 0})
        agent = make_agent(engine)   # starts on EW (index 0)
        agent.observe({})
        agent.set_emergency("NS")

        # Step 1: emergency preemption fires -> yellow begins.
        reason = agent.decide_and_act({})
        self.assertTrue(agent.is_yellow)
        self.assertFalse(agent.is_all_red)
        self.assertIn("Emergency", reason)

        # Drain the yellow interval.
        for _ in range(TrafficAgent.YELLOW_TIME):
            agent.observe({})
            reason = agent.decide_and_act({})

        # Emergency switches must NOT go straight to green — all-red first.
        self.assertFalse(agent.is_yellow)
        self.assertTrue(agent.is_all_red)
        self.assertEqual(agent.phase_names[agent.current_phase], "EW")  # hasn't actually switched yet

        # Drain the all-red interval.
        for _ in range(TrafficAgent.ALL_RED_TIME):
            agent.observe({})
            reason = agent.decide_and_act({})

        self.assertFalse(agent.is_all_red)
        self.assertEqual(agent.phase_names[agent.current_phase], "NS")
        self.assertIn("Emergency", reason)

    def test_ordinary_adaptive_switch_has_no_all_red(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 0, "road_B_0": 14}, lane_waiting={"road_A_0": 0, "road_B_0": 14})
        agent = make_agent(engine)
        for _ in range(TrafficAgent.MIN_GREEN):
            agent.observe({})
            agent.decide_and_act({})
        self.assertTrue(agent.is_yellow)
        for _ in range(TrafficAgent.YELLOW_TIME):
            agent.observe({})
            agent.decide_and_act({})
        # Ordinary switches go straight from yellow to green — no all-red stop-over.
        self.assertFalse(agent.is_all_red)
        self.assertEqual(agent.phase_names[agent.current_phase], "NS")


# ---------------------------------------------------------------------------
# TrafficAgent: starved-first recovery phase
# ---------------------------------------------------------------------------

class TestRecoveryPhase(unittest.TestCase):
    def test_clearing_emergency_enters_recovery_mode(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 2, "road_B_0": 2}, lane_waiting={"road_A_0": 0, "road_B_0": 0})
        agent = make_agent(engine)
        agent.set_emergency("EW")
        self.assertFalse(agent.recovery_mode)
        agent.set_emergency(None)
        self.assertTrue(agent.recovery_mode)

    def test_recovery_boosts_the_most_starved_phase(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 2, "road_B_0": 2}, lane_waiting={"road_A_0": 0, "road_B_0": 0})
        agent = make_agent(engine)
        agent.observe({})
        agent.local_obs["NS"]["waiting_time"] = 40   # NS is the starved one
        agent.local_obs["EW"]["waiting_time"] = 0

        without_recovery = agent.compute_priority("NS", {})
        agent.recovery_mode = True
        with_recovery = agent.compute_priority("NS", {})

        self.assertGreater(with_recovery, without_recovery)
        self.assertAlmostEqual(with_recovery - without_recovery, TrafficAgent.RECOVERY_PRIORITY_BOOST)

    def test_recovery_mode_expires_after_window(self):
        engine = FakeEngine(lane_vehicles={"road_A_0": 2, "road_B_0": 2}, lane_waiting={"road_A_0": 0, "road_B_0": 0})
        agent = make_agent(engine)
        agent.set_emergency("EW")
        agent.set_emergency(None)
        self.assertTrue(agent.recovery_mode)

        for _ in range(TrafficAgent.RECOVERY_WINDOW + 1):
            agent.observe({})
            agent.decide_and_act({})

        self.assertFalse(agent.recovery_mode)


# ---------------------------------------------------------------------------
# MultiAgentCoordinator: end-to-end corridor behaviour
# ---------------------------------------------------------------------------

class TestCoordinatorAmbulanceIntegration(unittest.TestCase):
    def _build(self):
        engine = FakeEngine(lane_vehicles={}, lane_waiting={})
        return MultiAgentCoordinator(engine)

    @staticmethod
    def _drive(coordinator):
        """One corridor update plus one signal decision per junction, keeping each agent's local_obs as the test set it."""
        coordinator._update_ambulances()
        for agent in coordinator.agents.values():
            agent.decide_and_act({})

    def test_dispatch_picks_nearest_hospital_and_route(self):
        coordinator = self._build()
        result = coordinator.dispatch_ambulance(origin_junction="J1")
        self.assertIsNotNone(result)
        self.assertIn(result["hospital"], coordinator.hospitals)
        self.assertEqual(result["route_junctions"][0], "J1")
        self.assertEqual(result["route_junctions"][-1], coordinator.hospitals[result["hospital"]])

    def test_unreachable_origin_returns_none_and_stays_inactive(self):
        coordinator = self._build()
        result = coordinator.dispatch_ambulance(origin_junction="NOWHERE")
        self.assertIsNone(result)
        self.assertFalse(coordinator.ambulance["active"])

    def test_early_start_triggers_sooner_with_queued_traffic(self):
        coordinator = self._build()
        # Heavy queue on J1's EW approach -> should trigger well before the
        # ambulance is physically close, thanks to the early-start allowance.
        coordinator.agents["J1"].local_obs = {"EW": {"queue_length": 10}, "NS": {"queue_length": 0}}
        coordinator.dispatch_ambulance(origin_junction="J1", hospitals={"City Hospital": "J4"})

        steps_to_trigger = None
        for step in range(1, 15):
            self._drive(coordinator)
            if "J1" in coordinator.ambulance["triggered_junctions"]:
                steps_to_trigger = step
                break
        self.assertIsNotNone(steps_to_trigger)
        self.assertEqual(coordinator.agents["J1"].emergency_override, "EW")

    def test_no_queue_triggers_later_than_with_queue(self):
        # J1 (the origin, i=0) has ETA 0 from the very first step regardless
        # of queue, and the network's default 200m links are short enough
        # that even J3 (i=1) starts inside the base lead time either way.
        # Use a longer link distance here so the comparison actually has
        # room to show a difference: base travel time to J3 exceeds the
        # plain 15s lead time, so only the queued approach's early-start
        # allowance triggers it immediately.
        coordinator_busy = self._build()
        coordinator_busy.AMBULANCE_LINK_DISTANCE_M = 500.0
        coordinator_busy.agents["J3"].local_obs = {"EW": {"queue_length": 10}, "NS": {"queue_length": 0}}
        coordinator_busy.dispatch_ambulance(origin_junction="J1", hospitals={"City Hospital": "J4"})

        coordinator_quiet = self._build()
        coordinator_quiet.AMBULANCE_LINK_DISTANCE_M = 500.0
        coordinator_quiet.agents["J3"].local_obs = {"EW": {"queue_length": 0}, "NS": {"queue_length": 0}}
        coordinator_quiet.dispatch_ambulance(origin_junction="J1", hospitals={"City Hospital": "J4"})

        def steps_until_triggered(coordinator):
            for step in range(1, 25):
                self._drive(coordinator)
                if "J3" in coordinator.ambulance["triggered_junctions"]:
                    return step
            return None

        busy_step = steps_until_triggered(coordinator_busy)
        quiet_step = steps_until_triggered(coordinator_quiet)
        self.assertIsNotNone(busy_step)
        self.assertIsNotNone(quiet_step)
        self.assertLess(busy_step, quiet_step)

    def test_junction_clears_and_ambulance_eventually_arrives(self):
        coordinator = self._build()
        coordinator.dispatch_ambulance(origin_junction="J1", hospitals={"City Hospital": "J4"})
        for agent in coordinator.agents.values():
            agent.local_obs = {"EW": {"queue_length": 0}, "NS": {"queue_length": 0}}

        j1_recovered = False
        for _ in range(200):   # plenty of steps to traverse J1 -> J3 -> J4
            self._drive(coordinator)
            j1_recovered = j1_recovered or coordinator.agents["J1"].recovery_mode
            if not coordinator.ambulance["active"]:
                break

        self.assertFalse(coordinator.ambulance["active"])
        self.assertIsNone(coordinator.agents["J1"].emergency_override)
        self.assertIsNone(coordinator.agents["J3"].emergency_override)
        # J1 should have entered its recovery phase as soon as it was cleared.
        self.assertTrue(j1_recovered)


class TestMultipleAmbulances(unittest.TestCase):
    """Several ambulances at once: J1->J3->J4 runs EW through J3, J2->J3->J5 runs NS through it."""

    EAST = {"City Hospital": "J4"}
    SOUTH = {"Capital Hospital": "J5"}

    def _build(self):
        coordinator = MultiAgentCoordinator(FakeEngine(lane_vehicles={}, lane_waiting={}))
        for agent in coordinator.agents.values():
            agent.local_obs = {"EW": {"queue_length": 0}, "NS": {"queue_length": 0}}
        return coordinator

    @staticmethod
    def _drive(coordinator):
        coordinator._update_ambulances()
        for agent in coordinator.agents.values():
            agent.decide_and_act({})

    def _run(self, coordinator, steps=400, on_step=None):
        for _ in range(steps):
            before = {a["id"]: a["progress_m"] for a in coordinator.active_ambulances()}
            self._drive(coordinator)
            if on_step:
                on_step(before)
            if not coordinator.ambulances:
                return
        self.fail("ambulances never all arrived")

    def _conflict(self, coordinator, head_start=4):
        """First ambulance heads east from J1; after a head start a second heads south from J2. Both cross J3."""
        first = coordinator.dispatch_ambulance("J1", self.EAST)["id"]
        for _ in range(head_start):
            self._drive(coordinator)
        second = coordinator.dispatch_ambulance("J2", self.SOUTH)["id"]
        return first, second

    def test_two_ambulances_can_be_dispatched(self):
        coordinator = self._build()
        a = coordinator.dispatch_ambulance("J1", self.EAST)
        b = coordinator.dispatch_ambulance("J2", self.SOUTH)
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(len(coordinator.active_ambulances()), 2)

    def test_conflict_goes_to_the_one_arriving_first_and_both_arrive(self):
        coordinator = self._build()
        first, second = self._conflict(coordinator)
        link = coordinator.AMBULANCE_LINK_DISTANCE_M
        j3_owners, crossings, finished = [], [], {}

        def watch(before):
            holder = coordinator.junction_holder.get("J3")
            if holder and (not j3_owners or j3_owners[-1] != holder):
                j3_owners.append(holder)
            for amb in coordinator.active_ambulances():
                finished[amb["id"]] = amb
            last = coordinator._last_ambulance
            if last:
                finished[last["id"]] = last
            for amb_id, prev in before.items():
                if prev <= link < finished[amb_id]["progress_m"]:
                    crossings.append(amb_id)

        self._run(coordinator, on_step=watch)
        self.assertEqual(j3_owners[0], first)
        self.assertIn(second, j3_owners)
        self.assertEqual(crossings, [first, second])
        self.assertGreater(finished[second]["held_steps"], 0, "the second ambulance should have had to wait")

    def test_loser_waits_at_red_instead_of_running_it(self):
        coordinator = self._build()
        _, second = self._conflict(coordinator)
        link = coordinator.AMBULANCE_LINK_DISTANCE_M
        violations = []

        def watch(before):
            amb = coordinator.ambulances.get(second)
            if amb and before.get(second, 0) <= link < amb["progress_m"]:
                j3 = coordinator.agents["J3"]
                if j3.emergency_override != "NS" or j3.phase_names[j3.current_phase] != "NS":
                    violations.append(amb["progress_m"])

        self._run(coordinator, on_step=watch)
        self.assertEqual(violations, [])

    def test_same_direction_ambulances_share_the_green(self):
        coordinator = self._build()
        coordinator.dispatch_ambulance("J1", self.EAST)
        coordinator.dispatch_ambulance("J2", self.EAST)   # J2->J3->J4: EW through J3 as well
        yielded = []

        def watch(_before):
            for amb in coordinator.active_ambulances():
                if "J3" in amb["yielding_to"]:
                    yielded.append(amb["id"])

        self._run(coordinator, on_step=watch)
        self.assertEqual(yielded, [])

    def test_tie_goes_to_the_more_severe_case(self):
        coordinator = self._build()
        routine = coordinator.dispatch_ambulance("J1", self.EAST, severity=1)["id"]
        critical = coordinator.dispatch_ambulance("J2", self.SOUTH, severity=3)["id"]
        for _ in range(30):
            self._drive(coordinator)
            if "J3" in coordinator.junction_holder:
                break
        self.assertEqual(coordinator.junction_holder["J3"], critical)
        self.assertEqual(coordinator.ambulances[routine]["yielding_to"].get("J3"), critical)

    def test_stand_down_one_keeps_the_other_running(self):
        coordinator = self._build()
        a = coordinator.dispatch_ambulance("J1", self.EAST)["id"]
        b = coordinator.dispatch_ambulance("J2", self.SOUTH)["id"]
        self._drive(coordinator)
        self.assertTrue(coordinator.stand_down_ambulance(a))
        self.assertEqual([x["id"] for x in coordinator.active_ambulances()], [b])
        self.assertIsNone(coordinator.agents["J1"].emergency_override)
        self.assertEqual(coordinator.agents["J2"].emergency_override, "NS")
        self.assertFalse(coordinator.stand_down_ambulance("AMB-404"))

    def test_fleet_limit(self):
        coordinator = self._build()
        for _ in range(coordinator.MAX_ACTIVE_AMBULANCES):
            self.assertIsNotNone(coordinator.dispatch_ambulance("J1", self.EAST))
        self.assertIsNone(coordinator.dispatch_ambulance("J1", self.EAST))

    def test_old_single_ambulance_view_follows_the_earliest_dispatch(self):
        coordinator = self._build()
        a = coordinator.dispatch_ambulance("J1", self.EAST)["id"]
        coordinator.dispatch_ambulance("J2", self.SOUTH)
        self.assertEqual(coordinator.ambulance["id"], a)
        self.assertTrue(coordinator.ambulance["active"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
