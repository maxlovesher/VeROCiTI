"""
test_standalone_server.py — integration tests for server_standalone.py

Checks that the default server really runs the multi-agent controller
(agent.MultiAgentCoordinator) over its built-in vehicle simulator, and that
every operator endpoint for Features 5, 7, 9, 10, 11 and 24 works through
Flask's test client.

The ANPR tracking routes are stubbed out and the detection database is
pointed at a temporary file, so running this never touches real data.
Run with:
    python "city flow model/scripts/test_standalone_server.py"
"""

import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
PROTOTYPE_DIR = os.path.join(os.path.dirname(SERVER_DIR), "prototype")
sys.path.insert(0, SERVER_DIR)
sys.path.append(PROTOTYPE_DIR)

_TMP = tempfile.mkdtemp(prefix="verociti_test_")
import database  # noqa: E402
database.DB_PATH = os.path.join(_TMP, "traffic.db")

# Stub the ANPR/Firebase routes: they start camera simulators and model downloads.
sys.modules["tracking_api"] = types.SimpleNamespace(register_tracking_routes=lambda app: None)

import server_standalone as srv  # noqa: E402
from server_standalone import TrafficSim, ContinuousVehicle, AGENT_INTERVAL  # noqa: E402

srv.ctrl["paused"] = True


def run(sim, seconds):
    for _ in range(seconds):
        sim.advance()


def run_server(seconds):
    for _ in range(seconds):
        srv._advance_all()
    srv._refresh_state()


class TestSimulationDrivesRealController(unittest.TestCase):
    def test_agents_are_the_real_traffic_agents(self):
        sim = TrafficSim("t", seed=1)
        from agent import TrafficAgent
        self.assertEqual(set(sim.coordinator.agents), {"J1", "J2", "J3", "J4", "J5"})
        for agent in sim.coordinator.agents.values():
            self.assertIsInstance(agent, TrafficAgent)

    def test_every_junction_switches_phase_within_ten_minutes(self):
        sim = TrafficSim("t", seed=2)
        seen = {jid: {a.current_phase} for jid, a in sim.coordinator.agents.items()}
        for _ in range(600):
            sim.advance()
            for jid, a in sim.coordinator.agents.items():
                seen[jid].add(a.current_phase)
        for jid, phases in seen.items():
            self.assertEqual(phases, {0, 1}, f"{jid} never switched")

    def test_observations_carry_feature_fields(self):
        sim = TrafficSim("t", seed=3)
        run(sim, 60)
        obs = sim.coordinator.agents["J3"].local_obs["EW"]
        for key in ("vehicle_count_pcu", "queue_length_pcu", "predicted_queue_5min",
                    "queue_trend", "speed_advisory", "pedestrian_waiting"):
            self.assertIn(key, obs)

    def test_queue_prediction_never_exceeds_approach_storage(self):
        sim = TrafficSim("t", seed=8)
        worst = 0.0
        for _ in range(900):
            sim.advance()
            for agent in sim.coordinator.agents.values():
                for obs in agent.local_obs.values():
                    self.assertLessEqual(obs["predicted_queue_5min"], obs["lane_capacity"])
                    worst = max(worst, obs["predicted_queue_5min"])
        self.assertGreater(worst, 0.0)

    def test_pcu_counts_trucks_heavier_than_raw_count(self):
        sim = TrafficSim("t", seed=4)
        sim.fleet.clear()
        for i in range(2):
            v = ContinuousVehicle(f"t{i}", ["road_J1_J3", "road_J3_J4"], "truck", sim.rng, 0)
            v.dist, v.speed, v.is_waiting = 150.0 + 13 * i, 0.0, True
            sim.fleet.append(v)
        sim.engine.snapshot()
        obs = sim.coordinator.agents["J3"].observe(sim._vehicle_info_map())
        self.assertEqual(obs["EW"]["vehicle_count"], 2)
        self.assertAlmostEqual(obs["EW"]["vehicle_count_pcu"], 6.0)
        self.assertAlmostEqual(obs["EW"]["queue_length_pcu"], 6.0)

    def test_vehicle_held_at_stop_line_stays_on_red(self):
        # Regression: a vehicle stopped exactly on the stop line used to creep through the red.
        sim = TrafficSim("t", seed=5)
        v = ContinuousVehicle("x", ["road_J1_J3", "road_J3_J4"], "car", sim.rng, 0)
        v.dist, v.speed = 182.0, 0.0      # 200 m road, stop line at 182 m
        for _ in range(100):
            v.update(0.1, None, signal_red=True)
        self.assertEqual(v.dist, 182.0)
        self.assertEqual(v.speed, 0.0)

    def test_vehicles_released_on_green(self):
        sim = TrafficSim("t", seed=5)
        v = ContinuousVehicle("x", ["road_J1_J3", "road_J3_J4"], "car", sim.rng, 0)
        v.dist, v.speed = 182.0, 0.0
        for _ in range(100):
            v.update(0.1, None, signal_red=False)
        self.assertEqual(v.road, "road_J3_J4")

    def test_late_buses_raise_priority_and_on_time_ones_do_not(self):
        sim = TrafficSim("t", seed=6)
        sim.fleet.clear()
        late = ContinuousVehicle("late", ["road_J1_J3", "road_J3_J4"], "bus", sim.rng, 0)
        on_time = ContinuousVehicle("ontime", ["road_J2_J3", "road_J3_J5"], "bus", sim.rng, 0)
        late.lateness_min, on_time.lateness_min = 8.0, 0.5
        sim.fleet.extend([late, on_time])
        sim._sync_bus_priority()
        j3 = sim.coordinator.agents["J3"]
        self.assertGreater(j3.bus_priority.boost_for_phase("EW"), 0.0)
        self.assertEqual(j3.bus_priority.boost_for_phase("NS"), 0.0)

        sim.fleet.clear()           # both buses have left J3's approaches
        sim._sync_bus_priority()
        self.assertEqual(j3.bus_priority.boost_for_phase("EW"), 0.0)

    def test_adaptive_control_idles_less_than_fixed_time(self):
        adaptive = TrafficSim("a", seed=11)
        fixed = TrafficSim("f", fixed_time=True, seed=11)
        for _ in range(2400):
            adaptive.advance()
            fixed.advance()
        a, f = adaptive.trip_stats(), fixed.trip_stats()
        self.assertGreater(a["trips"], 100)
        self.assertGreater(f["trips"], 100)
        self.assertLess(a["avg_idle_s"], f["avg_idle_s"])

    def test_fixed_time_baseline_survives_reset(self):
        sim = TrafficSim("f", fixed_time=True, seed=12)
        sim.reset()
        run(sim, AGENT_INTERVAL * 2)
        self.assertEqual(sim.coordinator.degradation.current, "FIXED_TIME")


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = srv.app.test_client()
        self.client.post("/api/control", json={"cmd": "reset"})
        srv.ctrl["paused"] = True

    def state(self):
        return self.client.get("/api/state").get_json()

    def test_state_shape_matches_dashboard_contract(self):
        run_server(30)
        s = self.state()
        for key in ("step", "agents", "tl_phases", "agent_messages", "ambulance",
                    "control_mode", "green_wave", "bus_priority", "emissions"):
            self.assertIn(key, s)
        j1 = s["agents"]["J1"]
        for key in ("current_phase", "is_yellow", "overall_density", "total_queue",
                    "available_capacity", "decision_reason", "local_obs", "control_mode"):
            self.assertIn(key, j1)
        self.assertIn("phase_idx", s["tl_phases"]["J1"])
        self.assertEqual(s["control_mode"], "FULL_AI")

    def test_ambulance_preempts_corridor_then_recovers(self):
        # Put the corridor junctions on the cross phase so preemption has to switch them.
        for jid in ("J1", "J3"):
            self.client.post("/api/override", json={"junction": jid, "phase": 1})

        r = self.client.post("/api/ambulance", json={"active": True, "origin": "J1"}).get_json()
        self.assertTrue(r["ok"])
        self.assertTrue(r["ambulance"]["active"])
        self.assertEqual(r["ambulance"]["route_junctions"], ["J1", "J3", "J4"])
        self.assertEqual(r["ambulance"]["hospital"], "City Hospital")

        coord = srv.live.coordinator
        saw_all_red, saw_emergency, saw_recovery = set(), set(), set()
        for _ in range(400):
            srv._advance_all()
            for jid in ("J1", "J3"):
                a = coord.agents[jid]
                if a.emergency_override == "EW":
                    saw_emergency.add(jid)
                if a.is_all_red:
                    saw_all_red.add(jid)
                if a.recovery_mode:
                    saw_recovery.add(jid)
            if not coord.ambulance["active"]:
                break
        self.assertEqual(saw_emergency, {"J1", "J3"})
        self.assertEqual(saw_all_red, {"J1", "J3"}, "emergency green without an all-red clearance")
        self.assertEqual(saw_recovery, {"J1", "J3"}, "no recovery phase after the ambulance cleared")
        self.assertFalse(coord.ambulance["active"], "ambulance never arrived")
        self.assertEqual(sorted(coord.ambulance["cleared_junctions"]), ["J1", "J3"])
        self.assertIsNone(coord.agents["J1"].emergency_override)

    def test_ambulance_stand_down_releases_signals(self):
        self.client.post("/api/ambulance", json={"active": True})
        agents = srv.live.coordinator.agents.values()
        for _ in range(AGENT_INTERVAL * 10):
            srv._advance_all()
            if any(a.emergency_override for a in agents):
                break
        self.assertTrue(any(a.emergency_override for a in agents))
        r = self.client.post("/api/ambulance", json={"active": False}).get_json()
        self.assertFalse(r["ambulance"]["active"])
        self.assertFalse(any(a.emergency_override for a in srv.live.coordinator.agents.values()))

    def test_ambulance_rejects_unknown_origin(self):
        r = self.client.post("/api/ambulance", json={"active": True, "origin": "J9"})
        self.assertEqual(r.status_code, 400)

    def test_sensor_fault_walks_down_the_ladder_and_back(self):
        self.client.post("/api/sensor_fault", json={"active": True})
        modes = []
        for _ in range(3 * 3):
            run_server(AGENT_INTERVAL * 3)     # 3 bad steps per rung
            modes.append(self.state()["control_mode"])
        self.assertIn("HISTORICAL_PROFILE", modes)
        self.assertIn("LOCAL_ACTUATED", modes)
        self.assertEqual(modes[-1], "FIXED_TIME")
        self.assertTrue(all(a.control_mode == "FIXED_TIME" for a in srv.live.coordinator.agents.values()))

        self.client.post("/api/sensor_fault", json={"active": False})
        run_server(AGENT_INTERVAL * 10)          # 10 good steps climb one rung
        self.assertEqual(self.state()["control_mode"], "LOCAL_ACTUATED")

    def test_operator_can_pin_and_release_control_mode(self):
        r = self.client.post("/api/control_mode", json={"mode": "HISTORICAL_PROFILE"}).get_json()
        self.assertEqual(r["control_mode"], "HISTORICAL_PROFILE")
        run_server(AGENT_INTERVAL * 2)
        s = self.state()
        self.assertEqual(s["control_mode"], "HISTORICAL_PROFILE")
        self.assertTrue(s["control_mode_forced"])
        self.assertEqual(s["agents"]["J2"]["control_mode"], "HISTORICAL_PROFILE")

        self.client.post("/api/control_mode", json={"mode": "AUTO"})
        run_server(AGENT_INTERVAL * 2)
        self.assertEqual(self.state()["control_mode"], "FULL_AI")
        self.assertEqual(self.client.post("/api/control_mode", json={"mode": "TURBO"}).status_code, 400)

    def test_pedestrian_call_is_visible_to_the_agent(self):
        r = self.client.post("/api/pedestrian", json={"junction": "J2", "phase": "NS", "waiting": True}).get_json()
        self.assertTrue(r["pedestrian_demand"]["NS"])
        self.assertTrue(srv.live.coordinator.agents["J2"].pedestrian_demand["NS"])
        self.assertEqual(self.client.post("/api/pedestrian", json={"junction": "J2", "phase": "XX"}).status_code, 400)

    def test_override_and_incident(self):
        self.client.post("/api/override", json={"junction": "J2", "phase": 1})
        self.assertEqual(self.state()["tl_phases"]["J2"]["phase_idx"], 1)

        r = self.client.post("/api/incident", json={"junction": "J3", "road": "road_J3_J2", "active": True}).get_json()
        self.assertEqual(len(r["incidents"]), 1)
        self.assertIn("road_J3_J2", srv.live.coordinator.agents["J3"].incidents)
        self.client.post("/api/incident", json={"junction": "J3", "road": "road_J3_J2", "active": False})
        self.assertEqual(self.state()["active_incidents"], [])

    def test_green_wave_follows_real_links(self):
        run_server(900)
        r = self.client.get("/api/green_wave").get_json()
        self.assertTrue(r["corridors"])
        adjacency = srv.live.coordinator._adjacency()
        for corridor in r["corridors"]:
            path = corridor["junctions"]
            for a, b in zip(path, path[1:]):
                self.assertIn(b, adjacency[a])
            self.assertEqual(len(corridor["offsets_s"]), len(path))

    def test_emissions_compare_against_fixed_time(self):
        run_server(1800)
        e = self.client.get("/api/emissions").get_json()
        self.assertTrue(e["ready"])
        self.assertEqual(e["baseline"]["control_mode"], "FIXED_TIME")
        self.assertGreater(e["co2_saved_kg"], 0.0)
        self.assertEqual(sum(c["trips"] for c in e["by_class"]), e["live"]["trips"])

    def test_pause_step_and_reset(self):
        self.client.post("/api/control", json={"cmd": "pause"})
        before = self.state()["step"]
        self.client.post("/api/control", json={"cmd": "step"})
        self.assertEqual(self.state()["step"], before + 1)
        self.client.post("/api/control", json={"cmd": "reset"})
        srv.ctrl["paused"] = True
        self.assertEqual(self.state()["step"], 0)
        self.assertEqual(self.client.post("/api/control", json={"cmd": "speed", "value": "fast"}).status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
