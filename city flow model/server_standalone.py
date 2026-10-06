"""
server_standalone.py — Standalone CityFlow-compatible Server (No C++ dependency)
=================================================================================
Runs the same multi-agent signal controller as server.py
(agent.MultiAgentCoordinator, with PCU queues, spillback prevention,
starvation caps, the degradation ladder, green waves, queue prediction, speed
advisory, bus priority, pedestrian walk phases and the smarter ambulance
corridor), but over a lightweight built-in vehicle simulator instead of the
CityFlow C++ engine. Use this when the cityflow module is not available.

A second copy of the simulation runs alongside the live one with its signals
pinned to FIXED_TIME. It sees the same kind of demand but no operator events,
and serves as the measured baseline for the idle-time / emissions-saved
estimate (Feature 24).
"""

import json
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from typing import Dict, Any, List, Optional

from flask import Flask, jsonify, request, send_from_directory

from agent import MultiAgentCoordinator, TrafficAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROTOTYPE_DIR = os.path.join(os.path.dirname(BASE_DIR), "prototype")
if os.path.isdir(PROTOTYPE_DIR) and PROTOTYPE_DIR not in sys.path:
    sys.path.append(PROTOTYPE_DIR)

import emissions  # noqa: E402  (lives in prototype/, added to the path above)

app = Flask(__name__, static_folder=STATIC_DIR)
app.config["JSON_SORT_KEYS"] = False


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Mount ANPR Vehicle Tracking & Firebase Cloud Engine ──
try:
    from tracking_api import register_tracking_routes
    register_tracking_routes(app)
except Exception as _track_err:
    print(f"[Server] Note: could not mount tracking routes: {_track_err}")


# ── Camera-network analytics & enforcement (Features 2, 12–23) ──
try:
    from features_api import register_feature_routes
    register_feature_routes(app)
except Exception as _feat_err:
    print(f"[Server] Note: could not mount analytics/enforcement routes: {_feat_err}")


# -- Team sign-in (accounts come from the TEAM_MEMBERS environment variable) --
from auth_api import register_auth_routes
register_auth_routes(app)


# ── Junction & Road Network Definition (matches roadnet_5j.json) ──
JUNCTIONS = ["J1", "J2", "J3", "J4", "J5"]

PHASE_NAMES = ["EW GREEN", "NS GREEN"]

# Road topology and lengths (from roadnet_5j.json)
ROAD_LENGTHS = {
    "road_VW1_J1": 200.0, "road_J1_VW1": 200.0,
    "road_J1_J3": 200.0, "road_J3_J1": 200.0,
    "road_VN1_J1": 200.0, "road_J1_VN1": 200.0,
    "road_VS1_J1": 200.0, "road_J1_VS1": 200.0,
    "road_VN2_J2": 200.0, "road_J2_VN2": 200.0,
    "road_J2_J3": 200.0, "road_J3_J2": 200.0,
    "road_VW2_J2": 200.0, "road_J2_VW2": 200.0,
    "road_VE2_J2": 200.0, "road_J2_VE2": 200.0,
    "road_J3_J4": 200.0, "road_J4_J3": 200.0,
    "road_J3_J5": 200.0, "road_J5_J3": 200.0,
    "road_VE4_J4": 200.0, "road_J4_VE4": 200.0,
    "road_VN4_J4": 200.0, "road_J4_VN4": 200.0,
    "road_VS4_J4": 200.0, "road_J4_VS4": 200.0,
    "road_VS5_J5": 200.0, "road_J5_VS5": 200.0,
    "road_VW5_J5": 200.0, "road_J5_VW5": 200.0,
    "road_VE5_J5": 200.0, "road_J5_VE5": 200.0,
}

# Incoming junction and approach phase for each road
ROAD_JUNCTION_PHASE = {
    "road_VW1_J1": ("J1", "EW"),
    "road_J3_J1": ("J1", "EW"),
    "road_VN1_J1": ("J1", "NS"),
    "road_VS1_J1": ("J1", "NS"),
    "road_VW2_J2": ("J2", "EW"),
    "road_VE2_J2": ("J2", "EW"),
    "road_VN2_J2": ("J2", "NS"),
    "road_J3_J2": ("J2", "NS"),
    "road_J1_J3": ("J3", "EW"),
    "road_J4_J3": ("J3", "EW"),
    "road_J2_J3": ("J3", "NS"),
    "road_J5_J3": ("J3", "NS"),
    "road_J3_J4": ("J4", "EW"),
    "road_VE4_J4": ("J4", "EW"),
    "road_VN4_J4": ("J4", "NS"),
    "road_VS4_J4": ("J4", "NS"),
    "road_VW5_J5": ("J5", "EW"),
    "road_VE5_J5": ("J5", "EW"),
    "road_J3_J5": ("J5", "NS"),
    "road_VS5_J5": ("J5", "NS"),
}

# Realistic multi-junction routes (from flow_5j.json)
ROUTES = [
    ["road_VW1_J1", "road_J1_J3", "road_J3_J4", "road_J4_VE4"],
    ["road_VE4_J4", "road_J4_J3", "road_J3_J1", "road_J1_VW1"],
    ["road_VN2_J2", "road_J2_J3", "road_J3_J5", "road_J5_VS5"],
    ["road_VS5_J5", "road_J5_J3", "road_J3_J2", "road_J2_VN2"],
    ["road_VN1_J1", "road_J1_VS1"],
    ["road_VS1_J1", "road_J1_VN1"],
    ["road_VW2_J2", "road_J2_VE2"],
    ["road_VE2_J2", "road_J2_VW2"],
    ["road_VN4_J4", "road_J4_VS4"],
    ["road_VS4_J4", "road_J4_VN4"],
    ["road_VW5_J5", "road_J5_VE5"],
    ["road_VE5_J5", "road_J5_VW5"],
]

# Junction-to-junction link lengths, for green-wave offsets (Feature 5).
LINK_DISTANCES_M = {
    ("J1", "J3"): ROAD_LENGTHS["road_J1_J3"],
    ("J2", "J3"): ROAD_LENGTHS["road_J2_J3"],
    ("J3", "J4"): ROAD_LENGTHS["road_J3_J4"],
    ("J3", "J5"): ROAD_LENGTHS["road_J3_J5"],
}

# Mixed Indian urban traffic, so PCU weighting (Feature 1) has something to weigh.
VEHICLE_MIX = [
    ("car", 0.48),
    ("motorbike", 0.26),
    ("auto-rickshaw", 0.12),
    ("bus", 0.07),
    ("truck", 0.07),
]

SIM_SECONDS_PER_TICK = 1          # each worker tick advances one simulated second, like CityFlow's next_step()
PHYSICS_SUBSTEPS = 10             # vehicle motion is integrated in 0.1 s slices so nobody overshoots a stop line
AGENT_INTERVAL = 3                # controller decides every 3 simulated seconds, as in server.py
FLEET_TARGET = 64                 # vehicles kept in the network
SPAWN_CLEARANCE_M = 14.0          # a new vehicle only enters if the first road has this much free space
PED_ARRIVAL_PROB = 0.02           # per decision step, per green approach, in the live simulation
TRIP_HISTORY = 400                # completed trips kept for travel-time / emissions stats
IDLE_MPS = emissions.IDLE_SPEED_THRESHOLD_KMPH / 3.6

# Control flags
ctrl = {
    "paused": False,
    "step_delay": 0.10,
}

sim_state: Dict[str, Any] = {
    "step": 0,
    "running": False,
    "total_vehicles": 0,
    "avg_travel_time": 0.0,
    "avg_speed": 0.0,
    "network_density": 0.0,
    "total_waiting": 0,
    "vehicles": [],
    "lane_vehicles": {},
    "lane_waiting": {},
    "tl_phases": {},
    "agents": {},
    "agent_messages": [],
    "active_incidents": [],
    "ambulance": {"active": False},
}
state_lock = threading.Lock()
sim_lock = threading.RLock()


def _pick_vehicle_class(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for cls, share in VEHICLE_MIX:
        cumulative += share
        if roll <= cumulative:
            return cls
    return VEHICLE_MIX[-1][0]


def _route_junctions(route: List[str]) -> List[str]:
    """Junctions a route passes through, in order (road_A_B ends at B)."""
    junctions = []
    for road in route:
        end = road.rsplit("_", 1)[-1]
        if end in JUNCTIONS and (not junctions or junctions[-1] != end):
            junctions.append(end)
    return junctions


class ContinuousVehicle:
    def __init__(self, vid, route, vehicle_class, rng: random.Random, spawn_time: int, partial: bool = False):
        self.vid = vid
        self.route = list(route)
        self.route_idx = 0
        self.vehicle_class = vehicle_class
        self.dist = rng.uniform(0.0, 50.0)
        self.speed = rng.uniform(5.0, 9.0)
        heavy = vehicle_class in ("bus", "truck")
        self.target_speed = rng.uniform(8.5, 11.5) if heavy else rng.uniform(10.0, 14.5)
        self.is_waiting = False
        self.idle_seconds = 0.0
        self.spawn_time = spawn_time
        # Vehicles seeded mid-route at startup didn't make a whole trip, so
        # they're left out of travel-time / idle statistics.
        self.partial = partial
        # Feature 7: each bus runs some minutes off its timetable (positive = late).
        self.lateness_min = rng.uniform(-2.0, 9.0) if vehicle_class == "bus" else 0.0

    @property
    def road(self):
        return self.route[self.route_idx]

    def update(self, dt, leader_dist, signal_red):
        road_len = ROAD_LENGTHS.get(self.road, 200.0)
        stop_line = road_len - 18.0

        # Smooth Deceleration & Car-Following Physics (No stop-and-go jitter)
        dist_to_stop = stop_line - self.dist
        # >= 0 so a vehicle already held at the stop line keeps holding.
        signal_stopping = signal_red and (0.0 <= dist_to_stop < 55.0)

        min_gap = 12.0
        desired_gap = 22.0

        if leader_dist is not None and leader_dist < desired_gap:
            if leader_dist <= min_gap:
                self.speed = max(0.0, self.speed - 12.0 * dt)
            else:
                target_spd = self.target_speed * ((leader_dist - min_gap) / (desired_gap - min_gap))
                if self.speed > target_spd:
                    self.speed = max(target_spd, self.speed - 5.0 * dt)
                else:
                    self.speed = min(target_spd, self.speed + 3.0 * dt)
            if signal_stopping and dist_to_stop <= 2.0:
                self.dist = min(self.dist, stop_line)
                self.speed = 0.0
        elif signal_stopping:
            if dist_to_stop <= 2.0:
                self.dist = min(self.dist, stop_line)
                self.speed = 0.0
            else:
                brake_rate = max(2.5, min(8.5, (self.speed * self.speed) / (2.0 * max(1.5, dist_to_stop))))
                self.speed = max(0.0, self.speed - brake_rate * dt)
        else:
            self.speed = min(self.target_speed, self.speed + 4.5 * dt)

        self.is_waiting = self.speed < 0.3
        if self.speed <= IDLE_MPS:
            self.idle_seconds += dt

        self.dist += self.speed * dt

        # Transition to next road in route seamlessly
        if self.dist >= road_len:
            excess = self.dist - road_len
            self.route_idx += 1
            if self.route_idx >= len(self.route):
                return False  # Completed trip
            self.dist = excess

        return True


class FleetEngine:
    """
    Answers the handful of CityFlow Engine calls agent.py makes, from a
    TrafficSim's own fleet. Lane snapshots are refreshed once per controller
    step (FleetEngine.snapshot) instead of on every call.
    """

    def __init__(self, sim: "TrafficSim"):
        self.sim = sim
        self._lane_vehicles: Dict[str, List[str]] = {}
        self._lane_waiting: Dict[str, int] = {}

    def snapshot(self) -> None:
        lanes: Dict[str, List[str]] = {f"{r}_0": [] for r in ROAD_LENGTHS}
        waiting: Dict[str, int] = {lane: 0 for lane in lanes}
        for v in self.sim.fleet:
            lane = f"{v.road}_0"
            lanes.setdefault(lane, []).append(v.vid)
            if v.is_waiting:
                waiting[lane] = waiting.get(lane, 0) + 1
        self._lane_vehicles = lanes
        self._lane_waiting = waiting

    def get_current_time(self) -> float:
        return float(self.sim.time)

    def get_lane_vehicles(self) -> Dict[str, List[str]]:
        return self._lane_vehicles

    def get_lane_vehicle_count(self) -> Dict[str, int]:
        return {lane: len(ids) for lane, ids in self._lane_vehicles.items()}

    def get_lane_waiting_vehicle_count(self) -> Dict[str, int]:
        return dict(self._lane_waiting)

    def set_tl_phase(self, junction_id, phase_idx) -> None:
        # Signal state lives on the agents; vehicles read it from there directly.
        pass


class TrafficSim:
    """One vehicle fleet plus the multi-agent controller running its signals."""

    def __init__(self, name: str, fixed_time: bool = False, pedestrians: bool = False, seed: Optional[int] = None):
        self.name = name
        self.fixed_time = fixed_time
        self.pedestrians = pedestrians
        self.rng = random.Random(seed)
        self.engine = FleetEngine(self)
        self.coordinator = MultiAgentCoordinator(self.engine)
        self.fleet: List[ContinuousVehicle] = []
        self.trips: deque = deque(maxlen=TRIP_HISTORY)
        self.od_counts: Counter = Counter()
        self.sensor_fault = False
        self._bus_positions: Dict[str, tuple] = {}
        self.veh_counter = 0
        self.time = 0
        self.reset()

    # ── lifecycle ──
    def reset(self) -> None:
        self.coordinator.reset()
        if self.fixed_time:
            self.coordinator.force_control_mode("FIXED_TIME")
        self.fleet.clear()
        self.trips.clear()
        self.od_counts.clear()
        self._bus_positions.clear()
        self.sensor_fault = False
        self.time = 0
        while len(self.fleet) < FLEET_TARGET:
            v = self._new_vehicle(partial=True)
            v.dist = self.rng.uniform(10.0, 160.0)
            self.fleet.append(v)
        self.engine.snapshot()
        # One observation pass so the dashboard has numbers before the first decision.
        self.coordinator.step(self._vehicle_info_map(), data_ok=True)

    def _new_vehicle(self, partial: bool = False, route: Optional[List[str]] = None) -> ContinuousVehicle:
        self.veh_counter += 1
        route = route or self.rng.choice(ROUTES)
        return ContinuousVehicle(
            f"v_{self.veh_counter}", route, _pick_vehicle_class(self.rng), self.rng, self.time, partial
        )

    # ── signals ──
    def signal_red(self, road: str) -> bool:
        info = ROAD_JUNCTION_PHASE.get(road)
        if not info:
            return False
        jid, phase_name = info
        agent = self.coordinator.agents.get(jid)
        if not agent:
            return False
        if agent.is_yellow or agent.is_all_red:
            return True
        return agent.phase_names[agent.current_phase] != phase_name

    # ── simulation ──
    def advance(self) -> None:
        """Advance one simulated second, then let the controller decide on its cadence."""
        dt = SIM_SECONDS_PER_TICK / PHYSICS_SUBSTEPS
        for _ in range(PHYSICS_SUBSTEPS):
            self._physics_substep(dt)
        self.time += SIM_SECONDS_PER_TICK
        self._spawn()

        if self.time % AGENT_INTERVAL == 0:
            self.engine.snapshot()
            self._sync_bus_priority()
            if self.pedestrians:
                self._update_pedestrians()
            self.coordinator.step(self._vehicle_info_map(), data_ok=not self.sensor_fault)

    def _physics_substep(self, dt: float) -> None:
        road_groups: Dict[str, List[ContinuousVehicle]] = {}
        for v in self.fleet:
            road_groups.setdefault(v.road, []).append(v)
        for rlist in road_groups.values():
            rlist.sort(key=lambda x: x.dist, reverse=True)

        survivors = []
        for road, rlist in road_groups.items():
            red = self.signal_red(road)
            for i, v in enumerate(rlist):
                leader_dist = (rlist[i - 1].dist - v.dist) if i > 0 else None
                if v.update(dt, leader_dist, red):
                    survivors.append(v)
                else:
                    self._record_trip(v)
        self.fleet[:] = survivors

    def _record_trip(self, v: ContinuousVehicle) -> None:
        junctions = _route_junctions(v.route)
        if len(junctions) >= 2:
            self.od_counts[(junctions[0], junctions[-1])] += 1
        if v.partial:
            return
        self.trips.append({
            "vehicle_class": v.vehicle_class,
            "idle_seconds": round(v.idle_seconds, 1),
            "travel_seconds": self.time - v.spawn_time,
        })

    def _spawn(self) -> None:
        # Entry roads are shared, so check free space before adding a vehicle
        # (demand queues outside the network instead of stacking inside it).
        attempts = 0
        while len(self.fleet) < FLEET_TARGET and attempts < 6:
            attempts += 1
            route = self.rng.choice(ROUTES)
            entry = route[0]
            closest = min((v.dist for v in self.fleet if v.road == entry), default=None)
            if closest is not None and closest < SPAWN_CLEARANCE_M + 2.0:
                continue
            v = self._new_vehicle(route=route)
            v.dist = 0.0 if closest is None else self.rng.uniform(0.0, closest - SPAWN_CLEARANCE_M)
            self.fleet.append(v)

    def _vehicle_info_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            v.vid: {
                "speed": str(round(v.speed, 2)),
                "distance": str(round(v.dist, 2)),
                "road": v.road,
                "vehicle_type": v.vehicle_class,
            }
            for v in self.fleet
        }

    def _sync_bus_priority(self) -> None:
        """Feature 7: tell each junction about buses currently on its approaches, and forget ones that have left."""
        positions = {}
        for v in self.fleet:
            if v.vehicle_class == "bus" and v.road in ROAD_JUNCTION_PHASE:
                positions[v.vid] = (ROAD_JUNCTION_PHASE[v.road], v.lateness_min)
        for vid, ((jid, _phase), _late) in self._bus_positions.items():
            now = positions.get(vid)
            if now is None or now[0][0] != jid:
                self.coordinator.agents[jid].clear_bus(vid)
        for vid, ((jid, phase), late) in positions.items():
            self.coordinator.agents[jid].register_bus(vid, phase, 0.0, late)
        self._bus_positions = positions

    def _update_pedestrians(self) -> None:
        """Feature 10: people arrive to cross a road while its traffic has green, and cross once it turns red."""
        for agent in self.coordinator.agents.values():
            green = None if (agent.is_yellow or agent.is_all_red) else agent.phase_names[agent.current_phase]
            for phase in agent.phase_names:
                waiting = agent.pedestrian_demand.get(phase, False)
                if phase == green:
                    if not waiting and self.rng.random() < PED_ARRIVAL_PROB:
                        agent.set_pedestrian_demand(phase, True)
                elif waiting and green is not None:
                    agent.set_pedestrian_demand(phase, False)

    # ── reporting ──
    def trip_stats(self) -> Dict[str, Any]:
        trips = list(self.trips)
        n = len(trips)
        return {
            "trips": n,
            "avg_idle_s": round(sum(t["idle_seconds"] for t in trips) / n, 1) if n else 0.0,
            "avg_travel_s": round(sum(t["travel_seconds"] for t in trips) / n, 1) if n else 0.0,
        }

    def od_flow(self) -> List[Dict[str, Any]]:
        return [
            {"origin": o, "destination": d, "trips": n}
            for (o, d), n in self.od_counts.most_common()
        ]


# Live simulation (what the dashboard shows and operators control) and the
# fixed-time baseline it's compared against for Feature 24.
live = TrafficSim("live", pedestrians=True)
baseline = TrafficSim("baseline", fixed_time=True, seed=7)

EMISSIONS_MIN_TRIPS = 20


def _emissions_summary() -> Dict[str, Any]:
    """Feature 24: CO2 saved by the adaptive controller versus the fixed-time baseline, over recent trips."""
    live_stats = live.trip_stats()
    base_stats = baseline.trip_stats()
    summary = {
        "ready": live_stats["trips"] >= EMISSIONS_MIN_TRIPS and base_stats["trips"] >= EMISSIONS_MIN_TRIPS,
        "live": {**live_stats, "control_mode": live.coordinator.degradation.current},
        "baseline": {**base_stats, "control_mode": "FIXED_TIME"},
        "idle_seconds_saved_per_trip": 0.0,
        "co2_saved_kg": 0.0,
        "co2_saved_kg_per_100_trips": 0.0,
        "by_class": [],
        "note": (
            "Baseline is a parallel simulation with the same kind of demand under fixed-time signals. "
            "Emission factors are illustrative idle CO2 rates, not a certified inventory."
        ),
    }
    if not summary["ready"]:
        return summary

    base_by_class: Dict[str, List[float]] = {}
    for t in baseline.trips:
        base_by_class.setdefault(t["vehicle_class"], []).append(t["idle_seconds"])
    base_overall = base_stats["avg_idle_s"]

    live_by_class: Dict[str, List[float]] = {}
    for t in live.trips:
        live_by_class.setdefault(t["vehicle_class"], []).append(t["idle_seconds"])

    total_grams = 0.0
    by_class = []
    for cls, idles in sorted(live_by_class.items()):
        base_samples = base_by_class.get(cls)
        base_mean = (sum(base_samples) / len(base_samples)) if base_samples else base_overall
        baseline_total = base_mean * len(idles)
        actual_total = sum(idles)
        # Signed on purpose: if the adaptive controller idles more, that shows as a negative saving.
        grams = (baseline_total - actual_total) * emissions.idle_emission_factor(cls)
        total_grams += grams
        by_class.append({
            "vehicle_class": cls,
            "trips": len(idles),
            "baseline_avg_idle_s": round(base_mean, 1),
            "actual_avg_idle_s": round(actual_total / len(idles), 1),
            "co2_saved_g": round(grams, 1),
        })

    trips = live_stats["trips"]
    summary.update({
        "idle_seconds_saved_per_trip": round(base_stats["avg_idle_s"] - live_stats["avg_idle_s"], 1),
        "co2_saved_kg": round(total_grams / 1000.0, 3),
        "co2_saved_kg_per_100_trips": round(total_grams / 1000.0 / trips * 100, 3) if trips else 0.0,
        "by_class": by_class,
    })
    return summary


def _green_wave() -> List[Dict[str, Any]]:
    od = live.od_flow()
    if not od:
        return []
    return live.coordinator.compute_green_wave(od, LINK_DISTANCES_M, travel_speed_kmph=40.0, top_n=2)


def _bus_priority_state() -> Dict[str, Dict[str, Any]]:
    out = {}
    for jid, agent in live.coordinator.agents.items():
        out[jid] = {
            p: {
                "late_buses": agent.bus_priority.active_requests_for_phase(p),
                "boost": agent.bus_priority.boost_for_phase(p),
            }
            for p in agent.phase_names
        }
    return out


def _ambulance_state() -> Dict[str, Any]:
    coord = live.coordinator
    amb = dict(coord.ambulance)
    route = amb.get("route_junctions") or []
    if len(route) >= 2:
        amb["corridor"] = coord._phase_towards(route[0], route[1]) or amb.get("corridor")
    return amb


def _refresh_state():
    with sim_lock:
        vehicles = []
        speeds = []
        lane_vehs: Dict[str, int] = {}
        lane_wait: Dict[str, int] = {}

        for v in live.fleet:
            speeds.append(v.speed * 3.6)
            vehicles.append({
                "id": v.vid,
                "speed": str(round(v.speed, 2)),
                "distance": str(round(v.dist, 2)),
                "road": v.road,
                "type": v.vehicle_class,
                "running": "1" if v.speed >= 0.5 else "0",
            })
            key = f"{v.road}_0"
            lane_vehs[key] = lane_vehs.get(key, 0) + 1
            if v.is_waiting:
                lane_wait[key] = lane_wait.get(key, 0) + 1

        avg_spd = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
        total_capacity = len(ROAD_LENGTHS) * TrafficAgent.LANE_CAPACITY
        net_density = round(len(vehicles) / total_capacity * 100, 1) if total_capacity else 0.0

        coord = live.coordinator
        agent_states = {}
        tl_phases = {}
        for jid, agent in coord.agents.items():
            agent_states[jid] = agent.get_broadcast_message()
            tl_phases[jid] = {
                "phase_idx": agent.current_phase,
                "phase_name": PHASE_NAMES[agent.current_phase],
                "is_yellow": agent.is_yellow,
                "is_all_red": agent.is_all_red,
            }

        messages = [
            {
                "sender": m["sender"],
                "junction_id": m["junction_id"],
                "timestamp": m["timestamp"],
                "decision_reason": m["decision_reason"],
            }
            for m in coord.message_history[-15:]
        ]

        snapshot = {
            "step": int(live.time),
            "running": not ctrl["paused"],
            "total_vehicles": len(vehicles),
            "avg_travel_time": live.trip_stats()["avg_travel_s"],
            "avg_speed": avg_spd,
            "network_density": net_density,
            "total_waiting": sum(lane_wait.values()),
            "vehicles": vehicles,
            "lane_vehicles": lane_vehs,
            "lane_waiting": lane_wait,
            "tl_phases": tl_phases,
            "agents": agent_states,
            "agent_messages": messages,
            "active_incidents": list(coord.active_incidents.values()),
            "ambulance": _ambulance_state(),
            "hospitals": dict(coord.hospitals),
            "control_mode": coord.degradation.current,
            "control_mode_forced": coord.degradation._forced is not None,
            "sensor_fault": live.sensor_fault,
            "green_wave": _green_wave(),
            "bus_priority": _bus_priority_state(),
            "emissions": _emissions_summary(),
        }

    with state_lock:
        sim_state.update(snapshot)


def _advance_all():
    with sim_lock:
        live.advance()
        baseline.advance()


def _sim_worker():
    while True:
        try:
            if not ctrl["paused"]:
                _advance_all()
                _refresh_state()
        except Exception as e:
            print(f"[CityFlow Worker] Exception during step: {e}", flush=True)

        time.sleep(ctrl["step_delay"])


# Initial state population and background thread launch (runs under Gunicorn & Standalone)
_refresh_state()
_sim_thread = threading.Thread(target=_sim_worker, daemon=True)
_sim_thread.start()


REACT_DIST = os.environ.get(
    "REACT_DIST_DIR",
    os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist"))
)


# ── Routes ──

@app.route("/cityflow-sim")
def cityflow_sim():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404

    if os.path.exists(REACT_DIST):
        target = os.path.join(REACT_DIST, path)
        if path and os.path.exists(target) and not os.path.isdir(target):
            return send_from_directory(REACT_DIST, path)
        return send_from_directory(REACT_DIST, "index.html")

    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/state")
def get_state():
    with state_lock:
        return jsonify(dict(sim_state))


@app.route("/api/roadnet")
def get_roadnet():
    path = os.path.join(BASE_DIR, "roadnet_5j.json")
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/control", methods=["POST"])
def control():
    data = request.get_json(silent=True) or {}
    cmd = data.get("cmd")

    if cmd == "start":
        ctrl["paused"] = False
    elif cmd == "pause":
        ctrl["paused"] = True
    elif cmd == "step":
        ctrl["paused"] = True
        _advance_all()
    elif cmd == "reset":
        ctrl["paused"] = False
        with sim_lock:
            live.reset()
            baseline.reset()
    elif cmd == "speed":
        try:
            ctrl["step_delay"] = max(0.02, float(data.get("value", 0.10)))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "value must be a number"}), 400

    _refresh_state()
    with state_lock:
        return jsonify({"ok": True, "state": dict(sim_state)})


@app.route("/api/incident", methods=["POST"])
def handle_incident():
    data = request.get_json(silent=True) or {}
    junction = data.get("junction", "J3")
    road = data.get("road", "road_J3_J2")
    itype = data.get("type", "ACCIDENT")
    active = bool(data.get("active", True))

    with sim_lock:
        live.coordinator.set_incident(junction, road, itype, active)
    _refresh_state()
    return jsonify({"ok": True, "incidents": list(live.coordinator.active_incidents.values())})


@app.route("/api/ambulance", methods=["POST"])
def handle_ambulance():
    """
    Feature 11: dispatch an ambulance from `origin` to the nearest hospital,
    or stand it down. Pass {"active": true|false}; omit it to toggle.
    """
    data = request.get_json(silent=True) or {}
    coord = live.coordinator
    with sim_lock:
        want_active = data.get("active")
        if want_active is None:
            want_active = not coord.ambulance.get("active", False)

        if want_active:
            if not coord.ambulance.get("active"):
                origin = data.get("origin", "J1")
                if origin not in coord.agents:
                    return jsonify({"ok": False, "error": f"Unknown origin junction '{origin}'"}), 400
                if coord.dispatch_ambulance(origin) is None:
                    return jsonify({"ok": False, "error": f"No hospital reachable from {origin}"}), 409
        else:
            coord.ambulance["active"] = False
            coord.ambulance["progress_m"] = 0.0
            for agent in coord.agents.values():
                if agent.emergency_override is not None:
                    agent.set_emergency(None)
    _refresh_state()
    return jsonify({"ok": True, "ambulance": _ambulance_state()})


@app.route("/api/override", methods=["POST"])
def handle_override():
    data = request.get_json(silent=True) or {}
    junction = data.get("junction")
    try:
        phase = int(data.get("phase", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "phase must be 0 or 1"}), 400

    agent = live.coordinator.agents.get(junction)
    if agent and phase in (0, 1):
        with sim_lock:
            agent.current_phase = phase
            agent.target_phase = phase
            agent.is_yellow = False
            agent.is_all_red = False
            agent._pending_all_red = False
            agent.steps_on_phase = 0
            agent.last_decision_reason = f"👤 Manual override — forced {PHASE_NAMES[phase]}"
        _refresh_state()
        return jsonify({"ok": True})

    return jsonify({"ok": False}), 400


@app.route("/api/control_mode", methods=["POST"])
def handle_control_mode():
    """Feature 9: pin the degradation ladder to a rung, or {"mode": "AUTO"} to release it."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    with sim_lock:
        try:
            current = live.coordinator.force_control_mode(None if mode in (None, "AUTO") else mode)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
    _refresh_state()
    return jsonify({"ok": True, "control_mode": current, "forced": mode not in (None, "AUTO")})


@app.route("/api/sensor_fault", methods=["POST"])
def handle_sensor_fault():
    """Feature 9: simulate the vehicle feed failing, so the ladder steps down on its own (and back up once cleared)."""
    data = request.get_json(silent=True) or {}
    with sim_lock:
        live.sensor_fault = bool(data.get("active", not live.sensor_fault))
    _refresh_state()
    return jsonify({"ok": True, "sensor_fault": live.sensor_fault})


@app.route("/api/pedestrian", methods=["POST"])
def handle_pedestrian():
    """Feature 10: a pedestrian push-button / detector call to cross `phase`'s road at `junction`."""
    data = request.get_json(silent=True) or {}
    agent = live.coordinator.agents.get(data.get("junction"))
    phase = data.get("phase")
    if not agent or phase not in agent.phase_names:
        return jsonify({"ok": False, "error": "junction must be J1–J5 and phase EW or NS"}), 400
    with sim_lock:
        agent.set_pedestrian_demand(phase, bool(data.get("waiting", True)))
    _refresh_state()
    return jsonify({"ok": True, "pedestrian_demand": dict(agent.pedestrian_demand)})


@app.route("/api/green_wave")
def handle_green_wave():
    """Feature 5: corridors worth synchronising, picked from the simulation's origin–destination flows."""
    with sim_lock:
        return jsonify({"ok": True, "od_flow": live.od_flow(), "corridors": _green_wave()})


@app.route("/api/emissions")
def handle_emissions():
    """Feature 24: idle time and CO2 saved versus the fixed-time baseline."""
    with sim_lock:
        return jsonify(_emissions_summary())


# ── Physical demo board: overhead camera -> Signal AI -> LEDs over USB ──
if os.environ.get("BOARD_DISABLE") != "1":
    try:
        from board_api import agent_signals, register_board_routes
        register_board_routes(
            app,
            sim_signals=lambda: {jid: agent_signals(a) for jid, a in live.coordinator.agents.items()},
        )
    except Exception as _board_err:
        print(f"[Server] Note: could not mount board routes: {_board_err}")


if __name__ == "__main__":
    print("=" * 60)
    print("  CityFlow Standalone Server (Simulated Mode)")
    print("  Running at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
