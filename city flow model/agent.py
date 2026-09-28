"""
agent.py — Distributed Multi-Agent Traffic Light Control System
================================================================
"""

import time
from typing import Dict, List, Any, Optional

import pcu
from degradation import DegradationLadder


class TrafficAgent:
    LANE_CAPACITY = 14.0
    MIN_GREEN = 10
    MAX_GREEN = 40
    YELLOW_TIME = 2
    STARVATION_LIMIT = 35
    SWITCH_RATIO = 1.25
    # Feature 3: once the outgoing road for the current phase is at/above this
    # density, stop feeding it more vehicles (box-blocking / spillback risk).
    SPILLBACK_DENSITY_THRESHOLD = 0.90
    # Feature 4: absolute wait ceiling — past this, the starved approach gets
    # green unconditionally, regardless of relative priority or spillback risk.
    STARVATION_HARD_CAP = 55

    # Feature 9: degradation ladder. Each rung needs strictly less live data
    # than the one before it, so the controller can keep running signal
    # timing safely as inputs get worse:
    #   FULL_AI            — everything: PCU density, downstream/spillback
    #                         awareness, incident penalties, live priorities.
    #   HISTORICAL_PROFILE — no live density/priority at all, just a
    #                         pre-set time-of-day green split per phase.
    #   LOCAL_ACTUATED     — live local queues/waits only, no cross-junction
    #                         (downstream/incident) awareness.
    #   FIXED_TIME         — ignores all sensor data; plain round-robin
    #                         fixed-duration phases (the classic hardware
    #                         fail-safe behaviour).
    # Emergency ambulance preemption and the minimum/maximum green
    # interlocks stay active at every rung — those are safety behaviour,
    # not "smart" control, and shouldn't degrade with data quality.
    CONTROL_MODES = ("FULL_AI", "HISTORICAL_PROFILE", "LOCAL_ACTUATED", "FIXED_TIME")
    FIXED_TIME_GREEN = 25
    HISTORICAL_CYCLE_LENGTH = 50

    # Feature 10: pedestrian-demand walk phase. Demand for phase P means
    # "people are waiting to cross P's road" — they can only walk while P is
    # red (i.e. the other phase is green). PED_MAX_WAIT forces P to end early
    # if that wait drags on; PED_MIN_WALK then protects the crossing for a
    # minimum interval by refusing to switch back to P prematurely.
    # Kept below MAX_GREEN on purpose — a pedestrian shouldn't have to wait
    # as long as a full vehicle max-green cycle before getting a call served.
    PED_MAX_WAIT = 30
    PED_MIN_WALK = 12

    def __init__(
        self,
        agent_id: str,
        junction_id: str,
        engine: Any,
        incoming_roads: Dict[str, List[str]],
        outgoing_neighbors: Dict[str, Optional[str]]
    ):
        self.agent_id = agent_id
        self.junction_id = junction_id
        self.engine = engine
        self.incoming_roads = incoming_roads
        self.outgoing_neighbors = outgoing_neighbors
        self.phase_names = list(incoming_roads.keys())   # ['EW', 'NS']

        self.current_phase = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.target_phase = 0
        self.steps_on_phase = 0
        self.allocated_green = self.MIN_GREEN

        self.waiting_time_tracker = {p: 0 for p in self.phase_names}
        self.local_obs = {}
        self.congestion_scores = {}
        self.priorities = {}
        self.last_decision_reason = "System Initialized."
        self.incidents: Dict[str, Any] = {}
        self.emergency_override: Optional[str] = None
        self.step_counter = 0

        self.control_mode = "FULL_AI"
        self.historical_green_split = {p: 1.0 / len(self.phase_names) for p in self.phase_names}

        # Feature 10: pedestrian demand. Keyed by phase_name = "people
        # waiting to cross this phase's road" (so they walk while it's red).
        self.pedestrian_demand: Dict[str, bool] = {p: False for p in self.phase_names}
        self.pedestrian_wait_tracker: Dict[str, int] = {p: 0 for p in self.phase_names}
        self.pedestrian_protection_remaining: Dict[str, int] = {p: 0 for p in self.phase_names}

    def set_control_mode(self, mode: str) -> None:
        """Move this agent to a rung of the degradation ladder (Feature 9)."""
        if mode not in self.CONTROL_MODES:
            raise ValueError(f"Unknown control mode '{mode}', expected one of {self.CONTROL_MODES}")
        self.control_mode = mode

    def set_historical_profile(self, green_split: Dict[str, float]) -> None:
        """Set the fixed time-of-day green split used by HISTORICAL_PROFILE mode. Values are normalised to sum to 1."""
        total = sum(max(0.0, v) for v in green_split.values()) or 1.0
        self.historical_green_split = {
            p: max(0.0, green_split.get(p, 0.0)) / total for p in self.phase_names
        }

    def set_pedestrian_demand(self, phase_name: str, waiting: bool) -> None:
        """Feature 10: called from a camera/detector — people are (or are no longer) waiting to cross phase_name's road."""
        if phase_name not in self.pedestrian_demand:
            return
        self.pedestrian_demand[phase_name] = waiting
        if not waiting:
            self.pedestrian_wait_tracker[phase_name] = 0

    def _pcu_counts_for_lane(self, lane_id: str, raw_count: int, vehicle_info_map: Dict[str, Any]) -> float:
        """
        PCU-weighted vehicle count for one lane (Feature 1). Uses the engine's
        per-lane vehicle id list when available so mixed traffic (buses,
        trucks, two-wheelers) counts for more/less than one car each. Falls
        back to the raw count untouched when the engine can't hand back ids
        (older CityFlow builds, or a lightweight test double), so behaviour
        never regresses when this information just isn't there.
        """
        get_lane_vehicles = getattr(self.engine, "get_lane_vehicles", None)
        if not callable(get_lane_vehicles):
            return float(raw_count)
        try:
            ids = get_lane_vehicles().get(lane_id, [])
        except Exception:
            return float(raw_count)
        if not ids:
            return 0.0
        _, weighted = pcu.weighted_count(ids, vehicle_info_map)
        return weighted

    def observe(self, vehicle_info_map: Dict[str, Any]) -> Dict[str, Any]:
        lane_waiting = self.engine.get_lane_waiting_vehicle_count()
        lane_vehicles = self.engine.get_lane_vehicle_count()

        obs = {}
        for phase_name, roads in self.incoming_roads.items():
            total_veh = 0        # raw vehicle count (kept for API/back-compat)
            waiting_veh = 0      # raw waiting count (kept for API/back-compat)
            total_pcu = 0.0      # PCU-weighted count, drives density/control decisions
            waiting_pcu = 0.0    # PCU-weighted waiting count
            speeds = []

            for r in roads:
                lane_id = f"{r}_0"
                w = lane_waiting.get(lane_id, 0)
                v = lane_vehicles.get(lane_id, 0)
                total_veh += v
                waiting_veh += w
                total_pcu += self._pcu_counts_for_lane(lane_id, v, vehicle_info_map)
                # No per-lane "which ids are waiting" call exists on the engine,
                # so approximate weighted waiting by applying the lane's
                # observed PCU/raw ratio to the raw waiting count.
                lane_pcu_ratio = (total_pcu / v) if v else 1.0
                waiting_pcu += w * lane_pcu_ratio

            for v_data in vehicle_info_map.values():
                if v_data.get("road") in roads:
                    try:
                        speeds.append(float(v_data.get("speed", 0.0)))
                    except (ValueError, TypeError):
                        pass

            avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 16.67
            total_cap = len(roads) * self.LANE_CAPACITY
            total_pcu = round(total_pcu, 3)
            waiting_pcu = round(waiting_pcu, 3)

            density = min(1.0, round(total_pcu / total_cap, 3))
            queue_score = min(1.0, round(waiting_pcu / total_cap, 3))

            is_active = (self.phase_names[self.current_phase] == phase_name and not self.is_yellow)
            if is_active:
                self.waiting_time_tracker[phase_name] = max(0, self.waiting_time_tracker[phase_name] - 2)
            else:
                if waiting_veh > 0:
                    self.waiting_time_tracker[phase_name] += 1
                else:
                    self.waiting_time_tracker[phase_name] = 0

            # Feature 10: pedestrians waiting to cross phase_name's road can
            # only be counted against while phase_name is actually green
            # (that's when their crossing is blocked); once it goes red
            # they're free to walk, so the wait resets.
            if is_active and self.pedestrian_demand.get(phase_name):
                self.pedestrian_wait_tracker[phase_name] = self.pedestrian_wait_tracker.get(phase_name, 0) + 1
            elif not is_active:
                self.pedestrian_wait_tracker[phase_name] = 0
            if not is_active and self.pedestrian_protection_remaining.get(phase_name, 0) > 0:
                self.pedestrian_protection_remaining[phase_name] -= 1

            wait_sec = self.waiting_time_tracker[phase_name]
            waiting_score = min(1.0, round(wait_sec / self.STARVATION_LIMIT, 3))

            cong_score = round(
                0.50 * density + 0.30 * queue_score + 0.20 * waiting_score,
                3
            )

            obs[phase_name] = {
                "vehicle_count": total_veh,
                "queue_length": waiting_veh,
                "vehicle_count_pcu": total_pcu,
                "queue_length_pcu": waiting_pcu,
                "average_speed": avg_speed,
                "lane_capacity": total_cap,
                "density": density,
                "queue_score": queue_score,
                "waiting_time": wait_sec,
                "waiting_score": waiting_score,
                "congestion_score": cong_score,
                "status": "HIGH" if density > 0.55 else "MEDIUM" if density > 0.22 else "LOW",
                "pedestrian_waiting": self.pedestrian_demand.get(phase_name, False),
                "pedestrian_wait_time": self.pedestrian_wait_tracker.get(phase_name, 0),
                "pedestrian_walk_protected": self.pedestrian_protection_remaining.get(phase_name, 0) > 0
            }

        self.local_obs = obs
        return obs

    def compute_priority(
        self,
        phase_name: str,
        neighbor_states: Dict[str, Any],
        use_downstream: bool = True
    ) -> float:
        obs = self.local_obs.get(phase_name, {})
        local_cong = obs.get("congestion_score", 0.0)

        if not use_downstream:
            # LOCAL_ACTUATED rung (Feature 9): decide from this junction's own
            # queues only, with no cross-junction/incident awareness.
            priority = local_cong
        else:
            # Check if an incident is active on any outgoing road for this phase
            incident_penalty = 1.0
            for inc in self.incidents.values():
                if inc.get("active"):
                    inc_road = inc.get("road", "")
                    # If accident is on road_J3_J2 (outgoing North road), penalize NS phase at J3
                    if "J3_J2" in inc_road and phase_name == "NS" and self.junction_id == "J3":
                        incident_penalty = 0.08

            neighbor_id = self.outgoing_neighbors.get(phase_name)
            downstream_density = 0.0

            if neighbor_id and neighbor_id in neighbor_states:
                n_data = neighbor_states[neighbor_id]
                downstream_density = n_data.get("overall_density", 0.0)

            downstream_cap_factor = max(0.20, 1.0 - downstream_density) * incident_penalty
            priority = local_cong * downstream_cap_factor

        wait_time = obs.get("waiting_time", 0)
        if wait_time > self.STARVATION_LIMIT:
            boost = min(0.6, (wait_time - self.STARVATION_LIMIT) / self.STARVATION_LIMIT * 0.5)
            priority += boost

        if self.emergency_override == phase_name:
            priority += 10.0

        return round(priority, 3)

    def decide_and_act(self, neighbor_states: Dict[str, Any]) -> str:
        self.step_counter += 1

        if self.is_yellow:
            self.yellow_timer -= 1
            if self.yellow_timer <= 0:
                self.is_yellow = False
                self.current_phase = self.target_phase
                self.steps_on_phase = 0
                self.engine.set_tl_phase(self.junction_id, self.current_phase)
                new_name = self.phase_names[self.current_phase]
                self.last_decision_reason = f"Green activated for {new_name} ({self.allocated_green}s allocated)."
                return self.last_decision_reason
            else:
                self.last_decision_reason = f"Yellow clearance interval ({self.yellow_timer}s remaining)."
                return self.last_decision_reason

        self.steps_on_phase += 1

        # 1. Emergency Ambulance Preemption (Takes absolute precedence)
        if self.emergency_override:
            target_phase_name = self.emergency_override
            target_idx = self.phase_names.index(target_phase_name) if target_phase_name in self.phase_names else 0
            if self.current_phase != target_idx:
                self._initiate_switch(target_idx, 1.0, "🚨 Emergency Ambulance Corridor Preemption")
                return self.last_decision_reason
            else:
                self.last_decision_reason = "🚨 Holding GREEN for Ambulance Corridor."
                return self.last_decision_reason

        cur_phase_name = self.phase_names[self.current_phase]
        other_phase_idx = 1 - self.current_phase
        other_phase_name = self.phase_names[other_phase_idx]

        # Feature 9, rung 4 (most degraded): ignores all sensor data.
        if self.control_mode == "FIXED_TIME":
            return self._decide_fixed_time(cur_phase_name, other_phase_idx)

        use_downstream = self.control_mode != "LOCAL_ACTUATED"

        # 2. Autonomous Multi-Agent Adaptive Control
        priorities = {}
        for p in self.phase_names:
            priorities[p] = self.compute_priority(p, neighbor_states, use_downstream=use_downstream)
        self.priorities = priorities

        cur_priority = priorities.get(cur_phase_name, 0.0)
        other_priority = priorities.get(other_phase_name, 0.0)

        # If an active incident is on our outgoing link, notify in reason
        active_inc = [i for i in self.incidents.values() if i.get("active")]
        incident_note = ""
        if active_inc:
            incident_note = f" [⚠️ Blockade on {active_inc[0].get('road')}]"

        # Minimum green constraint
        if self.steps_on_phase < self.MIN_GREEN:
            remaining_min = self.MIN_GREEN - self.steps_on_phase
            self.last_decision_reason = f"Holding {cur_phase_name}-Green (Min hold: {remaining_min}s left){incident_note}."
            return self.last_decision_reason

        # Feature 9, rung 2: no live priorities at all, just the historical split.
        if self.control_mode == "HISTORICAL_PROFILE":
            return self._decide_historical_profile(cur_phase_name, other_phase_idx, other_phase_name)

        # Feature 10: pedestrian call — people have been waiting too long to
        # cross the currently-green approach, so force the switch and open a
        # protected walk window on the way out.
        ped_wait = self.pedestrian_wait_tracker.get(cur_phase_name, 0)
        if self.pedestrian_demand.get(cur_phase_name) and ped_wait > self.PED_MAX_WAIT:
            self._initiate_switch(
                other_phase_idx, max(other_priority, 1.0),
                f"🚶 Pedestrian call: {cur_phase_name} crossing waited {ped_wait}s ≥ {self.PED_MAX_WAIT}s"
            )
            self.pedestrian_protection_remaining[cur_phase_name] = self.PED_MIN_WALK
            return self.last_decision_reason

        # Feature 10 continued: don't switch back into a phase whose
        # pedestrians are still inside their protected walk window.
        ped_protected_other = self.pedestrian_protection_remaining.get(other_phase_name, 0) > 0

        # Maximum green constraint
        if self.steps_on_phase >= self.MAX_GREEN:
            self._initiate_switch(other_phase_idx, other_priority, "Max green duration reached")
            return self.last_decision_reason

        other_wait = self.local_obs.get(other_phase_name, {}).get("waiting_time", 0)

        # Feature 4: starvation hard cap — past this wait, force the switch no
        # matter what the priority comparison says (safety ceiling, not a
        # preference). Checked before spillback so a starved approach is
        # never left waiting behind a spillback hold. Active on every rung
        # that still tracks live waits (FULL_AI and LOCAL_ACTUATED).
        if other_wait > self.STARVATION_HARD_CAP:
            self._initiate_switch(
                other_phase_idx, max(other_priority, 1.0),
                f"Hard starvation cap reached ({other_phase_name} waited {other_wait}s ≥ {self.STARVATION_HARD_CAP}s)"
            )
            return self.last_decision_reason

        # Feature 3: spillback / box-blocking prevention — needs downstream
        # density, so it only runs at the FULL_AI rung.
        if use_downstream:
            cur_neighbor = self.outgoing_neighbors.get(cur_phase_name)
            cur_downstream_density = 0.0
            if cur_neighbor and cur_neighbor in neighbor_states:
                cur_downstream_density = neighbor_states[cur_neighbor].get("overall_density", 0.0)
            if cur_downstream_density >= self.SPILLBACK_DENSITY_THRESHOLD:
                self._initiate_switch(
                    other_phase_idx, other_priority,
                    f"Spillback prevention: exit toward {cur_neighbor} at {int(cur_downstream_density * 100)}% capacity"
                )
                return self.last_decision_reason

        # Starvation trigger (soft — only switches early if the other phase
        # also has real need, and not into a phase still protecting a walk)
        if other_wait > self.STARVATION_LIMIT and other_priority > cur_priority and not ped_protected_other:
            self._initiate_switch(
                other_phase_idx,
                other_priority,
                f"Starvation prevented ({other_phase_name} waited {other_wait}s)"
            )
            return self.last_decision_reason

        # Adaptive switch (also respects an active walk-protection window)
        if other_priority > cur_priority * self.SWITCH_RATIO and other_priority > 0.15 and not ped_protected_other:
            neighbor = self.outgoing_neighbors.get(cur_phase_name)
            reason = f"Higher load on {other_phase_name} (P={other_priority} vs {cur_priority}){incident_note}"
            if use_downstream and neighbor and neighbor in neighbor_states:
                n_cap = 1.0 - neighbor_states[neighbor].get("overall_density", 0.0)
                reason += f" · Neighbor {neighbor} cap={int(n_cap*100)}%"
            self._initiate_switch(other_phase_idx, other_priority, reason)
            return self.last_decision_reason

        mode_tag = "" if self.control_mode == "FULL_AI" else f"[{self.control_mode}] "
        ped_note = f" [🚶 protecting {other_phase_name} walk, {self.pedestrian_protection_remaining.get(other_phase_name, 0)}s left]" if ped_protected_other else ""
        self.last_decision_reason = (
            f"{mode_tag}Maintaining {cur_phase_name}-Green (P={cur_priority} vs {other_phase_name} P={other_priority}){incident_note}{ped_note}."
        )
        return self.last_decision_reason

    def _decide_fixed_time(self, cur_phase_name: str, other_phase_idx: int) -> str:
        """Feature 9 rung 4: plain round-robin fixed-duration phases, no sensor input at all."""
        if self.steps_on_phase < self.MIN_GREEN:
            remaining = self.MIN_GREEN - self.steps_on_phase
            self.last_decision_reason = f"[FIXED_TIME] Holding {cur_phase_name}-Green (min hold: {remaining}s left)."
            return self.last_decision_reason
        if self.steps_on_phase >= self.FIXED_TIME_GREEN:
            self._initiate_switch(other_phase_idx, 1.0, f"[FIXED_TIME] Fixed {self.FIXED_TIME_GREEN}s cycle elapsed")
            self.allocated_green = self.FIXED_TIME_GREEN
            return self.last_decision_reason
        remaining = self.FIXED_TIME_GREEN - self.steps_on_phase
        self.last_decision_reason = f"[FIXED_TIME] Holding {cur_phase_name}-Green ({remaining}s left in fixed cycle)."
        return self.last_decision_reason

    def _decide_historical_profile(self, cur_phase_name: str, other_phase_idx: int, other_phase_name: str) -> str:
        """Feature 9 rung 2: a pre-set time-of-day green split, no live priority computation."""
        other_wait = self.local_obs.get(other_phase_name, {}).get("waiting_time", 0)
        # Even a non-adaptive fallback must not starve an approach indefinitely.
        if other_wait > self.STARVATION_HARD_CAP:
            other_target = max(self.MIN_GREEN, round(
                self.historical_green_split.get(other_phase_name, 0.5) * self.HISTORICAL_CYCLE_LENGTH))
            self._initiate_switch(
                other_phase_idx, 1.0,
                f"[HISTORICAL_PROFILE] Hard starvation cap reached ({other_phase_name} waited {other_wait}s)"
            )
            self.allocated_green = other_target
            return self.last_decision_reason

        target_green = max(self.MIN_GREEN, round(
            self.historical_green_split.get(cur_phase_name, 0.5) * self.HISTORICAL_CYCLE_LENGTH))
        if self.steps_on_phase >= target_green:
            other_target = max(self.MIN_GREEN, round(
                self.historical_green_split.get(other_phase_name, 0.5) * self.HISTORICAL_CYCLE_LENGTH))
            self._initiate_switch(other_phase_idx, 1.0, f"[HISTORICAL_PROFILE] Historical split reached ({target_green}s)")
            self.allocated_green = other_target
            return self.last_decision_reason

        remaining = target_green - self.steps_on_phase
        self.last_decision_reason = (
            f"[HISTORICAL_PROFILE] Holding {cur_phase_name}-Green ({remaining}s left of historical {target_green}s)."
        )
        return self.last_decision_reason

    def _initiate_switch(self, new_phase_idx: int, priority: float, reason: str):
        self.allocated_green = int(self.MIN_GREEN + min(1.0, priority) * (self.MAX_GREEN - self.MIN_GREEN))
        self.target_phase = new_phase_idx
        self.is_yellow = True
        self.yellow_timer = self.YELLOW_TIME
        old_name = self.phase_names[self.current_phase]
        new_name = self.phase_names[new_phase_idx]
        self.last_decision_reason = f"Switching {old_name} → {new_name} ({reason}). Allocated: {self.allocated_green}s."

    def get_broadcast_message(self) -> Dict[str, Any]:
        densities = [d.get("density", 0.0) for d in self.local_obs.values()]
        queues = [d.get("queue_length", 0) for d in self.local_obs.values()]
        speeds = [d.get("average_speed", 16.67) for d in self.local_obs.values()]

        avg_density = round(sum(densities) / len(densities), 2) if densities else 0.0
        total_queue = sum(queues)
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 16.67
        avail_capacity = max(0.0, round(1.0 - avg_density, 2))

        active_incidents = [i for i in self.incidents.values() if i.get("active")]

        return {
            "sender": self.agent_id,
            "junction_id": self.junction_id,
            "timestamp": int(self.engine.get_current_time()),
            "current_phase": self.phase_names[self.current_phase] if not self.is_yellow else "YELLOW",
            "phase_idx": self.current_phase,
            "is_yellow": self.is_yellow,
            "steps_on_phase": self.steps_on_phase,
            "allocated_green": self.allocated_green,
            "overall_density": avg_density,
            "total_queue": total_queue,
            "average_speed": avg_speed,
            "available_capacity": avail_capacity,
            "incident_active": len(active_incidents) > 0,
            "incidents": active_incidents,
            "emergency_active": self.emergency_override is not None,
            "local_obs": self.local_obs,
            "priorities": self.priorities,
            "decision_reason": self.last_decision_reason,
            "control_mode": self.control_mode
        }

    def set_incident(self, road_id: str, incident_type: str, active: bool = True):
        if active:
            self.incidents[road_id] = {
                "junction": self.junction_id,
                "road": road_id,
                "type": incident_type,
                "active": True,
                "timestamp": time.time()
            }
        else:
            self.incidents.pop(road_id, None)

    def set_emergency(self, phase_name: Optional[str]):
        self.emergency_override = phase_name

    def reset(self):
        self.current_phase = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.target_phase = 0
        self.steps_on_phase = 0
        self.allocated_green = self.MIN_GREEN
        self.waiting_time_tracker = {p: 0 for p in self.phase_names}
        self.local_obs = {}
        self.priorities = {}
        self.incidents = {}
        self.emergency_override = None
        self.control_mode = "FULL_AI"
        self.pedestrian_demand = {p: False for p in self.phase_names}
        self.pedestrian_wait_tracker = {p: 0 for p in self.phase_names}
        self.pedestrian_protection_remaining = {p: 0 for p in self.phase_names}
        self.last_decision_reason = "Reset Complete."


class MultiAgentCoordinator:
    def __init__(self, engine: Any):
        self.engine = engine
        self.agents: Dict[str, TrafficAgent] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.degradation = DegradationLadder()

        self.ambulance: Dict[str, Any] = {
            "active": False,
            "progress_m": 0.0,
            "speed": 16.0,
            "route_roads": ["road_VW1_J1", "road_J1_J3", "road_J3_J4", "road_J4_VE4"],
            "current_road": "road_VW1_J1",
            "road_dist": 0.0,
            "corridor": "EW"
        }

        self.active_incidents: Dict[str, Any] = {}
        self._setup_network()

    def _setup_network(self):
        self.agents["J1"] = TrafficAgent(
            agent_id="Agent-J1", junction_id="J1", engine=self.engine,
            incoming_roads={"EW": ["road_VW1_J1", "road_J3_J1"], "NS": ["road_VN1_J1", "road_VS1_J1"]},
            outgoing_neighbors={"EW": "J3", "NS": None}
        )

        self.agents["J2"] = TrafficAgent(
            agent_id="Agent-J2", junction_id="J2", engine=self.engine,
            incoming_roads={"EW": ["road_VW2_J2", "road_VE2_J2"], "NS": ["road_VN2_J2", "road_J3_J2"]},
            outgoing_neighbors={"EW": None, "NS": "J3"}
        )

        self.agents["J3"] = TrafficAgent(
            agent_id="Agent-J3", junction_id="J3", engine=self.engine,
            incoming_roads={"EW": ["road_J1_J3", "road_J4_J3"], "NS": ["road_J2_J3", "road_J5_J3"]},
            outgoing_neighbors={"EW": "J4", "NS": "J5"}
        )

        self.agents["J4"] = TrafficAgent(
            agent_id="Agent-J4", junction_id="J4", engine=self.engine,
            incoming_roads={"EW": ["road_J3_J4", "road_VE4_J4"], "NS": ["road_VN4_J4", "road_VS4_J4"]},
            outgoing_neighbors={"EW": "J3", "NS": None}
        )

        self.agents["J5"] = TrafficAgent(
            agent_id="Agent-J5", junction_id="J5", engine=self.engine,
            incoming_roads={"EW": ["road_VW5_J5", "road_VE5_J5"], "NS": ["road_J3_J5", "road_VS5_J5"]},
            outgoing_neighbors={"EW": None, "NS": "J3"}
        )

    def step(self, vehicle_info_map: Dict[str, Any], data_ok: bool = True) -> Dict[str, Any]:
        """
        data_ok: health signal for this step's sensor data (Feature 9). Pass
        False when the caller knows this step's input was bad/missing/stale
        (e.g. the vehicle feed timed out) to push the degradation ladder
        toward a safer, less data-hungry control mode; defaults to True so
        existing callers that never pass it keep running at FULL_AI exactly
        as before.
        """
        current_rung = self.degradation.record_health(data_ok)
        for agent in self.agents.values():
            if agent.control_mode != current_rung:
                agent.set_control_mode(current_rung)

        self._update_ambulance()

        for agent in self.agents.values():
            agent.observe(vehicle_info_map)

        current_broadcasts = {}
        for jid, agent in self.agents.items():
            msg = agent.get_broadcast_message()
            current_broadcasts[jid] = msg

        for jid, msg in current_broadcasts.items():
            if len(self.message_history) > 100:
                self.message_history.pop(0)
            self.message_history.append(msg)

        decisions = {}
        for jid, agent in self.agents.items():
            reason = agent.decide_and_act(current_broadcasts)
            decisions[jid] = reason

        return {
            "broadcasts": current_broadcasts,
            "decisions": decisions,
            "ambulance": dict(self.ambulance),
            "control_mode": current_rung
        }

    def force_control_mode(self, rung: Optional[str]) -> str:
        """Operator override for the degradation ladder (Feature 9). Pass None to release the pin."""
        return self.degradation.force(rung)

    def _update_ambulance(self):
        if not self.ambulance["active"]:
            return

        self.ambulance["progress_m"] += self.ambulance["speed"]
        prog = self.ambulance["progress_m"]

        if prog < 200:
            self.ambulance["current_road"] = "road_VW1_J1"
            self.ambulance["road_dist"] = prog
            self.agents["J1"].set_emergency("EW")
            self.agents["J3"].set_emergency("EW")
        elif prog < 400:
            self.ambulance["current_road"] = "road_J1_J3"
            self.ambulance["road_dist"] = prog - 200
            self.agents["J1"].set_emergency(None)
            self.agents["J3"].set_emergency("EW")
            self.agents["J4"].set_emergency("EW")
        elif prog < 600:
            self.ambulance["current_road"] = "road_J3_J4"
            self.ambulance["road_dist"] = prog - 400
            self.agents["J3"].set_emergency(None)
            self.agents["J4"].set_emergency("EW")
        elif prog < 800:
            self.ambulance["current_road"] = "road_J4_VE4"
            self.ambulance["road_dist"] = prog - 600
            self.agents["J4"].set_emergency(None)
        else:
            self.ambulance["active"] = False
            self.ambulance["progress_m"] = 0.0
            for agent in self.agents.values():
                agent.set_emergency(None)

    def dispatch_ambulance(self):
        self.ambulance["active"] = True
        self.ambulance["progress_m"] = 0.0
        self.ambulance["current_road"] = "road_VW1_J1"
        self.ambulance["road_dist"] = 0.0
        self.agents["J1"].set_emergency("EW")
        self.agents["J3"].set_emergency("EW")

    def set_incident(self, junction_id: str, road_id: str, incident_type: str, active: bool = True):
        if active:
            self.active_incidents[road_id] = {
                "junction": junction_id,
                "road": road_id,
                "type": incident_type,
                "active": True
            }
            # Notify the local agent J3 of the incident on road_J3_J2
            if junction_id in self.agents:
                self.agents[junction_id].set_incident(road_id, incident_type, True)
        else:
            self.active_incidents.pop(road_id, None)
            if junction_id in self.agents:
                self.agents[junction_id].set_incident(road_id, incident_type, False)

    def reset(self):
        for agent in self.agents.values():
            agent.reset()
        self.message_history.clear()
        self.active_incidents.clear()
        self.degradation.reset()
        self.ambulance["active"] = False
        self.ambulance["progress_m"] = 0.0
