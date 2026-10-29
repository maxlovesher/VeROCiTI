"""
scripts/compare_schedulers.py — signal scheduling algorithms, head to head
==========================================================================
Runs the same 5-junction network and the same random demand (same seeds)
under several signal scheduling policies, and measures what each one does
to throughput, delay and — the point of fair scheduling — the worst wait any
single vehicle suffers. Then runs the ambulance scenarios: one ambulance,
and two ambulances that need the same junction in different directions,
with and without green-corridor preemption.

Policies (all share the same safety rules: yellow, minimum green and
ambulance preemption; only "who goes next" differs):
  Fixed time          — round robin, fixed cycle (today's signals).
  FCFS                — serve the other approach as soon as anyone is waiting
                        there (vehicles served roughly in arrival order).
  SRTF                — shortest remaining job first: switch when the other
                        queue is smaller than what's left on the green one.
  Longest queue first — pure greedy on queue size.
  Max-pressure        — upstream minus downstream load (Varaiya, 2013).
  Congestion only     — VeROCiTI's congestion score with aging and the
                        wait guarantees switched off.
  VeROCiTI fair       — congestion score + aging + MAX_RED_WAIT_S guarantee.

Uses the built-in simulator from server_standalone.py, so no CityFlow is
needed. Writes docs/scheduler_comparison.md and .json at the repo root.
Run with:
    python "city flow model/scripts/compare_schedulers.py" [--seeds 8] [--minutes 30]
"""

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(SERVER_DIR)
sys.path.insert(0, SERVER_DIR)
sys.path.append(os.path.join(REPO_DIR, "prototype"))

# Same isolation as the server tests: no real detection DB, no camera/ANPR routes.
import database  # noqa: E402
database.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="verociti_bench_"), "traffic.db")
sys.modules["tracking_api"] = types.SimpleNamespace(register_tracking_routes=lambda app: None)

import server_standalone as srv  # noqa: E402
from agent import TrafficAgent  # noqa: E402

WARMUP_S = 300
TICK_S = srv.AGENT_INTERVAL
MAIN_CORRIDOR = {0, 1}          # ROUTES[0] / [1]: the J1-J3-J4 east-west through route
LONG_WAIT_S = 120.0

RUSH = [8 if i in MAIN_CORRIDOR else 1 for i in range(len(srv.ROUTES))]
SCENARIOS = {
    "balanced": {
        "title": "Balanced demand",
        "desc": "Every route equally likely, 64 vehicles in the network.",
        "weights": None,
        "fleet": 64,
    },
    "rush_hour": {
        "title": "Rush hour on the main road",
        "desc": "The J1-J3-J4 east-west corridor carries 8x the demand of each side street, 64 vehicles.",
        "weights": RUSH,
        "fleet": 64,
    },
    "heavy_rush_hour": {
        "title": "Heavy rush hour (near saturation)",
        "desc": "Same 8x main-road split with 130 vehicles in the network.",
        "weights": RUSH,
        "fleet": 130,
    },
}


# ── policies ────────────────────────────────────────────────────────────────

class PolicyAgent(TrafficAgent):
    """TrafficAgent's safety behaviour (yellow, all-red, min green, ambulances) with a pluggable 'who goes next'."""

    LABEL = "policy"

    def decide_and_act(self, neighbor_states):
        if self.is_yellow or self.is_all_red or self.emergency_override:
            return super().decide_and_act(neighbor_states)
        self.step_counter += 1
        self.steps_on_phase += 1
        if self.steps_on_phase < self.MIN_GREEN:
            self.last_decision_reason = f"[{self.LABEL}] minimum green"
            return self.last_decision_reason
        cur = self.phase_names[self.current_phase]
        other_idx = 1 - self.current_phase
        other = self.phase_names[other_idx]
        if self.should_switch(self.local_obs.get(cur, {}), self.local_obs.get(other, {}), cur, other, neighbor_states):
            self._initiate_switch(other_idx, 1.0, self.LABEL)
        else:
            self.last_decision_reason = f"[{self.LABEL}] holding {cur}"
        return self.last_decision_reason

    def should_switch(self, cur_obs, other_obs, cur, other, neighbor_states) -> bool:
        raise NotImplementedError


class FCFSAgent(PolicyAgent):
    LABEL = "FCFS"

    def should_switch(self, cur_obs, other_obs, cur, other, neighbor_states):
        return other_obs.get("queue_length", 0) > 0


class SRTFAgent(PolicyAgent):
    LABEL = "SRTF"

    def should_switch(self, cur_obs, other_obs, cur, other, neighbor_states):
        waiting = other_obs.get("queue_length_pcu", 0.0)
        remaining = cur_obs.get("vehicle_count_pcu", 0.0)
        if waiting <= 0:
            return False
        return remaining == 0 or waiting < remaining


class LongestQueueAgent(PolicyAgent):
    LABEL = "Longest queue first"

    def should_switch(self, cur_obs, other_obs, cur, other, neighbor_states):
        return other_obs.get("queue_length_pcu", 0.0) > cur_obs.get("vehicle_count_pcu", 0.0)


class MaxPressureAgent(PolicyAgent):
    LABEL = "Max-pressure"

    def _pressure(self, phase, obs, neighbor_states):
        downstream = 0.0
        neighbor = self.outgoing_neighbors.get(phase)
        if neighbor and neighbor in neighbor_states:
            downstream = neighbor_states[neighbor].get("overall_density", 0.0) * obs.get("lane_capacity", 0.0)
        return obs.get("vehicle_count_pcu", 0.0) - downstream

    def should_switch(self, cur_obs, other_obs, cur, other, neighbor_states):
        return self._pressure(other, other_obs, neighbor_states) > self._pressure(cur, cur_obs, neighbor_states)


class CongestionOnlyAgent(TrafficAgent):
    """VeROCiTI's controller with every waiting-time fairness rule turned off."""
    AGING_FULL_S = math.inf
    MAX_RED_WAIT_S = math.inf
    STARVATION_HARD_CAP = math.inf


POLICIES = [
    # (name, agent class, fixed-time?)
    ("Fixed time (round robin)", TrafficAgent, True),
    ("FCFS", FCFSAgent, False),
    ("SRTF", SRTFAgent, False),
    ("Longest queue first", LongestQueueAgent, False),
    ("Max-pressure", MaxPressureAgent, False),
    ("Congestion only", CongestionOnlyAgent, False),
    ("VeROCiTI fair", TrafficAgent, False),
]


# ── measurement ─────────────────────────────────────────────────────────────

class MeasuredSim(srv.TrafficSim):
    """TrafficSim that also tracks each vehicle's longest single standstill."""

    def __init__(self, *args, **kwargs):
        self.finished = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.finished = []
        super().reset()

    def _physics_substep(self, dt):
        super()._physics_substep(dt)
        for v in self.fleet:
            if v.is_waiting:
                v.cur_stop = getattr(v, "cur_stop", 0.0) + dt
                v.longest_stop = max(getattr(v, "longest_stop", 0.0), v.cur_stop)
            else:
                v.cur_stop = 0.0

    def _record_trip(self, v):
        super()._record_trip(v)
        if not v.partial and v.spawn_time >= WARMUP_S:
            self.finished.append({
                "travel_s": self.time - v.spawn_time,
                "idle_s": v.idle_seconds,
                "longest_stop_s": getattr(v, "longest_stop", 0.0),
                "main_road": srv.ROUTES.index(v.route) in MAIN_CORRIDOR,
            })


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run_policy(agent_cls, fixed_time, weights, seed, measure_s, fleet=None):
    default_fleet = srv.FLEET_TARGET
    srv.FLEET_TARGET = fleet or default_fleet   # read by TrafficSim when it tops up the network
    try:
        sim = MeasuredSim("bench", fixed_time=fixed_time, seed=seed, agent_cls=agent_cls, route_weights=weights)
        for _ in range(WARMUP_S + measure_s):
            sim.advance()
    finally:
        srv.FLEET_TARGET = default_fleet
    trips = sim.finished
    # Vehicles still stuck at the end count too — starvation hides in exactly those.
    still_waiting = [getattr(v, "longest_stop", 0.0) for v in sim.fleet if v.spawn_time >= WARMUP_S]
    stops = [t["longest_stop_s"] for t in trips] + still_waiting
    side = [t["idle_s"] for t in trips if not t["main_road"]]
    main = [t["idle_s"] for t in trips if t["main_road"]]
    n = len(trips) or 1
    return {
        "throughput_vph": len(trips) / (measure_s / 3600.0),
        "avg_travel_s": sum(t["travel_s"] for t in trips) / n,
        "avg_delay_s": sum(t["idle_s"] for t in trips) / n,
        "p95_wait_s": percentile(stops, 0.95),
        "worst_wait_s": max(stops) if stops else 0.0,
        "pct_wait_over_2min": 100.0 * sum(s > LONG_WAIT_S for s in stops) / (len(stops) or 1),
        "main_delay_s": sum(main) / len(main) if main else 0.0,
        "side_delay_s": sum(side) / len(side) if side else 0.0,
    }


def summarise(runs):
    out = {}
    for key in runs[0]:
        vals = [r[key] for r in runs]
        out[key] = [round(statistics.mean(vals), 1), round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0]
    return out


# ── ambulances ──────────────────────────────────────────────────────────────

def _disable_preemption(coord):
    """Ambulance gets no green corridor: it waits for the ordinary signal like everyone else."""
    def green_for(junction_id, next_junction_id):
        agent = coord.agents.get(junction_id)
        phase = coord._phase_towards(junction_id, next_junction_id)
        if agent is None or phase is None:
            return True
        return not agent.is_yellow and not agent.is_all_red and agent.phase_names[agent.current_phase] == phase
    coord._green_for = green_for
    coord._refresh_corridors = lambda: None


def run_ambulances(dispatches, preempt, weights, seed):
    """dispatches: [(origin, hospital_name)] all sent at the same moment. Returns trip seconds per ambulance."""
    sim = MeasuredSim("amb", seed=seed, route_weights=weights)
    if not preempt:
        _disable_preemption(sim.coordinator)
    for _ in range(WARMUP_S):
        sim.advance()
    coord = sim.coordinator
    records = []
    for origin, hospital in dispatches:
        amb_id = coord.dispatch_ambulance(origin, hospitals={hospital: coord.hospitals[hospital]})["id"]
        records.append(coord.ambulances[amb_id])   # the same dict keeps its arrival time once it finishes
    for _ in range(3600):
        sim.advance()
        if not coord.ambulances:
            break
    return [
        (r["arrived_at"] - r["dispatched_at"]) if r.get("arrived_at") is not None else float("nan")
        for r in records
    ]


def free_flow_s(links):
    c = srv.MultiAgentCoordinator
    steps = (links * c.AMBULANCE_LINK_DISTANCE_M + c.AMBULANCE_CLEAR_MARGIN_M) / c.AMBULANCE_SPEED
    return steps * TICK_S


# ── report ──────────────────────────────────────────────────────────────────

METRICS = [
    ("throughput_vph", "Throughput (veh/h)", "higher"),
    ("avg_travel_s", "Avg travel time (s)", "lower"),
    ("avg_delay_s", "Avg stopped time (s)", "lower"),
    ("p95_wait_s", "95th pct longest wait (s)", "lower"),
    ("worst_wait_s", "Worst single wait (s)", "lower"),
    ("pct_wait_over_2min", "Waits over 2 min (%)", "lower"),
]


def table(results, metrics):
    head = "| Policy | " + " | ".join(label for _, label, _ in metrics) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    best = {}
    for key, _, better in metrics:
        vals = [r[key][0] for r in results.values()]
        best[key] = max(vals) if better == "higher" else min(vals)
    rows = [head, sep]
    for name, r in results.items():
        cells = []
        for key, _, _ in metrics:
            mean, sd = r[key]
            cell = f"{mean:.0f} ± {sd:.0f}" if key != "pct_wait_over_2min" else f"{mean:.1f} ± {sd:.1f}"
            cells.append(f"**{cell}**" if mean == best[key] else cell)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description="Compare signal scheduling policies on the built-in simulator.")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--minutes", type=int, default=30)
    args = ap.parse_args()
    measure_s = args.minutes * 60
    seeds = list(range(1, args.seeds + 1))

    report = {
        "generated": datetime.date.today().isoformat(),
        "config": {
            "seeds": seeds, "warmup_s": WARMUP_S, "measure_s": measure_s, "decision_interval_s": TICK_S,
            "min_green_s": TrafficAgent.MIN_GREEN * TICK_S, "max_green_s": TrafficAgent.MAX_GREEN * TICK_S,
            "yellow_s": TrafficAgent.YELLOW_TIME * TICK_S,
            "aging_full_s": TrafficAgent.AGING_FULL_S, "max_red_wait_s": TrafficAgent.MAX_RED_WAIT_S,
        },
        "scenarios": {},
        "ambulance": {},
    }

    for key, sc in SCENARIOS.items():
        report["scenarios"][key] = {}
        for name, cls, fixed in POLICIES:
            runs = [run_policy(cls, fixed, sc["weights"], s, measure_s, sc["fleet"]) for s in seeds]
            report["scenarios"][key][name] = summarise(runs)
            print(f"[{key}] {name}: done", flush=True)

    east = ("J1", "City Hospital")     # J1 -> J3 -> J4, east-west through J3
    south = ("J2", "Capital Hospital")  # J2 -> J3 -> J5, north-south through J3
    amb_cases = [
        ("One ambulance", [east]),
        ("Two ambulances, same moment, crossing at J3", [east, south]),
    ]
    rush = RUSH
    for label, dispatches in amb_cases:
        report["ambulance"][label] = {}
        for preempt in (False, True):
            times = [run_ambulances(dispatches, preempt, rush, s) for s in seeds]
            per_amb = list(zip(*times))
            report["ambulance"][label]["green corridor" if preempt else "no preemption"] = [
                [round(statistics.mean(t), 1), round(max(t), 1)] for t in per_amb
            ]
        print(f"[ambulance] {label}: done", flush=True)

    write_report(report)


def write_report(report):
    cfg = report["config"]
    out = [
        "# Signal scheduling: comparison",
        "",
        f"Generated {report['generated']} by `city flow model/scripts/compare_schedulers.py` on the built-in "
        f"5-junction simulator (mixed Indian traffic: cars, two-wheelers, autos, buses, trucks). Each figure is the mean ± standard deviation over "
        f"{len(cfg['seeds'])} random seeds; every policy sees the same seeds. {cfg['warmup_s']} s warm-up, then "
        f"{cfg['measure_s'] // 60} min measured.",
        "",
        f"All policies share the same safety rules: the controller decides every {cfg['decision_interval_s']} s, "
        f"minimum green is {cfg['min_green_s']} s, yellow {cfg['yellow_s']} s. VeROCiTI fair adds aging (one full "
        f"point of priority per {cfg['aging_full_s']:.0f} s on red) and a {cfg['max_red_wait_s']:.0f} s wait guarantee.",
        "",
        "**Longest wait** is the longest single standstill of each vehicle, including vehicles still stuck when the "
        "run ended, so a policy can't hide starvation by never letting the starved cars finish.",
        "",
    ]
    for key, sc in SCENARIOS.items():
        res = report["scenarios"][key]
        out += [f"## {sc['title']}", "", sc["desc"], "", table(res, METRICS), ""]
        if sc["weights"]:
            out += [
                "Who pays for the main road's priority (average stopped time per trip):",
                "",
                "| Policy | Main road (s) | Side streets (s) |",
                "|---|---:|---:|",
            ]
            out += [f"| {n} | {r['main_delay_s'][0]:.0f} | {r['side_delay_s'][0]:.0f} |" for n, r in res.items()]
            out.append("")

    out += [
        "## Ambulances",
        "",
        f"Rush-hour demand. Ambulance trip time from dispatch to hospital, mean (worst) over the seeds. "
        f"Free-flow time for a 2-link trip is {free_flow_s(2):.0f} s.",
        "",
        "| Case | Ambulance | No preemption | Green corridor |",
        "|---|---|---:|---:|",
    ]
    for label, modes in report["ambulance"].items():
        names = ["East (J1→J3→J4)", "South (J2→J3→J5)"]
        for i, (no, yes) in enumerate(zip(modes["no preemption"], modes["green corridor"])):
            out.append(f"| {label if i == 0 else ''} | {names[i]} | {no[0]:.0f} ({no[1]:.0f}) | {yes[0]:.0f} ({yes[1]:.0f}) |")
    out += [
        "",
        "With two ambulances, J3 can only show one emergency direction, so it goes to whichever ambulance will "
        "reach it first (ties: more severe case, then earlier dispatch). The other waits at the stop line, gets J3 "
        "as soon as the first has cleared it, and never runs a red.",
        "",
    ]

    docs = os.path.join(REPO_DIR, "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "scheduler_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    with open(os.path.join(docs, "scheduler_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n".join(out))


if __name__ == "__main__":
    main()
