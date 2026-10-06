"""
board_api.py — The physical demo board: camera -> Signal AI -> LEDs.
====================================================================
Runs its own copy of agent.MultiAgentCoordinator (the same controller the
simulation uses, unchanged) with its sensor input coming from the overhead
camera instead of simulated vehicles, and pushes the resulting signal
states to the Arduino over USB serial every tick.

Layouts (board_layout.LAYOUTS): "single" — just J1 on the board, or
"full" — all five junctions. The controller always runs the whole network;
the layout decides which junctions the camera watches and the LEDs show.

Modes:
  camera : the AI drives the board from what the camera sees (the demo)
  sim    : the LEDs mirror the live simulation's signals (fallback if the
           camera misbehaves on the day)
  test   : lamp test — every colour on every head in turn, for checking wiring

A red box on the board is the ambulance: the junction it's queued at and
the next junction straight ahead are preempted (with the controller's usual
yellow + all-red clearance), and released into the recovery phase once it's
gone. Losing the camera feeds the degradation ladder, so the board steps
down to fixed-time on its own instead of freezing.
"""

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from flask import Response, jsonify, request

import board_layout as bl
from agent import MultiAgentCoordinator
from board_serial import BoardSerial, encode_frame, list_ports
from board_vision import (
    BoardVision, CameraSource, CameraTracker, SyntheticBoard, calibration_shape_problem, find_box_corners,
    parse_plate_list,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("BOARD_CONFIG", os.path.join(BASE_DIR, "board_config.json"))
DEFAULTS_PATH = os.path.join(BASE_DIR, "board_defaults.json")

TICK_S = float(os.environ.get("BOARD_TICK_S", "0.8"))   # one controller step; min green 10 steps = 8 s
# One model car stands for a platoon of this many PCU, so a few boxes read as a real queue.
PCU_PER_MODEL_CAR = 4
FRAME_STALE_S = 2.0
# Auto-calibration switches the lamps off so their glow can't hide the road-edge strips, then
# waits this long for the (laggy) phone stream to show it, and again after switching them back.
AUTOCAL_SETTLE_S = float(os.environ.get("BOARD_AUTOCAL_SETTLE_S", "3.5"))
AUTOCAL_MAX_ERROR = 6.0
# Keep an ambulance corridor open this long after the ambulance was last seen, so a blink in
# detection (a hand passing over, a glare) doesn't drop it and restart the amber/all-red cycle.
AMBULANCE_HOLD_S = 3.0
TRACK_CHECK_S = float(os.environ.get("BOARD_TRACK_CHECK_S", "2.0"))   # how often to check whether the phone moved       # roadnet units; a worse fit means a box was found wrongly
MODES = ("camera", "sim", "test", "manual")
LAMP_STATES = set("RYGO")      # red / yellow / green / off, per head group
DIRECTIONS = ("EW", "NS")


def agent_signals(agent) -> Dict[str, str]:
    """R/Y/G for each phase group of one junction, from the agent's own state."""
    out = {p: "R" for p in agent.phase_names}
    if agent.is_all_red:
        return out
    out[agent.phase_names[agent.current_phase]] = "Y" if agent.is_yellow else "G"
    return out


def to_hardware(signals: Dict[str, Dict[str, str]], swapped) -> Dict[str, Dict[str, str]]:
    """
    What to actually send for each junction. On a junction whose two head pairs were wired
    the other way round (its EW pins drive the north/south heads), swap EW and NS here, so
    the AI, the dashboard and manual control keep their normal meaning and the right heads light.
    """
    out = {}
    for jid, st in signals.items():
        st = dict(st)
        if jid in swapped:
            st["EW"], st["NS"] = st.get("NS", "R"), st.get("EW", "R")
        out[jid] = st
    return out


def lamp_test(now: float, active=bl.SIGNALLED) -> Dict[str, Dict[str, str]]:
    """All red, all amber, all green, then each head group green on its own — so a mis-wired lamp stands out."""
    steps: List[Dict[str, Any]] = [{"all": "R"}, {"all": "Y"}, {"all": "G"}]
    steps += [{"only": (jid, phase)} for jid in active for phase in ("EW", "NS")]
    step = steps[int(now / 0.7) % len(steps)]
    if "all" in step:
        return {jid: {"EW": step["all"], "NS": step["all"]} for jid in bl.SIGNALLED}
    jid, phase = step["only"]
    out = {j: {"EW": "O", "NS": "O"} for j in bl.SIGNALLED}
    out[jid][phase] = "G"
    return out


class BoardEngine:
    """The engine interface the agents read, filled from camera detections instead of a simulator."""

    def __init__(self):
        self.time = 0
        self._lane_vehicles: Dict[str, List[str]] = {}
        self._lane_waiting: Dict[str, int] = {}
        self.vehicle_info: Dict[str, Dict[str, Any]] = {}

    def load(self, vehicles: List[Dict[str, Any]]) -> None:
        lanes: Dict[str, List[str]] = {}
        waiting: Dict[str, int] = {}
        info: Dict[str, Dict[str, Any]] = {}
        for v in vehicles:
            if v["kind"] != "car" or not v.get("road") or v.get("in_junction"):
                continue
            lane = f"{v['road']}_0"
            is_waiting = v.get("waiting", True)
            for k in range(PCU_PER_MODEL_CAR * v.get("count", 1)):
                vid = f"b{v['id']}_{k}"
                lanes.setdefault(lane, []).append(vid)
                info[vid] = {
                    "speed": "0.0" if is_waiting else "3.0",
                    "distance": str(v.get("distance_m") or 0.0),
                    "road": v["road"],
                    "vehicle_type": "car",
                }
                if is_waiting:
                    waiting[lane] = waiting.get(lane, 0) + 1
        self._lane_vehicles, self._lane_waiting, self.vehicle_info = lanes, waiting, info

    def get_current_time(self) -> float:
        return float(self.time)

    def get_lane_vehicles(self) -> Dict[str, List[str]]:
        return self._lane_vehicles

    def get_lane_vehicle_count(self) -> Dict[str, int]:
        return {lane: len(ids) for lane, ids in self._lane_vehicles.items()}

    def get_lane_waiting_vehicle_count(self) -> Dict[str, int]:
        return dict(self._lane_waiting)

    def set_tl_phase(self, junction_id, phase_idx) -> None:
        pass


class BoardController:
    def __init__(self, sim_signals: Optional[Callable[[], Dict[str, Dict[str, str]]]] = None, ocr: bool = True,
                 config_path: str = CONFIG_PATH, start: bool = True):
        self.sim_signals = sim_signals
        self.config_path = config_path
        self.lock = threading.RLock()
        self.engine = BoardEngine()
        self.coordinator = MultiAgentCoordinator(self.engine)
        cfg = self._load_config()
        self.layout = bl.get_layout(os.environ.get("BOARD_LAYOUT", cfg.get("layout")))
        self._ocr = ocr
        self.camera = CameraSource()
        self.vision = BoardVision(self.layout, ocr=ocr)
        self.synthetic = SyntheticBoard(self.layout, signal_getter=lambda: self.signals)
        self.signals: Dict[str, Dict[str, str]] = {j: {"EW": "R", "NS": "R"} for j in bl.SIGNALLED}
        self.vehicles: List[Dict[str, Any]] = []
        self.data_ok = False
        self.ambulance: Dict[str, str] = {}
        self._lamps_off = False
        self._ambulance_seen = 0.0
        self.tracker = CameraTracker()
        self._last_track = 0.0
        self.realigned_at = 0.0
        self.calib_frame_size: Optional[List[int]] = cfg.get("calibration_frame_size")
        self._board_jpeg: Optional[bytes] = None
        self._last_encode = 0.0
        self._last_frame_ts = 0.0

        self.vision.registered = parse_plate_list(cfg.get("plates") or [])
        self.swapped = {j for j in (cfg.get("swapped") or []) if j in bl.SIGNALLED}
        self.mode = cfg.get("mode") if cfg.get("mode") in MODES and cfg.get("mode") != "manual" else "camera"
        # Manual control: the operator's own lamp states, and the mode to go back to on revert.
        self.manual_signals: Dict[str, Dict[str, str]] = {j: {"EW": "R", "NS": "R"} for j in bl.SIGNALLED}
        self._pre_manual_mode = self.mode
        self.serial = BoardSerial(os.environ.get("BOARD_SERIAL_PORT", cfg.get("serial_port") or "auto"))
        if cfg.get("corners"):
            try:
                self._migrate_calibration(cfg)
                self.vision.set_corners(cfg["corners"])
                self._load_reference(cfg)
                self._load_anchor(cfg)
            except ValueError:
                pass
        camera = os.environ.get("BOARD_CAMERA", cfg.get("camera"))
        if camera:
            self.set_camera(camera, save=False)

        if start:
            threading.Thread(target=self._vision_loop, daemon=True).start()
            threading.Thread(target=self._control_loop, daemon=True).start()

    # ── config ──
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception:
            pass
        # First run on this laptop: start from the settings shipped for the physical board
        # (layout, J3's swapped wiring). Camera, calibration and COM port are per-machine.
        try:
            with open(DEFAULTS_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    def _migrate_calibration(self, cfg: Dict[str, Any]) -> None:
        """
        Saved corners are where the layout's calibration points were clicked. If they were clicked
        on an older set of points (road tips), re-project them onto today's points (junction-box
        corners) through the same camera mapping, so an existing calibration keeps working.
        """
        saved_points = cfg.get("calibration_points") or self.layout.legacy_calibration_points
        current = [list(p) for p in self.layout.calibration_points]
        if [list(p) for p in saved_points] == current:
            return
        world_to_image = cv2.getPerspectiveTransform(np.array(saved_points, np.float32),
                                                     np.array(cfg["corners"], np.float32))
        def project(corners):
            return cv2.perspectiveTransform(np.array(self.layout.calibration_points, np.float32).reshape(-1, 1, 2),
                                            world_to_image).reshape(-1, 2).tolist()
        if cfg.get("reference_corners") == cfg["corners"]:
            cfg["reference_corners"] = project(cfg["corners"])
        cfg["corners"] = project(cfg["corners"])
        cfg["calibration_points"] = current
        try:
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    @property
    def anchor_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.config_path)), "board_anchor.png")

    def _load_anchor(self, cfg: Dict[str, Any]) -> None:
        """Pick the camera-tracking snapshot back up after a restart."""
        anchor = cfg.get("anchor")
        if not anchor or cfg.get("layout") != self.layout.name:
            return
        gray = cv2.imread(self.anchor_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            self.tracker.load_anchor(gray, anchor["scale"], anchor["corners"])

    def _set_anchor(self, frame) -> None:
        if frame is None or not self.vision.corners:
            return
        self.tracker.set_anchor(frame, self.vision.corners)
        try:
            cv2.imwrite(self.anchor_path, self.tracker.anchor["gray"])
        except Exception:
            pass

    def _track_camera(self, frame, now: float) -> None:
        """Every few seconds: has the phone moved since calibrating? If so, move the calibration with it."""
        if now - self._last_track < TRACK_CHECK_S or not self.vision.calibrated or self._lamps_off:
            return
        self._last_track = now
        if self.tracker.anchor is None:
            # Calibrated before tracking existed (or the snapshot was lost): take it from this view.
            with self.lock:
                self._set_anchor(frame)
                self._save_config()
            return
        moved = self.tracker.check(frame, self.vision.corners)
        if moved:
            with self.lock:
                self.vision.set_corners(moved, realign=True)
                self.calib_frame_size = [int(frame.shape[1]), int(frame.shape[0])]
                self.realigned_at = time.time()
                self._save_config()
            print(f"[Board] camera moved {self.tracker.last.get('moved_px')} px: calibration re-aligned", flush=True)

    @property
    def reference_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(self.config_path)), "board_reference.png")

    def _load_reference(self, cfg: Dict[str, Any]) -> None:
        """Reuse the saved empty-board shot after a restart, but only if the camera calibration is unchanged."""
        if cfg.get("reference_corners") != self.vision.corners or cfg.get("layout") != self.layout.name:
            return
        ref = cv2.imread(self.reference_path, cv2.IMREAD_GRAYSCALE)
        if ref is not None and ref.shape == (self.vision.size, self.vision.size):
            self.vision.reference = ref

    def _save_config(self) -> None:
        cfg = {
            "layout": self.layout.name,
            "camera": self.camera.source,
            "serial_port": self.serial.port_setting,
            "corners": self.vision.corners,
            # Manual control is never saved: a restart must not leave the lights under hand control.
            "mode": self._pre_manual_mode if self.mode == "manual" else self.mode,
            "plates": self.vision.registered,
            "swapped": sorted(self.swapped),
            "calibration_frame_size": self.calib_frame_size,
            "anchor": ({"scale": self.tracker.anchor["scale"], "corners": self.tracker.anchor["corners"]}
                       if self.tracker.anchor else None),
            "calibration_points": [list(p) for p in self.layout.calibration_points],
            "reference_corners": self.vision.corners if self.vision.reference is not None else None,
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"[Board] Could not save {self.config_path}: {e}", flush=True)

    # ── operator actions ──
    def set_camera(self, source: Optional[str], save: bool = True) -> None:
        with self.lock:
            synthetic = (source or "").strip() == "synthetic"
            self.camera.set_source(source, synthetic=self.synthetic if synthetic else None)
            if synthetic:
                # The rendered board's calibration points are known exactly, so it needs no clicking.
                self.vision.set_corners(self.synthetic.true_points)
            if save:
                self._save_config()

    def set_layout(self, name: str) -> None:
        if name not in bl.LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(bl.LAYOUTS)}")
        with self.lock:
            if name == self.layout.name:
                return
            self.layout = bl.LAYOUTS[name]
            # A new layout means new geometry: old calibration, tracks and preemptions no longer apply.
            registered = self.vision.registered
            self.vision = BoardVision(self.layout, ocr=self._ocr)
            self.vision.registered = registered
            self.synthetic.set_layout(self.layout)
            self.vehicles = []
            for jid in self.ambulance:
                self.coordinator.agents[jid].set_emergency(None)
            self.ambulance = {}
            if self.camera.source == "synthetic":
                self.vision.set_corners(self.synthetic.true_points)
            self._save_config()

    def set_swapped(self, junction: str, swapped: bool) -> List[str]:
        """Mark a junction whose EW and NS head pairs are wired the other way round."""
        if junction not in bl.SIGNALLED:
            raise ValueError(f"junction must be one of {', '.join(bl.SIGNALLED)}")
        with self.lock:
            (self.swapped.add if swapped else self.swapped.discard)(junction)
            self._save_config()
            return sorted(self.swapped)

    def set_registered_plates(self, plates: List[str]) -> List[str]:
        """The plates printed on the model cars; reads one character off one of these snap to it."""
        with self.lock:
            self.vision.registered = parse_plate_list(plates)
            self._save_config()
            return self.vision.registered

    def set_corners(self, corners: Optional[List[List[float]]]) -> None:
        with self.lock:
            self.vision.set_corners(corners)
            # Remember the picture size it was made for: if the phone changes resolution it no longer fits.
            frame, _ = self.camera.latest()
            self.calib_frame_size = ([int(frame.shape[1]), int(frame.shape[0])] if corners and frame is not None
                                     else (self.calib_frame_size if corners else None))
            if corners:
                self._set_anchor(frame)        # what the board looks like now, to follow the phone if it moves
            else:
                self.tracker.anchor = None
            self._save_config()

    def auto_calibrate(self, centres: List[List[float]], settle_s: float = AUTOCAL_SETTLE_S) -> Dict[str, Any]:
        """
        Calibrate from one click in the middle of each junction box on the board (in layout order).
        The lamps go dark for a few seconds, each box's inside corners are found from the grey
        road-edge strips around the click, all of them are fitted at once, and the empty-board
        shot is re-taken. Raises ValueError with a plain reason if anything doesn't add up.
        """
        active = list(self.layout.active)
        if len(centres) != len(active):
            raise ValueError(f"click the middle of each junction box ({', '.join(active)}): {len(active)} click(s)")
        self._lamps_off = True
        try:
            time.sleep(settle_s)
            frame, _ = self.camera.latest()
        finally:
            self._lamps_off = False
        if frame is None:
            raise ValueError("no camera picture yet")
        img_pts, world = [], []
        h = bl.ROAD_HALF_WIDTH
        for jid, c in zip(active, centres):
            corners = find_box_corners(frame, (float(c[0]), float(c[1])))
            if corners is None:
                raise ValueError(f"couldn't find {jid}'s junction box there; click inside the dark square of the box")
            jx, jy = bl.NODES[jid]
            img_pts += corners
            world += [(jx - h, jy + h), (jx + h, jy + h), (jx + h, jy - h), (jx - h, jy - h)]
        world_np, img_np = np.array(world, np.float32), np.array(img_pts, np.float32)
        H, _ = cv2.findHomography(world_np, img_np)
        if H is None:
            raise ValueError("couldn't fit the calibration; try again with the board empty")
        back = cv2.perspectiveTransform(img_np.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
        error = float(np.max(np.linalg.norm(back - world_np, axis=1)))
        if error > AUTOCAL_MAX_ERROR:
            raise ValueError(f"the junction boxes don't line up (off by {error:.0f} units); check you clicked {', '.join(active)} in that order")
        pts = np.array(self.layout.calibration_points, np.float32).reshape(-1, 1, 2)
        self.set_corners(cv2.perspectiveTransform(pts, H).reshape(-1, 2).tolist())
        time.sleep(settle_s)                 # lamps back on and visible again before the empty-board shot
        captured = self.capture_reference()
        return {"fit_error_units": round(error, 1), "boxes": len(active), "empty_board_captured": captured}

    def capture_reference(self) -> bool:
        frame, _ = self.camera.latest()
        if frame is None or not self.vision.calibrated:
            return False
        with self.lock:
            self.vision.capture_reference(frame)
            # Kept on disk so a restart doesn't bring the white strips back as "cars".
            cv2.imwrite(self.reference_path, self.vision.reference)
            self._save_config()
        return True

    def clear_reference(self) -> None:
        with self.lock:
            self.vision.reference = None
            try:
                os.remove(self.reference_path)
            except OSError:
                pass
            self._save_config()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self.lock:
            if mode == "manual" and self.mode != "manual":
                # Take over from whatever is lit right now, so nothing jumps on entry.
                self._pre_manual_mode = self.mode
                self.manual_signals = {j: dict(self.signals.get(j, {"EW": "R", "NS": "R"})) for j in bl.SIGNALLED}
            self.mode = mode
            self._save_config()

    def set_manual(self, junction: str, direction: str, state: str) -> None:
        """
        Manual control: set one head group (junction + EW/NS) to R / Y / G / O.
        junction "*" means every junction, direction "*" means both directions.
        Enters manual mode if the board isn't in it yet.
        """
        state = str(state or "").upper()
        if state not in LAMP_STATES:
            raise ValueError("state must be R, Y, G or O")
        junctions = list(bl.SIGNALLED) if junction == "*" else [junction]
        directions = list(DIRECTIONS) if direction == "*" else [direction]
        if any(j not in bl.SIGNALLED for j in junctions):
            raise ValueError(f"junction must be one of {', '.join(bl.SIGNALLED)} or *")
        if any(d not in DIRECTIONS for d in directions):
            raise ValueError("direction must be EW, NS or *")
        with self.lock:
            if self.mode != "manual":
                self.set_mode("manual")
            for j in junctions:
                for d in directions:
                    self.manual_signals[j][d] = state

    def revert_manual(self) -> str:
        """Leave manual control and hand the lights back to the mode that was running before."""
        with self.lock:
            if self.mode == "manual":
                self.mode = self._pre_manual_mode if self._pre_manual_mode in MODES and self._pre_manual_mode != "manual" else "camera"
                self._save_config()
            return self.mode

    def set_serial(self, port: str) -> None:
        self.serial.set_port(port)
        self._save_config()

    def reset_ai(self) -> None:
        with self.lock:
            self.coordinator.reset()
            self.ambulance = {}

    # ── loops ──
    def _vision_loop(self) -> None:
        while True:
            frame, ts = self.camera.latest()
            if frame is None or ts == self._last_frame_ts:
                time.sleep(0.03)
                continue
            self._last_frame_ts = ts
            try:
                self._track_camera(frame, ts)
                with self.lock:
                    vehicles = self.vision.process(frame, now=ts)
                    self.vehicles = vehicles
                    signals = self.signals
                if ts - self._last_encode > 0.15:
                    self._last_encode = ts
                    img = self.vision.annotate(vehicles, signals)
                    if img is not None:
                        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self._board_jpeg = buf.tobytes()
            except Exception as e:
                print(f"[Board] vision error: {e}", flush=True)
                time.sleep(0.2)

    def _control_loop(self) -> None:
        while True:
            started = time.time()
            try:
                self.tick(started)
            except Exception as e:
                print(f"[Board] control error: {e}", flush=True)
            time.sleep(max(0.0, TICK_S - (time.time() - started)))

    def tick(self, now: Optional[float] = None) -> Dict[str, Dict[str, str]]:
        now = now if now is not None else time.time()
        with self.lock:
            if self.mode == "test":
                signals = lamp_test(now, self.layout.active)
            elif self.mode == "manual":
                signals = {j: dict(v) for j, v in self.manual_signals.items()}
            elif self.mode == "sim" and self.sim_signals:
                signals = self.sim_signals()
            else:
                _, ts = self.camera.latest()
                self.data_ok = self.vision.calibrated and ts > 0 and (now - ts) < FRAME_STALE_S
                vehicles = self.vehicles if self.data_ok else []
                self._apply_ambulance(vehicles, now)
                self.engine.load(vehicles)
                self.engine.time += 1
                self.coordinator.step(self.engine.vehicle_info, data_ok=self.data_ok)
                signals = {jid: agent_signals(a) for jid, a in self.coordinator.agents.items()}
            self.signals = signals
        if self._lamps_off:
            # Calibrating: everything dark for a moment; the AI and dashboard carry on as normal.
            self.serial.send(encode_frame({j: {"EW": "O", "NS": "O"} for j in bl.SIGNALLED}))
        else:
            self.serial.send(encode_frame(to_hardware(signals, self.swapped)))
        return signals

    def _apply_ambulance(self, vehicles: List[Dict[str, Any]], now: Optional[float] = None) -> None:
        """
        Open a green corridor along the ambulance's road: the board junction behind it, the one it's
        heading into, and the one after that, all green in its direction. The corridor stays open for
        as long as the ambulance is on the board (plus a short hold for detection blinks), then each
        junction is released into its recovery phase.
        """
        now = time.time() if now is None else now
        want: Dict[str, str] = {}
        for v in vehicles:
            if v["kind"] != "ambulance":
                continue
            if v.get("road"):
                for jid in (self.layout.previous_junction(v["road"]), v["junction"],
                            self.layout.next_junction_straight(v["road"])):
                    if jid:
                        want.setdefault(jid, v["phase"])
            elif v.get("in_junction") and v.get("junction") in self.ambulance:
                # Crossing a box: keep the corridor it already has.
                want.update(self.ambulance)
        if want:
            self._ambulance_seen = now
        elif self.ambulance and now - self._ambulance_seen < AMBULANCE_HOLD_S:
            want = dict(self.ambulance)
        agents = self.coordinator.agents
        for jid, phase in want.items():
            if self.ambulance.get(jid) != phase:
                agents[jid].set_emergency(phase)
        for jid in self.ambulance:
            if jid not in want:
                agents[jid].set_emergency(None)
        self.ambulance = want

    # ── reporting ──
    def state(self) -> Dict[str, Any]:
        with self.lock:
            frame, ts = self.camera.latest()
            now = time.time()
            counts: Dict[str, Dict[str, int]] = {j: {"EW": 0, "NS": 0} for j in bl.SIGNALLED}
            for v in self.vehicles:
                if v["kind"] == "car" and v.get("road") and not v.get("in_junction"):
                    counts[v["junction"]][v["phase"]] += v.get("count", 1)
            junctions = {}
            for jid, agent in self.coordinator.agents.items():
                junctions[jid] = {
                    "signals": self.signals.get(jid, {}),
                    "cars": counts[jid],
                    "reason": agent.last_decision_reason,
                    "emergency": agent.emergency_override,
                    "recovery": agent.recovery_mode,
                    "steps_on_phase": agent.steps_on_phase,
                    "allocated_green": agent.allocated_green,
                    "priorities": dict(agent.priorities),
                }
            return {
                "ok": True,
                "mode": self.mode,
                "manual": {"previous_mode": self._pre_manual_mode if self.mode == "manual" else None},
                "tick_s": TICK_S,
                "layout": self.layout.name,
                "layouts": list(bl.LAYOUTS),
                "active_junctions": list(self.layout.active),
                "control_mode": self.coordinator.degradation.current,
                "data_ok": self.data_ok,
                "camera": {
                    "source": self.camera.source,
                    "live": frame is not None and (now - ts) < FRAME_STALE_S,
                    "fps": self.camera.fps,
                    "error": self.camera.error,
                    "frame_size": self.vision.last_frame_size,
                },
                "calibration": {
                    "calibrated": self.vision.calibrated,
                    "corners": self.vision.corners,
                    "labels": self.layout.calibration_labels,
                    "hint": self.layout.hint,
                    "auto_labels": [f"middle of {j}'s junction box" for j in self.layout.active],
                    # The camera's picture size changed since calibrating (phone resolution changed): it no longer fits.
                    "stale": bool(self.vision.calibrated and self.calib_frame_size and self.vision.last_frame_size
                                  and list(self.calib_frame_size) != list(self.vision.last_frame_size)),
                    "frame_size": self.calib_frame_size,
                    "tracking": self.tracker.anchor is not None,
                    "realigned_s_ago": round(time.time() - self.realigned_at) if self.realigned_at else None,
                    "reference": self.vision.reference is not None,
                },
                "ocr": None if not self.vision.plates else self.vision.plates.available,
                "registered_plates": list(self.vision.registered),
                "swapped_junctions": sorted(self.swapped),
                "serial": self.serial.status(),
                "signals": self.signals,
                "junctions": junctions,
                "vehicles": [{k: val for k, val in v.items() if k != "bbox"} for v in self.vehicles],
                "ambulance": {"active": bool(self.ambulance), "junctions": dict(self.ambulance)},
            }

    def board_jpeg(self) -> Optional[bytes]:
        return self._board_jpeg

    def raw_jpeg(self) -> Optional[bytes]:
        frame, _ = self.camera.latest()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None


def register_board_routes(app, sim_signals: Optional[Callable[[], Dict[str, Dict[str, str]]]] = None) -> BoardController:
    board = BoardController(sim_signals=sim_signals)

    def _ok():
        return jsonify({"ok": True, "state": board.state()})

    def _bad(msg: str, code: int = 400):
        return jsonify({"ok": False, "error": msg}), code

    @app.route("/api/board/state")
    def board_state():
        return jsonify(board.state())

    @app.route("/api/board/frame.jpg")
    def board_frame():
        data = board.raw_jpeg() if request.args.get("view") == "raw" else board.board_jpeg()
        if data is None:
            return _bad("no frame yet", 404)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/api/board/camera", methods=["POST"])
    def board_camera():
        data = request.get_json(silent=True) or {}
        board.set_camera(str(data.get("source") or "").strip() or None)
        return _ok()

    @app.route("/api/board/calibrate", methods=["POST"])
    def board_calibrate():
        data = request.get_json(silent=True) or {}
        if data.get("reset"):
            board.set_corners(None)
            return _ok()
        try:
            corners = [[float(x), float(y)] for x, y in data.get("corners") or []]
            if len(corners) != 4:
                raise ValueError("need exactly 4 points")
            problem = calibration_shape_problem(corners, board.layout.calibration_points)
            if problem:
                raise ValueError(f"Calibration not saved: {problem}.")
            board.set_corners(corners)
        except (ValueError, TypeError) as e:
            return _bad(str(e))
        return _ok()

    @app.route("/api/board/autocalibrate", methods=["POST"])
    def board_autocalibrate():
        """{"centres": [[x, y], ...]}: one click in the middle of each junction box, in layout order."""
        data = request.get_json(silent=True) or {}
        try:
            result = board.auto_calibrate([[float(x), float(y)] for x, y in data.get("centres") or []])
        except (ValueError, TypeError) as e:
            return _bad(str(e))
        return jsonify({"ok": True, "result": result, "state": board.state()})

    @app.route("/api/board/reference", methods=["POST"])
    def board_reference():
        data = request.get_json(silent=True) or {}
        if data.get("clear"):
            board.clear_reference()
            return _ok()
        if not board.capture_reference():
            return _bad("Need a live, calibrated camera before capturing the empty board", 409)
        return _ok()

    @app.route("/api/board/mode", methods=["POST"])
    def board_mode():
        data = request.get_json(silent=True) or {}
        try:
            board.set_mode(str(data.get("mode", "")))
        except ValueError as e:
            return _bad(str(e))
        return _ok()

    @app.route("/api/board/manual", methods=["POST"])
    def board_manual():
        """Manual control: {"junction": "J1"|"*", "direction": "EW"|"NS"|"*", "state": "R"|"Y"|"G"|"O"}."""
        data = request.get_json(silent=True) or {}
        try:
            board.set_manual(str(data.get("junction", "")), str(data.get("direction", "")), str(data.get("state", "")))
        except ValueError as e:
            return _bad(str(e))
        return _ok()

    @app.route("/api/board/manual/revert", methods=["POST"])
    def board_manual_revert():
        board.revert_manual()
        return _ok()

    @app.route("/api/board/layout", methods=["POST"])
    def board_layout():
        data = request.get_json(silent=True) or {}
        try:
            board.set_layout(str(data.get("layout", "")))
        except ValueError as e:
            return _bad(str(e))
        return _ok()

    @app.route("/api/board/wiring", methods=["POST"])
    def board_wiring():
        """{"junction": "J3", "swapped": true} when that junction's EW pins drive its north/south heads."""
        data = request.get_json(silent=True) or {}
        try:
            board.set_swapped(str(data.get("junction", "")), bool(data.get("swapped", True)))
        except ValueError as e:
            return _bad(str(e))
        return _ok()

    @app.route("/api/board/plates", methods=["POST"])
    def board_plates():
        """Registered plates, as a list or one comma/space separated string ("TN07, KA01 MH12")."""
        data = request.get_json(silent=True) or {}
        plates = data.get("plates") or []
        if isinstance(plates, str):
            plates = plates.replace(",", " ").split()
        board.set_registered_plates([str(p) for p in plates])
        return _ok()

    @app.route("/api/board/serial", methods=["POST"])
    def board_serial():
        data = request.get_json(silent=True) or {}
        board.set_serial(str(data.get("port") or "auto"))
        return _ok()

    @app.route("/api/board/serial/ports")
    def board_serial_ports():
        return jsonify({"ok": True, "ports": list_ports()})

    @app.route("/api/board/reset", methods=["POST"])
    def board_reset():
        board.reset_ai()
        return _ok()

    @app.route("/api/board/synthetic/toggle", methods=["POST"])
    def board_synthetic_toggle():
        """Synthetic camera only: click the board view to drop / remove a car (u, v are 0..1 across the view)."""
        data = request.get_json(silent=True) or {}
        try:
            u, v = float(data["u"]), float(data["v"])
        except (KeyError, TypeError, ValueError):
            return _bad("u and v (0..1) are required")
        lay = board.layout
        x, y = lay.from_canvas(u * lay.span, v * lay.span, 1.0)
        board.synthetic.toggle_at(x, y, kind="ambulance" if data.get("kind") == "ambulance" else "car")
        return _ok()

    @app.route("/api/board/synthetic/clear", methods=["POST"])
    def board_synthetic_clear():
        board.synthetic.set_cars([])
        return _ok()

    return board
