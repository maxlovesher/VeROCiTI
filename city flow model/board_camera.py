"""
board_camera.py — the board's eyes: overhead phone camera -> cars on the roads -> Signal AI.
===========================================================================================
  CameraSource  keeps the newest frame from the phone (IP Webcam / DroidCam URL or a webcam index)
  calibration   you pick four corners of the black junction boxes on the live picture; they give a
                perspective map from the camera onto a flat top-down plan of the board
  detection     on that plan: white cards = cars, a pale pink card = the ambulance. Compared with a
                photo of the empty board, with lit LEDs masked, so lane paint, glare and lamp glow
                never turn into cars
  CameraPilot   runs its own copy of agent.MultiAgentCoordinator fed by those cars, opens the
                green corridor for an ambulance, and falls back through the degradation ladder
                (fixed-time at the end) when the camera goes quiet

Board plan (roadnet units, north up, 2 px per unit):
    J1 at (100, 300) and J3 at (300, 300), 200 apart, roads 68 wide (junction boxes 68 square).
"""

import json
import os
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from board_plates import AMBULANCE, STATE_NAMES, PlateReader, rectify_label

# ── board geometry ──
NODES = {"J1": (100.0, 300.0), "J3": (300.0, 300.0)}
HALF = 34.0                    # half the road width = half a junction box
JUNCTION_HALF = 36.0           # a car this close to a junction centre counts as inside it
MARGIN = 12.0                  # a car may overhang the road edge this much
VIEW = (-100.0, 500.0, 100.0, 500.0)    # plan covers x -100..500, y 100..500 (north up)
SCALE = 2.0
CANVAS = (int((VIEW[1] - VIEW[0]) * SCALE), int((VIEW[3] - VIEW[2]) * SCALE))   # (w, h) px

CAL_LABELS = ["J1 box, north-west corner", "J3 box, north-east corner",
              "J3 box, south-east corner", "J1 box, south-west corner"]
CAL_WORLD = [(NODES["J1"][0] - HALF, NODES["J1"][1] + HALF), (NODES["J3"][0] + HALF, NODES["J3"][1] + HALF),
             (NODES["J3"][0] + HALF, NODES["J3"][1] - HALF), (NODES["J1"][0] - HALF, NODES["J1"][1] - HALF)]

# ── detection tuning ──
# The ambulance card is pale pink: its hue drifts with the LEDs and the phone's white balance
# (165 one day, 176-179 the next), but it is always pale and bright, while LED glow on the dark road is
# the same hues at saturation 113-176. When a green lamp lights beside it the phone re-exposes and the card
# washes out to saturation ~25-55; the white plate cards stay at 5-28 and are almost never pink-hued.
PINK_HUE = (150, 8)            # 150..179 then 0..8 (pink round to red)
PINK_SAT = (26, 108)
PINK_MIN_V = 150
CARD_UNITS = (60.0, 25.0)      # the AMBULANCE card, long side x short side
CARD_MIN_LONG, CARD_MIN_SHORT, CARD_MIN_ASPECT, CARD_MIN_RECT = 40.0, 15.0, 1.6, 0.72
CARD_SOLIDITY = 0.6
CAR_MIN_SIDE, CAR_MAX_SIDE = 10.0, 80.0
OPEN_KERNEL_UNITS = 7.0        # erases anything thinner than this (lane dashes, zebra stripes)
CAR_AREA_UNITS = 30.0 * 22.0   # first guess for one model car's footprint
CHANGE_ABS, CHANGE_REL = 28, 0.22
LED_CORE_V, LED_RING_SAT = 245, 100
LED_RING_UNITS, LED_CORE_MAX_UNITS, LED_PAD_UNITS, LED_HALO_UNITS = 15.0, 16.0, 5.0, 14.0
LED_HALO_MAX_FRAC, LED_MEMORY_DECAY = 0.65, 0.75      # a card beside a lamp is ~50% under its halo; the glow ghosts are ~100%
CONFIRM_S = {"car": 0.7, "ambulance": 1.0}      # a detection must hold this long to count
CONFIRM_MIN_SIGHTINGS = 3
TRACK_MATCH_UNITS, TRACK_TTL_S, STATIONARY_UNITS = 30.0, 1.0, 6.0

# ── plates ──
OCR_RETRY_S = 1.2              # how often a car whose plate isn't settled is read again
OCR_CONFIDENT_SCORE = 1.6      # summed read confidence (~two good reads) before a car stops being re-read
AMBULANCE_TEXT_SCORE = 0.4     # a card whose text reads AMBULANCE at least once fairly clearly is an ambulance

# ── following the phone ──
TRACK_CHECK_S = 2.0            # how often to check whether the phone has moved

# ── control ──
STEP_S = 0.8                  # one AI step; min green is 10 steps = 8 s
PCU_PER_MODEL_CAR = 4          # one model car stands for a platoon, so a few cards read as a real queue
FRAME_STALE_S = 2.0
AMBULANCE_HOLD_S = 5.0         # a dropout shorter than this (the phone re-exposing as a lamp changes) must not flap the corridor
CONFIG_PATH = os.environ.get("LED_BOARD_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "led_board_config.json"))


def to_canvas(x: float, y: float) -> Tuple[float, float]:
    return (x - VIEW[0]) * SCALE, (VIEW[3] - y) * SCALE


def from_canvas(px: float, py: float) -> Tuple[float, float]:
    return px / SCALE + VIEW[0], VIEW[3] - py / SCALE


def locate(x: float, y: float) -> Optional[Dict[str, Any]]:
    """Which road a point on the plan is on, which junction it queues at, and which phase it holds up."""
    for jid, (jx, jy) in NODES.items():
        if abs(x - jx) <= JUNCTION_HALF and abs(y - jy) <= JUNCTION_HALF:
            return {"in_junction": True, "junction": jid, "road": None, "phase": None}

    def approach(road: str, jid: str, phase: str) -> Dict[str, Any]:
        return {"in_junction": False, "junction": jid, "road": road, "phase": phase}

    if abs(y - NODES["J1"][1]) <= HALF + MARGIN:                       # the east-west road
        if x < NODES["J1"][0]:
            return approach("road_VW1_J1", "J1", "EW")
        if x < (NODES["J1"][0] + NODES["J3"][0]) / 2:
            return approach("road_J3_J1", "J1", "EW")
        if x < NODES["J3"][0]:
            return approach("road_J1_J3", "J3", "EW")
        return approach("road_J4_J3", "J3", "EW")
    for jid, (jx, jy) in NODES.items():                                  # the north-south arms
        if abs(x - jx) <= HALF + MARGIN:
            north = y > jy
            road = {("J1", True): "road_VN1_J1", ("J1", False): "road_VS1_J1",
                    ("J3", True): "road_J2_J3", ("J3", False): "road_J5_J3"}[(jid, north)]
            return approach(road, jid, "NS")
    return None


def road_mask() -> np.ndarray:
    mask = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
    half = HALF + MARGIN
    x0, y0 = to_canvas(VIEW[0], NODES["J1"][1] + half)
    x1, y1 = to_canvas(VIEW[1], NODES["J1"][1] - half)
    cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    for jx, _ in NODES.values():
        x0, y0 = to_canvas(jx - half, VIEW[3])
        x1, y1 = to_canvas(jx + half, VIEW[2])
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    return mask


def calibration_problem(points_px: List[List[float]]) -> Optional[str]:
    """A plain reason if four picked corners can't be the J1/J3 boxes, else None."""
    pts = np.array(points_px, float)
    world = np.array(CAL_WORLD, float)
    signed = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
    if signed <= 0:
        return "the corners are in the wrong order: go clockwise from J1's north-west corner"
    ratios = []
    for i in range(4):
        j = (i + 1) % 4
        length = float(np.linalg.norm(pts[j] - pts[i]))
        if length < 10:
            return "two of the corners are almost on top of each other"
        ratios.append(length / float(np.linalg.norm(world[j] - world[i])))
    if max(ratios) / min(ratios) > 1.6:
        return "the shape doesn't match the two boxes: pick the outer corners of J1's and J3's black boxes"
    return None


class CameraSource:
    """Keeps only the newest frame from a camera and reconnects on its own if the feed drops."""

    def __init__(self):
        self.source: Optional[str] = None
        self.fps = 0.0
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_source(self, source: Optional[str]) -> None:
        self.stop()
        self.source = (str(source).strip() or None) if source is not None else None
        self.error, self.fps = None, 0.0
        with self._lock:
            self._frame, self._ts = None, 0.0
        if self.source:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame, self._ts

    def _run(self, stop: threading.Event) -> None:
        last = 0.0
        src: Any = int(self.source) if self.source.isdigit() else self.source
        while not stop.is_set():
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(src)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                self.error = f"Couldn't open the camera '{self.source}'"
                cap.release()
                stop.wait(2.0)
                continue
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = "The camera feed dropped, reconnecting…"
                    break
                now = time.time()
                with self._lock:
                    self._frame, self._ts = frame, now
                if last:
                    self.fps = round(0.8 * self.fps + 0.2 / max(1e-3, now - last), 1)
                last, self.error = now, None
            cap.release()
            stop.wait(1.0)


class BoardEngine:
    """The engine interface the Signal AI agents read, filled from camera detections instead of a simulator."""

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
            for k in range(PCU_PER_MODEL_CAR * v.get("count", 1)):
                vid = f"b{v['id']}_{k}"
                lanes.setdefault(lane, []).append(vid)
                info[vid] = {"speed": "0.0" if v.get("waiting", True) else "3.0", "distance": "0.0",
                             "road": v["road"], "vehicle_type": "car"}
                if v.get("waiting", True):
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


class Track:
    def __init__(self, tid: int, det: Dict[str, Any], now: float):
        self.tid, self.kind = tid, det["kind"]
        self.x, self.y, self.bbox, self.count = det["x"], det["y"], det["bbox"], det.get("count", 1)
        self.rect = det.get("rect")
        self.first_seen = self.last_seen = now
        self.sightings = 1
        self.history = [(now, self.x, self.y)]
        self.votes: Counter = Counter()                    # plate -> summed read confidence
        self.last_ocr = 0.0

    def update(self, det: Dict[str, Any], now: float) -> None:
        self.x, self.y, self.bbox, self.count, self.last_seen = det["x"], det["y"], det["bbox"], det.get("count", 1), now
        self.rect = det.get("rect")
        self.sightings += 1
        self.history = [(t, x, y) for t, x, y in self.history if now - t <= 2.0] + [(now, self.x, self.y)]

    @property
    def plate(self) -> Optional[str]:
        return self.votes.most_common(1)[0][0] if self.votes else None

    @property
    def plate_score(self) -> float:
        return self.votes.most_common(1)[0][1] if self.votes else 0.0

    @property
    def says_ambulance(self) -> bool:
        """A white card whose text reads AMBULANCE (its best reading, read at least once fairly clearly)."""
        return self.kind == "car" and self.plate == AMBULANCE and self.votes[AMBULANCE] >= AMBULANCE_TEXT_SCORE

    def confirmed(self, now: float) -> bool:
        return self.sightings >= CONFIRM_MIN_SIGHTINGS and now - self.first_seen >= CONFIRM_S[self.kind]

    def waiting(self, now: float) -> bool:
        recent = [(x, y) for t, x, y in self.history if now - t <= 1.0]
        if len(recent) < 2:
            return True
        xs, ys = zip(*recent)
        return (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 <= STATIONARY_UNITS ** 2


class CameraTracker:
    """
    Keeps the four corners valid when the phone is nudged, turned or switched to another resolution.
    At calibration it keeps a snapshot of the camera picture; afterwards it matches the live picture
    against that snapshot (ORB features on the board's fixed paint; moving cars are just outliers) and
    works out where the corners are now. Two checks in a row must agree before the corners move.
    """

    WIDTH = 1280                # both pictures are compared at this width
    MIN_INLIERS = 50            # matches that agree on one exact movement; chance mismatches never line up this well
    MIN_INLIER_SHARE = 0.12
    MOVE_PX = 8.0               # full-resolution pixels; less than this is noise, not a moved phone
    AGREE_PX = 10.0

    def __init__(self):
        self._orb = cv2.ORB_create(nfeatures=3000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.anchor: Optional[Dict[str, Any]] = None
        self._pending: Optional[np.ndarray] = None
        self.last: Dict[str, Any] = {}

    def _prepare(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        scale = self.WIDTH / gray.shape[1]
        small = cv2.resize(gray, (self.WIDTH, int(round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        kp, des = self._orb.detectAndCompute(small, None)
        return small, scale, kp, des

    def set_anchor(self, frame: np.ndarray, corners: List[List[float]]) -> None:
        small, scale, kp, des = self._prepare(frame)
        self.anchor = {"gray": small, "scale": scale, "kp": kp, "des": des, "corners": [list(map(float, c)) for c in corners]}
        self._pending = None

    def load_anchor(self, gray_small: np.ndarray, scale: float, corners: List[List[float]]) -> None:
        kp, des = self._orb.detectAndCompute(gray_small, None)
        self.anchor = {"gray": gray_small, "scale": float(scale), "kp": kp, "des": des,
                       "corners": [list(map(float, c)) for c in corners]}
        self._pending = None

    def check(self, frame: np.ndarray, current: List[List[float]]) -> Optional[List[List[float]]]:
        """Where the anchored corners are in this frame (full-resolution pixels) if the phone has moved; else None."""
        a = self.anchor
        if a is None or a["des"] is None or len(a["kp"]) < self.MIN_INLIERS:
            return None
        _, scale, kp, des = self._prepare(frame)
        self.last = {"time": time.time(), "inliers": 0}
        if des is None or len(kp) < self.MIN_INLIERS:
            return None
        pairs = self._matcher.knnMatch(a["des"], des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < self.MIN_INLIERS:
            return None
        src = np.float32([a["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        self.last.update(inliers=inliers, matches=len(good))
        if H is None or inliers < self.MIN_INLIERS or inliers < self.MIN_INLIER_SHARE * len(good):
            return None
        pts = np.float32(a["corners"]).reshape(-1, 1, 2) * a["scale"]
        moved = cv2.perspectiveTransform(pts, H).reshape(-1, 2) / scale
        shift = float(np.max(np.linalg.norm(moved - np.float32(current), axis=1)))
        self.last["moved_px"] = round(shift, 1)
        if shift < self.MOVE_PX or shift > 0.6 * frame.shape[1]:
            self._pending = None
            return None
        if self._pending is None or float(np.max(np.linalg.norm(moved - self._pending, axis=1))) > self.AGREE_PX:
            self._pending = moved                          # wait for the next check to confirm before moving the corners
            return None
        self._pending = None
        return moved.tolist()


class CameraPilot:
    def __init__(self, config_path: str = CONFIG_PATH, start: bool = True, ocr: bool = True):
        from agent import MultiAgentCoordinator           # the same controller the simulation uses, unchanged
        self.config_path = config_path
        folder = os.path.dirname(os.path.abspath(config_path))
        self.reference_path = os.path.join(folder, "led_board_reference.png")
        self.anchor_path = os.path.join(folder, "led_board_anchor.png")
        self.lock = threading.RLock()
        self.camera = CameraSource()
        self.tracker = CameraTracker()
        self.plates: Optional[PlateReader] = PlateReader() if ocr else None
        self.realigned_at: Optional[float] = None
        self._last_track = 0.0
        self.engine = BoardEngine()
        self.coordinator = MultiAgentCoordinator(self.engine)
        self.points_norm: Optional[List[List[float]]] = None       # the four corners, as fractions of the picture
        self.reference: Optional[np.ndarray] = None
        self.vehicles: List[Dict[str, Any]] = []
        self.ambulance: Dict[str, str] = {}
        self.control_mode = "FULL_AI"
        self._roi = road_mask()
        k = max(3, int(OPEN_KERNEL_UNITS * SCALE) | 1)
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._pad = self._disk(LED_PAD_UNITS * SCALE)
        self._halo = self._disk(LED_HALO_UNITS * SCALE)
        self._ring = int(LED_RING_UNITS * SCALE) | 1
        self._led_memory: Optional[np.ndarray] = None
        self._areas: List[float] = []
        self._tracks: Dict[int, Track] = {}
        self._next_tid = 1
        self._last_step = 0.0
        self._amb_seen = 0.0
        self._last_warp: Optional[np.ndarray] = None
        self._frame_ts_done = 0.0
        self._load_config()
        if start:
            threading.Thread(target=self._vision_loop, daemon=True).start()

    # ── saved settings ──
    @staticmethod
    def _disk(radius_px: float) -> np.ndarray:
        d = max(3, int(radius_px) * 2 + 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    def _read_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _write_config(self) -> None:
        cfg = self._read_config()
        cfg["camera"] = self.camera.source
        cfg["calibration"] = self.points_norm
        a = self.tracker.anchor
        cfg["anchor"] = {"scale": a["scale"], "corners": a["corners"]} if a else None
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except OSError:
            pass

    def _load_config(self) -> None:
        cfg = self._read_config()
        if cfg.get("calibration"):
            self.points_norm = cfg["calibration"]
        ref = cv2.imread(self.reference_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(self.reference_path) else None
        if ref is not None and ref.shape == (CANVAS[1], CANVAS[0]):
            self.reference = ref
        anchor = cv2.imread(self.anchor_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(self.anchor_path) else None
        if self.points_norm and anchor is not None and cfg.get("anchor"):
            self.tracker.load_anchor(anchor, cfg["anchor"]["scale"], cfg["anchor"]["corners"])
        if cfg.get("camera"):
            self.camera.set_source(cfg["camera"])

    # ── camera + calibration ──
    def set_source(self, source: Optional[str]) -> None:
        self.camera.set_source(source)
        self._write_config()

    @property
    def calibrated(self) -> bool:
        return self.points_norm is not None

    def _homography(self, frame_shape) -> Optional[np.ndarray]:
        if not self.points_norm:
            return None
        h, w = frame_shape[:2]
        src = np.array([[u * w, v * h] for u, v in self.points_norm], np.float32)
        dst = np.array([to_canvas(*p) for p in CAL_WORLD], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def calibrate(self, points_norm: List[List[float]]) -> None:
        if len(points_norm) != 4 or any(not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0) for u, v in points_norm):
            raise ValueError("need four corners, all inside the picture")
        frame, _ = self.camera.latest()
        if frame is None:
            raise ValueError("no camera picture yet")
        h, w = frame.shape[:2]
        problem = calibration_problem([[u * w, v * h] for u, v in points_norm])
        if problem:
            raise ValueError(f"Calibration not saved: {problem}.")
        with self.lock:
            self.points_norm = [[float(u), float(v)] for u, v in points_norm]
            self.reference = None                          # an old empty-board photo no longer lines up
            self._areas.clear()
            self._tracks.clear()
            self._remove_reference_file()
            self._set_anchor(frame)
            self._write_config()

    def _set_anchor(self, frame: np.ndarray) -> None:
        """Remember what the board looks like now, so a moved phone can be followed."""
        h, w = frame.shape[:2]
        self.tracker.set_anchor(frame, [[u * w, v * h] for u, v in self.points_norm])
        cv2.imwrite(self.anchor_path, self.tracker.anchor["gray"])

    def reset_calibration(self) -> None:
        with self.lock:
            self.points_norm, self.reference = None, None
            self.tracker.anchor = None
            self._tracks.clear()
            self._remove_reference_file()
            try:
                os.remove(self.anchor_path)
            except OSError:
                pass
            self._write_config()

    def _check_motion(self, frame: np.ndarray, now: float) -> None:
        """Every few seconds: has the phone moved since calibrating? If so, move the corners with it."""
        if not self.calibrated:
            return
        h, w = frame.shape[:2]
        if self.tracker.anchor is None:                    # calibrated before this run (or the snapshot was lost): trust this view
            with self.lock:
                self._set_anchor(frame)
                self._write_config()
            return
        moved = self.tracker.check(frame, [[u * w, v * h] for u, v in self.points_norm])
        if moved:
            with self.lock:
                self.points_norm = [[x / w, y / h] for x, y in moved]    # the flat plan lands where it did, so the empty-board photo stays valid
                self.realigned_at = now
                self._write_config()
            print(f"[Camera] the phone moved {self.tracker.last.get('moved_px')} px: corners re-aligned", flush=True)

    def _remove_reference_file(self) -> None:
        try:
            os.remove(self.reference_path)
        except OSError:
            pass

    def capture_reference(self) -> bool:
        frame, _ = self.camera.latest()
        if frame is None or not self.calibrated:
            return False
        warped = cv2.warpPerspective(frame, self._homography(frame.shape), CANVAS)
        ref = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        with self.lock:
            self.reference = ref
            self._tracks.clear()
            cv2.imwrite(self.reference_path, ref)
        return True

    def clear_reference(self) -> None:
        with self.lock:
            self.reference = None
            self._remove_reference_file()

    # ── seeing ──
    def _vision_loop(self) -> None:
        while True:
            try:
                frame, ts = self.camera.latest()
                if frame is not None and ts != self._frame_ts_done and self.calibrated:
                    self._frame_ts_done = ts
                    self._process(frame, ts)
                    if ts - self._last_track >= TRACK_CHECK_S:
                        self._last_track = ts
                        self._check_motion(frame, ts)
            except Exception as e:
                print(f"[Camera] vision error: {e}", flush=True)
            time.sleep(0.08)

    def _process(self, frame: np.ndarray, now: float) -> None:
        H = self._homography(frame.shape)
        warped = cv2.warpPerspective(frame, H, CANVAS)
        dets = self.detect(warped)
        with self.lock:
            self._last_warp = warped
            self._update_tracks(dets, now)
            self._read_plates(frame, H, now)
            out = []
            for tr in self._tracks.values():
                if tr.last_seen != now or not tr.confirmed(now):
                    continue
                where = locate(tr.x, tr.y) or {}
                is_car = tr.kind == "car" and not tr.says_ambulance
                out.append({"id": tr.tid, "kind": "ambulance" if tr.says_ambulance else tr.kind,
                            "x": round(tr.x, 1), "y": round(tr.y, 1), "bbox": tr.bbox,
                            "count": tr.count, "waiting": tr.waiting(now), "road": where.get("road"),
                            "junction": where.get("junction"), "phase": where.get("phase"),
                            "in_junction": bool(where.get("in_junction")),
                            "plate": tr.plate if is_car else None,
                            "plate_state": STATE_NAMES.get((tr.plate or "")[:2]) if is_car else None})
            self.vehicles = out

    def _read_plates(self, frame: np.ndarray, H: np.ndarray, now: float) -> None:
        """Collect what the OCR has read, and send it the next car whose plate isn't settled (cut from the full-size picture)."""
        if self.plates is None:
            return
        for tid, reads in self.plates.take_results().items():
            tr = self._tracks.get(tid)
            if not tr:
                continue
            for plate, conf in reads:
                if plate == AMBULANCE or tr.count == 1:        # a merged nose-to-tail blob holds several plates: only AMBULANCE counts
                    tr.votes[plate] += conf
        H_inv = np.linalg.inv(H)
        for tr in self._tracks.values():
            if tr.kind != "car" or tr.last_seen != now or tr.rect is None or tr.count > 3:
                continue                                       # the wide AMBULANCE card can look like two or three cars, so those are read too
            if tr.plate_score >= OCR_CONFIDENT_SCORE or now - tr.last_ocr < OCR_RETRY_S:
                continue
            quad = cv2.perspectiveTransform(cv2.boxPoints(tr.rect).reshape(-1, 1, 2).astype(np.float32), H_inv).reshape(4, 2)
            crop = rectify_label(frame, quad)
            if crop.size and self.plates.submit(tr.tid, crop):
                tr.last_ocr = now

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def _one_car_area(self) -> float:
        if len(self._areas) >= 20:
            return float(np.percentile(self._areas, 30))
        return CAR_AREA_UNITS * SCALE ** 2

    def _blobs(self, mask: np.ndarray, kind: str) -> List[Dict[str, Any]]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo, hi = (CAR_MIN_SIDE * SCALE) ** 2, (CAR_MAX_SIDE * SCALE) ** 2
        one = self._one_car_area()
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < lo or area > hi * 4:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) > 3.5 * max(1, min(w, h)) and area < 2 * one:
                continue                                    # long thin strips are road edges, not cars
            m = cv2.moments(c)
            ux, uy = from_canvas(m["m10"] / m["m00"], m["m01"] / m["m00"])
            rect = cv2.minAreaRect(c)
            if kind == "car" and area >= 0.6 * CAR_AREA_UNITS * SCALE ** 2 and area / max(1.0, rect[1][0] * rect[1][1]) >= 0.75:
                self._areas.append(area)                    # only whole, tidy cards teach the size: fragments cut by a lamp's glow would shrink it
                self._areas = self._areas[-300:]
            out.append({"kind": kind, "x": ux, "y": uy, "bbox": [int(x), int(y), int(w), int(h)],
                        "count": int(min(6, max(1, round(area / one)))), "rect": rect,
                        "rectangularity": area / max(1.0, rect[1][0] * rect[1][1])})
        return out

    def _lit_leds(self, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Lit signal lamps: a blown-out core ringed by strongly coloured glow (paper cards never clip)."""
        core = (v >= LED_CORE_V).astype(np.float32)
        rest = 1.0 - core
        k = (self._ring, self._ring)
        ring_sat = cv2.boxFilter(s.astype(np.float32) * rest, -1, k, normalize=False) / np.maximum(
            cv2.boxFilter(rest, -1, k, normalize=False), 1.0)
        now = ((core > 0) & (ring_sat > LED_RING_SAT)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(now)
        big = np.nonzero(stats[1:, cv2.CC_STAT_AREA] > (LED_CORE_MAX_UNITS * SCALE) ** 2)[0] + 1
        if big.size:
            now[np.isin(labels, big)] = 0
        now = now.astype(np.float32)
        if self._led_memory is None or self._led_memory.shape != now.shape:
            self._led_memory = now
        else:
            self._led_memory = np.maximum(self._led_memory * LED_MEMORY_DECAY, now)     # a lamp stays masked ~5 frames
        return (self._led_memory > 0.2).astype(np.uint8) * 255

    @staticmethod
    def _overlap(blob: Dict[str, Any], mask: np.ndarray) -> float:
        x, y, w, h = blob["bbox"]
        region = mask[y:y + h, x:x + w]
        return float(np.count_nonzero(region)) / max(1, region.size)

    @staticmethod
    def _solidity(blob: Dict[str, Any], mask: np.ndarray) -> float:
        x, y, w, h = blob["bbox"]
        (_, _), (rw, rh), _ = blob["rect"]
        return float(np.count_nonzero(mask[y:y + h, x:x + w])) / max(1.0, rw * rh)

    @staticmethod
    def _is_card(blob: Dict[str, Any]) -> bool:
        """Shaped like the AMBULANCE card, not a patch of lamp glow or a lens-flare streak."""
        rw, rh = blob["rect"][1]
        long_side, short_side = max(rw, rh) / SCALE, max(1.0, min(rw, rh)) / SCALE
        return (long_side >= CARD_MIN_LONG and short_side >= CARD_MIN_SHORT
                and long_side / short_side >= CARD_MIN_ASPECT and blob["rectangularity"] >= CARD_MIN_RECT)

    def detect(self, warped: np.ndarray) -> List[Dict[str, Any]]:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        white = ((s < 70) & (v > 150)).astype(np.uint8) * 255
        pink = (((h >= PINK_HUE[0]) | (h <= PINK_HUE[1])) & (s >= PINK_SAT[0]) & (s <= PINK_SAT[1])
                & (v >= PINK_MIN_V)).astype(np.uint8) * 255
        if self.reference is not None:
            gray = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
            ref = self.reference.astype(np.float32)
            road = self._roi > 0
            gain = float(np.median(gray[road])) / max(1.0, float(np.median(ref[road])))    # the phone re-exposes whenever the LEDs change
            ref *= min(2.0, max(0.5, gain))
            diff = np.abs(gray - ref)
            changed = cv2.dilate(((diff > CHANGE_ABS) & (diff > CHANGE_REL * ref)).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            white &= changed
            pink &= changed
        leds = self._lit_leds(s, v)
        core, halo = cv2.dilate(leds, self._pad), cv2.dilate(leds, self._halo)
        white = self._fill_holes(white & cv2.bitwise_not(core) & self._roi)
        colour = pink & self._roi
        cars = [b for b in self._blobs(white, "car") if self._overlap(b, halo) <= LED_HALO_MAX_FRAC]
        ambulances = [b for b in self._blobs(self._fill_holes(colour), "ambulance")
                      if self._solidity(b, colour) >= CARD_SOLIDITY and self._is_card(b)]
        return cars + ambulances

    def _update_tracks(self, dets: List[Dict[str, Any]], now: float) -> None:
        unmatched = list(range(len(dets)))
        for tr in sorted(self._tracks.values(), key=lambda t: -t.last_seen):
            best, best_d = None, TRACK_MATCH_UNITS
            for i in unmatched:
                if dets[i]["kind"] != tr.kind:
                    continue
                d = ((dets[i]["x"] - tr.x) ** 2 + (dets[i]["y"] - tr.y) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                unmatched.remove(best)
                tr.update(dets[best], now)
        for i in unmatched:
            self._tracks[self._next_tid] = Track(self._next_tid, dets[i], now)
            self._next_tid += 1
        for tid in [t for t, tr in self._tracks.items() if now - tr.last_seen > TRACK_TTL_S]:
            del self._tracks[tid]

    # ── driving the Signal AI ──
    def tick(self, now: Optional[float] = None) -> None:
        """One AI step at most every STEP_S: cars in, the ambulance corridor, then the agents decide."""
        now = time.time() if now is None else now
        if now - self._last_step < STEP_S:
            return
        self._last_step = now
        _, ts = self.camera.latest()
        ok = self.calibrated and ts > 0 and (time.time() - ts) < FRAME_STALE_S
        with self.lock:
            vehicles = list(self.vehicles) if ok else []
        self._apply_ambulance(vehicles, now)
        self.engine.load(vehicles)
        self.engine.time += 1
        self.control_mode = self.coordinator.step(self.engine.vehicle_info, data_ok=ok)["control_mode"]

    def _apply_ambulance(self, vehicles: List[Dict[str, Any]], now: float) -> None:
        """Green corridor for an ambulance on the east-west road (both junctions) or on a north-south arm (that junction)."""
        want: Dict[str, str] = {}
        for v in vehicles:
            if v["kind"] != "ambulance":
                continue
            if v.get("road"):
                for jid in (NODES if v["phase"] == "EW" else [v["junction"]]):
                    want.setdefault(jid, v["phase"])
            elif v.get("in_junction") and v.get("junction") in self.ambulance:
                want.update(self.ambulance)                    # crossing a box: keep the corridor it has
        if want:
            self._amb_seen = now
        elif self.ambulance and now - self._amb_seen < AMBULANCE_HOLD_S:
            want = dict(self.ambulance)                        # a blink in detection must not drop the corridor
        agents = self.coordinator.agents
        for jid, phase in want.items():
            if self.ambulance.get(jid) != phase:
                agents[jid].set_emergency(phase)
        for jid in self.ambulance:
            if jid not in want:
                agents[jid].set_emergency(None)
        self.ambulance = want

    # ── pictures + state ──
    def frame_jpeg(self, view: str) -> Optional[bytes]:
        if view == "board":
            img = self.annotated()
        else:
            frame, _ = self.camera.latest()
            img = frame
            if img is not None and img.shape[1] > 1280:
                img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None

    def annotated(self) -> Optional[np.ndarray]:
        """The flat plan with the two junction boxes and every detected car or ambulance drawn on."""
        with self.lock:
            if self._last_warp is None:
                return None
            img = self._last_warp.copy()
            vehicles = list(self.vehicles)
        for jid, (jx, jy) in NODES.items():
            x0, y0 = to_canvas(jx - HALF, jy + HALF)
            x1, y1 = to_canvas(jx + HALF, jy - HALF)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 1)
            cv2.putText(img, jid, (int(x1) + 6, int(y0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        for v in vehicles:
            x, y, w, h = v["bbox"]
            amb = v["kind"] == "ambulance"
            colour = (60, 60, 255) if amb else ((80, 230, 80) if v["road"] else (180, 180, 180))
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 3 if amb else 2)
            label = "AMBULANCE" if amb else (v.get("plate") or f"car {v['id']}") + (f" x{v['count']}" if v["count"] > 1 else "")
            cv2.putText(img, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        return img

    def state(self) -> Dict[str, Any]:
        frame, ts = self.camera.latest()
        with self.lock:
            vehicles = [{k: v[k] for k in ("id", "kind", "road", "junction", "phase", "waiting", "count", "plate", "plate_state")}
                        for v in self.vehicles]
        return {
            "source": self.camera.source,
            "live": frame is not None and (time.time() - ts) < FRAME_STALE_S,
            "fps": self.camera.fps,
            "error": self.camera.error,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
            "calibrated": self.calibrated,
            "points": self.points_norm,
            "labels": CAL_LABELS,
            "has_reference": self.reference is not None,
            "following": self.tracker.anchor is not None,
            "realigned_s_ago": round(time.time() - self.realigned_at) if self.realigned_at else None,
            "plates_available": None if self.plates is None else self.plates.available,
            "vehicles": vehicles,
            "ambulance": dict(self.ambulance),
            "control_mode": self.control_mode,
        }


def register_camera_routes(app, pilot: CameraPilot, board) -> None:
    from flask import Response, jsonify, request

    def bad(msg: str, code: int = 400):
        return jsonify({"ok": False, "error": msg}), code

    @app.route("/api/board/camera", methods=["POST"])
    def cam_source():
        pilot.set_source(str((request.get_json(silent=True) or {}).get("source") or "").strip() or None)
        return jsonify(board.state())

    @app.route("/api/board/calibrate", methods=["POST"])
    def cam_calibrate():
        d = request.get_json(silent=True) or {}
        try:
            if d.get("reset"):
                pilot.reset_calibration()
            else:
                pilot.calibrate([[float(u), float(v)] for u, v in d.get("points") or []])
        except (ValueError, TypeError) as e:
            return bad(str(e))
        return jsonify(board.state())

    @app.route("/api/board/reference", methods=["POST"])
    def cam_reference():
        if (request.get_json(silent=True) or {}).get("clear"):
            pilot.clear_reference()
        elif not pilot.capture_reference():
            return bad("Calibrate on a live camera first, then photograph the empty board", 409)
        return jsonify(board.state())

    @app.route("/api/board/frame.jpg")
    def cam_frame():
        data = pilot.frame_jpeg("board" if request.args.get("view") == "board" else "raw")
        if data is None:
            return Response(status=204)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
