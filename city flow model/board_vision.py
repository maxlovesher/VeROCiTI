"""
board_vision.py — Overhead camera -> cars on the physical demo board.
=====================================================================
  camera     : a phone running IP Webcam / DroidCam (MJPEG URL), a USB webcam
               index, or "synthetic" (a rendered board, for testing with no
               hardware at all)
  calibrate  : 4 clicked points (the corner bends) give a homography from the
               camera image onto a top-down canvas in roadnet units
  detect     : white boxes = cars, red box = ambulance. Colour + shape on the
               matte road, optionally AND-ed with a difference against an
               empty-board reference shot, then opened so lane markings,
               zebra stripes and LED glints drop out; lit signal lamps
               (clipped core + coloured glow) are masked so they never
               read as a car or an ambulance
  track      : nearest-neighbour tracks, so a plate read once sticks to its car
  ocr        : each car's top face is cut out along its own tilt and turned
               upright, then EasyOCR reads it in a background thread with
               Indian-plate positional fixes (O/0, I/1, S/5, B/8), the most
               confident valid reading wins, reads are vote-weighted by
               confidence across frames, and (optionally) snapped to the
               list of registered plates — a stand-in for the vehicle
               registry lookup a real ANPR camera does
"""

import math
import queue
import re
import threading
import time
from collections import Counter, deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import board_layout as bl

# Ambulance colours: a red box, or the pink AMBULANCE card. Light pink paper reads on the phone
# camera as hue ~165-172 (OpenCV's 0-179 scale), saturation ~90-140, very bright.
PINK_HUE_MIN = 150            # pink/magenta from here up to 179, plus 0-6 where it wraps round to red
PINK_SAT = (55, 200)
PINK_MIN_V = 140
# A card is a solid sheet of colour (apart from its lettering); a toy car's painted body is only a
# coloured ring around its white plate label. A candidate needs this much of its rectangle in colour.
AMBULANCE_SOLIDITY = 0.6
# Lamp glow is a round blob (rectangularity ~0.5-0.8 on the real board); a card is a crisp rectangle
# (~0.9+), so a card stays a card even when its bright paper has blown-out, lamp-like highlights.
CARD_RECTANGULARITY = 0.86
CAR_MIN_SIDE = 10.0           # roadnet units; smaller blobs are markings / glints
CAR_MAX_SIDE = 80.0
OPEN_KERNEL_UNITS = 7.0       # erases anything thinner than this (lane dashes, zebra stripes)
CAR_AREA_UNITS = 30.0 * 22.0  # starting guess for one model car's footprint, used to split nose-to-tail blobs
CAR_AREA_LEARN_MIN = 20       # blob samples needed before the learned footprint replaces the guess
TRACK_MATCH_UNITS = 30.0
TRACK_TTL_S = 1.0
# A detection only counts once it has been seen steadily for this long. Real cards sit still;
# flicker from the LEDs changing (and the phone's auto-exposure reacting) lasts a frame or two.
CONFIRM_S = {"car": 0.7, "ambulance": 1.0}
CONFIRM_MIN_SIGHTINGS = 3
# Empty-board comparison: a pixel has "changed" if it differs from the (exposure-matched)
# reference by more than this many grey levels, or this fraction of its reference brightness.
CHANGE_ABS = 28
CHANGE_REL = 0.22
STATIONARY_UNITS = 6.0
# The board's own LEDs: a lit lamp is a blown-out core (the paper cars never clip) ringed by
# strongly coloured glow. Found per frame and remembered briefly so a lamp mid-change can't
# flicker into a phantom car, or a red one into a phantom ambulance.
LED_CORE_V = 245
LED_RING_SAT = 100            # mean saturation around a core; card glare sits on white paper (~10)
LED_RING_UNITS = 15.0
LED_CORE_MAX_UNITS = 16.0     # a lamp's blown-out centre is small; a card-sized one is a white label on a red/pink car
LED_CORE_PAD_UNITS = 5.0      # cut out of the white mask so a car beside a lamp stays separate
LED_HALO_UNITS = 14.0         # a white blob mostly inside this is lamp glow on the zebra / road
LED_HALO_MAX_FRAC = 0.45
LED_MEMORY_DECAY = 0.75       # per frame; a lamp stays masked ~5 frames after it goes dark
OCR_RETRY_S = 1.2
OCR_CONFIDENT_SCORE = 1.6     # summed read confidence (~two good reads) before a car stops being re-read
OCR_EARLY_EXIT_CONF = 0.8     # one reading this confident is enough; skip the remaining orientations
LABEL_H = 96                  # height in pixels of the upright label crop handed to the OCR

PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}([A-Z]{1,3}\d{1,4})?$")
# Indian state / UT codes; an OCR read that doesn't start with one is noise, not a plate.
STATE_NAMES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh", "DD": "Daman & Diu", "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli", "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu & Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha", "PB": "Punjab", "PY": "Puducherry",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}
STATE_CODES = set(STATE_NAMES)
# The usual ANPR look-alike swaps, applied only where the plate format says a letter (or digit) must be.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "J", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "J": "3", "A": "4", "S": "5", "G": "6",
             "B": "8", "T": "7"}


AMBULANCE = "AMBULANCE"
# Reading weight a card needs on "AMBULANCE" before it's treated as one: a single fair read.
AMBULANCE_TEXT_SCORE = 0.4
_AMBULANCE_PIECES = ("AMBUL", "MBULA", "BULAN", "ULANC")     # not "LANCE": BALANCE has it too
_AMB_LETTER = {"4": "A", "8": "B", "0": "O", "1": "L", "5": "S", "6": "G", "3": "B", "7": "T", "2": "Z", "I": "L"}


def is_ambulance_text(text: str) -> bool:
    """
    True for OCR text that is the word AMBULANCE, allowing the usual misreads (AMBULANCF,
    4MBULANCE, AMBU LANCE, AMBULANC) without matching ordinary plates.
    """
    s = "".join(_AMB_LETTER.get(c, c) for c in re.sub(r"[^A-Z0-9]", "", (text or "").upper()))
    if len(s) < 5:
        return False
    if any(piece in s for piece in _AMBULANCE_PIECES):
        return True
    from difflib import SequenceMatcher
    return len(s) >= 7 and SequenceMatcher(None, s, AMBULANCE).ratio() >= 0.8


def normalise_plate(text: str) -> Optional[str]:
    """Clean an OCR string into a plate like TN07 / KA01AB1234, fixing the usual letter/digit swaps by position."""
    s = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if len(s) < 4:
        return None
    chars = list(s)
    for i in (0, 1):
        chars[i] = _TO_LETTER.get(chars[i], chars[i])
    for i in (2, 3):
        chars[i] = _TO_DIGIT.get(chars[i], chars[i])
    fixed = "".join(chars)
    if fixed[:2] not in STATE_CODES:
        return None
    if PLATE_RE.match(fixed):
        return fixed
    if PLATE_RE.match(fixed[:4]):
        return fixed[:4]
    return None


def parse_plate_list(plates: Iterable[str]) -> List[str]:
    """Registered plates as typed by the operator ('tn07, KA-01 ...') -> cleaned, valid, de-duplicated."""
    out: List[str] = []
    for raw in plates:
        p = normalise_plate(raw)
        if p and p not in out:
            out.append(p)
    return out


def snap_to_registered(plate: str, registered: Iterable[str]) -> Tuple[Optional[str], float]:
    """
    Match a read against the registered plates: exact -> itself (weight 1.0);
    one character off -> the registered plate it must have been (0.8), when
    that's unambiguous; anything else -> kept but down-weighted (0.3), so a
    stray misread can't outvote a registered plate.
    """
    registered = list(registered)
    if not registered:
        return plate, 1.0
    if plate in registered:
        return plate, 1.0
    near = [r for r in registered if len(r) == len(plate) and sum(a != b for a, b in zip(r, plate)) == 1]
    if len(near) == 1:
        return near[0], 0.8
    return plate, 0.3


def rectify_label(img: np.ndarray, rect, inset: float = 0.0) -> np.ndarray:
    """
    Cut a car's top face out of the canvas along its own tilt, as an upright
    rectangle with the long side horizontal, scaled to LABEL_H pixels tall.
    Whether the text ends up the right way up or upside down is left to the
    reader, which tries both. No inset by default: printed plates nearly fill
    the box, and trimming the edge clips the outer characters.
    """
    pts = cv2.boxPoints(rect).astype(np.float32)
    d01 = float(np.linalg.norm(pts[1] - pts[0]))
    d12 = float(np.linalg.norm(pts[2] - pts[1]))
    if d01 >= d12:
        a, b, c, long_side, short_side = pts[0], pts[1], pts[2], d01, d12
    else:
        a, b, c, long_side, short_side = pts[1], pts[2], pts[3], d12, float(np.linalg.norm(pts[3] - pts[2]))
    scale = LABEL_H / max(short_side, 1.0)
    w, h = max(8, int(round(long_side * scale))), LABEL_H
    # Keep the corner order's handedness so the crop is rotated, never mirrored.
    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    dst = [(0, 0), (w, 0), (w, h)] if cross > 0 else [(0, h), (w, h), (w, 0)]
    M = cv2.getAffineTransform(np.float32([a, b, c]), np.float32(dst))
    crop = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    dx, dy = int(w * inset), int(h * inset)
    return crop[dy:h - dy, dx:w - dx]


# ── calibration helpers ──

def _bright_strips(frame: np.ndarray, near: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    The light grey road-edge strips (and other white paint): colourless and clearly brighter than
    the road. "Clearly brighter" is measured against the road around `near` when given, so a strip
    in shadow on one side of the board still stands out.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, sat = hsv[..., 2], hsv[..., 1]
    threshold = 165
    if near is not None:
        x, y = near
        patch = v[max(0, y - 15): y + 16, max(0, x - 15): x + 16]
        if patch.size:
            threshold = int(min(165, max(105, float(np.median(patch)) + 40)))
    # Not "colourless" strictly: under warm room light the grey strips (and the yellow edge lines
    # beside them, which mark the same road edge) come out quite saturated on the phone camera.
    return (v > threshold) & (sat < 170)


def _first_run(line: np.ndarray, min_run: int = 3) -> Optional[int]:
    """Index where the first run of >= min_run bright pixels starts, scanning from index 0."""
    run = 0
    for i, v in enumerate(line):
        run = run + 1 if v else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def _robust_line(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Fit b = a*t + c through (t, b) points, ignoring outliers (a zebra bar, a lamp housing, a stray
    glint): the median of all pairwise slopes (Theil–Sen) shrugs off up to ~30% bad points, then a
    least-squares refit on the points within a few pixels of that line.
    """
    if len(pts) < 6:
        return None
    p = np.array(pts, float)
    i, j = np.triu_indices(len(p), 1)
    dt = p[j, 0] - p[i, 0]
    ok = np.abs(dt) > 1e-9
    a = float(np.median((p[j, 1] - p[i, 1])[ok] / dt[ok]))
    c = float(np.median(p[:, 1] - a * p[:, 0]))
    inl = np.abs(p[:, 1] - (a * p[:, 0] + c)) <= 4.0
    if inl.sum() < max(6, len(p) // 2):
        return None
    a, c = np.polyfit(p[inl, 0], p[inl, 1], 1)
    return float(a), float(c)


def find_box_corners(frame: np.ndarray, centre: Tuple[float, float], max_reach: int = 900) -> Optional[List[List[float]]]:
    """
    The four inside corners of the junction box around an image point (TL, TR, BR, BL), found by
    scanning outward from it to the grey road-edge strips on every side and fitting a line along
    each edge. Works best with the signal lamps switched off. None if the box can't be found.
    """
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    if not (0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]):
        return None
    bright = _bright_strips(frame, near=(cx, cy))
    h_img, w_img = bright.shape

    def reach(dx: int, dy: int, x0: int, y0: int) -> Optional[int]:
        """Distance from (x0, y0) along (dx, dy) to the first bright strip."""
        if dx:
            if not 0 <= y0 < h_img:
                return None
            line = bright[y0, x0::dx][:max_reach] if dx > 0 else bright[y0, x0::-1][:max_reach]
        else:
            if not 0 <= x0 < w_img:
                return None
            line = bright[y0::dy, x0][:max_reach] if dy > 0 else bright[y0::-1, x0][:max_reach]
        return _first_run(line)

    # Rough size first, from several lines around the click (not the click line itself: it may carry a
    # dash). A stretch of strip lost to shadow or a lamp housing makes a probe overshoot, so take a low
    # percentile rather than the median.
    probes = (-40, -30, -20, -10, 10, 20, 30, 40)

    def median_reach(dx, dy):
        vals = [reach(dx, dy, cx + (0 if dx else o), cy + (o if dx else 0)) for o in probes]
        vals = [v for v in vals if v is not None]
        return int(np.percentile(vals, 25)) if vals else None

    left, right, up, down = median_reach(-1, 0), median_reach(1, 0), median_reach(0, -1), median_reach(0, 1)
    if None in (left, right, up, down) or min(left, right, up, down) < 15:
        return None

    # Sample across the middle 70% of the box itself, wherever inside it the click landed.
    edges = {}
    top, bottom, lft, rgt = cy - up, cy + down, cx - left, cx + right
    rows = np.linspace(top + 0.15 * (bottom - top), bottom - 0.15 * (bottom - top), 40).astype(int)
    cols = np.linspace(lft + 0.15 * (rgt - lft), rgt - 0.15 * (rgt - lft), 40).astype(int)
    for name, dx, dy in (("L", -1, 0), ("R", 1, 0), ("T", 0, -1), ("B", 0, 1)):
        pts = []
        for k in (rows if dx else cols):
            if dx:
                d = reach(dx, 0, cx, int(k))
                if d is not None:
                    pts.append((float(k), cx + dx * (d - 0.5)))     # x of the strip's inner edge on row k
            else:
                d = reach(0, dy, int(k), cy)
                if d is not None:
                    pts.append((float(k), cy + dy * (d - 0.5)))     # y of the strip's inner edge on column k
        edges[name] = _robust_line(pts)
    if any(v is None for v in edges.values()):
        return None

    def corner(v, hz):   # vertical edge x = av*y + bv meets horizontal edge y = ah*x + bh
        av, bv = v
        ah, bh = hz
        y = (ah * bv + bh) / (1 - ah * av)
        return [av * y + bv, y]

    L, R, T, B = edges["L"], edges["R"], edges["T"], edges["B"]
    return [corner(L, T), corner(R, T), corner(R, B), corner(L, B)]


def calibration_shape_problem(corners: List[List[float]], world: List[Tuple[float, float]]) -> Optional[str]:
    """
    Sanity check for hand-clicked calibration points: each side of the clicked shape should be
    stretched by about the same amount relative to the real layout (an overhead camera only adds
    mild perspective). Clicking the wrong spots (road tips instead of box corners, or out of
    order) breaks that. Returns a reason, or None if the clicks look right.
    """
    img = np.array(corners, float)
    w = np.array(world, float)
    ratios = []
    for i in range(4):
        j = (i + 1) % 4
        img_len = float(np.linalg.norm(img[j] - img[i]))
        if img_len < 10:
            return "two of the points are almost on top of each other"
        ratios.append(img_len / float(np.linalg.norm(w[j] - w[i])))
    # The layout's points go clockwise on the board, so the clicks must too (image y points down).
    signed = sum(img[i][0] * img[(i + 1) % 4][1] - img[(i + 1) % 4][0] * img[i][1] for i in range(4))
    if signed <= 0:
        return "the points are in the wrong order (go clockwise, starting top-left)"
    if max(ratios) / min(ratios) > 1.5:
        return "the shape doesn't match the junction boxes; check you clicked the box corners, not the road ends"
    return None


# ── following the camera ──

class CameraTracker:
    """
    Keeps a calibration valid when the phone is nudged, rotated or switched to another resolution.
    At calibration it keeps a snapshot of the camera picture; afterwards it matches the live picture
    against that snapshot (ORB features on the board's fixed paint: zebras, strips, lane dashes;
    moving cars just become RANSAC outliers) and works out where the calibration points are now.
    """

    WIDTH = 1280                # both pictures are compared at this width
    MIN_INLIERS = 50            # matches agreeing on one exact movement; random mismatches never line up this well
    MIN_INLIER_SHARE = 0.12     # the real board agrees ~99%; repetitive paint (zebras, dashes) can drag this down
    MOVE_PX = 8.0               # full-resolution pixels; less than this is noise, not a moved phone
    AGREE_PX = 10.0             # two checks in a row must agree this closely before re-aligning

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
        self.anchor = {"gray": small, "scale": scale, "kp": kp, "des": des,
                       "corners": [list(map(float, c)) for c in corners]}
        self._pending = None

    def load_anchor(self, gray_small: np.ndarray, scale: float, corners: List[List[float]]) -> None:
        kp, des = self._orb.detectAndCompute(gray_small, None)
        self.anchor = {"gray": gray_small, "scale": float(scale), "kp": kp, "des": des,
                       "corners": [list(map(float, c)) for c in corners]}
        self._pending = None

    def check(self, frame: np.ndarray, current: List[List[float]]) -> Optional[List[List[float]]]:
        """Where the anchored calibration points are in this frame, if the phone has moved; else None."""
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
        moved = (cv2.perspectiveTransform(pts, H).reshape(-1, 2) / scale)
        shift = float(np.max(np.linalg.norm(moved - np.float32(current), axis=1)))
        self.last["moved_px"] = round(shift, 1)
        if shift < self.MOVE_PX or shift > 0.6 * frame.shape[1]:
            self._pending = None
            return None
        if self._pending is None or float(np.max(np.linalg.norm(moved - self._pending, axis=1))) > self.AGREE_PX:
            self._pending = moved           # wait for the next check to confirm it before re-aligning
            return None
        self._pending = None
        return moved.tolist()


# ── camera sources ──

class CameraSource:
    """Keeps only the newest frame from a camera, reconnecting on its own if the feed drops."""

    def __init__(self):
        self.source: Optional[str] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.error: Optional[str] = None
        self.synthetic: Optional["SyntheticBoard"] = None

    def set_source(self, source: Optional[str], synthetic: Optional["SyntheticBoard"] = None) -> None:
        self.stop()
        self.source = (str(source).strip() or None) if source is not None else None
        self.synthetic = synthetic
        self.error = None
        self.fps = 0.0
        with self._lock:
            self._frame = None
            self._frame_ts = 0.0
        if self.source:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest(self):
        with self._lock:
            return self._frame, self._frame_ts

    def _publish(self, frame: np.ndarray, last_ts: float) -> float:
        now = time.time()
        with self._lock:
            self._frame = frame
            self._frame_ts = now
        if last_ts:
            dt = now - last_ts
            if dt > 0:
                self.fps = round(0.8 * self.fps + 0.2 * (1.0 / dt), 1)
        self.error = None
        return now

    def _run(self, stop: threading.Event) -> None:
        last = 0.0
        if self.source == "synthetic":
            while not stop.is_set():
                if self.synthetic is not None:
                    last = self._publish(self.synthetic.render(), last)
                stop.wait(0.1)
            return

        src: Any = int(self.source) if self.source.isdigit() else self.source
        while not stop.is_set():
            if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                self.error = f"Could not open camera '{self.source}'"
                cap.release()
                stop.wait(2.0)
                continue
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = "Camera feed dropped, reconnecting…"
                    break
                last = self._publish(frame, last)
            cap.release()
            stop.wait(1.0)


class SyntheticBoard:
    """
    A rendered stand-in for the real board, seen by a slightly tilted camera:
    roads, lane dashes, zebra crossings, LED heads showing the live signal
    state, white plate-labelled cars and a red ambulance. Lets the whole
    pipeline run (and be tested) with no phone and no hardware.
    """

    FRAME_W, FRAME_H = 1280, 960
    # The synthetic camera: where four fixed roadnet points (each layout's original calibration
    # points) land in its frame. The points a layout asks you to click are projected through this.
    CAMERA = {
        "full": [[418.0, 262.0], [870.0, 250.0], [905.0, 708.0], [385.0, 722.0]],
        "single": [[655.0, 120.0], [1060.0, 470.0], [640.0, 840.0], [225.0, 490.0]],
        "double": [[460.0, 190.0], [1170.0, 540.0], [815.0, 900.0], [100.0, 550.0]],
    }
    S = 2.5   # render scale, px per roadnet unit, so plate text survives the camera warp
    PLATES = ["TN07", "KA01", "GJ05", "MH12", "DL03", "WB20", "OD02", "KL09", "AP16", "RJ14"]

    def __init__(self, layout: bl.Layout, signal_getter: Optional[Callable[[], Dict[str, Dict[str, str]]]] = None,
                 edge_strips: bool = False):
        self.edge_strips = edge_strips      # light grey road-edge strips, as on the real board
        self.cars: List[Dict[str, Any]] = []
        self.signal_getter = signal_getter
        self._lock = threading.Lock()
        self._plate_idx = 0
        self.set_layout(layout)

    def set_layout(self, layout: bl.Layout) -> None:
        self.layout = layout
        src = np.array([layout.to_canvas(x, y, self.S) for x, y in layout.legacy_calibration_points], dtype=np.float32)
        dst = np.array(self.CAMERA[layout.name], dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, dst)
        self._base = self._draw_base()
        self.set_cars([])

    @property
    def true_points(self) -> List[List[float]]:
        """Where this layout's calibration points appear in the synthetic frame (what a perfect click gives)."""
        pts = np.array([self.layout.to_canvas(x, y, self.S) for x, y in self.layout.calibration_points], dtype=np.float32)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self._H).reshape(-1, 2).tolist()

    def set_cars(self, cars: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.cars = [dict(c) for c in cars]

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(c) for c in self.cars]

    def toggle_at(self, x: float, y: float, kind: str = "car") -> None:
        """Add a car at a roadnet point, or remove the one already there."""
        with self._lock:
            for c in self.cars:
                if math.hypot(c["x"] - x, c["y"] - y) < 22:
                    self.cars.remove(c)
                    return
            plate = self.PLATES[self._plate_idx % len(self.PLATES)]
            self._plate_idx += 1
            self.cars.append({"x": x, "y": y, "plate": plate, "kind": kind})

    def _draw_base(self) -> np.ndarray:
        lay = self.layout
        size = lay.canvas_size(self.S)
        img = np.full((size, size, 3), (120, 170, 205), np.uint8)   # MDF tan (BGR)
        S = self.S
        half = int(bl.ROAD_HALF_WIDTH * S)
        for seg in lay.segments:
            p1 = tuple(int(v) for v in lay.to_canvas(*seg["pa"], self.S))
            p2 = tuple(int(v) for v in lay.to_canvas(*seg["pb"], self.S))
            cv2.line(img, p1, p2, (38, 38, 38), thickness=2 * half)
        if self.edge_strips:
            strip_w = int(6 * S)
            off = bl.ROAD_HALF_WIDTH + 3          # strip centre, just outside the road surface
            for seg in lay.segments:
                (ax, ay), (bx, by) = seg["pa"], seg["pb"]
                for sgn in (-1, 1):
                    if seg["axis"] == "EW":
                        q1, q2 = (min(ax, bx) - 40, ay + sgn * off), (max(ax, bx) + 40, ay + sgn * off)
                    else:
                        q1, q2 = (ax + sgn * off, min(ay, by) - 40), (ax + sgn * off, max(ay, by) + 40)
                    c1 = tuple(int(v) for v in lay.to_canvas(*q1, S))
                    c2 = tuple(int(v) for v in lay.to_canvas(*q2, S))
                    cv2.line(img, c1, c2, (215, 215, 212), strip_w)
        for seg in lay.segments:
            (ax, ay), (bx, by) = seg["pa"], seg["pb"]
            for t in np.arange(0.0, 1.0, 0.12):
                mx, my = ax + (bx - ax) * (t + 0.025), ay + (by - ay) * (t + 0.025)
                if any(abs(mx - bl.NODES[j][0]) < bl.JUNCTION_HALF and abs(my - bl.NODES[j][1]) < bl.JUNCTION_HALF
                       for j in lay.active):
                    continue          # no lane paint inside a junction box, like the real board
                q1 = lay.to_canvas(ax + (bx - ax) * t, ay + (by - ay) * t, S)
                q2 = lay.to_canvas(ax + (bx - ax) * (t + 0.05), ay + (by - ay) * (t + 0.05), S)
                cv2.line(img, tuple(int(v) for v in q1), tuple(int(v) for v in q2), (235, 235, 235), int(2 * S))
        for jid in lay.active:
            jx, jy = bl.NODES[jid]
            for k in range(-3, 4):
                off = k * 9
                for (x1, y1, x2, y2) in (
                    (jx + off, jy + 40, jx + off, jy + 50), (jx + off, jy - 40, jx + off, jy - 50),
                    (jx + 40, jy + off, jx + 50, jy + off), (jx - 40, jy + off, jx - 50, jy + off),
                ):
                    c1 = tuple(int(v) for v in lay.to_canvas(x1, y1, self.S))
                    c2 = tuple(int(v) for v in lay.to_canvas(x2, y2, self.S))
                    cv2.line(img, c1, c2, (235, 235, 235), int(3 * S))
        return img

    def render(self) -> np.ndarray:
        img = self._base.copy()
        signals = {}
        if self.signal_getter:
            try:
                signals = self.signal_getter() or {}
            except Exception:
                signals = {}
        colour = {"R": (40, 40, 255), "Y": (0, 200, 255), "G": (60, 230, 60)}
        lay = self.layout
        for jid in lay.active:
            jx, jy = bl.NODES[jid]
            st = signals.get(jid, {})
            heads = (("EW", ((jx - 44, jy + 44), (jx + 44, jy - 44))), ("NS", ((jx + 44, jy + 44), (jx - 44, jy - 44))))
            for phase, spots in heads:
                for sx, sy in spots:
                    c = lay.to_canvas(sx, sy, self.S)
                    cv2.circle(img, (int(c[0]), int(c[1])), int(4 * self.S), colour.get(st.get(phase, "R"), colour["R"]), -1)
        for car in self.snapshot():
            cx, cy = lay.to_canvas(car["x"], car["y"], self.S)
            w, h = car.get("w", 30) * self.S, car.get("h", 22) * self.S
            if car.get("body"):
                # A painted toy car under its white label: only a rim of paint shows round the label.
                bw, bh = w * 1.35, h * 1.5
                cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)),
                              tuple(car["body"]), -1)
            fill = tuple(car["colour"]) if car.get("colour") else (
                (40, 40, 220) if car.get("kind") == "ambulance" else (245, 245, 245))
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), fill, -1)
            if car.get("kind") != "ambulance" and car.get("plate"):
                # Big bold print filling the box, like the real labels should be.
                font, thick = cv2.FONT_HERSHEY_DUPLEX, max(2, int(self.S))
                (tw, th), _ = cv2.getTextSize(car["plate"], font, 1.0, thick)
                fs = 0.86 * w / tw
                (tw, th), _ = cv2.getTextSize(car["plate"], font, fs, thick)
                cv2.putText(img, car["plate"], (int(cx - tw / 2), int(cy + th / 2)), font, fs, (10, 10, 10), thick, cv2.LINE_AA)
        frame = cv2.warpPerspective(img, self._H, (self.FRAME_W, self.FRAME_H), borderValue=(70, 80, 90))
        noise = np.random.default_rng().normal(0, 3, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── tracking + OCR ──

class _Track:
    __slots__ = ("tid", "kind", "x", "y", "bbox", "rect", "count", "first_seen", "last_seen", "sightings",
                 "history", "votes", "last_ocr")

    def __init__(self, tid: int, det: Dict[str, Any], now: float):
        self.tid = tid
        self.kind = det["kind"]
        self.x, self.y = det["x"], det["y"]
        self.bbox = det["bbox"]
        self.rect = det.get("rect")
        self.count = det.get("count", 1)
        self.first_seen = now
        self.last_seen = now
        self.sightings = 1
        self.history = [(now, self.x, self.y)]
        self.votes: Counter = Counter()     # plate -> summed read confidence
        self.last_ocr = 0.0

    @property
    def plate(self) -> Optional[str]:
        return self.votes.most_common(1)[0][0] if self.votes else None

    @property
    def plate_score(self) -> float:
        return self.votes.most_common(1)[0][1] if self.votes else 0.0

    def confirmed(self, now: float) -> bool:
        return (now - self.first_seen >= CONFIRM_S.get(self.kind, 0.7)
                and self.sightings >= CONFIRM_MIN_SIGHTINGS)

    def stationary(self, now: float) -> bool:
        recent = [(x, y) for t, x, y in self.history if now - t <= 1.0]
        if len(recent) < 2:
            return True
        return math.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1]) < STATIONARY_UNITS


class PlateReader:
    """EasyOCR in a background thread, started on the first crop so the server starts instantly."""

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue(maxsize=6)
        self._reader = None
        self.available: Optional[bool] = None
        self._results: Dict[int, List[Tuple[str, float]]] = {}
        self._lock = threading.Lock()
        self._started = False

    def submit(self, tid: int, crop: np.ndarray) -> bool:
        if self.available is False:
            return False
        if not self._started:
            self._started = True
            threading.Thread(target=self._run, daemon=True).start()
        try:
            self._q.put_nowait((tid, crop))
            return True
        except queue.Full:
            return False

    def take_results(self) -> Dict[int, List[Tuple[str, float]]]:
        with self._lock:
            out, self._results = self._results, {}
        return out

    def _run(self) -> None:
        try:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self.available = True
        except Exception as e:
            print(f"[BoardVision] OCR unavailable ({e}); cars will be counted without plates.", flush=True)
            self.available = False
            return
        while True:
            tid, crop = self._q.get()
            plates = self.read(crop)
            if plates:
                with self._lock:
                    self._results.setdefault(tid, []).extend(plates)

    def _readings(self, img: np.ndarray) -> List[Tuple[str, float]]:
        """Valid plates EasyOCR sees in one upright image, each with its confidence."""
        try:
            found = self._reader.readtext(img, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        except Exception:
            return []
        found = sorted(found, key=lambda r: min(pt[0] for pt in r[0]))   # left to right
        # A card that says AMBULANCE, in one piece or several.
        whole = "".join(text for _, text, _ in found)
        if found and (is_ambulance_text(whole) or any(is_ambulance_text(t) for _, t, _ in found)):
            return [(AMBULANCE, float(max(conf for _, _, conf in found)))]
        out = [(p, float(conf)) for _, text, conf in found for p in [normalise_plate(text)] if p]
        if len(found) > 1:
            # A plate split into pieces ("TN" "07"): read the pieces together.
            joined = normalise_plate("".join(text for _, text, _ in found))
            if joined:
                out.append((joined, float(np.mean([conf for _, _, conf in found]))))
        return out

    def read(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """
        Plates read from an upright label crop (see rectify_label), best first.
        The crop is given a margin, then read as-is and
        turned 180° (the box may face either way); a squarish box is also tried
        sideways. EasyOCR's own rotation search happily prefers an upside-down
        "LONL" over "TN07", so we rotate ourselves and let the state-code check
        and the confidence decide. A binarised copy is the fallback when the
        plain one gives nothing.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        margin = max(6, gray.shape[0] // 6)
        gray = cv2.copyMakeBorder(gray, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = gray.shape[:2]
        rotations = [None, cv2.ROTATE_180]
        if w < 1.25 * h:
            rotations += [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
        best: Dict[str, float] = {}
        for variant in (gray, binary):
            for rot in rotations:
                img = variant if rot is None else cv2.rotate(variant, rot)
                for plate, conf in self._readings(img):
                    best[plate] = max(best.get(plate, 0.0), conf)
                if best and max(best.values()) >= OCR_EARLY_EXIT_CONF:
                    break
            if best:
                break
        return sorted(best.items(), key=lambda kv: -kv[1])


# ── the vision pipeline ──

class BoardVision:
    def __init__(self, layout: bl.Layout, ocr: bool = True):
        self.layout = layout
        self.scale = scale = layout.scale
        self.car_area_units = CAR_AREA_UNITS
        self._areas: deque = deque(maxlen=300)
        self.size = layout.canvas_size()
        self.corners: Optional[List[List[float]]] = None
        self.H: Optional[np.ndarray] = None
        self.reference: Optional[np.ndarray] = None
        self._roi = layout.road_mask()
        k = max(3, int(OPEN_KERNEL_UNITS * scale) | 1)
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._led_ring = max(3, int(LED_RING_UNITS * scale) | 1)
        self._led_pad = self._disk(LED_CORE_PAD_UNITS * scale)
        self._led_halo = self._disk(LED_HALO_UNITS * scale)
        self._led_memory: Optional[np.ndarray] = None
        self._tracks: Dict[int, _Track] = {}
        self._next_tid = 1
        self.plates = PlateReader() if ocr else None
        self.registered: List[str] = []
        self.last_warped: Optional[np.ndarray] = None
        self.last_frame_size: Optional[List[int]] = None

    # calibration
    def set_corners(self, corners: Optional[List[List[float]]], realign: bool = False) -> None:
        """
        realign=True is the camera tracker nudging an existing calibration after the phone moved:
        the board lands back in the same place on the canvas, so the empty-board shot and the
        learned car size stay valid.
        """
        if not corners:
            self.corners, self.H = None, None
            return
        if len(corners) != 4:
            raise ValueError("need exactly 4 corner points (NW, NE, SE, SW)")
        new = [[float(x), float(y)] for x, y in corners]
        if new == self.corners and self.H is not None:
            return              # same calibration: keep the empty-board shot that matches it
        src = np.array(corners, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, self.layout.calibration_canvas_points())
        self.corners = new
        if realign:
            return
        self.reference = None   # an old empty-board shot no longer lines up
        self._areas.clear()     # nor does the car size learned through the old view
        self._led_memory = None

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.H, (self.size, self.size))

    def capture_reference(self, frame: np.ndarray) -> None:
        self.reference = cv2.GaussianBlur(cv2.cvtColor(self.warp(frame), cv2.COLOR_BGR2GRAY), (7, 7), 0)

    # detection
    def _blobs(self, mask: np.ndarray, kind: str) -> List[Dict[str, Any]]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo = (CAR_MIN_SIDE * self.scale) ** 2
        hi = (CAR_MAX_SIDE * self.scale) ** 2
        one_car = self._one_car_area()
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < lo or area > hi * 4:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if max(w, h) > 3.5 * max(1, min(w, h)) and area < 2 * one_car:
                continue          # long thin strips are road edges / tape, not cars
            if kind == "car":
                self._areas.append(area)
            m = cv2.moments(c)
            ux, uy = self.layout.from_canvas(m["m10"] / m["m00"], m["m01"] / m["m00"])
            # Cars parked nose to tail read as one blob; count them by footprint.
            count = int(min(6, max(1, round(area / one_car))))
            rect = cv2.minAreaRect(c)
            out.append({"kind": kind, "x": ux, "y": uy, "bbox": [int(x), int(y), int(w), int(h)], "count": count,
                        "rect": rect, "rectangularity": area / max(1.0, rect[1][0] * rect[1][1])})
        return out

    def _one_car_area(self) -> float:
        """
        Footprint of a single car in canvas pixels. Learned from the blobs seen
        so far (most boxes sit on their own, so a low percentile is one car),
        so any box size works without tuning; falls back to the guess at start.
        """
        if len(self._areas) >= CAR_AREA_LEARN_MIN:
            return float(np.percentile(self._areas, 30))
        return self.car_area_units * self.scale ** 2

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Printed plate text punches holes in a white box; fill every blob solid before the open step."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    @staticmethod
    def _disk(radius_px: float) -> np.ndarray:
        d = max(3, int(radius_px) * 2 + 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    def _lit_leds(self, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Mask of lit signal lamps: clipped pixels whose surroundings are strongly coloured glow."""
        core = (v >= LED_CORE_V).astype(np.float32)
        rest = 1.0 - core
        k = (self._led_ring, self._led_ring)
        ring_sat = cv2.boxFilter(s.astype(np.float32) * rest, -1, k, normalize=False) / np.maximum(
            cv2.boxFilter(rest, -1, k, normalize=False), 1.0)
        now = ((core > 0) & (ring_sat > LED_RING_SAT)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(now)
        too_big = np.nonzero(stats[1:, cv2.CC_STAT_AREA] > (LED_CORE_MAX_UNITS * self.scale) ** 2)[0] + 1
        if too_big.size:
            now[np.isin(labels, too_big)] = 0
        now = now.astype(np.float32)
        if self._led_memory is None or self._led_memory.shape != now.shape:
            self._led_memory = now
        else:
            self._led_memory = np.maximum(self._led_memory * LED_MEMORY_DECAY, now)
        return (self._led_memory > 0.2).astype(np.uint8) * 255

    def detect(self, warped: np.ndarray) -> List[Dict[str, Any]]:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        white = ((s < 70) & (v > 150)).astype(np.uint8) * 255
        red = ((h < 10) | (h > 170)) & (s > 110) & (v > 80)
        pink = ((h >= PINK_HUE_MIN) | (h <= 6)) & (s >= PINK_SAT[0]) & (s <= PINK_SAT[1]) & (v >= PINK_MIN_V)
        red = (red | pink).astype(np.uint8) * 255
        if self.reference is not None:
            gray = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
            ref = self.reference.astype(np.float32)
            # The phone re-exposes whenever the LEDs change; match the reference's overall brightness
            # to this frame first (median over the road, which a few cards can't move much).
            road = self._roi > 0
            gain = float(np.median(gray[road])) / max(1.0, float(np.median(ref[road])))
            ref = ref * min(2.0, max(0.5, gain))
            diff = np.abs(gray - ref)
            changed = ((diff > CHANGE_ABS) & (diff > CHANGE_REL * ref)).astype(np.uint8) * 255
            changed = cv2.dilate(changed, np.ones((5, 5), np.uint8))
            white &= changed
            red &= changed
        leds = self._lit_leds(s, v)
        led_core = cv2.dilate(leds, self._led_pad)
        led_halo = cv2.dilate(leds, self._led_halo)
        white = self._fill_holes(white & ~led_core & self._roi)
        colour = red & self._roi
        red = self._fill_holes(colour)
        cars = [b for b in self._blobs(white, "car") if self._overlap(b, led_halo) <= LED_HALO_MAX_FRAC]
        # Red glow always surrounds a lamp's core; a real ambulance card doesn't glow. And it's a solid
        # sheet of colour, unlike a red or pink toy car, whose paint only rings its white label.
        ambulances = [b for b in self._blobs(red, "ambulance")
                      if self._solidity(b, colour) >= AMBULANCE_SOLIDITY
                      and (self._overlap(b, led_core) == 0.0 or b["rectangularity"] >= CARD_RECTANGULARITY)]
        return cars + ambulances

    @staticmethod
    def _solidity(blob: Dict[str, Any], mask: np.ndarray) -> float:
        """Share of the blob's (possibly tilted) rectangle that is actually the colour, before hole filling."""
        x, y, w, h = blob["bbox"]
        (_, _), (rw, rh), _ = blob["rect"]
        return float(np.count_nonzero(mask[y:y + h, x:x + w])) / max(1.0, rw * rh)

    @staticmethod
    def _overlap(blob: Dict[str, Any], mask: np.ndarray) -> float:
        x, y, w, h = blob["bbox"]
        region = mask[y:y + h, x:x + w]
        return float(np.count_nonzero(region)) / max(1, region.size)

    def _update_tracks(self, dets: List[Dict[str, Any]], now: float) -> None:
        unmatched = list(range(len(dets)))
        for tr in sorted(self._tracks.values(), key=lambda t: -t.last_seen):
            best, best_d = None, TRACK_MATCH_UNITS
            for i in unmatched:
                d = dets[i]
                if d["kind"] != tr.kind:
                    continue
                dist = math.hypot(d["x"] - tr.x, d["y"] - tr.y)
                if dist < best_d:
                    best, best_d = i, dist
            if best is not None:
                d = dets[best]
                unmatched.remove(best)
                tr.x, tr.y, tr.bbox, tr.count, tr.last_seen = d["x"], d["y"], d["bbox"], d.get("count", 1), now
                tr.rect = d.get("rect")
                tr.sightings += 1
                tr.history = [(t, x, y) for t, x, y in tr.history if now - t <= 2.0] + [(now, tr.x, tr.y)]
        for i in unmatched:
            tr = _Track(self._next_tid, dets[i], now)
            self._tracks[tr.tid] = tr
            self._next_tid += 1
        for tid in [t for t, tr in self._tracks.items() if now - tr.last_seen > TRACK_TTL_S]:
            del self._tracks[tid]

    def _ocr_pass(self, warped: np.ndarray, now: float) -> None:
        if not self.plates:
            return
        for tid, reads in self.plates.take_results().items():
            tr = self._tracks.get(tid)
            if not tr:
                continue
            for plate, conf in reads:
                if plate == AMBULANCE:
                    tr.votes[plate] += conf
                    continue
                if tr.count > 1:
                    continue          # a merged nose-to-tail blob holds several plates; only an AMBULANCE read counts
                plate, weight = snap_to_registered(plate, self.registered)
                tr.votes[plate] += conf * weight
        for tr in self._tracks.values():
            if tr.kind != "car" or tr.last_seen != now or tr.rect is None or tr.count > 3:
                continue
            # The wide AMBULANCE card can look like two cars by size, so blobs of up to three still get read.
            if tr.plate_score >= OCR_CONFIDENT_SCORE and (not self.registered or tr.plate in self.registered):
                continue
            if now - tr.last_ocr < OCR_RETRY_S:
                continue
            crop = rectify_label(warped, tr.rect)
            if crop.size and self.plates.submit(tr.tid, crop):
                tr.last_ocr = now

    @staticmethod
    def _is_ambulance_card(tr: "_Track") -> bool:
        """A white card whose text reads AMBULANCE (its best reading, read at least once fairly clearly)."""
        return tr.kind == "car" and tr.plate == AMBULANCE and tr.votes[AMBULANCE] >= AMBULANCE_TEXT_SCORE

    def process(self, frame: np.ndarray, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """One frame in, the current list of tracked vehicles out (with road / junction / phase / plate)."""
        now = now if now is not None else time.time()
        self.last_frame_size = [int(frame.shape[1]), int(frame.shape[0])]
        if not self.calibrated:
            self.last_warped = None
            return []
        warped = self.warp(frame)
        self.last_warped = warped
        self._update_tracks(self.detect(warped), now)
        self._ocr_pass(warped, now)
        out = []
        for tr in self._tracks.values():
            if tr.last_seen != now or not tr.confirmed(now):
                continue
            where = self.layout.locate(tr.x, tr.y) or {}
            out.append({
                "id": tr.tid,
                "kind": "ambulance" if self._is_ambulance_card(tr) else tr.kind,
                "x": round(tr.x, 1),
                "y": round(tr.y, 1),
                "bbox": tr.bbox,
                "count": tr.count,
                "plate": tr.plate,
                "plate_state": STATE_NAMES.get((tr.plate or "")[:2]),
                "registered": bool(tr.plate and tr.plate in self.registered),
                "waiting": tr.stationary(now),
                "road": where.get("road"),
                "junction": where.get("junction") or where.get("in_junction"),
                "phase": where.get("phase"),
                "in_junction": "in_junction" in where,
                "distance_m": where.get("distance_m"),
            })
        return out

    # drawing
    def annotate(self, vehicles: List[Dict[str, Any]], signals: Dict[str, Dict[str, str]]) -> Optional[np.ndarray]:
        """Top-down board view with detections and live signal state drawn on, like the dashboard feed."""
        if self.last_warped is None:
            return None
        img = self.last_warped.copy()
        overlay = img.copy()
        colour = {"R": (60, 60, 255), "Y": (0, 210, 255), "G": (80, 230, 80)}
        s = self.scale
        bar = int(6 * s)
        for jid in self.layout.active:
            jx, jy = bl.NODES[jid]
            st = signals.get(jid, {})
            cx, cy = self.layout.to_canvas(jx, jy)
            half = bl.JUNCTION_HALF * s
            ew, ns = colour.get(st.get("EW", "R")), colour.get(st.get("NS", "R"))
            # EW bars on the west/east edges of the box, NS bars on north/south.
            cv2.rectangle(overlay, (int(cx - half - bar), int(cy - half)), (int(cx - half), int(cy + half)), ew, -1)
            cv2.rectangle(overlay, (int(cx + half), int(cy - half)), (int(cx + half + bar), int(cy + half)), ew, -1)
            cv2.rectangle(overlay, (int(cx - half), int(cy - half - bar)), (int(cx + half), int(cy - half)), ns, -1)
            cv2.rectangle(overlay, (int(cx - half), int(cy + half)), (int(cx + half), int(cy + half + bar)), ns, -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        for jid in self.layout.active:
            cx, cy = self.layout.to_canvas(*bl.NODES[jid])
            half = bl.JUNCTION_HALF * s
            cv2.putText(img, jid, (int(cx + half + bar + 4), int(cy - half - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        for v in vehicles:
            x, y, w, h = v["bbox"]
            c = (60, 60, 255) if v["kind"] == "ambulance" else ((80, 230, 80) if v["road"] else (180, 180, 180))
            cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
            label = "AMBULANCE" if v["kind"] == "ambulance" else (v["plate"] or f"car {v['id']}")
            if v.get("count", 1) > 1:
                label += f" x{v['count']}"
            cv2.putText(img, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        return img
