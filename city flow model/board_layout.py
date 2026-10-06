"""
board_layout.py — Geometry of the physical demo board.
=======================================================
Two board layouts, both carved out of the same 5-junction network the
controller already runs (roadnet_5j.json):

  single : one signalled junction (J1) with its four arms — the quick build.

                  N
                  |
            W -- J1 -- E          click the four road tips N, E, S, W
                  |
                  S

  double : J1 and J3 side by side, joined by one road, each with its own
           three outer arms — click J1's north tip, J3's east tip, J3's
           south tip and J1's west tip.

  full   : the plus of five junctions inside a 3x3 grid of roads, J2 north,
           J5 south, J1 west, J4 east, J3 in the middle; the four grid
           corners are plain unsignalled bends (click those to calibrate).

Everything works in roadnet units (x east, y north, junctions 200 units
apart). The camera image is mapped onto those units by a 4-point
calibration, and the top-down "canvas" the vision code works on is just a
square window of roadnet units scaled to pixels. The controller always runs
all five junctions; a layout only decides which of them are physically on
the board (and therefore watched by the camera and lit by the LEDs).
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROADNET_PATH = os.path.join(BASE_DIR, "roadnet_5j.json")

SIGNALLED = ("J1", "J2", "J3", "J4", "J5")

# Road half-width and junction-box half-size, in roadnet units (junctions are 200 apart,
# so roads are about a third as wide as a junction-to-junction link is long).
ROAD_HALF_WIDTH = 34.0
JUNCTION_HALF = 36.0
# Cars are allowed to overhang the road edge a little before they stop counting.
ROAD_MARGIN = 12.0

ROAD_LENGTH_M = 200.0


def _load_roadnet() -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, str]]]:
    with open(ROADNET_PATH) as f:
        data = json.load(f)
    nodes = {i["id"]: (float(i["point"]["x"]), float(i["point"]["y"])) for i in data["intersections"]}
    roads = [(r["id"], r["startIntersection"], r["endIntersection"]) for r in data["roads"]]
    return nodes, roads


NODES, ROADS = _load_roadnet()
ROAD_ENDS = {rid: (a, b) for rid, a, b in ROADS}


def phase_for_road(road_id: str) -> Optional[Tuple[str, str]]:
    """(junction, phase) for a road that feeds a signalled junction: horizontal roads feed EW, vertical NS."""
    ends = ROAD_ENDS.get(road_id)
    if not ends or ends[1] not in SIGNALLED:
        return None
    pa, pb = NODES[ends[0]], NODES[ends[1]]
    return ends[1], ("EW" if abs(pa[1] - pb[1]) < 1e-6 else "NS")


class Layout:
    def __init__(self, name: str, active: Tuple[str, ...], x_min: float, y_max: float, span: float,
                 scale: float, calibration: List[Tuple[str, Tuple[float, float]]], hint: str,
                 legacy_calibration: Optional[List[Tuple[float, float]]] = None):
        self.name = name
        self.active = active
        self.x_min, self.y_max, self.span = x_min, y_max, span
        self.scale = scale                      # canvas pixels per roadnet unit (canvas is ~1200 px square)
        self.calibration_labels = [label for label, _ in calibration]
        self.calibration_points = [pt for _, pt in calibration]
        self.hint = hint
        # What older saved calibrations clicked, so they can be converted instead of thrown away.
        self.legacy_calibration_points = legacy_calibration or self.calibration_points
        self.segments = self._build_segments()

    def _build_segments(self) -> List[Dict]:
        """One physical segment per pair of connected nodes that touches a junction on the board."""
        segments: Dict[frozenset, Dict] = {}
        for rid, a, b in ROADS:
            if a not in self.active and b not in self.active:
                continue
            seg = segments.setdefault(frozenset((a, b)), {"a": a, "b": b, "into": {}})
            # road_X_Y runs X -> Y, so it is the incoming road for Y.
            if b in self.active:
                seg["into"][b] = rid
        out = []
        for seg in segments.values():
            pa, pb = NODES[seg["a"]], NODES[seg["b"]]
            seg["pa"], seg["pb"] = pa, pb
            seg["axis"] = "EW" if abs(pa[1] - pb[1]) < 1e-6 else "NS"
            out.append(seg)
        return out

    def previous_junction(self, road_id: str) -> Optional[str]:
        """The board junction this road comes out of (behind anything travelling along it), if there is one."""
        ends = ROAD_ENDS.get(road_id)
        return ends[0] if ends and ends[0] in self.active else None

    def next_junction_straight(self, road_id: str) -> Optional[str]:
        """The board junction straight ahead after crossing this road's junction, if there is one."""
        ends = ROAD_ENDS.get(road_id)
        if not ends:
            return None
        ax, ay = NODES[ends[0]]
        bx, by = NODES[ends[1]]
        target = (2 * bx - ax, 2 * by - ay)
        for jid in self.active:
            if NODES[jid] == target:
                return jid
        return None

    def locate(self, x: float, y: float) -> Optional[Dict]:
        """
        Where a car at roadnet point (x, y) is queued: the road it's on, the
        junction and phase it's waiting for, and how far along the road it is.
        Returns {"in_junction": J} for a car inside a junction box (it's
        crossing, not queued), or None when the point isn't on any road.
        """
        for jid in self.active:
            jx, jy = NODES[jid]
            if abs(x - jx) <= JUNCTION_HALF and abs(y - jy) <= JUNCTION_HALF:
                return {"in_junction": jid}

        best = None
        for seg in self.segments:
            (ax, ay), (bx, by) = seg["pa"], seg["pb"]
            dx, dy = bx - ax, by - ay
            t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
            if t < -0.05 or t > 1.05:
                continue
            off = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
            if off > ROAD_HALF_WIDTH + ROAD_MARGIN:
                continue
            if best is None or off < best[0]:
                best = (off, seg, min(1.0, max(0.0, t)))
        if best is None:
            return None

        _, seg, t = best
        into, a, b = seg["into"], seg["a"], seg["b"]
        if a in into and b in into:
            # Between two board junctions: a car queues for whichever one it's nearer.
            target = b if t >= 0.5 else a
        elif b in into:
            target = b
        elif a in into:
            target = a
        else:
            return None
        along = t if target == b else 1.0 - t
        return {
            "road": into[target],
            "junction": target,
            "phase": seg["axis"],
            "distance_m": round(along * ROAD_LENGTH_M, 1),
        }

    # ── canvas mapping ──
    def to_canvas(self, x: float, y: float, scale: Optional[float] = None) -> Tuple[float, float]:
        s = self.scale if scale is None else scale
        return (x - self.x_min) * s, (self.y_max - y) * s

    def from_canvas(self, cx: float, cy: float, scale: Optional[float] = None) -> Tuple[float, float]:
        s = self.scale if scale is None else scale
        return cx / s + self.x_min, self.y_max - cy / s

    def canvas_size(self, scale: Optional[float] = None) -> int:
        return int(round(self.span * (self.scale if scale is None else scale)))

    def calibration_canvas_points(self, scale: Optional[float] = None) -> np.ndarray:
        return np.array([self.to_canvas(x, y, scale) for x, y in self.calibration_points], dtype=np.float32)

    def road_mask(self, scale: Optional[float] = None) -> np.ndarray:
        """Canvas mask of every road surface (junction boxes included), widened by the overhang margin."""
        import cv2
        s = self.scale if scale is None else scale
        size = self.canvas_size(s)
        mask = np.zeros((size, size), np.uint8)
        half = int((ROAD_HALF_WIDTH + ROAD_MARGIN) * s)
        for seg in self.segments:
            p1 = tuple(int(v) for v in self.to_canvas(*seg["pa"], s))
            p2 = tuple(int(v) for v in self.to_canvas(*seg["pb"], s))
            cv2.line(mask, p1, p2, 255, thickness=2 * half)
        return mask


def _box(cx: float, cy: float, corner: str) -> Tuple[float, float]:
    """A junction box corner (the inside corner where the road-edge strips cross), in roadnet units."""
    h = ROAD_HALF_WIDTH
    return (cx + (h if "right" in corner else -h), cy + (h if "top" in corner else -h))


# Single and double boards calibrate on junction-box corners rather than road tips: the box is
# the same on any build, while how far each road runs past the junction (and how much of it
# the camera sees) varies board to board, and tip clicks then squash the whole picture.
_BOX_HINT = ("Click the inside corner of each junction box, where the grey road-edge strips cross "
             "(north is the top of the board)")

LAYOUTS: Dict[str, Layout] = {
    "single": Layout(
        "single", ("J1",), x_min=-100.0, y_max=500.0, span=400.0, scale=3.0,
        calibration=[
            ("J1 box top-left corner", _box(100, 300, "top-left")),
            ("J1 box top-right corner", _box(100, 300, "top-right")),
            ("J1 box bottom-right corner", _box(100, 300, "bottom-right")),
            ("J1 box bottom-left corner", _box(100, 300, "bottom-left")),
        ],
        hint=_BOX_HINT + ": top-left, top-right, bottom-right, bottom-left.",
        legacy_calibration=[(100.0, 500.0), (300.0, 300.0), (100.0, 100.0), (-100.0, 300.0)],
    ),
    "double": Layout(
        "double", ("J1", "J3"), x_min=-100.0, y_max=600.0, span=600.0, scale=2.0,
        calibration=[
            ("J1 box top-left corner", _box(100, 300, "top-left")),
            ("J3 box top-right corner", _box(300, 300, "top-right")),
            ("J3 box bottom-right corner", _box(300, 300, "bottom-right")),
            ("J1 box bottom-left corner", _box(100, 300, "bottom-left")),
        ],
        hint=_BOX_HINT + ": J1's top-left, J3's top-right, J3's bottom-right, J1's bottom-left.",
        legacy_calibration=[(100.0, 500.0), (500.0, 300.0), (300.0, 100.0), (-100.0, 300.0)],
    ),
    "full": Layout(
        "full", SIGNALLED, x_min=-100.0, y_max=700.0, span=800.0, scale=1.5,
        calibration=[
            ("NW corner", (100.0, 500.0)),
            ("NE corner", (500.0, 500.0)),
            ("SE corner", (500.0, 100.0)),
            ("SW corner", (100.0, 100.0)),
        ],
        hint="Click the centre of each corner bend of the road grid in order: NW (top-left), NE, SE, SW. North is the J2 side.",
    ),
}
DEFAULT_LAYOUT = "single"


def get_layout(name: Optional[str]) -> Layout:
    return LAYOUTS.get(name or DEFAULT_LAYOUT, LAYOUTS[DEFAULT_LAYOUT])
