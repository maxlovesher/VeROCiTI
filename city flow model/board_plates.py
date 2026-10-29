"""
board_plates.py — reading number plates off the model cars.
===========================================================
  normalise_plate    an OCR string -> a valid Indian plate (TN07, KA01AB1234), fixing letter/digit
                     look-alikes by position, or None if it isn't one
  is_ambulance_text  "AMBULANCE" allowing the usual misreads (AMBULANCF, 4MBULANCE)
  rectify_label      a car's top face cut out of the camera picture as an upright rectangle
  PlateReader        EasyOCR in a background thread, started on the first crop so the server starts instantly
"""

import queue
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

RETRY_SLEEP = time.sleep        # a name of its own so tests don't have to wait
LABEL_H = 96                    # height in pixels of the upright label crop handed to the OCR
OCR_EARLY_EXIT_CONF = 0.8       # one reading this confident is enough; skip the remaining orientations

PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}([A-Z]{1,3}\d{1,4})?$")
# Indian state / UT codes: an OCR read that doesn't start with one is noise, not a plate.
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
# The usual look-alike swaps, applied only where the plate format says a letter (or a digit) must be.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "J", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "J": "3", "A": "4", "S": "5", "G": "6",
             "B": "8", "T": "7"}

AMBULANCE = "AMBULANCE"
_AMBULANCE_PIECES = ("AMBUL", "MBULA", "BULAN", "ULANC")     # not "LANCE": BALANCE has it too
_AMB_LETTER = {"4": "A", "8": "B", "0": "O", "1": "L", "5": "S", "6": "G", "3": "B", "7": "T", "2": "Z", "I": "L"}


def is_ambulance_text(text: str) -> bool:
    s = "".join(_AMB_LETTER.get(c, c) for c in re.sub(r"[^A-Z0-9]", "", (text or "").upper()))
    if len(s) < 5:
        return False
    if any(piece in s for piece in _AMBULANCE_PIECES):
        return True
    return len(s) >= 7 and SequenceMatcher(None, s, AMBULANCE).ratio() >= 0.8


def normalise_plate(text: str) -> Optional[str]:
    """Clean an OCR string into a plate like TN07 / KA01AB1234, or None if it can't be one."""
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


def rectify_label(img: np.ndarray, quad) -> np.ndarray:
    """
    A car's top face (four corners in the picture, in order round the rectangle) cut out as an upright
    rectangle with the long side horizontal, LABEL_H pixels tall. Rotated, never mirrored. Which way up
    the text ends up is left to the reader, which tries both.
    """
    pts = np.float32(quad).reshape(4, 2)
    d01, d12 = float(np.linalg.norm(pts[1] - pts[0])), float(np.linalg.norm(pts[2] - pts[1]))
    if d12 > d01:
        pts, d01, d12 = np.roll(pts, -1, axis=0), d12, d01        # start the corner list on a long side
    w, h = max(8, int(round(d01 * LABEL_H / max(d12, 1.0)))), LABEL_H
    area = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
    dst = [(0, 0), (w, 0), (w, h), (0, h)] if area >= 0 else [(0, h), (w, h), (w, 0), (0, 0)]
    M = cv2.getPerspectiveTransform(pts, np.float32(dst))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class PlateReader:
    """EasyOCR in a background thread. submit() a crop, take_results() later."""

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

    def _load(self):
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)

    def _run(self) -> None:
        # Another part of the server imports PyTorch at startup too, and two threads importing torchvision at
        # once trip over each other's half-finished import. That passes in seconds, so wait and try again.
        for attempt in range(5):
            try:
                self._reader = self._load()
                self.available = True
                break
            except ImportError as e:
                if attempt == 4:
                    print(f"[Plates] OCR unavailable ({e}); cars will be counted without plates.", flush=True)
                    self.available = False
                    return
                RETRY_SLEEP(3.0)
            except Exception as e:
                print(f"[Plates] OCR unavailable ({e}); cars will be counted without plates.", flush=True)
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
        found = sorted(found, key=lambda r: min(pt[0] for pt in r[0]))     # left to right
        whole = "".join(text for _, text, _ in found)
        if found and (is_ambulance_text(whole) or any(is_ambulance_text(t) for _, t, _ in found)):
            return [(AMBULANCE, float(max(conf for _, _, conf in found)))]
        out = [(p, float(conf)) for _, text, conf in found for p in [normalise_plate(text)] if p]
        if len(found) > 1:                                                  # a plate split in pieces ("TN" "07")
            joined = normalise_plate("".join(text for _, text, _ in found))
            if joined:
                out.append((joined, float(np.mean([conf for _, _, conf in found]))))
        return out

    def read(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """
        Plates read from an upright label crop, best first. The crop gets a margin, then is read as-is and
        turned 180° (the card may face either way); a squarish one is also tried sideways. EasyOCR's own
        rotation search prefers an upside-down "LONL" over "TN07", so we rotate ourselves and let the
        state-code check and the confidence decide. A binarised copy is the fallback.
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
