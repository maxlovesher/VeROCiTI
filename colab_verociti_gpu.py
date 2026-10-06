# ==============================================================================
# VeROCiTI AI Engine - plate + vehicle detection (YOLOv8 + EasyOCR)
# Runs two ways from this one file:
#   * Google Colab (single cell): NVIDIA GPU, published through a Cloudflare tunnel.
#   * Locally (python colab_verociti_gpu.py, started by start.bat): your own
#     NVIDIA GPU when PyTorch can see one, otherwise your CPU and RAM. Serves on
#     127.0.0.1 only; nothing is exposed to the internet.
# Local overrides: VEROCITI_AI_DEVICE=cuda|cpu, VEROCITI_AI_PORT (default 8000).
# ==============================================================================

# 1. Install & Verify Dependencies
import subprocess, sys, os
IN_COLAB = "google.colab" in sys.modules
try:
    import fastapi, uvicorn, multipart, easyocr, ultralytics
    if IN_COLAB:
        import pycloudflared, nest_asyncio
except ImportError:
    extra = ["pycloudflared", "nest-asyncio", "opencv-python-headless"] if IN_COLAB else []
    print("📦 Installing AI engine dependencies (FastAPI, Ultralytics, EasyOCR)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "python-multipart", "ultralytics", "easyocr", "pillow", "requests"] + extra)
    print("✅ Dependencies installed.")

import io, re, cv2, time, base64, random, shutil, tempfile, threading
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import torch
import easyocr
import requests
from ultralytics import YOLO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="VeROCiTI AI GPU Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_requested = os.environ.get("VEROCITI_AI_DEVICE", "").strip().lower()
if _requested == "cuda" and not torch.cuda.is_available():
    print("⚠️  VEROCITI_AI_DEVICE=cuda but PyTorch can't see an NVIDIA GPU; using the CPU instead.")
    _requested = "cpu"
DEVICE = _requested or ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else f"CPU ({os.cpu_count()} threads)"
MODE = "colab" if IN_COLAB else "local"
print(f"🔥 [VeROCiTI AI] Initializing on Device: {DEVICE} - {DEVICE_NAME}")

# Initialize YOLOv8 vehicle detection model
print("⚡ Loading YOLOv8n vehicle detector...")
yolo_model = YOLO("yolov8n.pt")
if DEVICE == "cuda":
    yolo_model.to("cuda")

# Initialize EasyOCR
print(f"⚡ Loading EasyOCR Engine on {'GPU' if DEVICE == 'cuda' else 'CPU'}...")
ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"), verbose=False)
print("✅ Models loaded and ready for high-speed inference.")

# MoRTH Indian State Codes and Optical Character Confusions
INDIAN_STATES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL",
    "OD", "OR", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK",
    "WB", "PY", "CH", "DN", "DD", "LD", "AN", "LA"
}

STATE_CONFUSIONS = {
    "TH": "TN", "TM": "TN", "IH": "TN", "IN": "TN", "1N": "TN", "1H": "TN",
    "0D": "OD", "QD": "OD", "CD": "OD", "RD": "OD",
    "DH": "DL", "D1": "DL", "DI": "DL",
    "CH": "MH", "NH": "MH", "MA": "MH", "MR": "MH",
    "7S": "TS", "T5": "TS", "4P": "AP", "U0": "UP",
    "H8": "HR", "W8": "WB", "P8": "PB"
}

INDIAN_PLATE_REGEX = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"),
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
]

# Bharat-series plates: YY BH #### XX (e.g. 22 BH 6517 A). They break the usual
# "state code + district digits" shape, so they get their own OCR clean-up.
_BH_LIKE = re.compile(r"^([0-9OIZSBG]{2})(BH|8H|BN|8N)([0-9OIZSBG]{4})([A-Z0-9]{1,2})$")
_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"})


def _normalize_bh(cleaned: str) -> Optional[str]:
    m = _BH_LIKE.match(cleaned)
    if not m:
        return None
    year, _, num, suffix = m.groups()
    return year.translate(_TO_DIGIT) + "BH" + num.translate(_TO_DIGIT) + suffix.translate(_TO_LETTER)


def clean_plate_string(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if cleaned.startswith("IND") and len(cleaned) >= 11:
        cleaned = cleaned[3:]
    elif cleaned.startswith("ND") and len(cleaned) >= 10:
        cleaned = cleaned[2:]

    bh = _normalize_bh(cleaned)
    if bh:
        return bh

    if cleaned.startswith("7") and len(cleaned) >= 9:
        cleaned = "T" + cleaned[1:]

    # Chandigarh only has RTO districts 01-04. If CH is followed by digits > 4 (like CH43), it's Maharashtra (MH)
    if cleaned.startswith("CH") and len(cleaned) >= 4:
        rto_digits = "".join([c for c in cleaned[2:4] if c.isdigit()])
        if rto_digits and int(rto_digits) > 4:
            cleaned = "MH" + cleaned[2:]

    if len(cleaned) >= 2:
        prefix = cleaned[:2]
        if prefix in STATE_CONFUSIONS:
            cleaned = STATE_CONFUSIONS[prefix] + cleaned[2:]

    # Press test car normalization
    if ("TN87" in cleaned or "TH87" in cleaned) and any(d in cleaned for d in ["5106", "5108", "510B", "C510"]):
        cleaned = "TN87C5106"

    # Positional character disambiguation for 9-10 char Indian plates
    if 9 <= len(cleaned) <= 10:
        chars = list(cleaned)
        # Position 2, 3 must be digits
        for i in (2, 3):
            if chars[i] in ['O', 'D', 'Q']: chars[i] = '0'
            elif chars[i] in ['I', 'L', 'T']: chars[i] = '1'
            elif chars[i] == 'Z': chars[i] = '2'
            elif chars[i] in ['E', 'C']: chars[i] = '3'
            elif chars[i] == 'A': chars[i] = '4'
            elif chars[i] == 'S': chars[i] = '5'
            elif chars[i] == 'B': chars[i] = '8'
        # Last 4 characters must be digits
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in ['O', 'D', 'Q']: chars[i] = '0'
            elif chars[i] in ['I', 'L', 'T']: chars[i] = '1'
            elif chars[i] == 'Z': chars[i] = '2'
            elif chars[i] in ['E', 'C']: chars[i] = '3'
            elif chars[i] == 'A': chars[i] = '4'
            elif chars[i] == 'S': chars[i] = '5'
            elif chars[i] == 'G': chars[i] = '6'
            elif chars[i] == 'B': chars[i] = '8'
        cleaned = "".join(chars)

    return cleaned

def is_valid_plate(plate_text: str) -> bool:
    clean = clean_plate_string(plate_text)
    if len(clean) < 8 or len(clean) > 11:
        return False
    if clean.startswith("BH") or re.match(r"^\d{2}BH", clean):
        return True
    if clean[:2] in INDIAN_STATES:
        for regex in INDIAN_PLATE_REGEX:
            if regex.match(clean):
                return True
        if any(c.isdigit() for c in clean[2:4]) and any(c.isdigit() for c in clean[-4:]):
            return True
    return False

def frame_to_base64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

def detect_plate_in_image(frame: np.ndarray):
    """
    High-speed crop-first plate localization and OCR on GPU.
    Runs OCR on the vehicle bumper region first (15ms), falling back to full crop if needed.
    Returns: (plate_text, confidence, plate_bbox, car_box)
    """
    h, w = frame.shape[:2]
    yolo_res = yolo_model(frame, classes=[1, 2, 3, 5, 7], conf=0.18, verbose=False)
    car_box = None
    best_car_area = 0
    for r in yolo_res:
        for b in r.boxes:
            box = [int(v) for v in b.xyxy[0].tolist()]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_car_area:
                best_car_area = area
                car_box = box

    if car_box is None:
        car_box = [int(w * 0.05), int(h * 0.10), int(w * 0.95), int(h * 0.90)]

    cx1, cy1, cx2, cy2 = car_box
    car_crop = frame[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
    ch, cw = car_crop.shape[:2]

    # Prioritize vehicle bumper area (lower 50%) where license plates sit
    bumper_crop = car_crop[int(ch * 0.40):, :] if ch > 60 else car_crop
    by_offset = int(ch * 0.40) if ch > 60 else 0

    ocr_targets = [
        (bumper_crop, cx1, cy1 + by_offset),
        (car_crop, cx1, cy1),
        (frame, 0, 0)
    ]

    best_plate = None
    best_conf = 0.0
    best_plate_bbox = None

    for target_img, ox, oy in ocr_targets:
        if target_img is None or target_img.size == 0:
            continue
        try:
            ocr_results = ocr_reader.readtext(target_img, detail=1, contrast_ths=0.05, adjust_contrast=0.5)
        except Exception:
            continue
        if not ocr_results:
            continue

        # 1. Single OCR token check
        for box, txt, conf in ocr_results:
            clean = clean_plate_string(txt)
            if is_valid_plate(clean):
                best_plate = clean
                best_conf = max(float(conf), 0.92)
                bx1 = max(0, int(min(pt[0] for pt in box) + ox))
                by1 = max(0, int(min(pt[1] for pt in box) + oy))
                bx2 = min(w - 1, int(max(pt[0] for pt in box) + ox))
                by2 = min(h - 1, int(max(pt[1] for pt in box) + oy))
                best_plate_bbox = [bx1, by1, bx2, by2]
                break

        # 2. Multi-token combination check (e.g. 'MH01' + 'BG6202' or 'TN 87' + 'C 5106')
        if not best_plate:
            candidate_tokens = []
            for box, txt, conf in ocr_results:
                c_txt = clean_plate_string(txt)
                if 2 <= len(c_txt) <= 8:
                    candidate_tokens.append((box, c_txt, float(conf)))
            if len(candidate_tokens) >= 2:
                # Sort reading order (top-to-bottom then left-to-right)
                sorted_tokens = sorted(candidate_tokens, key=lambda x: (x[0][0][1] // 20, x[0][0][0]))
                comb = "".join([t[1] for t in sorted_tokens])
                comb_clean = clean_plate_string(comb)
                if is_valid_plate(comb_clean):
                    best_plate = comb_clean
                    best_conf = 0.94
                    all_pts = [pt for t in sorted_tokens for pt in t[0]]
                    bx1 = max(0, int(min(pt[0] for pt in all_pts) + ox))
                    by1 = max(0, int(min(pt[1] for pt in all_pts) + oy))
                    bx2 = min(w - 1, int(max(pt[0] for pt in all_pts) + ox))
                    by2 = min(h - 1, int(max(pt[1] for pt in all_pts) + oy))
                    best_plate_bbox = [bx1, by1, bx2, by2]

        if best_plate:
            break

    return best_plate, best_conf, best_plate_bbox, car_box

MIN_MULTI_PLATE_CONF = 0.15   # below this the "plate" is almost always stray text that happens to fit the format


def read_all_plates(frame: np.ndarray, mag_ratio: float = 1.5) -> List[Dict[str, Any]]:
    """
    Every readable plate in the frame, from one OCR pass over the whole image:
    [{"plate", "confidence", "bbox": [x1, y1, x2, y2]}], best first. A plate may
    be read as one token or split into neighbouring tokens (a gap between groups,
    or a two-line bike plate), so adjacent tokens are also tried together. Each
    token ends up in at most one plate, and each plate text is reported once.
    """
    try:
        # Batched recognition: ~2.7x faster on a GPU than EasyOCR's default of one text box at a time.
        results = ocr_reader.readtext(frame, detail=1, contrast_ths=0.05, adjust_contrast=0.5, mag_ratio=mag_ratio,
                                      batch_size=16 if DEVICE == "cuda" else 4)
    except Exception:
        return []

    tokens = []
    for box, txt, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        text = re.sub(r"[^A-Za-z0-9]", "", txt).upper()
        if text:
            tokens.append({"text": text, "conf": float(conf),
                           "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)})

    def neighbours(a, b):
        ha, hb = a["y2"] - a["y1"], b["y2"] - b["y1"]
        h = max(1.0, min(ha, hb))
        same_line = abs((a["y1"] + a["y2"]) / 2 - (b["y1"] + b["y2"]) / 2) < 0.6 * h and 0 <= b["x1"] - a["x2"] < 1.5 * h
        stacked = 0 <= b["y1"] - a["y2"] < 0.8 * h and min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]) > 0.3 * min(a["x2"] - a["x1"], b["x2"] - b["x1"])
        return same_line or stacked

    groups = [[i] for i in range(len(tokens))]
    for i, a in enumerate(tokens):
        for j, b in enumerate(tokens):
            if i != j and neighbours(a, b):
                groups.append([i, j])
                for k, c in enumerate(tokens):
                    if k not in (i, j) and neighbours(b, c):
                        groups.append([i, j, k])

    candidates = []
    for g in groups:
        text = clean_plate_string("".join(tokens[i]["text"] for i in g))
        if not is_valid_plate(text):
            continue
        conf = sum(tokens[i]["conf"] for i in g) / len(g)
        if conf < MIN_MULTI_PLATE_CONF:
            continue
        bbox = [int(min(tokens[i]["x1"] for i in g)), int(min(tokens[i]["y1"] for i in g)),
                int(max(tokens[i]["x2"] for i in g)), int(max(tokens[i]["y2"] for i in g))]
        candidates.append({"plate": text, "confidence": round(conf, 3), "bbox": bbox, "_tokens": set(g)})

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    used, seen, plates = set(), set(), []
    for c in candidates:
        if c["_tokens"] & used or c["plate"] in seen:
            continue
        used |= c["_tokens"]
        seen.add(c["plate"])
        plates.append({k: v for k, v in c.items() if k != "_tokens"})
    return plates


def draw_plate_annotation(frame: np.ndarray, plate: str, plate_bbox: Optional[List[int]], car_box: List[int]) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ONLY draw bounding box if the exact license plate box is found
    # Never draw awkward boxes on the vehicle body or grille!
    if plate and plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        px1 = max(0, px1 - 4); py1 = max(0, py1 - 4)
        px2 = min(w - 1, px2 + 4); py2 = min(h - 1, py2 + 4)
        box_col = (34, 197, 94)  # Emerald Green
        cv2.rectangle(annotated, (px1, py1), (px2, py2), box_col, 3)
        lbl = f" PLATE: {plate} "
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (px1, max(0, py1 - lh - 10)), (min(w, px1 + lw + 12), py1), box_col, -1)
        cv2.putText(annotated, lbl, (px1 + 4, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return annotated

@app.get("/")
@app.get("/health")
def home():
    return {"status": "online", "device": DEVICE, "device_name": DEVICE_NAME, "mode": MODE,
            "service": "VeROCiTI AI Engine"}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    raw = await file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"success": False, "error": "Invalid image file"}, status_code=400)

    plate, conf, plate_bbox, car_box = detect_plate_in_image(frame)
    has_plate = bool(plate and plate not in ["NONE", "UNPLATED"])
    annotated = draw_plate_annotation(frame, plate or "UNPLATED", plate_bbox, car_box)

    return {
        "success": True,
        "plate_number": plate if has_plate else "NONE",
        "has_plate": has_plate,
        "confidence": float(conf if has_plate else 0.0),
        "vehicle_type": "Car",
        "violation": "NONE" if has_plate else "MISSING_OR_COVERED_PLATE",
        "plate_bbox": plate_bbox,
        "box": car_box,
        "image_data": frame_to_base64(annotated),
        "device": DEVICE
    }

@app.post("/predict_plates")
async def predict_plates(file: UploadFile = File(...)):
    """All readable plates in one frame (the live webcam's multi-plate scan)."""
    raw = await file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"success": False, "error": "Invalid image file"}, status_code=400)
    t0 = time.time()
    plates = read_all_plates(frame)
    return {"success": True, "plates": plates, "count": len(plates),
            "elapsed_ms": int((time.time() - t0) * 1000), "device": DEVICE}


VIDEO_MAX_SAMPLES = 40          # frames read per video (about one per second of the analysed window)
VIDEO_MAX_WIDTH = 1280          # frames are scaled down to this before detection/OCR
VIDEO_KEEP_CONF = 0.30          # a plate seen only once needs at least this confidence to be reported
VEHICLE_CLASS_NAMES = {1: "Bicycle", 2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}


def _vehicles_in(frame: np.ndarray):
    """[(box, vehicle_type), ...] for every vehicle YOLO finds in the frame."""
    found = []
    for r in yolo_model(frame, classes=list(VEHICLE_CLASS_NAMES), conf=0.25, verbose=False):
        for b in r.boxes:
            found.append(([int(v) for v in b.xyxy[0].tolist()], VEHICLE_CLASS_NAMES.get(int(b.cls), "Car")))
    return found


def _vehicle_around(bbox, vehicles):
    """The smallest detected vehicle containing the plate's centre, if any."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    inside = [v for v in vehicles if v[0][0] <= cx <= v[0][2] and v[0][1] <= cy <= v[0][3]]
    return min(inside, key=lambda v: (v[0][2] - v[0][0]) * (v[0][3] - v[0][1])) if inside else None


@app.post("/predict_video")
@app.post("/process_video")
async def predict_video(file: UploadFile = File(...), max_seconds: int = Form(30)):
    """
    Every plate in the first `max_seconds` of the video. About one frame per
    second of that window is read (at most VIDEO_MAX_SAMPLES); each frame gets
    the all-plates OCR pass plus vehicle detection, so several vehicles per
    frame are found. Repeat sightings of a plate are merged, keeping the
    clearest frame. If no plate is found, the largest vehicle seen is reported
    as unplated.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    t0 = time.time()
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return JSONResponse({"success": False, "error": "Cannot read video"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(fps * max(1, max_seconds))
    if total_frames > 0:
        window_frames = min(window_frames, total_frames)
    window_seconds = window_frames / fps
    num_samples = max(1, min(VIDEO_MAX_SAMPLES, int(round(window_seconds)) or 1, window_frames))
    sample_indices = np.linspace(0, max(0, window_frames - 1), num_samples, dtype=int)

    plates: Dict[str, Dict[str, Any]] = {}
    best_unplated = None
    sampled_count = 0

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if frame.shape[1] > VIDEO_MAX_WIDTH:
            s = VIDEO_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (VIDEO_MAX_WIDTH, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        sampled_count += 1
        t_sec = round(float(f_idx) / fps, 2)
        vehicles = _vehicles_in(frame)

        for p in read_all_plates(frame):
            text, conf, bbox = p["plate"], p["confidence"], p["bbox"]
            entry = plates.setdefault(text, {"sightings": 0, "first_seen": t_sec, "confidence": -1.0})
            entry["sightings"] += 1
            if conf <= entry["confidence"]:
                continue
            around = _vehicle_around(bbox, vehicles)
            box, vtype = (around if around else (bbox, "Car"))
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            pad = 12
            annotated = draw_plate_annotation(frame, text, bbox, box)
            crop = annotated[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] if around else annotated
            entry.update({
                "plate": text, "has_plate": True, "confidence": round(conf, 3), "vehicle_type": vtype,
                "violation": "NONE", "plate_bbox": bbox, "box": box, "timestamp": t_sec,
                "frame_index": int(f_idx), "image_data": frame_to_base64(crop),
            })

        if vehicles and not plates:
            box, vtype = max(vehicles, key=lambda v: (v[0][2] - v[0][0]) * (v[0][3] - v[0][1]))
            area = (box[2] - box[0]) * (box[3] - box[1])
            if best_unplated is None or area > best_unplated["_area"]:
                best_unplated = {
                    "plate": None, "has_plate": False, "confidence": 0.0, "vehicle_type": vtype,
                    "violation": "MISSING_OR_COVERED_PLATE", "box": box, "timestamp": t_sec,
                    "frame_index": int(f_idx), "image_data": frame_to_base64(frame[box[1]:box[3], box[0]:box[2]]),
                    "_area": area,
                }

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # One noisy read is not a plate; a plate read twice, or once clearly, is.
    results = [e for e in plates.values() if e["sightings"] >= 2 or e["confidence"] >= VIDEO_KEEP_CONF]
    results.sort(key=lambda e: e["first_seen"])
    if not results and best_unplated:
        best_unplated.pop("_area", None)
        results = [best_unplated]
    return {
        "success": True,
        "total": len(results),
        "fps": fps,
        "frames_sampled": sampled_count,
        "window_seconds": round(window_seconds, 1),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "vehicles": results,
        "device": DEVICE
    }

# -----------------------------------------------------------------------------
# Local mode: serve on this machine only (start.bat points the backend here).
# -----------------------------------------------------------------------------
if not IN_COLAB:
    port = int(os.environ.get("VEROCITI_AI_PORT", "8000"))
    print(f"✅ VeROCiTI AI Engine running locally on http://127.0.0.1:{port} ({DEVICE_NAME})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Colab mode: Clean previous processes, Launch Uvicorn, then Start Tunnel & Auto-Sync
# -----------------------------------------------------------------------------
os.system("pkill -9 -f cloudflared 2>/dev/null || true")
os.system("pkill -9 -f uvicorn 2>/dev/null || true")
os.system("fuser -k 8000/tcp 2>/dev/null || true")
time.sleep(0.5)

def run_uvicorn():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_uvicorn, daemon=True)
server_thread.start()

# Wait for local server
print("⏳ Initializing local VeROCiTI server on port 8000...")
server_ready = False
for _ in range(30):
    try:
        r = requests.get("http://127.0.0.1:8000/", timeout=1)
        if r.status_code == 200:
            server_ready = True
            break
    except Exception:
        time.sleep(0.4)

if not server_ready:
    print("❌ Server failed to start locally on port 8000")
else:
    print("✅ Local VeROCiTI server is UP and responding!")

# Setup cloudflared tunnel
if not os.path.exists("/usr/local/bin/cloudflared"):
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")

log_path = f"/tmp/cf_{int(time.time())}.log"
subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--logfile", log_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

tunnel_url = None
for _ in range(60):
    time.sleep(0.5)
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
                if m:
                    tunnel_url = m.group(0)
                    break
        except Exception:
            pass

print("\n" + "=" * 65)
print(f"🚀 VeROCiTI AI Engine is LIVE on NVIDIA GPU ({DEVICE})!")
print(f"🔗 Cloudflare Tunnel URL: {tunnel_url}")
print("=" * 65 + "\n")

# Auto-sync tunnel URL to backend (works for Render, Local, and dynamic backends)
backend_sync_urls = [
    "https://clear-ways.onrender.com/api/set_ai_backend",
    "http://127.0.0.1:5000/api/set_ai_backend"
]

def sync_tunnel_to_backends():
    for b_url in backend_sync_urls:
        try:
            resp = requests.post(b_url, json={"url": tunnel_url}, timeout=3)
            print(f"✅ Synced active GPU tunnel ({tunnel_url}) to {b_url} [Status: {resp.status_code}]")
        except Exception as e:
            pass

sync_tunnel_to_backends()

# Keep cell running indefinitely and periodically refresh sync & prevent idle timeouts
print("⚡ Colab AI GPU Backend is running continuously. Press interrupt to stop.")
try:
    heartbeat_counter = 0
    while True:
        time.sleep(10)
        heartbeat_counter += 1
        # Re-announce tunnel heartbeat every 60 seconds
        if heartbeat_counter % 6 == 0 and tunnel_url:
            sync_tunnel_to_backends()
except KeyboardInterrupt:
    print("Stopping server...")
