"""
features_api.py — Camera-network analytics & enforcement endpoints
==================================================================
Wires the pure-logic prototype modules into HTTP routes, fed from the ANPR
detection database (prototype/database.py):

  Feature 2   trajectories with hashed IDs      trajectory_events, identity
  12–17       automated enforcement checks      enforcement
  18          fuzzy plate matching              plate_match
  19          incident detection                incident_detect (+ green_wave.QueuePredictor)
  20          camera health monitoring          camera_health
  21          privacy by design                 privacy, identity
  22          travel-time reliability           travel_time_analytics
  23          unusual-demand alerts             demand_alerts

Routes live under /api/insights/. Everything here only reads the detection
table except POST /api/insights/privacy/purge, which applies the retention
limit.
"""

import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List, Optional

from flask import jsonify, request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTYPE_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "prototype")
if os.path.isdir(PROTOTYPE_DIR) and PROTOTYPE_DIR not in sys.path:
    sys.path.append(PROTOTYPE_DIR)

import database as db                      # noqa: E402
import camera_health                       # noqa: E402
import demand_alerts                       # noqa: E402
import enforcement                         # noqa: E402
import identity                            # noqa: E402
import incident_detect                     # noqa: E402
import plate_match                         # noqa: E402
import privacy                             # noqa: E402
import trajectory_events                   # noqa: E402
import travel_time_analytics               # noqa: E402
from analytics import haversine_km         # noqa: E402
from green_wave import QueuePredictor      # noqa: E402

# ── Enforcement configuration ──
# Straight-line camera-to-camera distance understates the road distance, so
# section speeds computed from it are a lower bound: this check can miss
# speeders but won't accuse anyone who wasn't speeding.
SECTION_SPEED_LIMIT_KMPH = 60.0
SECTION_MAX_GAP_MIN = 30.0          # sightings further apart than this aren't treated as one trip
# City-limits heavy-vehicle rule: goods vehicles may only run at night. Buses
# count as heavy in enforcement.py, but city buses run all day by design, so
# the time window is only applied to goods classes.
HEAVY_VEHICLE_WINDOWS = [(dtime(22, 0), dtime(6, 0))]
GOODS_VEHICLE_CLASSES = {"truck", "lorry", "trailer", "heavy"}
# Cameras covering a school frontage, with their timed limits.
SCHOOL_ZONES = {
    "CAM_SAINIK": [(dtime(7, 30), dtime(9, 0), 25.0), (dtime(13, 30), dtime(15, 30), 25.0)],
}

MAX_ROWS = 20000


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Detections are stored in server-local time; drop any offset so comparisons stay naive.
    return ts.astimezone().replace(tzinfo=None) if ts.tzinfo else ts


def _detections(hours: float) -> List[Dict]:
    """Detection rows from the last `hours`, oldest first, with a parsed `_ts`."""
    conn = db.get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, plate, camera_id, timestamp, speed_kmph, vehicle_type, violation, direction, lat, lon
            FROM detections ORDER BY id DESC LIMIT ?
            """,
            (MAX_ROWS,),
        ).fetchall()
    finally:
        conn.close()
    cutoff = datetime.now() - timedelta(hours=hours)
    out = []
    for r in rows:
        item = dict(r)
        ts = _parse_ts(item.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        item["_ts"] = ts
        out.append(item)
    out.sort(key=lambda d: d["_ts"])
    return out


def _is_real_plate(plate: str) -> bool:
    p = (plate or "").upper()
    return bool(p) and "NO PLATE" not in p and "UNREADABLE" not in p


def _camera_meta() -> Dict[str, Dict]:
    return {c["id"]: c for c in db.get_all_cameras()}


def _cam_name(cams: Dict[str, Dict], cam_id: str) -> str:
    return cams.get(cam_id, {}).get("name", cam_id)


def _camera_point(det: Dict, cams: Dict[str, Dict]):
    meta = cams.get(det["camera_id"], {})
    lat = meta.get("lat") or det.get("lat")
    lon = meta.get("lon") or det.get("lon")
    return (lat, lon) if lat is not None and lon is not None else None


def _hops(rows: List[Dict], cams: Dict[str, Dict], max_gap_min: float = SECTION_MAX_GAP_MIN) -> List[Dict]:
    """
    Consecutive sightings of the same plate at two different cameras, within
    max_gap_min. Each hop is marked `feasible` unless covering the
    straight-line distance in that time would take more than
    plate_match's 140 km/h ceiling — on a real network that means a misread
    or cloned plate, not one vehicle driving, so such hops are kept out of
    travel-time and incident statistics and surfaced for review instead.
    """
    by_plate: Dict[str, List[Dict]] = defaultdict(list)
    for d in rows:
        if _is_real_plate(d["plate"]):
            by_plate[d["plate"]].append(d)
    hops = []
    for plate, seen in by_plate.items():
        for a, b in zip(seen, seen[1:]):
            if a["camera_id"] == b["camera_id"]:
                continue
            minutes = (b["_ts"] - a["_ts"]).total_seconds() / 60.0
            if not 0 < minutes <= max_gap_min:
                continue
            pa, pb = _camera_point(a, cams), _camera_point(b, cams)
            dist_km = haversine_km(pa[0], pa[1], pb[0], pb[1]) if pa and pb else None
            feasible = dist_km is None or plate_match.travel_time_feasible(dist_km, minutes)
            hops.append({
                "plate": plate, "from": a, "to": b, "minutes": minutes,
                "distance_km": dist_km, "feasible": feasible,
                "required_kmph": round(dist_km / (minutes / 60.0), 1) if dist_km is not None else None,
            })
    return hops


# ── Feature 22: travel-time reliability ──
def travel_time_report(hops: List[Dict], cams: Dict[str, Dict]) -> Dict:
    samples: Dict[str, List[float]] = defaultdict(list)
    for hop in hops:
        if not hop["feasible"]:
            continue
        a, b = hop["from"]["camera_id"], hop["to"]["camera_id"]
        samples[f"{a}→{b}"].append(round(hop["minutes"], 2))
    stats = travel_time_analytics.road_travel_time_stats(samples)
    links = []
    for link, s in stats.items():
        a, b = link.split("→")
        links.append({"link": link, "from_name": _cam_name(cams, a), "to_name": _cam_name(cams, b), **s})
    links.sort(key=lambda l: l["sample_size"], reverse=True)
    bottlenecks = travel_time_analytics.identify_bottlenecks(stats)
    for b in bottlenecks:
        a, z = b["road"].split("→")
        b["from_name"], b["to_name"] = _cam_name(cams, a), _cam_name(cams, z)
    return {"links": links, "bottlenecks": bottlenecks, "unit": "minutes"}


# ── Feature 20: camera health ──
def camera_health_report(rows: List[Dict], cams: Dict[str, Dict], timeout_s: float) -> Dict:
    monitor = camera_health.CameraHealthMonitor(timeout_s=timeout_s)
    last_seen: Dict[str, datetime] = {}
    for d in rows:
        last_seen[d["camera_id"]] = d["_ts"]
    for cam_id, ts in last_seen.items():
        monitor.heartbeat(cam_id, ts.isoformat())
    now = datetime.now()
    fleet = sorted(last_seen)
    ratio = monitor.fleet_health_ratio(fleet, now.isoformat())
    cameras = [{
        "camera_id": cam_id,
        "name": _cam_name(cams, cam_id),
        "healthy": monitor.is_healthy(cam_id, now.isoformat()),
        "last_seen": last_seen[cam_id].isoformat(timespec="seconds"),
        "seconds_since_last": round((now - last_seen[cam_id]).total_seconds()),
    } for cam_id in fleet]
    cameras.sort(key=lambda c: (c["healthy"], -c["seconds_since_last"]))
    return {
        "fleet_size": len(fleet),
        "healthy": sum(1 for c in cameras if c["healthy"]),
        "health_ratio": ratio,
        "fallback_level": camera_health.fallback_level_for(ratio) if fleet else "NO_DATA",
        "timeout_s": timeout_s,
        "cameras": cameras,
    }


# ── Feature 23: unusual demand ──
def demand_report(rows: List[Dict], cams: Dict[str, Dict], window_min: int, history_windows: int) -> Dict:
    now = datetime.now()
    counts: Dict[str, List[set]] = defaultdict(lambda: [set() for _ in range(history_windows + 1)])
    for d in rows:
        idx = int((now - d["_ts"]).total_seconds() // (window_min * 60))   # 0 = current window, 1.. = history
        if 0 <= idx <= history_windows:
            counts[d["camera_id"]][idx].add(d["plate"])

    span_min = (now - rows[0]["_ts"]).total_seconds() / 60.0 if rows else 0.0
    needed_min = (history_windows + 1) * window_min
    # Too little history makes every camera look "unusual" against a zero baseline.
    warming_up = span_min < needed_min

    current = {cam: len(w[0]) for cam, w in counts.items()}
    histories = {cam: [len(x) for x in w[1:]] for cam, w in counts.items()}
    alerts = [] if warming_up else demand_alerts.check_unusual_demand_for_cameras(current, histories)
    for a in alerts:
        a["name"] = _cam_name(cams, a["camera_id"])
    return {
        "window_minutes": window_min,
        "history_windows": history_windows,
        "warming_up": warming_up,
        "minutes_of_data": round(span_min, 1),
        "minutes_needed": needed_min,
        "cameras_tracked": len(current),
        "alerts": alerts,
    }


# ── Feature 19: incident detection ──
def incident_report(rows: List[Dict], hops: List[Dict], cams: Dict[str, Dict], travel: Dict) -> Dict:
    now = datetime.now()
    typical = {l["link"]: l["median"] for l in travel["links"] if l["sample_size"] >= 2 and l["median"] > 0}

    # Where does traffic usually go next from each camera?
    successors: Dict[str, Counter] = defaultdict(Counter)
    for hop in hops:
        if hop["feasible"]:
            successors[hop["from"]["camera_id"]][hop["to"]["camera_id"]] += 1

    last_by_plate: Dict[str, Dict] = {}
    for d in rows:
        if _is_real_plate(d["plate"]):
            last_by_plate[d["plate"]] = d

    detectors: Dict[str, incident_detect.IncidentDetector] = {}
    for plate, d in last_by_plate.items():
        if (now - d["_ts"]).total_seconds() > 3600:
            continue   # left the network long ago, not "expected" anywhere
        nxt = successors[d["camera_id"]].most_common(1)
        if not nxt:
            continue
        link = f"{d['camera_id']}→{nxt[0][0]}"
        if link not in typical:
            continue
        det = detectors.setdefault(
            link, incident_detect.IncidentDetector(typical[link], tolerance_minutes=max(2.0, typical[link]))
        )
        det.record_departure(plate, d["_ts"].isoformat())

    # Queue-growth proxy at the downstream camera: 5-minute arrival counts over the last 30 minutes.
    def arrivals_trend(cam_id: str) -> str:
        predictor = QueuePredictor(history_len=6, sample_interval_s=300)
        for i in range(5, -1, -1):
            lo, hi = now - timedelta(minutes=5 * (i + 1)), now - timedelta(minutes=5 * i)
            predictor.record(sum(1 for d in rows if d["camera_id"] == cam_id and lo <= d["_ts"] < hi))
        return predictor.trend()

    links = []
    for link, det in detectors.items():
        downstream = link.split("→")[1]
        overdue = det.overdue_vehicles(now.isoformat())
        verdict = incident_detect.detect_incident(len(overdue), arrivals_trend(downstream))
        links.append({
            "link": link,
            "from_name": _cam_name(cams, link.split("→")[0]),
            "to_name": _cam_name(cams, downstream),
            "typical_minutes": typical[link],
            "pending": det.pending_count(),
            "overdue": len(overdue),
            **verdict,
        })
    links.sort(key=lambda l: (l["incident_likely"], l["overdue"]), reverse=True)
    return {"links_watched": links, "incidents": [l for l in links if l["incident_likely"]]}


# ── Features 12, 15, 16: enforcement scan over stored detections ──
def enforcement_scan(rows: List[Dict], hops: List[Dict], cams: Dict[str, Dict], limit: int = 40) -> Dict:
    findings = []

    for hop in hops:
        a, b = hop["from"], hop["to"]
        pa, pb = _camera_point(a, cams), _camera_point(b, cams)
        if not pa or not pb:
            continue
        where = f"{_cam_name(cams, a['camera_id'])} → {_cam_name(cams, b['camera_id'])}"
        if not hop["feasible"]:
            # Not a speeding ticket: no vehicle covers this in the time, so one of the reads is wrong.
            findings.append({
                "plate": hop["plate"], "time": b["timestamp"], "camera_id": b["camera_id"], "where": where,
                "violation": False, "type": "IMPOSSIBLE_TRAVEL", "review": True,
                "details": {"required_kmph": hop["required_kmph"], "minutes": round(hop["minutes"], 2),
                            "note": "Possible misread or cloned plate"},
            })
            continue
        result = enforcement.check_section_speed(
            pa[0], pa[1], a["_ts"].isoformat(), pb[0], pb[1], b["_ts"].isoformat(), SECTION_SPEED_LIMIT_KMPH
        )
        if result["violation"]:
            findings.append({
                "plate": hop["plate"], "time": b["timestamp"], "camera_id": b["camera_id"], "where": where,
                **result,
            })

    seen_heavy = set()
    for d in rows:
        if not _is_real_plate(d["plate"]):
            continue
        vehicle_type = (d.get("vehicle_type") or "").strip()
        if vehicle_type.lower() in GOODS_VEHICLE_CLASSES:
            result = enforcement.check_heavy_vehicle_window(vehicle_type, d["_ts"].isoformat(), HEAVY_VEHICLE_WINDOWS)
            key = (d["plate"], d["camera_id"], d["_ts"].date())
            if result["violation"] and key not in seen_heavy:
                seen_heavy.add(key)
                findings.append({
                    "plate": d["plate"], "time": d["timestamp"], "camera_id": d["camera_id"],
                    "where": _cam_name(cams, d["camera_id"]), **result,
                })

        schedule = SCHOOL_ZONES.get(d["camera_id"])
        if schedule and d.get("speed_kmph"):
            result = enforcement.check_school_zone_speed(float(d["speed_kmph"]), d["_ts"].isoformat(), schedule)
            if result["violation"]:
                findings.append({
                    "plate": d["plate"], "time": d["timestamp"], "camera_id": d["camera_id"],
                    "where": _cam_name(cams, d["camera_id"]), **result,
                })

    findings.sort(key=lambda f: f["time"], reverse=True)
    return {
        "total": sum(1 for f in findings if f["violation"]),
        "needs_review": sum(1 for f in findings if f.get("review")),
        "by_type": dict(Counter(f["type"] for f in findings)),
        "findings": findings[:limit],
        "rules": {
            "section_speed_limit_kmph": SECTION_SPEED_LIMIT_KMPH,
            "heavy_vehicle_allowed": [f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')}" for s, e in HEAVY_VEHICLE_WINDOWS],
            "heavy_vehicle_classes": sorted(GOODS_VEHICLE_CLASSES),
            "school_zones": {
                _cam_name(cams, cam): [f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')} @ {lim:g} km/h" for s, e, lim in sched]
                for cam, sched in SCHOOL_ZONES.items()
            },
            # Stored ANPR sightings carry no signal state, lane direction or dwell time,
            # so these rules only run on evidence posted to /api/insights/enforcement/check.
            "on_demand_only": ["RED_LIGHT_VIOLATION", "WRONG_WAY", "JUNCTION_MOUTH_PARKING"],
        },
    }


# ── Feature 2: hashed trajectories ──
def trajectories_report(rows: List[Dict], cams: Dict[str, Dict], limit: int = 12) -> Dict:
    events = trajectory_events.events_from_detections(r for r in rows if _is_real_plate(r["plate"]))
    trajectories = trajectory_events.build_trajectories(events)
    items = [{
        "hashed_id": hashed_id,
        "sightings": len(traj),
        "recent_cameras": [_cam_name(cams, e["camera"]) for e in traj[-5:]],
        "first_seen": traj[0]["time"],
        "last_seen": traj[-1]["time"],
        "dwell_seconds": trajectory_events.dwell_seconds(traj),
    } for hashed_id, traj in trajectories.items()]
    items.sort(key=lambda t: t["sightings"], reverse=True)
    return {"identities": len(items), "trajectories": items[:limit]}


# ── Feature 21: privacy / retention ──
def _all_retention_rows(conn) -> List[Dict]:
    return [dict(r) for r in conn.execute("SELECT id, timestamp, violation FROM detections").fetchall()]


def privacy_report() -> Dict:
    conn = db.get_conn()
    try:
        rows = _all_retention_rows(conn)
    finally:
        conn.close()
    kept = privacy.purge_expired(rows, datetime.now().isoformat())
    return {
        "what_we_store": privacy.what_we_store_summary(),
        "retention": {
            "rows_total": len(rows),
            "rows_past_retention": len(rows) - len(kept),
            "violation_rows": sum(1 for r in rows if r.get("violation") and r.get("violation") != "NONE"),
        },
        "identity_secret_configured": bool(os.environ.get("IDENTITY_HASH_SECRET")),
    }


def _hashed_watchlist() -> List[str]:
    try:
        return [identity.hash_id(plate_match._clean(item["plate"])) for item in db.get_blacklist()]
    except Exception:
        return []


def build_summary(hours: float = 24, timeout_s: float = 300, window_min: int = 5, history_windows: int = 6) -> Dict:
    rows = _detections(hours)
    cams = _camera_meta()
    hops = _hops(rows, cams)
    travel = travel_time_report(hops, cams)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "detections_analysed": len(rows),
        "hours": hours,
        "camera_health": camera_health_report(rows, cams, timeout_s),
        "travel_time": travel,
        "incidents": incident_report(rows, hops, cams, travel),
        "demand": demand_report(rows, cams, window_min, history_windows),
        "enforcement": enforcement_scan(rows, hops, cams),
        "trajectories": trajectories_report(rows, cams),
        "privacy": privacy_report(),
    }


def register_feature_routes(app):

    @app.route("/api/insights/summary")
    def insights_summary():
        try:
            params = dict(
                hours=float(request.args.get("hours", 24)),
                timeout_s=float(request.args.get("camera_timeout_s", 300)),
                window_min=max(1, int(request.args.get("window_min", 5))),
                history_windows=max(2, int(request.args.get("history_windows", 6))),
            )
        except ValueError:
            return jsonify({"ok": False, "error": "query parameters must be numbers"}), 400
        return jsonify(build_summary(**params))

    @app.route("/api/insights/plate_match")
    def insights_plate_match():
        """Feature 18 (+21 edge watchlist): look a possibly-misread plate up against every plate seen."""
        candidate = (request.args.get("plate") or "").strip()
        if not candidate:
            return jsonify({"ok": False, "error": "plate is required"}), 400
        conn = db.get_conn()
        try:
            known = [r[0] for r in conn.execute("SELECT DISTINCT plate FROM detections").fetchall()]
        finally:
            conn.close()
        known = [p for p in known if _is_real_plate(p)]

        distance_km = request.args.get("distance_km", type=float)
        minutes = request.args.get("minutes", type=float)
        result = plate_match.match_with_fallback(candidate, known, distance_km=distance_km, minutes=minutes)

        nearest = sorted(
            ({"plate": p, "similarity": plate_match.fuzzy_plate_similarity(candidate, p)} for p in known),
            key=lambda x: x["similarity"], reverse=True,
        )[:5]
        watchlist = _hashed_watchlist()
        hashes = {identity.hash_id(plate_match._clean(candidate))}
        if result.get("plate"):
            hashes.add(identity.hash_id(plate_match._clean(result["plate"])))
        return jsonify({
            "ok": True,
            "candidate": candidate,
            "result": result,
            "nearest": nearest,
            "known_plates": len(known),
            "watchlist_hit": any(privacy.match_watchlist_hash(h, watchlist) for h in hashes),
        })

    @app.route("/api/insights/enforcement/check", methods=["POST"])
    def insights_enforcement_check():
        """Run any single Feature 12–17 rule on caller-supplied evidence."""
        data = request.get_json(silent=True) or {}
        check = data.get("check")

        def _t(s):
            return datetime.strptime(s, "%H:%M").time()

        try:
            if check == "section_speed":
                r = enforcement.check_section_speed(
                    float(data["lat1"]), float(data["lon1"]), data["t1"],
                    float(data["lat2"]), float(data["lon2"]), data["t2"],
                    float(data.get("limit_kmph", SECTION_SPEED_LIMIT_KMPH)),
                )
            elif check == "red_light":
                r = enforcement.check_red_light_violation(
                    data["crossing_time"], data["red_start_time"], float(data.get("grace_period_s", 1.0)))
            elif check == "wrong_way":
                r = enforcement.check_wrong_way(data["observed_direction"], data["allowed_directions"])
            elif check == "heavy_vehicle":
                windows = [(_t(w[0]), _t(w[1])) for w in data.get("allowed_windows", [])] or HEAVY_VEHICLE_WINDOWS
                r = enforcement.check_heavy_vehicle_window(data["vehicle_class"], data["timestamp"], windows)
            elif check == "school_zone":
                sched = [(_t(s[0]), _t(s[1]), float(s[2])) for s in data.get("schedule", [])] \
                    or SCHOOL_ZONES["CAM_SAINIK"]
                r = enforcement.check_school_zone_speed(float(data["speed_kmph"]), data["timestamp"], sched)
            elif check == "junction_parking":
                r = enforcement.check_junction_mouth_parking(
                    float(data["dwell_seconds"]), float(data.get("threshold_seconds", 90.0)))
            else:
                return jsonify({"ok": False, "error": "check must be one of section_speed, red_light, wrong_way, "
                                                      "heavy_vehicle, school_zone, junction_parking"}), 400
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"missing or invalid field: {e}"}), 400
        return jsonify({"ok": True, "check": check, **r})

    @app.route("/api/insights/privacy/purge", methods=["POST"])
    def insights_privacy_purge():
        """Feature 21: apply the retention limit — delete detection rows past it."""
        conn = db.get_conn()
        try:
            rows = _all_retention_rows(conn)
            kept_ids = {r["id"] for r in privacy.purge_expired(rows, datetime.now().isoformat())}
            expired = [r["id"] for r in rows if r["id"] not in kept_ids]
            for i in range(0, len(expired), 500):
                chunk = expired[i:i + 500]
                conn.execute(f"DELETE FROM detections WHERE id IN ({','.join('?' * len(chunk))})", chunk)
            conn.commit()
        finally:
            conn.close()
        return jsonify({"ok": True, "deleted": len(expired), "kept": len(kept_ids)})

    print("[Features API] Analytics & enforcement routes registered under /api/insights/")
