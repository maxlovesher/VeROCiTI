"""
test_features_api.py — endpoint tests for features_api.py (/api/insights/*)

Builds a throwaway detection database with hand-made sightings, each chosen
to trigger (or deliberately not trigger) one of the camera-network features:
travel-time reliability and bottlenecks, camera health, unusual demand,
incident detection, the enforcement scan, hashed trajectories, retention,
fuzzy plate matching with an edge watchlist, and the on-demand rule checks.

Uses a temporary SQLite file, never the real prototype/data/traffic.db.
Run with:
    python "city flow model/scripts/test_features_api.py"
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
PROTOTYPE_DIR = os.path.join(os.path.dirname(SERVER_DIR), "prototype")
sys.path.insert(0, SERVER_DIR)
sys.path.append(PROTOTYPE_DIR)

import database as db  # noqa: E402

from flask import Flask  # noqa: E402

import features_api  # noqa: E402

NOW = datetime.now().replace(microsecond=0)
YESTERDAY = NOW - timedelta(days=1)


def ts(dt):
    return dt.isoformat(timespec="seconds")


def insert(conn, plate, cam, when, speed=40.0, vtype="Car", violation="NONE"):
    conn.execute(
        "INSERT INTO detections (plate, camera_id, timestamp, speed_kmph, vehicle_type, violation) VALUES (?,?,?,?,?,?)",
        (plate, cam, ts(when), speed, vtype, violation),
    )


def build_fixture():
    conn = db.get_conn()

    # Travel time CAM_01 (Patia) -> CAM_02 (Jayadev Vihar), ~6 km: five normal
    # trips of 9-10 min plus one 20-min outlier (a bottleneck), and one
    # 3-minute trip (~120 km/h, a section-speed violation).
    for i, minutes in enumerate([9.0, 9.5, 10.0, 9.2, 20.0]):
        start = NOW - timedelta(minutes=120 - i * 5)
        insert(conn, f"OD01TT{1000 + i}", "CAM_01", start)
        insert(conn, f"OD01TT{1000 + i}", "CAM_02", start + timedelta(minutes=minutes))
    insert(conn, "OD09SP7777", "CAM_01", NOW - timedelta(minutes=70))
    insert(conn, "OD09SP7777", "CAM_02", NOW - timedelta(minutes=67), speed=110.0)
    # Same plate 6 km apart one minute later (~360 km/h): a misread or cloned
    # plate, not a speeder, and not a travel-time sample either.
    insert(conn, "OD09CL0000", "CAM_01", NOW - timedelta(minutes=64))
    insert(conn, "OD09CL0000", "CAM_02", NOW - timedelta(minutes=63))

    # Incident on CAM_03 -> CAM_04 (~1.1 km, typically 3 min): four earlier
    # trips set the norm, then four vehicles leave CAM_03 20+ min ago and never
    # arrive, while arrivals at CAM_04 keep climbing.
    for i in range(4):
        start = NOW - timedelta(minutes=100 - i * 10)
        insert(conn, f"OD03HH{2000 + i}", "CAM_03", start)
        insert(conn, f"OD03HH{2000 + i}", "CAM_04", start + timedelta(minutes=3))
    for i in range(4):
        insert(conn, f"OD03XX{3000 + i}", "CAM_03", NOW - timedelta(minutes=20 + i))
    rising = [0, 0, 1, 2, 3, 5]            # arrivals per 5-min bucket, oldest first
    n = 0
    for bucket, count in enumerate(rising):
        for _ in range(count):
            n += 1
            insert(conn, f"OD04QQ{4000 + n}", "CAM_04", NOW - timedelta(minutes=5 * (5 - bucket) + 2))

    # Unusual demand at CAM_05: 1-2 vehicles per 5-min window, then 12 now.
    for w in range(1, 7):
        for k in range(1 + w % 2):
            insert(conn, f"OD05DD{w}{k}", "CAM_05", NOW - timedelta(minutes=5 * w + 2))
    for k in range(12):
        insert(conn, f"OD05NOW{k:02d}", "CAM_05", NOW - timedelta(seconds=30 + k))

    # A truck in daytime and at night; a speeder in the Sainik School zone at 08:00.
    insert(conn, "MH12TR5555", "CAM_06", YESTERDAY.replace(hour=14, minute=0, second=0),
           vtype="Truck", violation="HEAVY_VEHICLE_RESTRICTED_HOURS")
    insert(conn, "MH12TR6666", "CAM_06", YESTERDAY.replace(hour=23, minute=0, second=0),
           vtype="Truck", violation="LOGGED")
    # A city bus in daytime is not a goods vehicle, so it's not restricted.
    insert(conn, "OD02BU1234", "CAM_06", YESTERDAY.replace(hour=13, minute=0, second=0),
           vtype="Bus", violation="LOGGED")
    insert(conn, "OD02SC8888", "CAM_SAINIK", YESTERDAY.replace(hour=8, minute=0, second=0), speed=41.0,
           violation="SCHOOL_ZONE_SPEED")

    # Retention: one ordinary sighting and one violation, both 3 days old.
    insert(conn, "OD07OLD001", "CAM_07", NOW - timedelta(days=3))
    insert(conn, "OD07OLD002", "CAM_07", NOW - timedelta(days=3), violation="STOLEN_VEHICLE_ALERT")

    # Plate-match target, also on the watchlist.
    insert(conn, "OD02BA4455", "CAM_08", NOW - timedelta(minutes=50))
    conn.commit()
    conn.close()
    db.add_to_blacklist("OD02BA4455", "Test watchlist entry")


class TestInsightsApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="verociti_insights_")
        db.DB_PATH = os.path.join(cls._tmp, "traffic.db")
        db.init_db()
        build_fixture()
        app = Flask(__name__)
        features_api.register_feature_routes(app)
        cls.client = app.test_client()
        cls.summary = cls.client.get("/api/insights/summary?hours=48").get_json()

    def test_travel_time_stats_and_bottleneck(self):
        links = {l["link"]: l for l in self.summary["travel_time"]["links"]}
        link = links["CAM_01→CAM_02"]
        self.assertEqual(link["sample_size"], 6)
        self.assertGreater(link["p95"], link["median"])
        bottlenecks = [b["road"] for b in self.summary["travel_time"]["bottlenecks"]]
        self.assertIn("CAM_01→CAM_02", bottlenecks)
        self.assertNotIn("CAM_03→CAM_04", bottlenecks)   # consistent 3-minute trips

    def test_camera_health_and_fallback_level(self):
        health = self.summary["camera_health"]
        cams = {c["camera_id"]: c for c in health["cameras"]}
        self.assertTrue(cams["CAM_05"]["healthy"])
        self.assertFalse(cams["CAM_06"]["healthy"])          # last seen yesterday
        self.assertEqual(health["healthy"], sum(1 for c in cams.values() if c["healthy"]))
        self.assertAlmostEqual(health["health_ratio"], round(health["healthy"] / health["fleet_size"], 3))
        self.assertIn(health["fallback_level"], ("FULL_COVERAGE", "PARTIAL_COVERAGE", "MINIMAL_COVERAGE"))

    def test_unusual_demand_flags_the_spike_only(self):
        demand = self.summary["demand"]
        self.assertFalse(demand["warming_up"])
        flagged = {a["camera_id"]: a for a in demand["alerts"]}
        self.assertIn("CAM_05", flagged)
        self.assertGreaterEqual(flagged["CAM_05"]["z_score"], 2.5)
        self.assertNotIn("CAM_02", flagged)

    def test_demand_waits_for_enough_history(self):
        fresh = features_api.demand_report(
            [{"camera_id": "CAM_01", "plate": "X", "_ts": NOW}], {}, window_min=5, history_windows=6)
        self.assertTrue(fresh["warming_up"])
        self.assertEqual(fresh["alerts"], [])

    def test_incident_detected_on_link_with_missing_arrivals(self):
        incidents = {i["link"]: i for i in self.summary["incidents"]["incidents"]}
        self.assertIn("CAM_03→CAM_04", incidents)
        inc = incidents["CAM_03→CAM_04"]
        self.assertGreaterEqual(inc["overdue"], 3)
        self.assertEqual(inc["queue_trend"], "RISING")
        self.assertNotIn("CAM_01→CAM_02", incidents)

    def test_enforcement_scan(self):
        enf = self.summary["enforcement"]
        by_type = enf["by_type"]
        self.assertEqual(by_type.get("SECTION_SPEED"), 1)
        self.assertEqual(by_type.get("HEAVY_VEHICLE_RESTRICTED_HOURS"), 1)
        self.assertEqual(by_type.get("SCHOOL_ZONE_SPEED"), 1)
        plates = {f["type"]: f["plate"] for f in enf["findings"]}
        self.assertEqual(plates["SECTION_SPEED"], "OD09SP7777")
        self.assertEqual(plates["HEAVY_VEHICLE_RESTRICTED_HOURS"], "MH12TR5555")   # the night truck is fine
        self.assertEqual(plates["SCHOOL_ZONE_SPEED"], "OD02SC8888")

    def test_impossible_travel_is_flagged_for_review_not_ticketed(self):
        enf = self.summary["enforcement"]
        review = [f for f in enf["findings"] if f["type"] == "IMPOSSIBLE_TRAVEL"]
        self.assertEqual([f["plate"] for f in review], ["OD09CL0000"])
        self.assertFalse(review[0]["violation"])
        self.assertGreater(review[0]["details"]["required_kmph"], 140)
        self.assertEqual(enf["needs_review"], 1)
        self.assertEqual(enf["total"], 3)       # the three genuine violations only

    def test_trajectories_never_expose_plates(self):
        traj = self.summary["trajectories"]
        self.assertGreater(traj["identities"], 0)
        blob = json.dumps(traj)
        for plate in ("OD01TT1000", "OD09SP7777", "OD02BA4455", "MH12TR5555"):
            self.assertNotIn(plate, blob)
        for t in traj["trajectories"]:
            self.assertTrue(t["hashed_id"].startswith("HID_"))

    def test_privacy_retention_preview_and_purge(self):
        preview = self.summary["privacy"]
        self.assertIn("identity", preview["what_we_store"])
        self.assertEqual(preview["retention"]["rows_past_retention"], 1)

        # Purge on a copy of the database so other tests keep their fixture.
        original = db.DB_PATH
        copy = os.path.join(self._tmp, "purge_copy.db")
        shutil.copy(original, copy)
        db.DB_PATH = copy
        try:
            r = self.client.post("/api/insights/privacy/purge").get_json()
            self.assertEqual(r["deleted"], 1)
            conn = db.get_conn()
            remaining = {row[0] for row in conn.execute("SELECT plate FROM detections WHERE camera_id='CAM_07'")}
            conn.close()
            self.assertEqual(remaining, {"OD07OLD002"})    # the violation is kept for 90 days
        finally:
            db.DB_PATH = original

    def test_plate_match_exact_fuzzy_and_none(self):
        exact = self.client.get("/api/insights/plate_match?plate=OD02BA4455").get_json()
        self.assertEqual(exact["result"]["method"], "EXACT")

        fuzzy = self.client.get("/api/insights/plate_match?plate=0D02BA4455").get_json()   # zero for O
        self.assertEqual(fuzzy["result"]["method"], "FUZZY")
        self.assertEqual(fuzzy["result"]["plate"], "OD02BA4455")
        self.assertTrue(fuzzy["watchlist_hit"])

        none = self.client.get("/api/insights/plate_match?plate=ZZ99ZZ9999").get_json()
        self.assertEqual(none["result"]["method"], "NONE")
        self.assertFalse(none["watchlist_hit"])

        self.assertEqual(self.client.get("/api/insights/plate_match").status_code, 400)

    def test_plate_match_rejects_impossible_travel(self):
        r = self.client.get("/api/insights/plate_match?plate=0D02BA4455&distance_km=50&minutes=5").get_json()
        self.assertEqual(r["result"]["method"], "NONE")

    def test_on_demand_enforcement_checks(self):
        def post(body):
            return self.client.post("/api/insights/enforcement/check", json=body)

        cases = [
            ({"check": "red_light", "crossing_time": "2026-01-01T10:00:05",
              "red_start_time": "2026-01-01T10:00:00"}, True),
            ({"check": "red_light", "crossing_time": "2026-01-01T10:00:00.500000",
              "red_start_time": "2026-01-01T10:00:00"}, False),
            ({"check": "wrong_way", "observed_direction": "S", "allowed_directions": ["N"]}, True),
            ({"check": "junction_parking", "dwell_seconds": 200}, True),
            ({"check": "junction_parking", "dwell_seconds": 30}, False),
            ({"check": "heavy_vehicle", "vehicle_class": "truck", "timestamp": "2026-01-01T12:00:00"}, True),
            ({"check": "heavy_vehicle", "vehicle_class": "car", "timestamp": "2026-01-01T12:00:00"}, False),
            ({"check": "school_zone", "speed_kmph": 40, "timestamp": "2026-01-01T08:15:00"}, True),
            ({"check": "section_speed", "lat1": 20.3540, "lon1": 85.8170, "t1": "2026-01-01T10:00:00",
              "lat2": 20.3005, "lon2": 85.8228, "t2": "2026-01-01T10:03:00"}, True),
        ]
        for body, expected in cases:
            r = post(body)
            self.assertEqual(r.status_code, 200, body)
            self.assertEqual(r.get_json()["violation"], expected, body)

        self.assertEqual(post({"check": "teleport"}).status_code, 400)
        self.assertEqual(post({"check": "red_light"}).status_code, 400)

    def test_summary_rejects_bad_parameters(self):
        self.assertEqual(self.client.get("/api/insights/summary?hours=abc").status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
