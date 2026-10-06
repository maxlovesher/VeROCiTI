"""
test_webcam_live.py — regression tests for the live-webcam pipeline (webcam_pipeline.py)

Covers:
  - The tracker allocating a new track ID while fast_track holds _live_state_lock
    (this used to deadlock the webcam the first time a new vehicle appeared).
  - The multi-plate scan: every plate the AI engine reports becomes a track, a
    plate inside a tracked vehicle is attached to that vehicle, plates expire,
    and a scan that finishes after a session reset is discarded.

The AI engine is replaced by a fake HTTP response, so no models or GPU are needed.
Run with:
    python "city flow model/scripts/test_webcam_live.py"
"""

import os
import sys
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
sys.path.insert(0, SERVER_DIR)
sys.path.append(os.path.join(os.path.dirname(SERVER_DIR), "prototype"))

import numpy as np  # noqa: E402

import webcam_pipeline as wp  # noqa: E402

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class FakeResponse:
    status_code = 200

    def __init__(self, plates):
        self._plates = plates

    def json(self):
        return {"success": True, "plates": self._plates}


def fake_requests(plates, delay=0.0):
    def post(url, files=None, timeout=None):
        assert url.endswith("/predict_plates"), url
        time.sleep(delay)
        return FakeResponse(plates)
    return types.SimpleNamespace(post=post)


class TestTrackIdAllocation(unittest.TestCase):
    def setUp(self):
        wp._live_state.clear()

    def test_new_track_under_live_state_lock_does_not_deadlock(self):
        det = {"bbox": [100, 100, 300, 250], "cls": 2, "vtype": "Car", "conf": 0.9, "tid_hint": None}
        result = {}

        def run():
            with wp._live_state_lock:   # exactly what fast_track does
                result["matched"] = wp.match_detections_to_tracks(
                    [det], wp._live_state, now=time.time(), id_generator=wp._get_next_live_track_id)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "match_detections_to_tracks deadlocked allocating a new track id")
        self.assertEqual(len(result["matched"]), 1)


class TestMultiPlateScan(unittest.TestCase):
    PLATES = [
        {"plate": "21BH2345AA", "confidence": 0.84, "bbox": [570, 590, 694, 617]},
        {"plate": "OD02BA4455", "confidence": 0.71, "bbox": [150, 200, 250, 230]},
    ]

    def setUp(self):
        wp._live_state.clear()
        wp._reset_multi_plate()
        self._real_requests = sys.modules.get("requests")

    def tearDown(self):
        if self._real_requests is not None:
            sys.modules["requests"] = self._real_requests
        wp._reset_multi_plate()
        wp._live_state.clear()

    def scan(self, plates, session=None):
        sys.modules["requests"] = fake_requests(plates)
        s = wp._multi_plate_state["session"] if session is None else session
        wp._run_multi_plate_scan(FRAME, "CAM_TEST", "2026-01-01T10:00:00", "http://engine", s)

    def test_every_plate_becomes_a_track(self):
        self.scan(self.PLATES)
        out = wp._multi_plate_tracks_out("2026-01-01T10:00:00")
        self.assertEqual(sorted(t["plate"] for t in out), ["21BH2345AA", "OD02BA4455"])
        self.assertEqual(len({t["track_id"] for t in out}), 2)
        for t in out:
            self.assertEqual(t["status"], "CONFIRMED")

    def test_plate_inside_vehicle_is_attached_to_it(self):
        st = wp._new_track_state()
        st.update(bbox=[100, 150, 300, 260], vehicle_type="Car")
        wp._live_state[7] = st
        self.scan(self.PLATES)
        self.assertEqual(wp._live_state[7]["plate"], "OD02BA4455")
        out = wp._multi_plate_tracks_out("2026-01-01T10:00:00")
        # Shown on the car, not a second time as a loose plate.
        self.assertEqual([t["plate"] for t in out], ["21BH2345AA"])

    def test_plates_expire(self):
        self.scan(self.PLATES)
        for pt in wp._multi_plate_tracks.values():
            pt["last_seen"] -= wp.MULTI_PLATE_TTL_S + 1
        self.assertEqual(wp._multi_plate_tracks_out("t"), [])

    def test_scan_finishing_after_reset_is_discarded(self):
        stale_session = wp._multi_plate_state["session"]
        wp._reset_multi_plate()
        self.scan(self.PLATES, session=stale_session)
        self.assertEqual(wp._multi_plate_tracks_out("t"), [])
        self.assertFalse(wp._multi_plate_state["busy"])

    def test_only_one_scan_runs_at_a_time(self):
        sys.modules["requests"] = fake_requests(self.PLATES, delay=0.3)
        wp._maybe_start_multi_plate_scan(FRAME, "CAM_TEST", "t", "http://engine")
        self.assertTrue(wp._multi_plate_state["busy"])
        started = wp._multi_plate_state["last_start"]
        wp._multi_plate_state["last_start"] = 0.0      # interval passed, but a scan is still in flight
        wp._maybe_start_multi_plate_scan(FRAME, "CAM_TEST", "t", "http://engine")
        self.assertEqual(wp._multi_plate_state["last_start"], 0.0, "a second scan started while one was busy")
        deadline = time.time() + 5
        while wp._multi_plate_state["busy"] and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(wp._multi_plate_state["busy"])
        self.assertGreater(started, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
