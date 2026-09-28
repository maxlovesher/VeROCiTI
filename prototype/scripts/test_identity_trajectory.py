"""
test_identity_trajectory.py — offline unit tests for Feature 2
(hashed trajectory events) and the hashing core of Feature 21 (privacy).

Run with: python "prototype/scripts/test_identity_trajectory.py"
"""

import os
import sys
from datetime import date
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identity
import trajectory_events as te

DAY1 = date(2026, 1, 1)
DAY2 = date(2026, 1, 2)


class TestIdentityHashing(unittest.TestCase):
    def test_same_plate_same_day_hashes_identically(self):
        h1 = identity.hash_id("OD02BA4455", day=DAY1, secret="s")
        h2 = identity.hash_id("od02ba4455", day=DAY1, secret="s")   # case/whitespace shouldn't matter
        self.assertEqual(h1, h2)

    def test_same_plate_different_day_hashes_differently(self):
        h1 = identity.hash_id("OD02BA4455", day=DAY1, secret="s")
        h2 = identity.hash_id("OD02BA4455", day=DAY2, secret="s")
        self.assertNotEqual(h1, h2)

    def test_different_plates_hash_differently(self):
        h1 = identity.hash_id("OD02BA4455", day=DAY1, secret="s")
        h2 = identity.hash_id("MH12DE1234", day=DAY1, secret="s")
        self.assertNotEqual(h1, h2)

    def test_hash_has_stable_prefix_and_length(self):
        h = identity.hash_id("OD02BA4455", day=DAY1, secret="s")
        self.assertTrue(h.startswith(identity.HASH_ID_PREFIX))
        self.assertEqual(len(h), len(identity.HASH_ID_PREFIX) + 32)

    def test_empty_value_returns_empty_string(self):
        self.assertEqual(identity.hash_id("", day=DAY1, secret="s"), "")

    def test_verify_same_identity(self):
        self.assertTrue(identity.verify_same_identity("OD02BA4455", "od02ba4455", day=DAY1, secret="s"))
        self.assertFalse(identity.verify_same_identity("OD02BA4455", "OD02BA4456", day=DAY1, secret="s"))

    def test_different_secret_changes_hash(self):
        h1 = identity.hash_id("OD02BA4455", day=DAY1, secret="secret-a")
        h2 = identity.hash_id("OD02BA4455", day=DAY1, secret="secret-b")
        self.assertNotEqual(h1, h2)


class TestBuildTrajectories(unittest.TestCase):
    def test_groups_and_sorts_by_time(self):
        events = [
            {"hashed_id": "A", "camera": "CAM_02", "time": "2026-01-01T10:05:00"},
            {"hashed_id": "A", "camera": "CAM_01", "time": "2026-01-01T10:00:00"},
            {"hashed_id": "B", "camera": "CAM_03", "time": "2026-01-01T10:02:00"},
        ]
        trajectories = te.build_trajectories(events)
        self.assertEqual(list(trajectories.keys()).__len__(), 2)
        self.assertEqual([e["camera"] for e in trajectories["A"]], ["CAM_01", "CAM_02"])
        self.assertEqual(len(trajectories["B"]), 1)

    def test_malformed_events_are_skipped(self):
        events = [
            {"hashed_id": "A", "camera": "CAM_01", "time": "2026-01-01T10:00:00"},
            {"hashed_id": "", "camera": "CAM_02", "time": "2026-01-01T10:01:00"},   # missing hashed_id
            {"hashed_id": "C", "camera": "CAM_03"},                                 # missing time
            {"camera": "CAM_04", "time": "2026-01-01T10:03:00"},                    # missing hashed_id key entirely
        ]
        trajectories = te.build_trajectories(events)
        self.assertEqual(set(trajectories.keys()), {"A"})

    def test_trajectory_for_single_identity(self):
        events = [
            {"hashed_id": "A", "camera": "CAM_01", "time": "2026-01-01T10:00:00"},
            {"hashed_id": "B", "camera": "CAM_02", "time": "2026-01-01T10:01:00"},
        ]
        self.assertEqual(len(te.trajectory_for(events, "A")), 1)
        self.assertEqual(te.trajectory_for(events, "NOPE"), [])


class TestEventsFromDetections(unittest.TestCase):
    def test_bridges_detection_rows_without_leaking_plate(self):
        detections = [
            {"plate": "OD02BA4455", "camera_id": "CAM_01", "timestamp": "2026-01-01T10:00:00"},
            {"plate": "od02ba4455", "camera_id": "CAM_02", "timestamp": "2026-01-01T10:05:00"},
        ]
        events = te.events_from_detections(detections, day=DAY1, secret="s")
        self.assertEqual(len(events), 2)
        # Same plate at two cameras -> same hashed_id, and never the raw plate text.
        self.assertEqual(events[0]["hashed_id"], events[1]["hashed_id"])
        for e in events:
            self.assertNotIn("plate", e)

        trajectories = te.build_trajectories(events)
        self.assertEqual(len(trajectories), 1)
        traj = list(trajectories.values())[0]
        self.assertEqual([e["camera"] for e in traj], ["CAM_01", "CAM_02"])

    def test_skips_rows_missing_required_fields(self):
        detections = [{"plate": "OD02BA4455", "camera_id": "CAM_01"}]  # no timestamp
        self.assertEqual(te.events_from_detections(detections, day=DAY1, secret="s"), [])


class TestDwellSeconds(unittest.TestCase):
    def test_single_point_trajectory_has_zero_dwell(self):
        self.assertEqual(te.dwell_seconds([{"time": "2026-01-01T10:00:00"}]), 0.0)

    def test_multi_point_dwell(self):
        traj = [{"time": "2026-01-01T10:00:00"}, {"time": "2026-01-01T10:05:30"}]
        self.assertEqual(te.dwell_seconds(traj), 330.0)

    def test_empty_trajectory(self):
        self.assertEqual(te.dwell_seconds([]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
