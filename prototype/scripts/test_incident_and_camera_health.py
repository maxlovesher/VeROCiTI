"""
test_incident_and_camera_health.py — offline unit tests for Features 19
(incident detection) and 20 (camera health monitoring).

Run with: python "prototype/scripts/test_incident_and_camera_health.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incident_detect import IncidentDetector, detect_incident
from camera_health import CameraHealthMonitor, fallback_level_for


class TestIncidentDetector(unittest.TestCase):
    def test_arrival_clears_pending_departure(self):
        det = IncidentDetector(typical_travel_minutes=3.0, tolerance_minutes=1.0)
        det.record_departure("V1", "2026-01-01T10:00:00")
        det.record_arrival("V1")
        self.assertEqual(det.pending_count(), 0)
        self.assertEqual(det.overdue_vehicles("2026-01-01T10:10:00"), [])

    def test_vehicle_within_window_is_not_overdue(self):
        det = IncidentDetector(typical_travel_minutes=3.0, tolerance_minutes=1.0)
        det.record_departure("V1", "2026-01-01T10:00:00")
        self.assertEqual(det.overdue_vehicles("2026-01-01T10:03:30"), [])   # 3.5 min < 4 min deadline

    def test_vehicle_past_window_is_overdue(self):
        det = IncidentDetector(typical_travel_minutes=3.0, tolerance_minutes=1.0)
        det.record_departure("V1", "2026-01-01T10:00:00")
        self.assertEqual(det.overdue_vehicles("2026-01-01T10:05:00"), ["V1"])   # 5 min > 4 min deadline

    def test_multiple_vehicles_tracked_independently(self):
        det = IncidentDetector(typical_travel_minutes=3.0, tolerance_minutes=1.0)
        det.record_departure("V1", "2026-01-01T10:00:00")
        det.record_departure("V2", "2026-01-01T10:04:00")
        overdue = det.overdue_vehicles("2026-01-01T10:05:00")
        self.assertEqual(overdue, ["V1"])   # V2 only just departed, not overdue yet

    def test_clear_removes_all_pending(self):
        det = IncidentDetector(typical_travel_minutes=3.0)
        det.record_departure("V1", "2026-01-01T10:00:00")
        det.clear()
        self.assertEqual(det.pending_count(), 0)


class TestDetectIncident(unittest.TestCase):
    def test_missing_arrivals_and_rising_queue_is_an_incident(self):
        result = detect_incident(overdue_count=5, queue_trend="RISING", missing_threshold=3)
        self.assertTrue(result["incident_likely"])

    def test_missing_arrivals_alone_is_not_enough(self):
        result = detect_incident(overdue_count=5, queue_trend="STABLE", missing_threshold=3)
        self.assertFalse(result["incident_likely"])

    def test_rising_queue_alone_is_not_enough(self):
        result = detect_incident(overdue_count=1, queue_trend="RISING", missing_threshold=3)
        self.assertFalse(result["incident_likely"])

    def test_falling_queue_never_flags_even_with_missing_arrivals(self):
        result = detect_incident(overdue_count=10, queue_trend="FALLING", missing_threshold=3)
        self.assertFalse(result["incident_likely"])


class TestCameraHealthMonitor(unittest.TestCase):
    def test_recent_heartbeat_is_healthy(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        mon.heartbeat("CAM_01", "2026-01-01T10:00:00")
        self.assertTrue(mon.is_healthy("CAM_01", "2026-01-01T10:00:20"))

    def test_stale_heartbeat_is_unhealthy(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        mon.heartbeat("CAM_01", "2026-01-01T10:00:00")
        self.assertFalse(mon.is_healthy("CAM_01", "2026-01-01T10:01:00"))

    def test_never_seen_camera_is_unhealthy(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        self.assertFalse(mon.is_healthy("CAM_GHOST", "2026-01-01T10:00:00"))

    def test_unhealthy_cameras_lists_only_failed_ones(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        mon.heartbeat("CAM_01", "2026-01-01T10:00:00")
        mon.heartbeat("CAM_02", "2026-01-01T09:00:00")   # stale
        unhealthy = mon.unhealthy_cameras(["CAM_01", "CAM_02", "CAM_03"], "2026-01-01T10:00:10")
        self.assertEqual(set(unhealthy), {"CAM_02", "CAM_03"})

    def test_fleet_health_ratio(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        mon.heartbeat("CAM_01", "2026-01-01T10:00:00")
        mon.heartbeat("CAM_02", "2026-01-01T10:00:00")
        ratio = mon.fleet_health_ratio(["CAM_01", "CAM_02", "CAM_03", "CAM_04"], "2026-01-01T10:00:05")
        self.assertEqual(ratio, 0.5)

    def test_empty_fleet_is_fully_healthy_by_convention(self):
        mon = CameraHealthMonitor()
        self.assertEqual(mon.fleet_health_ratio([], "2026-01-01T10:00:00"), 1.0)

    def test_forget_removes_camera_from_tracking(self):
        mon = CameraHealthMonitor(timeout_s=30.0)
        mon.heartbeat("CAM_01", "2026-01-01T10:00:00")
        mon.forget("CAM_01")
        self.assertFalse(mon.is_healthy("CAM_01", "2026-01-01T10:00:05"))


class TestFallbackLevel(unittest.TestCase):
    def test_full_coverage_above_threshold(self):
        self.assertEqual(fallback_level_for(0.95), "FULL_COVERAGE")

    def test_partial_coverage_in_between(self):
        self.assertEqual(fallback_level_for(0.7), "PARTIAL_COVERAGE")

    def test_minimal_coverage_below_threshold(self):
        self.assertEqual(fallback_level_for(0.2), "MINIMAL_COVERAGE")

    def test_boundary_values_are_inclusive(self):
        self.assertEqual(fallback_level_for(0.9, full_threshold=0.9), "FULL_COVERAGE")
        self.assertEqual(fallback_level_for(0.5, partial_threshold=0.5), "PARTIAL_COVERAGE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
