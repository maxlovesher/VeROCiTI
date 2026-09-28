"""
test_enforcement.py — offline unit tests for Features 12–17
(automated enforcement checks).

Run with: python "prototype/scripts/test_enforcement.py"
"""

import os
import sys
from datetime import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import enforcement as enf

# Two points roughly 1.11 km apart (0.01 degrees latitude ~ 1.11 km).
LAT1, LON1 = 20.30, 85.82
LAT2, LON2 = 20.31, 85.82


class TestSectionSpeed(unittest.TestCase):
    def test_computes_plausible_speed(self):
        # ~1.11 km in 60s -> ~66.6 km/h
        speed = enf.section_speed_kmph(LAT1, LON1, "2026-01-01T10:00:00", LAT2, LON2, "2026-01-01T10:01:00")
        self.assertAlmostEqual(speed, 66.6, delta=1.0)

    def test_flags_violation_over_limit(self):
        result = enf.check_section_speed(LAT1, LON1, "2026-01-01T10:00:00", LAT2, LON2, "2026-01-01T10:01:00", limit_kmph=50.0)
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "SECTION_SPEED")

    def test_no_violation_under_limit(self):
        result = enf.check_section_speed(LAT1, LON1, "2026-01-01T10:00:00", LAT2, LON2, "2026-01-01T10:03:00", limit_kmph=50.0)
        self.assertFalse(result["violation"])

    def test_non_positive_elapsed_time_returns_none(self):
        self.assertIsNone(enf.section_speed_kmph(LAT1, LON1, "2026-01-01T10:01:00", LAT2, LON2, "2026-01-01T10:00:00"))

    def test_invalid_timestamp_returns_none(self):
        self.assertIsNone(enf.section_speed_kmph(LAT1, LON1, "not-a-time", LAT2, LON2, "2026-01-01T10:01:00"))


class TestRedLightViolation(unittest.TestCase):
    def test_crossing_well_after_red_is_a_violation(self):
        result = enf.check_red_light_violation("2026-01-01T10:00:05", "2026-01-01T10:00:00", grace_period_s=1.0)
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "RED_LIGHT_VIOLATION")

    def test_crossing_within_grace_period_is_not_a_violation(self):
        result = enf.check_red_light_violation("2026-01-01T10:00:00.5", "2026-01-01T10:00:00", grace_period_s=1.0)
        self.assertFalse(result["violation"])

    def test_crossing_before_red_is_not_a_violation(self):
        result = enf.check_red_light_violation("2026-01-01T09:59:59", "2026-01-01T10:00:00")
        self.assertFalse(result["violation"])


class TestWrongWay(unittest.TestCase):
    def test_direction_not_allowed_is_a_violation(self):
        result = enf.check_wrong_way("S", ["N"])
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "WRONG_WAY")

    def test_direction_allowed_is_not_a_violation(self):
        result = enf.check_wrong_way("N", ["N", "NE"])
        self.assertFalse(result["violation"])

    def test_case_insensitive(self):
        result = enf.check_wrong_way("n", ["N"])
        self.assertFalse(result["violation"])

    def test_missing_direction_is_not_flagged(self):
        result = enf.check_wrong_way("", ["N"])
        self.assertFalse(result["violation"])


class TestHeavyVehicleWindow(unittest.TestCase):
    NIGHT_ONLY = [(time(23, 0), time(6, 0))]   # wraps past midnight

    def test_car_never_violates(self):
        result = enf.check_heavy_vehicle_window("Car", "2026-01-01T14:00:00", self.NIGHT_ONLY)
        self.assertFalse(result["violation"])

    def test_truck_during_day_violates(self):
        result = enf.check_heavy_vehicle_window("Truck", "2026-01-01T14:00:00", self.NIGHT_ONLY)
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "HEAVY_VEHICLE_RESTRICTED_HOURS")

    def test_truck_during_permitted_night_window_ok(self):
        result = enf.check_heavy_vehicle_window("Truck", "2026-01-01T23:30:00", self.NIGHT_ONLY)
        self.assertFalse(result["violation"])

    def test_truck_during_permitted_early_morning_ok(self):
        result = enf.check_heavy_vehicle_window("Truck", "2026-01-01T05:00:00", self.NIGHT_ONLY)
        self.assertFalse(result["violation"])

    def test_bus_is_treated_as_heavy(self):
        result = enf.check_heavy_vehicle_window("Bus", "2026-01-01T14:00:00", self.NIGHT_ONLY)
        self.assertTrue(result["violation"])


class TestSchoolZoneSpeed(unittest.TestCase):
    MORNING_ZONE = [(time(7, 30), time(9, 0), 25.0)]

    def test_speeding_inside_active_window_violates(self):
        result = enf.check_school_zone_speed(40.0, "2026-01-01T08:00:00", self.MORNING_ZONE)
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "SCHOOL_ZONE_SPEED")

    def test_same_speed_outside_window_ok(self):
        result = enf.check_school_zone_speed(40.0, "2026-01-01T12:00:00", self.MORNING_ZONE)
        self.assertFalse(result["violation"])

    def test_within_limit_inside_window_ok(self):
        result = enf.check_school_zone_speed(20.0, "2026-01-01T08:00:00", self.MORNING_ZONE)
        self.assertFalse(result["violation"])


class TestJunctionMouthParking(unittest.TestCase):
    def test_short_dwell_ok(self):
        self.assertFalse(enf.check_junction_mouth_parking(30.0, threshold_seconds=90.0)["violation"])

    def test_long_dwell_violates(self):
        result = enf.check_junction_mouth_parking(120.0, threshold_seconds=90.0)
        self.assertTrue(result["violation"])
        self.assertEqual(result["type"], "JUNCTION_MOUTH_PARKING")


if __name__ == "__main__":
    unittest.main(verbosity=2)
