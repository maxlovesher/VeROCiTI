"""
test_analytics_features.py — offline unit tests for Features 22, 23, 24
(travel-time reliability/bottlenecks, unusual-demand alerts,
idle-time/emissions-saved estimate).

Run with: python "prototype/scripts/test_analytics_features.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import travel_time_analytics as tta
import demand_alerts as da
import emissions as em


class TestPercentile(unittest.TestCase):
    def test_median_of_odd_count(self):
        self.assertEqual(tta.percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_p95_of_larger_set(self):
        values = list(range(1, 101))   # 1..100
        self.assertAlmostEqual(tta.percentile(values, 95), 95.05, delta=0.5)

    def test_empty_list(self):
        self.assertEqual(tta.percentile([], 50), 0.0)

    def test_single_value(self):
        self.assertEqual(tta.percentile([7.0], 50), 7.0)


class TestRoadTravelTimeStats(unittest.TestCase):
    def test_computes_median_p95_and_reliability(self):
        stats = tta.road_travel_time_stats({"NH-16": [5, 5, 5, 5, 20]})
        road = stats["NH-16"]
        self.assertEqual(road["sample_size"], 5)
        self.assertGreater(road["p95"], road["median"])
        self.assertGreater(road["reliability_ratio"], 1.0)

    def test_consistent_road_has_reliability_near_one(self):
        stats = tta.road_travel_time_stats({"Steady Rd": [10, 10, 10, 10]})
        self.assertAlmostEqual(stats["Steady Rd"]["reliability_ratio"], 1.0, delta=0.01)

    def test_single_sample_has_no_reliability_ratio(self):
        stats = tta.road_travel_time_stats({"Rd": [10]})
        self.assertIsNone(stats["Rd"]["reliability_ratio"])

    def test_empty_samples_handled(self):
        stats = tta.road_travel_time_stats({"Rd": []})
        self.assertEqual(stats["Rd"]["sample_size"], 0)
        self.assertIsNone(stats["Rd"]["reliability_ratio"])

    def test_negative_and_non_numeric_samples_filtered(self):
        stats = tta.road_travel_time_stats({"Rd": [10, -5, 10, 10]})
        self.assertEqual(stats["Rd"]["sample_size"], 3)


class TestIdentifyBottlenecks(unittest.TestCase):
    def test_flags_unreliable_road_above_threshold(self):
        stats = {
            "Bottleneck Rd": {"median": 5, "p95": 20, "sample_size": 5, "reliability_ratio": 4.0},
            "Steady Rd": {"median": 10, "p95": 11, "sample_size": 5, "reliability_ratio": 1.1},
        }
        bottlenecks = tta.identify_bottlenecks(stats, min_samples=3, reliability_threshold=1.5)
        self.assertEqual([b["road"] for b in bottlenecks], ["Bottleneck Rd"])

    def test_excludes_roads_with_too_few_samples(self):
        stats = {"Sparse Rd": {"median": 5, "p95": 20, "sample_size": 2, "reliability_ratio": 4.0}}
        self.assertEqual(tta.identify_bottlenecks(stats, min_samples=3), [])

    def test_sorted_worst_first(self):
        stats = {
            "A": {"median": 5, "p95": 10, "sample_size": 5, "reliability_ratio": 2.0},
            "B": {"median": 5, "p95": 20, "sample_size": 5, "reliability_ratio": 4.0},
        }
        bottlenecks = tta.identify_bottlenecks(stats, min_samples=3, reliability_threshold=1.5)
        self.assertEqual([b["road"] for b in bottlenecks], ["B", "A"])


class TestUnusualDemand(unittest.TestCase):
    def test_normal_count_not_flagged(self):
        result = da.detect_unusual_demand(current_count=22, history=[20, 21, 19, 22, 20])
        self.assertFalse(result["unusual_demand"])

    def test_spike_is_flagged(self):
        result = da.detect_unusual_demand(current_count=90, history=[20, 21, 19, 22, 20])
        self.assertTrue(result["unusual_demand"])
        self.assertGreater(result["z_score"], 2.5)

    def test_flat_baseline_does_not_explode_on_tiny_bump(self):
        # std=0 baseline; a +1 bump shouldn't trigger a false "festival" alert.
        result = da.detect_unusual_demand(current_count=11, history=[10, 10, 10, 10])
        self.assertFalse(result["unusual_demand"])

    def test_empty_history_uses_zero_baseline(self):
        result = da.detect_unusual_demand(current_count=5, history=[])
        self.assertEqual(result["baseline_mean"], 0.0)

    def test_batch_check_flags_and_sorts(self):
        current = {"CAM_A": 20, "CAM_B": 90}
        history = {"CAM_A": [18, 20, 19, 21], "CAM_B": [20, 21, 19, 22]}
        flagged = da.check_unusual_demand_for_cameras(current, history)
        self.assertEqual([f["camera_id"] for f in flagged], ["CAM_B"])


class TestIdleTime(unittest.TestCase):
    def test_counts_only_idle_samples(self):
        speeds = [0, 0, 5, 20, 1, 0]
        self.assertEqual(em.idle_time_seconds(speeds, sample_interval_s=2.0, idle_threshold_kmph=2.0), 8.0)

    def test_no_idle_samples(self):
        self.assertEqual(em.idle_time_seconds([30, 40, 50]), 0.0)


class TestEmissionsSaved(unittest.TestCase):
    def test_saved_grams_scales_with_seconds_saved(self):
        grams = em.estimate_emissions_saved_grams(baseline_idle_seconds=100, actual_idle_seconds=40, vehicle_class="car")
        self.assertAlmostEqual(grams, 60 * em.idle_emission_factor("car"), delta=0.1)

    def test_heavier_vehicle_saves_more_for_same_idle_reduction(self):
        car_saved = em.estimate_emissions_saved_grams(100, 40, "car")
        truck_saved = em.estimate_emissions_saved_grams(100, 40, "truck")
        self.assertGreater(truck_saved, car_saved)

    def test_no_negative_savings_when_idle_increased(self):
        grams = em.estimate_emissions_saved_grams(baseline_idle_seconds=40, actual_idle_seconds=100, vehicle_class="car")
        self.assertEqual(grams, 0.0)

    def test_unknown_vehicle_class_uses_default_factor(self):
        self.assertEqual(em.idle_emission_factor("spaceship"), em.DEFAULT_IDLE_G_PER_S)

    def test_fleet_emissions_saved_aggregates_trips(self):
        trips = [
            {"vehicle_class": "car", "baseline_idle_seconds": 100, "actual_idle_seconds": 50},
            {"vehicle_class": "bus", "baseline_idle_seconds": 100, "actual_idle_seconds": 50},
        ]
        result = em.fleet_emissions_saved_grams(trips)
        self.assertEqual(result["trip_count"], 2)
        expected = em.estimate_emissions_saved_grams(100, 50, "car") + em.estimate_emissions_saved_grams(100, 50, "bus")
        self.assertAlmostEqual(result["total_grams_saved"], expected, delta=0.1)
        self.assertAlmostEqual(result["total_kg_saved"], expected / 1000.0, delta=0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
