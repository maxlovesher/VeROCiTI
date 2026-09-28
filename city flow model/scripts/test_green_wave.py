"""
test_green_wave.py — offline unit tests for Features 5, 6, 8
(green-wave corridor selection, queue prediction, speed advisory).

Run with: python "city flow model/scripts/test_green_wave.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_wave import select_corridors, corridor_offsets, QueuePredictor, green_speed_advisory

# Mirrors the real 5-junction layout in agent.py's MultiAgentCoordinator._setup_network.
ADJACENCY = {
    "J1": {"J3": "EW"},
    "J2": {"J3": "NS"},
    "J3": {"J4": "EW", "J5": "NS", "J1": "EW", "J2": "NS"},
    "J4": {"J3": "EW"},
    "J5": {"J3": "NS"},
}


class TestSelectCorridors(unittest.TestCase):
    def test_picks_highest_trip_corridor_first(self):
        od = [
            {"origin": "J1", "destination": "J4", "trips": 5},
            {"origin": "J2", "destination": "J5", "trips": 50},
        ]
        corridors = select_corridors(od, ADJACENCY, top_n=1)
        self.assertEqual(len(corridors), 1)
        self.assertEqual(corridors[0]["junctions"], ["J2", "J3", "J5"])
        self.assertEqual(corridors[0]["trips"], 50)

    def test_phase_sequence_matches_hops(self):
        od = [{"origin": "J1", "destination": "J4", "trips": 10}]
        corridors = select_corridors(od, ADJACENCY, top_n=1)
        self.assertEqual(corridors[0]["phase_sequence"], ["EW", "EW"])

    def test_unreachable_destination_is_skipped(self):
        od = [{"origin": "J1", "destination": "NOWHERE", "trips": 999}]
        corridors = select_corridors(od, ADJACENCY, top_n=3)
        self.assertEqual(corridors, [])

    def test_top_n_limits_results_and_dedupes(self):
        od = [
            {"origin": "J1", "destination": "J3", "trips": 30},
            {"origin": "J1", "destination": "J3", "trips": 10},  # duplicate path, lower trips
            {"origin": "J2", "destination": "J3", "trips": 20},
        ]
        corridors = select_corridors(od, ADJACENCY, top_n=2)
        self.assertEqual(len(corridors), 2)
        paths = [tuple(c["junctions"]) for c in corridors]
        self.assertEqual(len(set(paths)), 2)  # no duplicate path in the result


class TestCorridorOffsets(unittest.TestCase):
    def test_offsets_grow_with_distance(self):
        corridor = {"junctions": ["J2", "J3", "J5"]}
        distances = {("J2", "J3"): 400.0, ("J3", "J5"): 200.0}
        offsets = corridor_offsets(corridor, distances, travel_speed_kmph=36.0)  # 10 m/s
        self.assertEqual(offsets, [0.0, 40.0, 60.0])

    def test_missing_distance_defaults_to_zero_offset_contribution(self):
        corridor = {"junctions": ["J1", "J3"]}
        offsets = corridor_offsets(corridor, {}, travel_speed_kmph=36.0)
        self.assertEqual(offsets, [0.0, 0.0])


class TestQueuePredictor(unittest.TestCase):
    def test_empty_history_predicts_zero(self):
        self.assertEqual(QueuePredictor().predict(300), 0.0)

    def test_flat_history_predicts_same_value(self):
        qp = QueuePredictor(sample_interval_s=3.0)
        for _ in range(10):
            qp.record(5.0)
        self.assertAlmostEqual(qp.predict(300), 5.0, delta=0.01)
        self.assertEqual(qp.trend(), "STABLE")

    def test_rising_queue_extrapolates_upward(self):
        qp = QueuePredictor(sample_interval_s=3.0)
        for i in range(10):
            qp.record(i)   # 0,1,2,...,9 -> slope 1 per sample
        predicted = qp.predict(seconds_ahead=30)   # 10 samples ahead at 3s/sample
        self.assertGreater(predicted, 9.0)
        self.assertEqual(qp.trend(), "RISING")

    def test_falling_queue_never_predicts_negative(self):
        qp = QueuePredictor(sample_interval_s=3.0)
        for i in range(5, 0, -1):
            qp.record(i)   # 5,4,3,2,1 -> steep negative slope
        predicted = qp.predict(seconds_ahead=600)  # far ahead, should floor at 0
        self.assertEqual(predicted, 0.0)
        self.assertEqual(qp.trend(), "FALLING")

    def test_history_len_caps_window(self):
        qp = QueuePredictor(history_len=3, sample_interval_s=3.0)
        for v in [100, 100, 100, 1, 1, 1]:   # old high values should fall out of the window
            qp.record(v)
        self.assertAlmostEqual(qp.predict(0), 1.0, delta=0.01)

    def test_reset_clears_history(self):
        qp = QueuePredictor()
        qp.record(10)
        qp.reset()
        self.assertEqual(qp.predict(300), 0.0)


class TestGreenSpeedAdvisory(unittest.TestCase):
    def test_recommends_slower_speed_when_plenty_of_time(self):
        # 100m in 20s = 18 km/h required, well under the 50 km/h limit.
        result = green_speed_advisory(distance_m=100, time_to_green_s=20, speed_limit_kmph=50)
        self.assertAlmostEqual(result["advisory_kmph"], 18.0, delta=0.1)
        self.assertFalse(result["arrives_on_red"])

    def test_caps_at_speed_limit_and_flags_arriving_on_red(self):
        # 500m in 10s would need 180 km/h -> impossible/illegal, so cap at limit and flag red arrival.
        result = green_speed_advisory(distance_m=500, time_to_green_s=10, speed_limit_kmph=50)
        self.assertEqual(result["advisory_kmph"], 50.0)
        self.assertTrue(result["arrives_on_red"])

    def test_floors_at_minimum_speed(self):
        # 50m in 60s would need 3 km/h -> floor at min_kmph instead of telling anyone to crawl.
        result = green_speed_advisory(distance_m=50, time_to_green_s=60, speed_limit_kmph=50, min_kmph=15)
        self.assertEqual(result["advisory_kmph"], 15.0)
        self.assertFalse(result["arrives_on_red"])

    def test_already_green_returns_no_advisory(self):
        result = green_speed_advisory(distance_m=200, time_to_green_s=0)
        self.assertIsNone(result["advisory_kmph"])
        self.assertFalse(result["arrives_on_red"])

    def test_at_stop_line_returns_no_advisory(self):
        result = green_speed_advisory(distance_m=0, time_to_green_s=10)
        self.assertIsNone(result["advisory_kmph"])


class TestCoordinatorIntegration(unittest.TestCase):
    """compute_green_wave() end-to-end against the real 5-junction MultiAgentCoordinator."""

    def test_compute_green_wave_on_real_coordinator(self):
        from agent import MultiAgentCoordinator

        class FakeEngine:
            def get_lane_waiting_vehicle_count(self):
                return {}

            def get_lane_vehicle_count(self):
                return {}

            def get_current_time(self):
                return 0

            def set_tl_phase(self, junction_id, phase_idx):
                pass

        coordinator = MultiAgentCoordinator(FakeEngine())
        od = [
            {"origin": "J2", "destination": "J5", "trips": 40},
            {"origin": "J1", "destination": "J4", "trips": 5},
        ]
        result = coordinator.compute_green_wave(
            od, link_distances_m={("J2", "J3"): 300.0, ("J3", "J5"): 150.0}, travel_speed_kmph=36.0
        )
        self.assertEqual(result[0]["junctions"], ["J2", "J3", "J5"])
        self.assertEqual(result[0]["offsets_s"], [0.0, 30.0, 45.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
