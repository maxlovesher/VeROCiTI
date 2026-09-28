"""
test_plate_match.py — offline unit tests for Feature 18 (fuzzy plate matching).

Run with: python "prototype/scripts/test_plate_match.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plate_match as pm


class TestFuzzySimilarity(unittest.TestCase):
    def test_identical_plates_score_one(self):
        self.assertEqual(pm.fuzzy_plate_similarity("OD02BA4455", "OD02BA4455"), 1.0)

    def test_zero_confusion_treated_as_letter_o(self):
        self.assertTrue(pm.fuzzy_plate_equal("OD02BA4455", "0D02BA4455"))

    def test_eight_b_confusion(self):
        self.assertTrue(pm.fuzzy_plate_equal("OD02BA4455", "OD02BA4455".replace("B", "8")))

    def test_one_i_confusion(self):
        self.assertTrue(pm.fuzzy_plate_equal("MH12DE1234", "MH12DEI234"))

    def test_genuinely_different_plates_score_low(self):
        score = pm.fuzzy_plate_similarity("OD02BA4455", "KA01AB1111")
        self.assertLess(score, 0.85)

    def test_different_length_scores_zero(self):
        self.assertEqual(pm.fuzzy_plate_similarity("OD02BA4455", "OD02BA445"), 0.0)

    def test_empty_input_scores_zero(self):
        self.assertEqual(pm.fuzzy_plate_similarity("", "OD02BA4455"), 0.0)

    def test_case_and_whitespace_insensitive(self):
        self.assertTrue(pm.fuzzy_plate_equal("od 02 ba 4455", "OD02BA4455"))


class TestTravelTimeFeasibility(unittest.TestCase):
    def test_short_distance_short_time_feasible(self):
        self.assertTrue(pm.travel_time_feasible(distance_km=5, minutes=10, max_kmph=140))

    def test_long_distance_short_time_infeasible(self):
        # 500km in 10 minutes -> 3000 km/h, impossible.
        self.assertFalse(pm.travel_time_feasible(distance_km=500, minutes=10, max_kmph=140))

    def test_zero_distance_any_time_feasible(self):
        self.assertTrue(pm.travel_time_feasible(distance_km=0, minutes=0))

    def test_positive_distance_zero_time_infeasible(self):
        self.assertFalse(pm.travel_time_feasible(distance_km=1, minutes=0))

    def test_exactly_at_the_limit_is_feasible(self):
        # 140km in 60 minutes = exactly 140 km/h.
        self.assertTrue(pm.travel_time_feasible(distance_km=140, minutes=60, max_kmph=140))


class TestMatchWithFallback(unittest.TestCase):
    KNOWN = ["OD02BA4455", "MH12DE1234", "KA01AB1111"]

    def test_exact_match(self):
        result = pm.match_with_fallback("OD02BA4455", self.KNOWN)
        self.assertTrue(result["matched"])
        self.assertEqual(result["method"], "EXACT")
        self.assertEqual(result["plate"], "OD02BA4455")

    def test_fuzzy_match_without_feasibility_inputs(self):
        result = pm.match_with_fallback("0D02BA4455", self.KNOWN)   # 0 instead of O
        self.assertTrue(result["matched"])
        self.assertEqual(result["method"], "FUZZY")
        self.assertEqual(result["plate"], "OD02BA4455")

    def test_fuzzy_match_rejected_by_infeasible_travel_time(self):
        result = pm.match_with_fallback(
            "0D02BA4455", self.KNOWN, distance_km=500, minutes=1, max_kmph=140
        )
        self.assertFalse(result["matched"])

    def test_fuzzy_match_accepted_with_feasible_travel_time(self):
        result = pm.match_with_fallback(
            "0D02BA4455", self.KNOWN, distance_km=5, minutes=10, max_kmph=140
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["method"], "FUZZY")

    def test_falls_back_to_appearance_when_no_text_match(self):
        candidates = [
            {"plate": "GHOST_A", "color": "White"},
            {"plate": "GHOST_B", "color": "Red"},
        ]

        def score_fn(c):
            return 0.9 if c["plate"] == "GHOST_B" else 0.3

        result = pm.match_with_fallback(
            "ZZ99ZZ9999", self.KNOWN,
            appearance_candidates=candidates, appearance_score_fn=score_fn, appearance_threshold=0.74,
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["method"], "APPEARANCE")
        self.assertEqual(result["plate"], "GHOST_B")

    def test_appearance_fallback_below_threshold_is_no_match(self):
        candidates = [{"plate": "GHOST_A", "color": "White"}]

        def score_fn(c):
            return 0.5

        result = pm.match_with_fallback(
            "ZZ99ZZ9999", self.KNOWN,
            appearance_candidates=candidates, appearance_score_fn=score_fn, appearance_threshold=0.74,
        )
        self.assertFalse(result["matched"])
        self.assertEqual(result["method"], "NONE")

    def test_no_match_and_no_appearance_candidates(self):
        result = pm.match_with_fallback("ZZ99ZZ9999", self.KNOWN)
        self.assertFalse(result["matched"])
        self.assertEqual(result["method"], "NONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
