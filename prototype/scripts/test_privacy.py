"""
test_privacy.py — offline unit tests for Feature 21 (privacy by design).

Run with: python "prototype/scripts/test_privacy.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identity
import privacy


class TestShouldDiscardFrame(unittest.TestCase):
    def test_ordinary_sighting_frame_is_discarded(self):
        self.assertTrue(privacy.should_discard_frame(is_violation=False))

    def test_violation_frame_is_kept(self):
        self.assertFalse(privacy.should_discard_frame(is_violation=True))


class TestWatchlistMatching(unittest.TestCase):
    def test_matches_by_hash_only(self):
        h = identity.hash_id("OD05XX9999", secret="s")
        self.assertTrue(privacy.match_watchlist_hash(h, [h, "HID_other"]))

    def test_no_match_for_absent_hash(self):
        h = identity.hash_id("OD05XX9999", secret="s")
        other = identity.hash_id("MH12DE1234", secret="s")
        self.assertFalse(privacy.match_watchlist_hash(h, [other]))

    def test_empty_hash_never_matches(self):
        self.assertFalse(privacy.match_watchlist_hash("", ["HID_something"]))


class TestRetention(unittest.TestCase):
    def test_non_violation_expires_quickly(self):
        self.assertTrue(privacy.should_retain_row(is_violation=False, age_days=0.5, non_violation_days=1))
        self.assertFalse(privacy.should_retain_row(is_violation=False, age_days=1.5, non_violation_days=1))

    def test_violation_kept_much_longer(self):
        self.assertTrue(privacy.should_retain_row(is_violation=True, age_days=60, violation_days=90))
        self.assertFalse(privacy.should_retain_row(is_violation=True, age_days=120, violation_days=90))

    def test_purge_expired_filters_rows(self):
        rows = [
            {"timestamp": "2026-01-01T00:00:00", "violation": "NONE"},        # old non-violation -> purged
            {"timestamp": "2026-01-09T12:00:00", "violation": "NONE"},        # recent non-violation -> kept
            {"timestamp": "2026-01-01T00:00:00", "violation": "RED_LIGHT"},   # old violation, within limit -> kept
        ]
        kept = privacy.purge_expired(rows, now_iso="2026-01-10T00:00:00", non_violation_days=1, violation_days=90)
        self.assertEqual(len(kept), 2)
        violations_kept = [r for r in kept if r["violation"] != "NONE"]
        self.assertEqual(len(violations_kept), 1)

    def test_purge_expired_keeps_unparseable_timestamps(self):
        rows = [{"timestamp": "not-a-date", "violation": "NONE"}]
        kept = privacy.purge_expired(rows, now_iso="2026-01-10T00:00:00")
        self.assertEqual(len(kept), 1)

    def test_purge_expired_treats_missing_violation_field_as_non_violation(self):
        rows = [{"timestamp": "2026-01-01T00:00:00"}]   # no "violation" key at all
        kept = privacy.purge_expired(rows, now_iso="2026-01-10T00:00:00", non_violation_days=1)
        self.assertEqual(kept, [])


class TestWhatWeStoreSummary(unittest.TestCase):
    def test_reflects_configured_retention_periods(self):
        summary = privacy.what_we_store_summary(non_violation_days=2, violation_days=45)
        self.assertEqual(summary["retention_limit_days"], {"non_violation": 2, "violation": 45})
        self.assertIn("45", summary["raw_frames"])
        self.assertIn("2", summary["detection_records"])

    def test_mentions_hashing_and_edge_matching(self):
        summary = privacy.what_we_store_summary()
        self.assertIn("hash", summary["identity"].lower())
        self.assertIn("edge", summary["watchlist_matching"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
