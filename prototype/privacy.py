"""
privacy.py — Feature 21: privacy by design.
===========================================================================
"Privacy by design: HMAC hashing with a daily key, raw frames discarded,
watchlists matched at the edge, evidence kept only for violations, a
retention limit, a 'What we store' view."

The HMAC hashing piece lives in identity.py (shared with Feature 2). This
module covers the rest:
  - should_discard_frame(): raw frames are discarded unless the detection
    is a violation — evidence is kept only when there's a reason to.
  - match_watchlist_hash(): watchlist matching against hashed identities,
    so the raw plate never needs to leave the edge to check a hit.
  - purge_expired()/should_retain_row(): a retention limit, shorter for
    ordinary sightings than for violations.
  - what_we_store_summary(): a plain-language transparency view of the
    above, suitable for a "What we store" page.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

from datetime import datetime
from typing import Dict, Iterable, List, Optional

DEFAULT_NON_VIOLATION_RETENTION_DAYS = 1
DEFAULT_VIOLATION_RETENTION_DAYS = 90


def should_discard_frame(is_violation: bool) -> bool:
    """Raw frames are discarded unless this detection is a violation — evidence is kept only for a reason."""
    return not is_violation


def match_watchlist_hash(candidate_hash: str, watchlist_hashes: Iterable[str]) -> bool:
    """
    Edge watchlist matching: both sides are already hashed identities
    (identity.hash_id output) — no raw plate is needed to check a hit, and
    none is exposed by a match either.
    """
    return bool(candidate_hash) and candidate_hash in set(watchlist_hashes)


def should_retain_row(
    is_violation: bool,
    age_days: float,
    non_violation_days: float = DEFAULT_NON_VIOLATION_RETENTION_DAYS,
    violation_days: float = DEFAULT_VIOLATION_RETENTION_DAYS,
) -> bool:
    limit = violation_days if is_violation else non_violation_days
    return age_days <= limit


def purge_expired(
    rows: List[Dict],
    now_iso: str,
    non_violation_days: float = DEFAULT_NON_VIOLATION_RETENTION_DAYS,
    violation_days: float = DEFAULT_VIOLATION_RETENTION_DAYS,
) -> List[Dict]:
    """
    rows: detection-row-shaped dicts with "timestamp" and "violation"
    (database.py's convention — "NONE"/missing/falsy means not a violation).
    Returns only the rows that should be KEPT. A row whose timestamp can't
    be parsed is kept rather than dropped — this function decides retention,
    it shouldn't silently destroy data it's unsure about; a separate,
    explicit data-quality pass should handle malformed rows.
    """
    now = datetime.fromisoformat(now_iso)
    kept = []
    for row in rows:
        ts = row.get("timestamp")
        try:
            age_days = (now - datetime.fromisoformat(ts)).total_seconds() / 86400.0
        except (ValueError, TypeError):
            kept.append(row)
            continue
        is_violation = bool(row.get("violation")) and row.get("violation") != "NONE"
        if should_retain_row(is_violation, age_days, non_violation_days, violation_days):
            kept.append(row)
    return kept


def what_we_store_summary(
    non_violation_days: float = DEFAULT_NON_VIOLATION_RETENTION_DAYS,
    violation_days: float = DEFAULT_VIOLATION_RETENTION_DAYS,
) -> Dict:
    """Plain-language description of the data-handling rules above, for a public 'What we store' page."""
    return {
        "identity": (
            "Vehicles are tracked by an HMAC-SHA256 hash of the plate, keyed to the day. "
            "The same vehicle hashes the same way across cameras on one day, but hashes from "
            "different days can't be linked to each other, and the hash can't be reversed back "
            "to the plate without that day's key."
        ),
        "raw_frames": (
            f"Discarded immediately for ordinary sightings. Kept only when the detection is a "
            f"violation, for up to {violation_days} days."
        ),
        "detection_records": (
            f"Ordinary sighting records are retained {non_violation_days} day(s). "
            f"Violation records are retained up to {violation_days} days."
        ),
        "watchlist_matching": (
            "Performed against hashed identities at the edge — the watchlist never needs the "
            "raw plate text to check for a match."
        ),
        "retention_limit_days": {"non_violation": non_violation_days, "violation": violation_days},
    }
