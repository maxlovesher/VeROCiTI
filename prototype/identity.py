"""
identity.py — Privacy-preserving vehicle identity hashing.
===========================================================================
Foundation for Feature 2 ("trajectory events with hashed IDs") and the core
of Feature 21 ("Privacy by design: HMAC hashing with a daily key").

Instead of storing/matching raw plate text everywhere a vehicle needs to be
tracked across cameras, callers compute a keyed HMAC of the plate. The key
rotates daily, so:
  - the same vehicle gets the same hashed_id across cameras *within one day*
    (so trajectories/Re-ID still work), but
  - hashed_ids from different days can't be correlated with each other,
    and the hash can't be reversed back to the plate without the day's key
    (which itself is derived from a secret that should live outside source
    control in a real deployment — see IDENTITY_HASH_SECRET below).

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

import hashlib
import hmac
import os
from datetime import date as _date_type, datetime
from typing import Optional

# In a real deployment this MUST come from a secret manager / env var, never
# a literal in source. The fallback exists only so the system still runs
# (with a clearly weaker guarantee) in a dev/demo environment that hasn't
# set one — callers that care about real privacy guarantees should set
# IDENTITY_HASH_SECRET explicitly.
_DEFAULT_SECRET = "velociti-dev-only-change-me"
HASH_ID_PREFIX = "HID_"


def _secret() -> str:
    return os.environ.get("IDENTITY_HASH_SECRET", _DEFAULT_SECRET)


def _day_string(day: Optional[_date_type] = None) -> str:
    d = day or datetime.utcnow().date()
    return d.isoformat()


def daily_key(day: Optional[_date_type] = None, secret: Optional[str] = None) -> bytes:
    """HMAC-SHA256(secret, date) — a key that's stable for one calendar day (UTC) and changes the next."""
    base_secret = (secret if secret is not None else _secret()).encode("utf-8")
    day_str = _day_string(day).encode("utf-8")
    return hmac.new(base_secret, day_str, hashlib.sha256).digest()


def hash_id(value: str, day: Optional[_date_type] = None, secret: Optional[str] = None) -> str:
    """
    Keyed hash of `value` (typically a cleaned plate string) under the given
    day's key. Returns "HID_<32 hex chars>" — stable for repeated calls with
    the same value+day+secret, different for a different day.
    """
    if not value:
        return ""
    key = daily_key(day, secret)
    digest = hmac.new(key, value.strip().upper().encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{HASH_ID_PREFIX}{digest[:32]}"


def hashed_event(hashed_id: str, camera: str, time_iso: str) -> dict:
    """Constructs one {hashed_id, camera, time} event — the unit Feature 2's trajectories are built from."""
    return {"hashed_id": hashed_id, "camera": camera, "time": time_iso}


def verify_same_identity(value_a: str, value_b: str, day: Optional[_date_type] = None, secret: Optional[str] = None) -> bool:
    """True if two raw values (e.g. two plate reads) hash to the same identity on the given day."""
    return hash_id(value_a, day, secret) == hash_id(value_b, day, secret)
