"""
plate_match.py — Feature 18: fuzzy plate matching.
===========================================================================
"Fuzzy plate matching: handles 0/O, 8/B and 1/I confusion, plus a
travel-time feasibility check and a fallback to matching on appearance."

Three independent pieces, composed by match_with_fallback():
  1. Character-confusion-tolerant equality/similarity between two plate
     reads (OCR commonly confuses these pairs).
  2. A travel-time feasibility check — even a fuzzy text match is rejected
     if it would require an impossible average speed between two sightings
     (almost certainly two different vehicles, not the same one twice).
  3. A fallback to appearance similarity (colour/body-type/etc.) when the
     plate text doesn't confidently match at all — the caller supplies the
     actual similarity function (e.g. vehicle_reid.cosine_similarity over
     visual embeddings) so this module stays free of vision dependencies.

Pure logic, no DB/CV dependency, independently unit-testable.
"""

from typing import Callable, Dict, List, Optional

# Bidirectional OCR confusion pairs for Indian plates (upper-case, digits+letters mixed).
_CONFUSABLE_GROUPS = [{"0", "O"}, {"8", "B"}, {"1", "I"}]
_CONFUSABLE_OF: Dict[str, set] = {}
for _group in _CONFUSABLE_GROUPS:
    for _ch in _group:
        _CONFUSABLE_OF[_ch] = _group


def _clean(plate: str) -> str:
    return (plate or "").strip().upper().replace(" ", "").replace("-", "")


def _chars_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    return b in _CONFUSABLE_OF.get(a, set())


def fuzzy_plate_similarity(plate_a: str, plate_b: str) -> float:
    """
    Fraction of positions that match, treating confusable character pairs as
    equal. 0.0 if the cleaned plates have different lengths (a length
    mismatch means missed/extra characters, which this simple positional
    comparison can't meaningfully score).
    """
    a, b = _clean(plate_a), _clean(plate_b)
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for ca, cb in zip(a, b) if _chars_equal(ca, cb))
    return round(matches / len(a), 3)


def fuzzy_plate_equal(plate_a: str, plate_b: str, min_similarity: float = 0.85) -> bool:
    """True if two plate reads are the same plate, allowing for confusable-character OCR noise."""
    return fuzzy_plate_similarity(plate_a, plate_b) >= min_similarity


def travel_time_feasible(distance_km: float, minutes: float, max_kmph: float = 140.0) -> bool:
    """
    True if travelling `distance_km` in `minutes` doesn't require exceeding
    `max_kmph` (a generous ceiling — Indian arterial/highway speeds rarely
    exceed this even for a fast vehicle). minutes <= 0 with any positive
    distance is infeasible (can't cover ground in zero or negative time);
    minutes <= 0 with zero distance (same instant, same place) is feasible.
    """
    if distance_km <= 0:
        return True
    if minutes <= 0:
        return False
    required_kmph = distance_km / (minutes / 60.0)
    return required_kmph <= max_kmph


def match_with_fallback(
    candidate_plate: str,
    known_plates: List[str],
    distance_km: Optional[float] = None,
    minutes: Optional[float] = None,
    max_kmph: float = 140.0,
    min_similarity: float = 0.85,
    appearance_candidates: Optional[List[Dict]] = None,
    appearance_score_fn: Optional[Callable[[Dict], float]] = None,
    appearance_threshold: float = 0.74,
) -> Dict:
    """
    Tries, in order:
      1. Exact match against known_plates.
      2. Fuzzy (confusable-character) match against known_plates, validated
         by travel-time feasibility when distance_km/minutes are given.
      3. Appearance fallback: if appearance_candidates + appearance_score_fn
         are supplied, picks the best-scoring candidate at/above
         appearance_threshold (mirrors vehicle_reid.py's Re-ID threshold).

    Returns {"matched": bool, "method": "EXACT"|"FUZZY"|"APPEARANCE"|"NONE",
             "plate": str|None, "score": float, "candidate": dict|None}.
    """
    clean_candidate = _clean(candidate_plate)

    for known in known_plates:
        if _clean(known) == clean_candidate:
            return {"matched": True, "method": "EXACT", "plate": known, "score": 1.0, "candidate": None}

    best_fuzzy_plate, best_fuzzy_score = None, 0.0
    for known in known_plates:
        score = fuzzy_plate_similarity(clean_candidate, known)
        if score > best_fuzzy_score:
            best_fuzzy_plate, best_fuzzy_score = known, score

    if best_fuzzy_plate is not None and best_fuzzy_score >= min_similarity:
        feasible = True
        if distance_km is not None and minutes is not None:
            feasible = travel_time_feasible(distance_km, minutes, max_kmph)
        if feasible:
            return {"matched": True, "method": "FUZZY", "plate": best_fuzzy_plate, "score": best_fuzzy_score, "candidate": None}

    if appearance_candidates and appearance_score_fn:
        best_candidate, best_appearance_score = None, 0.0
        for candidate in appearance_candidates:
            score = appearance_score_fn(candidate)
            if score > best_appearance_score:
                best_candidate, best_appearance_score = candidate, score
        if best_candidate is not None and best_appearance_score >= appearance_threshold:
            return {
                "matched": True, "method": "APPEARANCE",
                "plate": best_candidate.get("plate"), "score": round(best_appearance_score, 3),
                "candidate": best_candidate,
            }

    return {"matched": False, "method": "NONE", "plate": None, "score": 0.0, "candidate": None}
