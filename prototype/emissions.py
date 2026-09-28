"""
emissions.py — Feature 24: idle-time and emissions-saved estimate.
===========================================================================
"Idle-time and emissions-saved estimate." Converts a reduction in idle
time (e.g. from better signal timing — fewer/shorter unnecessary red waits)
into an estimated CO2 saving, using rough, clearly-labelled idle emission
factors. These factors are illustrative approximations for a dashboard
figure, not a certified emissions inventory.

Pure logic, no DB/Flask dependency, independently unit-testable.
"""

from typing import Dict, List

# Grams of CO2 per second of idling, by vehicle class — rough, illustrative
# figures (idling passenger car ≈ 2.6 g CO2/s is a commonly cited ballpark;
# heavier classes scaled up accordingly). NOT a certified emissions source.
IDLE_CO2_G_PER_S: Dict[str, float] = {
    "motorbike": 0.9,
    "scooter": 0.9,
    "auto-rickshaw": 1.4,
    "car": 2.6,
    "suv": 3.4,
    "van": 3.4,
    "bus": 8.0,
    "truck": 9.5,
    "heavy": 9.5,
}
DEFAULT_IDLE_G_PER_S = IDLE_CO2_G_PER_S["car"]

IDLE_SPEED_THRESHOLD_KMPH = 2.0   # at/below this, a sample counts as "idling" not "moving slowly"


def idle_emission_factor(vehicle_class: str) -> float:
    return IDLE_CO2_G_PER_S.get((vehicle_class or "").strip().lower(), DEFAULT_IDLE_G_PER_S)


def idle_time_seconds(speed_samples_kmph: List[float], sample_interval_s: float = 1.0, idle_threshold_kmph: float = IDLE_SPEED_THRESHOLD_KMPH) -> float:
    """Total time a vehicle spent idling, from a series of speed samples spaced sample_interval_s apart."""
    idle_samples = sum(1 for s in speed_samples_kmph if s <= idle_threshold_kmph)
    return round(idle_samples * sample_interval_s, 1)


def estimate_emissions_saved_grams(baseline_idle_seconds: float, actual_idle_seconds: float, vehicle_class: str = "car") -> float:
    """
    CO2 saved (grams) by idling less than a baseline scenario would have
    (e.g. "before" vs "after" adaptive signal timing for comparable trips).
    Returns 0.0 rather than a negative number if actual idle time was
    somehow higher than baseline — this function reports savings, not losses.
    """
    seconds_saved = max(0.0, baseline_idle_seconds - actual_idle_seconds)
    return round(seconds_saved * idle_emission_factor(vehicle_class), 1)


def fleet_emissions_saved_grams(trips: List[Dict]) -> Dict:
    """
    trips: [{"vehicle_class": str, "baseline_idle_seconds": float, "actual_idle_seconds": float}, ...]
    Returns {"total_grams_saved": float, "total_kg_saved": float, "trip_count": int}.
    """
    total_grams = 0.0
    for trip in trips:
        total_grams += estimate_emissions_saved_grams(
            trip.get("baseline_idle_seconds", 0.0),
            trip.get("actual_idle_seconds", 0.0),
            trip.get("vehicle_class", "car"),
        )
    return {
        "total_grams_saved": round(total_grams, 1),
        "total_kg_saved": round(total_grams / 1000.0, 3),
        "trip_count": len(trips),
    }
