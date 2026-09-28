"""
pcu.py — Passenger Car Unit (PCU) weighting for queue/density measurement.
===========================================================================
Feature 1 (Foundations): "Queue measurement in PCU: vehicles weighted by
type, not raw counts."

CityFlow vehicle IDs are shaped like "flow_<flowIndex>_<n>" where
<flowIndex> is the 0-based position of the flow entry that generated the
vehicle inside the engine's flow file (flow_5j.json). We keep a small map
from flow index -> vehicle class so heterogeneous flows (buses, trucks,
two-wheelers mixed in with cars) can be weighted correctly. Anything that
doesn't match a known flow index, or comes from a source with no flow
concept at all (e.g. the standalone demo simulator), is treated as a
standard car (PCU 1.0) so behaviour never regresses when type information
just isn't available.
"""

import re
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

# Standard Indian Roads Congress (IRC:106) style PCU factors, simplified
# to the vehicle classes this project actually models.
PCU_WEIGHTS: Dict[str, float] = {
    "bicycle": 0.3,
    "motorbike": 0.5,
    "scooter": 0.5,
    "auto-rickshaw": 0.8,
    "car": 1.0,
    "hatchback": 1.0,
    "sedan": 1.0,
    "suv": 1.2,
    "van": 1.2,
    "pickup": 1.2,
    "minibus": 2.0,
    "bus": 3.0,
    "truck": 3.0,
    "heavy": 3.0,
    "ambulance": 1.0,   # emergency cars/vans still occupy ~1 car's worth of queue space
}
DEFAULT_PCU = 1.0

# Which CityFlow flow-file entry (0-based index into flow_5j.json) represents
# which vehicle class. Flows not listed here default to "car". Kept separate
# from flow_5j.json itself so this file is the single source of truth for
# "what does flow index N mean" and can be unit tested without CityFlow.
FLOW_INDEX_VEHICLE_TYPE: Dict[int, str] = {}

_FLOW_ID_RE = re.compile(r"^flow_(\d+)_\d+$")


def register_flow_types(mapping: Dict[int, str]) -> None:
    """Replace the flow-index -> vehicle-type table (used at simulation startup)."""
    FLOW_INDEX_VEHICLE_TYPE.clear()
    FLOW_INDEX_VEHICLE_TYPE.update(mapping)


def pcu_weight(vehicle_type: Optional[str]) -> float:
    """PCU factor for a vehicle class name (case-insensitive). Unknown -> 1.0."""
    if not vehicle_type:
        return DEFAULT_PCU
    return PCU_WEIGHTS.get(str(vehicle_type).strip().lower(), DEFAULT_PCU)


def vehicle_type_from_id(vehicle_id: str) -> str:
    """Best-effort vehicle class for a CityFlow vehicle id such as 'flow_2_17'."""
    m = _FLOW_ID_RE.match(vehicle_id or "")
    if not m:
        return "car"
    return FLOW_INDEX_VEHICLE_TYPE.get(int(m.group(1)), "car")


def resolve_vehicle_type(vehicle_id: str, vehicle_info_map: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve a vehicle's class for PCU weighting. Prefers an explicit
    'vehicle_type' key on the vehicle's info dict (present when the caller
    already knows the type, e.g. from an ANPR classification bridged in),
    then falls back to decoding it from the CityFlow flow index.
    """
    info = (vehicle_info_map or {}).get(vehicle_id) or {}
    explicit = info.get("vehicle_type") or info.get("type")
    if explicit:
        return str(explicit)
    return vehicle_type_from_id(vehicle_id)


def weighted_count(
    vehicle_ids: Iterable[str],
    vehicle_info_map: Optional[Dict[str, Any]] = None,
    type_resolver: Optional[Callable[[str], str]] = None,
) -> Tuple[int, float]:
    """
    Returns (raw_count, pcu_weighted_count) for an iterable of vehicle ids.
    `type_resolver`, if given, overrides how a vehicle id maps to a class
    name (used by tests and by callers with their own type source).
    """
    resolver = type_resolver or (lambda vid: resolve_vehicle_type(vid, vehicle_info_map))
    raw = 0
    weighted = 0.0
    for vid in vehicle_ids:
        raw += 1
        weighted += pcu_weight(resolver(vid))
    return raw, round(weighted, 3)
