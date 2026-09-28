"""
Edge AI Inference Engine.
Runs collision risk computation and generates safety alerts
within the < 20ms latency budget of 5G MEC.
Algorithms:
  - TTC (Time-to-Collision) for forward collision warning
  - Proximity check for pedestrian collision warning
  - Emergency vehicle approach detection
"""
import math
import time
import uuid
from core.v2x_models import SafetyAlert
from core.sensor_fusion import LocalDynamicMap


# Safety thresholds
TTC_FCW_THRESHOLD_SEC = 3.5     # Forward Collision Warning threshold
PCW_DISTANCE_THRESHOLD_M = 12.0 # Pedestrian proximity danger zone (metres)
EVP_APPROACH_THRESHOLD_M = 80.0 # Emergency vehicle preemption trigger (metres)
INTERSECTION_CENTER = (300, 300) # Canvas coordinates of intersection centre


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def _heading_vec(heading_deg: float) -> tuple[float, float]:
    """Convert heading (0=N, 90=E) to a unit direction vector."""
    rad = math.radians(heading_deg)
    return math.sin(rad), -math.cos(rad)  # x, y (canvas y-axis flipped)


def _compute_ttc(v1: dict, v2: dict) -> float | None:
    """
    Simplified 1-D TTC between two vehicles moving in the same lane/heading.
    Returns None if no collision is predicted.
    """
    if abs(v1.get("heading", 0) - v2.get("heading", 0)) > 30:
        return None  # not same direction
    dx = v2["x"] - v1["x"]
    dy = v2["y"] - v1["y"]
    dist = math.sqrt(dx * dx + dy * dy)
    relative_speed = v1.get("speed", 0) - v2.get("speed", 0)
    if relative_speed <= 0:
        return None  # leader is faster, no collision
    ttc = dist / relative_speed
    return ttc if ttc > 0 else None


class EdgeAIInference:
    """
    Edge AI module running on 5G MEC node.
    Evaluates all LDM participants every tick and emits safety alerts.
    """

    def __init__(self, ldm: LocalDynamicMap):
        self.ldm = ldm
        self.active_alerts: list[SafetyAlert] = []

    def run_inference(self) -> list[SafetyAlert]:
        t0 = time.perf_counter()
        alerts: list[SafetyAlert] = []
        vehicles = self.ldm.get_all_vehicles()
        vrus = self.ldm.get_all_vrus()

        # --- 1. Forward Collision Warning (FCW) ---
        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i + 1:]:
                ttc = _compute_ttc(v1, v2)
                if ttc is not None and ttc < TTC_FCW_THRESHOLD_SEC:
                    alerts.append(SafetyAlert(
                        alert_type="FCW",
                        severity="CRITICAL" if ttc < 1.5 else "WARNING",
                        target_vehicle_id=v1["station_id"],
                        hazard_entity_id=v2["station_id"],
                        ttc_seconds=round(ttc, 2),
                        recommended_action=f"BRAKE — collision in {ttc:.1f}s",
                        inference_latency_ms=0,
                    ))

        # --- 2. Pedestrian Collision Warning (PCW) ---
        for vru in vrus:
            for v in vehicles:
                dist = _distance(v["x"], v["y"], vru["x"], vru["y"])
                if dist < PCW_DISTANCE_THRESHOLD_M:
                    alerts.append(SafetyAlert(
                        alert_type="PCW",
                        severity="CRITICAL",
                        target_vehicle_id=v["station_id"],
                        hazard_entity_id=vru["vru_id"],
                        ttc_seconds=None,
                        recommended_action="STOP — pedestrian in path",
                        inference_latency_ms=0,
                    ))

        # --- 3. Emergency Vehicle Preemption Alert (EVP_APPROACH) ---
        for v in vehicles:
            if v.get("vehicle_type") == "emergency" and v.get("siren_active"):
                cx, cy = INTERSECTION_CENTER
                dist = _distance(v["x"], v["y"], cx, cy)
                if dist < EVP_APPROACH_THRESHOLD_M:
                    alerts.append(SafetyAlert(
                        alert_type="EVP_APPROACH",
                        severity="CRITICAL",
                        target_vehicle_id=v["station_id"],
                        hazard_entity_id=None,
                        ttc_seconds=None,
                        recommended_action="YIELD — emergency vehicle approaching",
                        inference_latency_ms=0,
                    ))

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        for a in alerts:
            a.inference_latency_ms = latency_ms

        self.active_alerts = alerts
        return alerts

    def has_emergency_preemption(self) -> bool:
        return any(a.alert_type == "EVP_APPROACH" for a in self.active_alerts)

    def get_active_alerts_dict(self) -> list[dict]:
        return [a.model_dump() for a in self.active_alerts]
