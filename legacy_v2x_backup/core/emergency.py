"""
Emergency Vehicle Priority (EVP) Coordination.
Handles route clearing and signal preemption for emergency vehicles.
"""
import time


class EmergencyCoordinator:
    """
    Coordinates emergency vehicle priority across the intersection.
    When an ambulance/fire/police vehicle is detected approaching with siren,
    it triggers a green-wave corridor on the preemption route.
    """

    def __init__(self):
        self.preemption_active: bool = False
        self.preempting_vehicle_id: str | None = None
        self.preemption_start: float | None = None
        self.preemption_route: str | None = None  # "NS" or "EW"
        self.events: list[dict] = []

    def trigger_preemption(self, vehicle_id: str, approach_direction: str):
        """
        Activate emergency preemption for a vehicle approaching from given direction.
        This will signal the ASC to hold NS or EW green.
        """
        if not self.preemption_active:
            self.preemption_active = True
            self.preempting_vehicle_id = vehicle_id
            self.preemption_route = approach_direction  # which axis to keep green
            self.preemption_start = time.time()
            event = {
                "event": "EVP_TRIGGERED",
                "vehicle_id": vehicle_id,
                "route": approach_direction,
                "timestamp": time.time(),
                "message": f"🚨 Emergency vehicle {vehicle_id} preemption on {approach_direction} corridor"
            }
            self.events.append(event)
            if len(self.events) > 50:
                self.events.pop(0)

    def release_preemption(self):
        """Release preemption once the emergency vehicle has cleared the intersection."""
        if self.preemption_active:
            event = {
                "event": "EVP_CLEARED",
                "vehicle_id": self.preempting_vehicle_id,
                "duration_sec": round(time.time() - (self.preemption_start or time.time()), 1),
                "timestamp": time.time(),
                "message": f"✅ Emergency corridor cleared — resuming normal signal cycle"
            }
            self.events.append(event)
            if len(self.events) > 50:
                self.events.pop(0)
        self.preemption_active = False
        self.preempting_vehicle_id = None
        self.preemption_route = None
        self.preemption_start = None

    def get_status(self) -> dict:
        return {
            "preemption_active": self.preemption_active,
            "vehicle_id": self.preempting_vehicle_id,
            "route": self.preemption_route,
            "elapsed_sec": round(time.time() - self.preemption_start, 1) if self.preemption_start else 0,
        }

    def get_recent_events(self, n: int = 5) -> list[dict]:
        return self.events[-n:]
