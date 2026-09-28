"""
Sensor Fusion — Local Dynamic Map (LDM) Builder.
Merges BSM, VRU, and RSU data into a unified spatial state,
simulating 5G MEC stream processing (local dynamic map layer).
"""
import time
from core.v2x_models import BSMMessage, VRUMessage, RSUDetection


class LocalDynamicMap:
    """
    Maintains a live, fused spatial state of all V2X participants at the intersection.
    Updated by the V2X Gateway at every simulation tick.
    """

    def __init__(self):
        self.vehicles: dict[str, dict] = {}    # station_id -> latest state
        self.vrus: dict[str, dict] = {}        # vru_id -> latest state
        self.rsu_objects: dict[str, dict] = {} # object_id -> latest RSU detection
        self.last_update: float = time.time()

    def update_from_bsm(self, bsm_dict: dict):
        sid = bsm_dict["station_id"]
        self.vehicles[sid] = bsm_dict
        self.last_update = time.time()

    def update_from_vru(self, vru_dict: dict):
        vid = vru_dict["vru_id"]
        self.vrus[vid] = vru_dict
        self.last_update = time.time()

    def update_from_rsu(self, rsu_dict: dict):
        oid = rsu_dict["object_id"]
        self.rsu_objects[oid] = rsu_dict
        self.last_update = time.time()

    def get_vehicles_in_lane(self, lane_id: str) -> list[dict]:
        return [v for v in self.vehicles.values() if v.get("lane_id") == lane_id]

    def get_all_vehicles(self) -> list[dict]:
        return list(self.vehicles.values())

    def get_all_vrus(self) -> list[dict]:
        return list(self.vrus.values())

    def vehicle_count_by_direction(self) -> dict:
        """Returns vehicle queue counts per approach direction."""
        ns_count = sum(
            1 for v in self.vehicles.values()
            if v.get("lane_id", "").startswith(("N", "S"))
        )
        ew_count = sum(
            1 for v in self.vehicles.values()
            if v.get("lane_id", "").startswith(("E", "W"))
        )
        return {"NS": ns_count, "EW": ew_count}

    def snapshot(self) -> dict:
        return {
            "vehicles": list(self.vehicles.values()),
            "vrus": list(self.vrus.values()),
            "rsu_objects": list(self.rsu_objects.values()),
            "last_update": self.last_update,
        }
