"""
V2X Simulation Engine — Road Edge Vehicle & Pedestrian Simulation.
Generates realistic vehicle telemetry (CAM/BSM) and VRU positions
for the 4-way intersection digital twin.
"""
import math
import time
import random
import uuid
from core.v2x_models import BSMMessage, VRUMessage
from core.gateway import V2XGateway
from core.sensor_fusion import LocalDynamicMap
from core.emergency import EmergencyCoordinator
from core.signal_controller import AdaptiveSignalController

# Canvas coordinates: 600x600 canvas, intersection center at (300,300)
CX, CY = 300, 300
ROAD_HALF = 40  # Half-width of road from center lane


class Vehicle:
    """A connected vehicle broadcasting BSM messages."""

    LANE_CONFIGS = {
        "N_IN":  {"start": (CX - 15, 0),     "target": (CX - 15, CY - ROAD_HALF), "heading": 180, "axis": "NS"},
        "S_IN":  {"start": (CX + 15, 600),   "target": (CX + 15, CY + ROAD_HALF), "heading": 0,   "axis": "NS"},
        "E_IN":  {"start": (600, CY - 15),   "target": (CX + ROAD_HALF, CY - 15), "heading": 270, "axis": "EW"},
        "W_IN":  {"start": (0, CY + 15),     "target": (CX - ROAD_HALF, CY + 15), "heading": 90,  "axis": "EW"},
    }

    def __init__(self, lane_id: str, vehicle_type: str = "civilian", speed: float = None):
        cfg = self.LANE_CONFIGS.get(lane_id, self.LANE_CONFIGS["N_IN"])
        sx, sy = cfg["start"]
        self.station_id = f"V-{vehicle_type[:3].upper()}-{str(uuid.uuid4())[:4]}"
        self.vehicle_type = vehicle_type
        self.x = float(sx + random.randint(-5, 5))
        self.y = float(sy + random.randint(-5, 5))
        self.speed = speed if speed else random.uniform(25, 45)  # pixels/sec
        self.base_speed = self.speed
        self.heading = cfg["heading"]
        self.lane_id = lane_id
        self.axis = cfg["axis"]
        self.target_x, self.target_y = cfg["target"]
        self.siren_active = (vehicle_type == "emergency")
        self.pki_signature = "EMERGENCY_CERT_ECC_384" if vehicle_type == "emergency" else "VALID_PKI_CERT_ECC_256"
        self.crossed = False      # Has passed through intersection
        self.waiting = False
        self.brake_active = False
        self.alive = True         # Remove when off canvas

    def _can_enter_intersection(self, signal_state: dict) -> bool:
        """Check if the signal for this vehicle's approach is green."""
        direction_map = {
            "N_IN": "north", "S_IN": "south",
            "E_IN": "east",  "W_IN": "west",
        }
        direction = direction_map.get(self.lane_id, "north")
        return signal_state.get(direction, "red") == "green"

    def update(self, dt: float, signal_state: dict, alerts: list[dict]):
        """Move vehicle, respecting signals and collision alerts."""
        if not self.alive:
            return

        # Check for alerts targeting this vehicle
        my_alerts = [a for a in alerts if a.get("target_vehicle_id") == self.station_id]
        if any(a["alert_type"] in ("FCW", "PCW") for a in my_alerts):
            self.brake_active = True
            self.speed = max(0, self.speed - 15 * dt)
        else:
            self.brake_active = False
            self.speed = min(self.base_speed, self.speed + 8 * dt)

        # Stop at red light if not yet crossed
        if not self.crossed:
            can_go = self._can_enter_intersection(signal_state)
            if not can_go and self._near_stop_line():
                self.waiting = True
                self.speed = 0
                return
            else:
                self.waiting = False

        # Move along heading
        rad = math.radians(self.heading)
        dx = math.sin(rad) * self.speed * dt
        dy = -math.cos(rad) * self.speed * dt
        self.x += dx
        self.y += dy

        # Mark as crossed once past intersection center
        if not self.crossed:
            dist_to_center = math.sqrt((self.x - CX)**2 + (self.y - CY)**2)
            if dist_to_center < ROAD_HALF:
                self.crossed = True

        # Remove when off canvas
        if self.x < -50 or self.x > 650 or self.y < -50 or self.y > 650:
            self.alive = False

    def _near_stop_line(self) -> bool:
        """Check if within stopping distance of intersection."""
        dist = math.sqrt((self.x - CX)**2 + (self.y - CY)**2)
        return dist < (ROAD_HALF + 50)

    def to_bsm(self) -> BSMMessage:
        return BSMMessage(
            station_id=self.station_id,
            vehicle_type=self.vehicle_type,
            x=round(self.x, 1),
            y=round(self.y, 1),
            speed=round(self.speed / 10, 2),  # convert to m/s (scaled)
            heading=self.heading,
            acceleration=0.0,
            lane_id=self.lane_id,
            brake_active=self.brake_active,
            siren_active=self.siren_active,
            pki_signature=self.pki_signature,
        )


class Pedestrian:
    """A VRU (pedestrian) crossing the intersection."""

    CROSSWALK_CONFIGS = [
        {"start": (CX - ROAD_HALF, CY - ROAD_HALF - 5), "end": (CX + ROAD_HALF, CY - ROAD_HALF - 5), "cw_id": "CW_NORTH"},
        {"start": (CX - ROAD_HALF, CY + ROAD_HALF + 5), "end": (CX + ROAD_HALF, CY + ROAD_HALF + 5), "cw_id": "CW_SOUTH"},
    ]

    def __init__(self):
        cfg = random.choice(self.CROSSWALK_CONFIGS)
        self.vru_id = f"P-{str(uuid.uuid4())[:4]}"
        self.x = float(cfg["start"][0])
        self.y = float(cfg["start"][1])
        self.target_x = float(cfg["end"][0])
        self.target_y = float(cfg["end"][1])
        self.crosswalk_id = cfg["cw_id"]
        self.speed = random.uniform(20, 30)  # pixels/sec
        self.alive = True

    def update(self, dt: float):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 3:
            self.alive = False
            return
        norm = dist
        self.x += (dx / norm) * self.speed * dt
        self.y += (dy / norm) * self.speed * dt

    def to_vru_msg(self) -> VRUMessage:
        heading = math.degrees(math.atan2(self.target_x - self.x, -(self.target_y - self.y))) % 360
        return VRUMessage(
            vru_id=self.vru_id,
            vru_type="pedestrian",
            x=round(self.x, 1),
            y=round(self.y, 1),
            speed=round(self.speed / 10, 2),
            heading=heading,
            crosswalk_id=self.crosswalk_id,
        )


class SimulationEngine:
    """
    Master simulation engine driving the V2X digital twin.
    Orchestrates vehicles, pedestrians, gateway, LDM, edge AI, and ASC.
    """

    def __init__(self, gateway: V2XGateway, ldm: LocalDynamicMap,
                 evp: EmergencyCoordinator, asc: AdaptiveSignalController):
        from core.edge_ai import EdgeAIInference
        self.gateway = gateway
        self.ldm = ldm
        self.evp = evp
        self.asc = asc
        self.edge_ai = EdgeAIInference(ldm)

        self.vehicles: list[Vehicle] = []
        self.pedestrians: list[Pedestrian] = []

        self.tick_count = 0
        self.spawn_timer = 0.0
        self.ped_spawn_timer = 0.0
        self.spawn_interval = 2.5       # seconds between vehicle spawns
        self.rush_hour = False          # Rush hour mode
        self.scenario_log: list[dict] = []

        self.kpis = {
            "total_vehicles_processed": 0,
            "avg_wait_time_sec": 0.0,
            "emergency_clearings": 0,
            "alerts_generated": 0,
            "cycles_completed": 0,
        }

        # Seed initial traffic
        self._spawn_vehicle("N_IN")
        self._spawn_vehicle("S_IN")
        self._spawn_vehicle("E_IN")
        self._spawn_vehicle("W_IN")

    def _spawn_vehicle(self, lane_id: str = None, vehicle_type: str = "civilian"):
        lanes = ["N_IN", "S_IN", "E_IN", "W_IN"]
        if lane_id is None:
            lane_id = random.choice(lanes)
        v = Vehicle(lane_id, vehicle_type=vehicle_type)
        self.vehicles.append(v)
        return v.station_id

    def _spawn_pedestrian(self):
        p = Pedestrian()
        self.pedestrians.append(p)

    def spawn_emergency_vehicle(self):
        """Inject an ambulance onto a random approach — triggers EVP."""
        lane = random.choice(["N_IN", "S_IN", "E_IN", "W_IN"])
        axis = Vehicle.LANE_CONFIGS[lane]["axis"]
        vid = self._spawn_vehicle(lane, vehicle_type="emergency")
        self.log_scenario(f"🚨 Ambulance {vid} spawned on {lane} approach")
        return vid

    def spawn_jaywalker(self):
        """Inject a pedestrian hazard (jaywalker) near intersection centre."""
        p = Pedestrian()
        # Place pedestrian right in the middle of active traffic
        p.x = float(CX + random.choice([-20, 20]))
        p.y = float(CY + random.choice([-20, 20]))
        p.target_x = float(CX - (p.x - CX))  # cross toward opposite side
        p.target_y = float(CY - (p.y - CY))
        self.pedestrians.append(p)
        self.log_scenario("⚠️ Jaywalker injected at intersection centre!")

    def set_rush_hour(self, enabled: bool):
        self.rush_hour = enabled
        self.spawn_interval = 0.8 if enabled else 2.5
        self.log_scenario(f"{'🚗 Rush hour traffic surge activated' if enabled else '✅ Normal traffic restored'}")

    def log_scenario(self, msg: str):
        self.scenario_log.append({"msg": msg, "time": time.time()})
        if len(self.scenario_log) > 50:
            self.scenario_log.pop(0)

    def tick(self, dt: float = 0.05) -> dict:
        """
        Main simulation tick — runs full V2X pipeline:
        1. Spawn vehicles/pedestrians
        2. Ingest BSM/VRU into gateway → LDM
        3. Edge AI inference
        4. ASC phase update
        5. Move vehicles/pedestrians respecting signals and alerts
        """
        self.tick_count += 1

        # --- Spawning ---
        self.spawn_timer += dt
        self.ped_spawn_timer += dt

        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            lanes = ["N_IN", "S_IN", "E_IN", "W_IN"]
            n_spawn = 2 if self.rush_hour else 1
            for _ in range(n_spawn):
                self._spawn_vehicle(random.choice(lanes))

        if self.ped_spawn_timer >= 6.0:
            self.ped_spawn_timer = 0
            if random.random() < 0.4:
                self._spawn_pedestrian()

        # --- Ingest BSMs into gateway → LDM ---
        for v in self.vehicles:
            bsm = v.to_bsm()
            msg = self.gateway.ingest_bsm(bsm)
            if msg:
                self.ldm.update_from_bsm(msg)

        # --- Ingest VRU messages ---
        for p in self.pedestrians:
            vru_msg = p.to_vru_msg()
            msg = self.gateway.ingest_vru(vru_msg)
            self.ldm.update_from_vru(msg)

        # --- Edge AI Inference ---
        alerts = self.edge_ai.run_inference()
        alerts_dict = self.edge_ai.get_active_alerts_dict()
        self.kpis["alerts_generated"] += len(alerts)

        # --- Emergency Preemption Detection ---
        if self.edge_ai.has_emergency_preemption():
            for a in alerts:
                if a.alert_type == "EVP_APPROACH":
                    # Determine route from vehicle lane
                    ev = self.ldm.vehicles.get(a.target_vehicle_id)
                    if ev:
                        axis = Vehicle.LANE_CONFIGS.get(ev.get("lane_id", "N_IN"), {}).get("axis", "NS")
                        self.evp.trigger_preemption(a.target_vehicle_id, axis)
        else:
            # Check if EVP vehicle has cleared
            if self.evp.preemption_active and self.evp.preempting_vehicle_id:
                ev = self.ldm.vehicles.get(self.evp.preempting_vehicle_id)
                if ev and (ev["x"] < -20 or ev["x"] > 620 or ev["y"] < -20 or ev["y"] > 620):
                    self.evp.release_preemption()
                    self.kpis["emergency_clearings"] += 1

        # --- Adaptive Signal Control tick ---
        spat = self.asc.tick(dt)
        signal_state = self.asc.get_signal_state()
        self.kpis["cycles_completed"] = self.asc.cycle_count

        # --- Move vehicles ---
        for v in self.vehicles:
            v.update(dt, signal_state, alerts_dict)

        # --- Move pedestrians ---
        for p in self.pedestrians:
            p.update(dt)

        # --- Clean up dead entities ---
        dead_vehicles = [v for v in self.vehicles if not v.alive]
        self.kpis["total_vehicles_processed"] += len(dead_vehicles)
        self.vehicles = [v for v in self.vehicles if v.alive]
        self.pedestrians = [p for p in self.pedestrians if p.alive]

        # Keep LDM clean: remove stale vehicles
        for dv in dead_vehicles:
            self.ldm.vehicles.pop(dv.station_id, None)

        return {
            "vehicles": [
                {
                    "id": v.station_id,
                    "x": v.x, "y": v.y,
                    "heading": v.heading,
                    "speed": round(v.speed, 1),
                    "type": v.vehicle_type,
                    "lane": v.lane_id,
                    "waiting": v.waiting,
                    "brake": v.brake_active,
                    "siren": v.siren_active,
                } for v in self.vehicles
            ],
            "pedestrians": [
                {"id": p.vru_id, "x": p.x, "y": p.y} for p in self.pedestrians
            ],
            "spat": spat,
            "signal_state": signal_state,
            "alerts": alerts_dict,
            "evp": self.evp.get_status(),
            "kpis": self.kpis.copy(),
            "recent_packets": self.gateway.get_recent_packets(8),
            "evp_events": self.evp.get_recent_events(5),
            "scenario_log": self.scenario_log[-5:],
            "gateway_stats": self.gateway.stats(),
        }
