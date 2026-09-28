"""
Adaptive Signal Controller (ASC).
Implements dynamic signal phase management based on:
  - Queue pressure from LDM vehicle counts
  - Emergency vehicle preemption requests
  - Green-wave corridor creation
  
Replaces fixed-time signaling with AI-driven RL-style phase optimization.
"""
import time
from core.sensor_fusion import LocalDynamicMap
from core.emergency import EmergencyCoordinator

# Timing constants (seconds)
PHASE_MIN_GREEN = 8       # Min green time before considering switch
PHASE_MAX_GREEN = 40      # Max green time before forced switch
PHASE_YELLOW = 3          # Yellow phase duration
EMERGENCY_GREEN = 30      # How long to hold green for emergency vehicle


class AdaptiveSignalController:
    """
    Adaptive signal controller for a 4-way intersection.
    Phase cycle: NS_GREEN → NS_YELLOW → EW_GREEN → EW_YELLOW → repeat
    With emergency preemption override.
    """

    PHASE_CYCLE = ["NS_GREEN", "NS_YELLOW", "EW_GREEN", "EW_YELLOW"]

    def __init__(self, ldm: LocalDynamicMap, evp: EmergencyCoordinator):
        self.ldm = ldm
        self.evp = evp
        self.current_phase: str = "NS_GREEN"
        self.phase_start_time: float = time.time()
        self.countdown: float = PHASE_MIN_GREEN
        self.cycle_count: int = 0
        self.ns_total_time: float = 0
        self.ew_total_time: float = 0
        self.phase_log: list[dict] = []

    def _phase_elapsed(self) -> float:
        return time.time() - self.phase_start_time

    def _set_phase(self, phase: str, duration: float):
        self.current_phase = phase
        self.phase_start_time = time.time()
        self.countdown = duration
        self.phase_log.append({"phase": phase, "duration": duration, "time": time.time()})
        if len(self.phase_log) > 100:
            self.phase_log.pop(0)

    def _compute_adaptive_duration(self, direction: str) -> float:
        """
        Queue-pressure-based green time computation.
        More vehicles in approach = longer green time (bounded).
        """
        counts = self.ldm.vehicle_count_by_direction()
        pressure = counts.get(direction, 0)
        # Base: 12s + 3s per vehicle, capped at MAX_GREEN
        duration = min(PHASE_MIN_GREEN + pressure * 3, PHASE_MAX_GREEN)
        return duration

    def tick(self, dt: float = 0.1) -> dict:
        """
        Called every simulation tick. Evaluates phase transitions.
        Returns current SPaT state dict.
        """
        elapsed = self._phase_elapsed()
        counts = self.ldm.vehicle_count_by_direction()

        # --- Emergency Preemption Override ---
        if self.evp.preemption_active:
            route = self.evp.preemption_route  # "NS" or "EW"
            target_phase = f"{route}_GREEN"
            if self.current_phase != "EMERGENCY_CORRIDOR_GREEN":
                self._set_phase("EMERGENCY_CORRIDOR_GREEN", EMERGENCY_GREEN)
            elif elapsed >= EMERGENCY_GREEN:
                # Hold until EVP is released by simulation
                self._set_phase("EMERGENCY_CORRIDOR_GREEN", EMERGENCY_GREEN)
            return self._spat_dict(counts)

        # --- Normal Adaptive Phase Logic ---
        if self.current_phase == "NS_GREEN":
            max_dur = self._compute_adaptive_duration("NS")
            if elapsed >= max_dur:
                self._set_phase("NS_YELLOW", PHASE_YELLOW)

        elif self.current_phase == "NS_YELLOW":
            if elapsed >= PHASE_YELLOW:
                dur = self._compute_adaptive_duration("EW")
                self._set_phase("EW_GREEN", dur)
                self.cycle_count += 1

        elif self.current_phase == "EW_GREEN":
            max_dur = self._compute_adaptive_duration("EW")
            if elapsed >= max_dur:
                self._set_phase("EW_YELLOW", PHASE_YELLOW)

        elif self.current_phase == "EW_YELLOW":
            if elapsed >= PHASE_YELLOW:
                dur = self._compute_adaptive_duration("NS")
                self._set_phase("NS_GREEN", dur)

        # If we came out of emergency, jump back to NS_GREEN
        elif self.current_phase == "EMERGENCY_CORRIDOR_GREEN":
            dur = self._compute_adaptive_duration("NS")
            self._set_phase("NS_GREEN", dur)

        return self._spat_dict(counts)

    def _spat_dict(self, counts: dict) -> dict:
        elapsed = self._phase_elapsed()
        remaining = max(0, self.countdown - elapsed)
        return {
            "intersection_id": "INT_MAIN_01",
            "current_phase": self.current_phase,
            "countdown": round(remaining, 1),
            "ns_queue": counts.get("NS", 0),
            "ew_queue": counts.get("EW", 0),
            "emergency_preemption_active": self.evp.preemption_active,
            "preempting_vehicle_id": self.evp.preempting_vehicle_id,
            "cycle_count": self.cycle_count,
        }

    def get_signal_state(self) -> dict:
        """Get current signal state (red/green/yellow) for each approach."""
        phase = self.current_phase
        evp = self.evp.preemption_active
        route = self.evp.preemption_route

        if evp:
            return {
                "north": "green" if route == "NS" else "red",
                "south": "green" if route == "NS" else "red",
                "east": "green" if route == "EW" else "red",
                "west": "green" if route == "EW" else "red",
            }
        elif phase == "NS_GREEN":
            return {"north": "green", "south": "green", "east": "red", "west": "red"}
        elif phase == "NS_YELLOW":
            return {"north": "yellow", "south": "yellow", "east": "red", "west": "red"}
        elif phase == "EW_GREEN":
            return {"north": "red", "south": "red", "east": "green", "west": "green"}
        elif phase == "EW_YELLOW":
            return {"north": "red", "south": "red", "east": "yellow", "west": "yellow"}
        else:
            return {"north": "red", "south": "red", "east": "red", "west": "red"}
