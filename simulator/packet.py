"""
Packet abstraction for realistic network queuing simulation.
Tracks creation time, queue dwell time, transmission time, and drop events.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class Packet:
    packet_id: int
    traffic_class: int
    size_bytes: int
    creation_time: float          # Simulation time when generated (seconds)
    queue_enter_time: float = 0.0 # Time when packet entered queue
    departure_time: float = 0.0   # Time when fully transmitted
    is_dropped: bool = False
    drop_reason: Optional[str] = None  # "buffer_overflow", "aqm_drop", "deadline_expired"

    @property
    def queueing_delay(self) -> float:
        """Queue waiting time in milliseconds."""
        if self.departure_time >= self.queue_enter_time and self.queue_enter_time > 0:
            return (self.departure_time - self.queue_enter_time) * 1000.0
        return 0.0

    @property
    def end_to_end_delay(self) -> float:
        """Total latency in milliseconds from creation to departure."""
        if self.departure_time >= self.creation_time and not self.is_dropped:
            return (self.departure_time - self.creation_time) * 1000.0
        return 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.packet_id,
            "class": self.traffic_class,
            "size_bytes": self.size_bytes,
            "creation_time": round(self.creation_time, 4),
            "delay_ms": round(self.end_to_end_delay, 2),
            "is_dropped": self.is_dropped,
            "drop_reason": self.drop_reason
        }
