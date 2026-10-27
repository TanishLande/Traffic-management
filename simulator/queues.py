"""
Multi-Queue buffer management and scheduler.
Supports Deficit Round Robin (DRR) / Weighted Fair Queueing (WFQ)
with dynamic weights and Active Queue Management (AQM) early drop thresholds.
"""
from collections import deque
from typing import List, Dict, Tuple
import random
from .packet import Packet
from config import NUM_CLASSES

class MultiQueueManager:
    def __init__(self, max_depth_per_queue: int = 120):
        self.max_depth = max_depth_per_queue
        # Queues per class
        self.queues: Dict[int, deque] = {c: deque() for c in range(NUM_CLASSES)}
        # Deficit counters for DRR
        self.deficit_counters: Dict[int, float] = {c: 0.0 for c in range(NUM_CLASSES)}
        # Dynamic early drop thresholds [0.5, 1.0] per class
        self.drop_thresholds: Dict[int, float] = {c: 0.95 for c in range(NUM_CLASSES)}
        # Statistics
        self.dropped_packets_count: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
        self.admitted_packets_count: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
        self.recent_drops: List[Packet] = []

    def set_drop_thresholds(self, thresholds: List[float]):
        """Set dynamic AQM thresholds from DL agent action [0.5, 1.0]."""
        for c in range(NUM_CLASSES):
            if c < len(thresholds):
                self.drop_thresholds[c] = float(max(0.3, min(1.0, thresholds[c])))

    def enqueue(self, packet: Packet, current_time: float) -> bool:
        """
        Attempts to enqueue a packet into its corresponding class queue.
        Returns True if admitted, False if dropped.
        """
        c = packet.traffic_class
        q = self.queues[c]
        current_len = len(q)
        max_allowed = self.max_depth
        thresh = self.drop_thresholds[c]

        # Check hard physical limit
        if current_len >= max_allowed:
            packet.is_dropped = True
            packet.drop_reason = "buffer_overflow"
            self.dropped_packets_count[c] += 1
            self.recent_drops.append(packet)
            if len(self.recent_drops) > 50:
                self.recent_drops.pop(0)
            return False

        # Active Queue Management early drop check
        # If queue occupancy > threshold * max_depth, calculate drop probability
        ratio = current_len / max_allowed
        if ratio > thresh:
            # Linear probabilistic drop up to max capacity
            p_drop = (ratio - thresh) / max(0.01, (1.0 - thresh))
            if random.random() < p_drop:
                packet.is_dropped = True
                packet.drop_reason = "aqm_congestion_drop"
                self.dropped_packets_count[c] += 1
                self.recent_drops.append(packet)
                if len(self.recent_drops) > 50:
                    self.recent_drops.pop(0)
                return False

        # Admitted
        packet.queue_enter_time = current_time
        q.append(packet)
        self.admitted_packets_count[c] += 1
        return True

    def schedule_and_dequeue(
        self,
        weights: List[float],
        link_capacity_bps: float,
        dt_seconds: float,
        current_time: float,
        propagation_delay_ms: float
    ) -> List[Packet]:
        """
        Dequeues packets using Deficit Weighted Round Robin based on dynamic weights.
        Total transmission budget = link_capacity_bps * dt_seconds (in bits).
        """
        total_bit_budget = link_capacity_bps * dt_seconds
        transmitted_packets: List[Packet] = []

        # Normalize weights to sum to 1.0
        w_sum = sum(weights)
        if w_sum <= 0:
            norm_weights = [1.0 / NUM_CLASSES] * NUM_CLASSES
        else:
            norm_weights = [w / w_sum for w in weights]

        # Allocate byte quantum per class
        for c in range(NUM_CLASSES):
            quantum_bits = total_bit_budget * norm_weights[c]
            self.deficit_counters[c] += quantum_bits

        # Round Robin servicing
        progress_made = True
        while progress_made:
            progress_made = False
            for c in range(NUM_CLASSES):
                q = self.queues[c]
                while q:
                    front_pkt: Packet = q[0]
                    pkt_bits = front_pkt.size_bytes * 8.0
                    if self.deficit_counters[c] >= pkt_bits:
                        pkt = q.popleft()
                        self.deficit_counters[c] -= pkt_bits
                        # Calculate departure time: transmission finish + propagation delay
                        tx_duration = pkt_bits / max(1.0, link_capacity_bps)
                        pkt.departure_time = current_time + tx_duration + (propagation_delay_ms / 1000.0)
                        transmitted_packets.append(pkt)
                        progress_made = True
                    else:
                        break

        # Bound deficit counters to prevent unbounded accumulation
        max_accumulated = total_bit_budget * 2.0
        for c in range(NUM_CLASSES):
            if len(self.queues[c]) == 0:
                self.deficit_counters[c] = 0.0
            else:
                self.deficit_counters[c] = min(max_accumulated, self.deficit_counters[c])

        return transmitted_packets

    def get_queue_lengths(self) -> Dict[int, int]:
        return {c: len(self.queues[c]) for c in range(NUM_CLASSES)}

    def get_queue_occupancy_ratios(self) -> Dict[int, float]:
        return {c: len(self.queues[c]) / self.max_depth for c in range(NUM_CLASSES)}

    def clear(self):
        for c in range(NUM_CLASSES):
            self.queues[c].clear()
            self.deficit_counters[c] = 0.0
            self.dropped_packets_count[c] = 0
            self.admitted_packets_count[c] = 0
        self.recent_drops.clear()
