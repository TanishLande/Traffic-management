"""
Stochastic traffic generator for heterogeneous network traffic classes.
Models VoIP (CBR/jittered), Video (Pareto on/off), IoT (Poisson), and Best-Effort (heavy-tailed).
"""
import random
import numpy as np
from typing import List, Dict
from .packet import Packet
from config import (
    CLASS_VOIP, CLASS_VIDEO, CLASS_IOT, CLASS_BEST_EFFORT,
    TRAFFIC_CLASSES, NUM_CLASSES
)

class TrafficGenerator:
    def __init__(self, base_arrival_rates: Dict[int, float]):
        self.base_arrival_rates = dict(base_arrival_rates)
        self.packet_counter = 0
        self.burst_multipliers = {c: 1.0 for c in range(NUM_CLASSES)}
        self.burst_remaining_time = {c: 0.0 for c in range(NUM_CLASSES)}
        self.scenario_name = "Normal Traffic"

    def set_scenario(self, scenario: str):
        """Configure predefined network traffic scenarios."""
        self.scenario_name = scenario
        # Reset multipliers
        self.burst_multipliers = {c: 1.0 for c in range(NUM_CLASSES)}
        self.burst_remaining_time = {c: 0.0 for c in range(NUM_CLASSES)}
        
        if scenario == "voip_rush":
            self.burst_multipliers[CLASS_VOIP] = 3.5
            self.burst_remaining_time[CLASS_VOIP] = 10.0
        elif scenario == "video_surge":
            self.burst_multipliers[CLASS_VIDEO] = 2.8
            self.burst_remaining_time[CLASS_VIDEO] = 10.0
        elif scenario == "iot_storm":
            self.burst_multipliers[CLASS_IOT] = 4.0
            self.burst_remaining_time[CLASS_IOT] = 10.0
        elif scenario == "bulk_download":
            self.burst_multipliers[CLASS_BEST_EFFORT] = 3.2
            self.burst_remaining_time[CLASS_BEST_EFFORT] = 10.0
        elif scenario == "congested_all":
            for c in range(NUM_CLASSES):
                self.burst_multipliers[c] = 2.0
                self.burst_remaining_time[c] = 10.0

    def trigger_spike(self, class_id: int, multiplier: float = 3.0, duration: float = 5.0):
        """Dynamically inject an on-demand traffic surge into a specific class."""
        if 0 <= class_id < NUM_CLASSES:
            self.burst_multipliers[class_id] = max(1.0, multiplier)
            self.burst_remaining_time[class_id] = max(1.0, duration)

    def generate_packets_for_interval(self, current_time: float, dt_seconds: float) -> List[Packet]:
        """
        Generates packets arriving during time interval [current_time, current_time + dt_seconds].
        """
        generated_packets: List[Packet] = []

        # Decrement burst remaining times
        for c in range(NUM_CLASSES):
            if self.burst_remaining_time[c] > 0:
                self.burst_remaining_time[c] = max(0.0, self.burst_remaining_time[c] - dt_seconds)
                if self.burst_remaining_time[c] == 0:
                    self.burst_multipliers[c] = 1.0

        for class_info in TRAFFIC_CLASSES:
            c_id = class_info["id"]
            base_rate = self.base_arrival_rates.get(c_id, 200.0)
            effective_rate = base_rate * self.burst_multipliers[c_id]
            expected_count = effective_rate * dt_seconds

            # Traffic generation mechanism per class
            if c_id == CLASS_VOIP:
                # Periodic / Constant Bit Rate with slight Gaussian variance
                count = np.random.poisson(expected_count)
                for _ in range(count):
                    offset = random.uniform(0, dt_seconds)
                    size = int(np.random.normal(160, 10))
                    size = max(64, min(240, size))
                    self.packet_counter += 1
                    pkt = Packet(
                        packet_id=self.packet_counter,
                        traffic_class=c_id,
                        size_bytes=size,
                        creation_time=current_time + offset
                    )
                    generated_packets.append(pkt)

            elif c_id == CLASS_VIDEO:
                # Pareto distributed burstiness (I-frames vs P-frames)
                count = np.random.poisson(expected_count)
                for _ in range(count):
                    offset = random.uniform(0, dt_seconds)
                    is_iframe = (random.random() < 0.15)
                    if is_iframe:
                        size = int(np.random.normal(1480, 20))
                    else:
                        size = int(np.random.normal(900, 150))
                    size = max(400, min(1500, size))
                    self.packet_counter += 1
                    pkt = Packet(
                        packet_id=self.packet_counter,
                        traffic_class=c_id,
                        size_bytes=size,
                        creation_time=current_time + offset
                    )
                    generated_packets.append(pkt)

            elif c_id == CLASS_IOT:
                # Poisson arrival with small deterministic packets
                count = np.random.poisson(expected_count)
                for _ in range(count):
                    offset = random.uniform(0, dt_seconds)
                    size = int(np.random.normal(80, 8))
                    size = max(40, min(128, size))
                    self.packet_counter += 1
                    pkt = Packet(
                        packet_id=self.packet_counter,
                        traffic_class=c_id,
                        size_bytes=size,
                        creation_time=current_time + offset
                    )
                    generated_packets.append(pkt)

            elif c_id == CLASS_BEST_EFFORT:
                # Heavy-tailed bursty arrivals
                count = np.random.poisson(expected_count)
                for _ in range(count):
                    offset = random.uniform(0, dt_seconds)
                    size = 1500 if random.random() < 0.8 else random.randint(500, 1400)
                    self.packet_counter += 1
                    pkt = Packet(
                        packet_id=self.packet_counter,
                        traffic_class=c_id,
                        size_bytes=size,
                        creation_time=current_time + offset
                    )
                    generated_packets.append(pkt)

        # Sort all generated packets chronologically
        generated_packets.sort(key=lambda p: p.creation_time)
        return generated_packets
