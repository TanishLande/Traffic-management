"""
Configuration parameters for Multi-Objective Deep Learning QoS Framework.
Defines network topology, traffic classes, SLA targets, and DL hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List

# Traffic Class IDs
CLASS_VOIP = 0
CLASS_VIDEO = 1
CLASS_IOT = 2
CLASS_BEST_EFFORT = 3

TRAFFIC_CLASSES = [
    {
        "id": CLASS_VOIP,
        "name": "VoIP / Voice",
        "description": "Ultra-low delay and jitter critical audio traffic",
        "avg_packet_size_bytes": 160,
        "target_delay_ms": 25.0,
        "target_jitter_ms": 5.0,
        "target_loss_ratio": 0.01,
        "priority_weight": 0.25,
        "color": "#10B981"  # Emerald green
    },
    {
        "id": CLASS_VIDEO,
        "name": "Video Streaming",
        "description": "High bandwidth, bursty video conferencing frames",
        "avg_packet_size_bytes": 1280,
        "target_delay_ms": 60.0,
        "target_jitter_ms": 15.0,
        "target_loss_ratio": 0.015,
        "priority_weight": 0.35,
        "color": "#3B82F6"  # Blue
    },
    {
        "id": CLASS_IOT,
        "name": "Mission-Critical IoT",
        "description": "Deterministic, zero-loss control telemetry",
        "avg_packet_size_bytes": 80,
        "target_delay_ms": 20.0,
        "target_jitter_ms": 4.0,
        "target_loss_ratio": 0.001,
        "priority_weight": 0.20,
        "color": "#F59E0B"  # Amber
    },
    {
        "id": CLASS_BEST_EFFORT,
        "name": "Best Effort (FTP/Web)",
        "description": "Elastic bulk transfer, delay tolerant",
        "avg_packet_size_bytes": 1500,
        "target_delay_ms": 120.0,
        "target_jitter_ms": 30.0,
        "target_loss_ratio": 0.05,
        "priority_weight": 0.20,
        "color": "#8B5CF6"  # Purple
    }
]

NUM_CLASSES = len(TRAFFIC_CLASSES)

# QoS Objective Indices
OBJ_THROUGHPUT = 0
OBJ_DELAY = 1
OBJ_JITTER = 2
OBJ_LOSS = 3
OBJ_FAIRNESS = 4

OBJECTIVES = [
    {"id": OBJ_THROUGHPUT, "name": "Throughput", "unit": "Mbps", "type": "maximize", "color": "#06B6D4"},
    {"id": OBJ_DELAY, "name": "Delay", "unit": "ms", "type": "minimize", "color": "#EF4444"},
    {"id": OBJ_JITTER, "name": "Jitter", "unit": "ms", "type": "minimize", "color": "#F97316"},
    {"id": OBJ_LOSS, "name": "Packet Loss", "unit": "%", "type": "minimize", "color": "#EC4899"},
    {"id": OBJ_FAIRNESS, "name": "Fairness (JFI)", "unit": "ratio", "type": "maximize", "color": "#10B981"}
]

NUM_OBJECTIVES = len(OBJECTIVES)

@dataclass
class NetworkConfig:
    # Router Link Specifications
    link_capacity_mbps: float = 40.0       # Bottleneck capacity
    propagation_delay_ms: float = 5.0      # Base propagation delay
    max_queue_depth_packets: int = 120     # Max buffer per class queue
    time_step_ms: float = 50.0             # Control decision interval (50ms)
    
    # Traffic Generation Defaults (packets per second mean)
    base_arrival_rates: Dict[int, float] = field(default_factory=lambda: {
        CLASS_VOIP: 180.0,         # ~230 kbps
        CLASS_VIDEO: 320.0,        # ~3.28 Mbps
        CLASS_IOT: 120.0,          # ~76 kbps
        CLASS_BEST_EFFORT: 350.0   # ~4.2 Mbps
    })

@dataclass
class DLConfig:
    state_dim: int = 24             # 4 classes * 5 metrics + 4 global metrics
    action_dim: int = 8             # 4 scheduling weights + 4 drop thresholds
    hidden_dim: int = 128
    learning_rate: float = 0.0005
    gamma: float = 0.95
    tau: float = 0.005
    batch_size: int = 64
    buffer_capacity: int = 20000
    chebyshev_rho: float = 0.05     # Augmented Chebyshev scalarization parameter

# Predefined Preference Profiles (Alpha weights [T, D, J, L, F])
PREFERENCE_PROFILES = {
    "balanced": {
        "name": "Balanced Multi-Objective",
        "description": "Equal balance across all 5 QoS dimensions",
        "weights": [0.20, 0.20, 0.20, 0.20, 0.20]
    },
    "gaming_urllc": {
        "name": "Ultra-Low Latency (URLLC/Gaming)",
        "description": "Prioritizes minimal delay & jitter at the cost of peak throughput",
        "weights": [0.05, 0.40, 0.40, 0.10, 0.05]
    },
    "streaming": {
        "name": "Max Throughput & Reliability",
        "description": "Optimized for video streaming and file transfer delivery",
        "weights": [0.45, 0.10, 0.10, 0.25, 0.10]
    },
    "fairness_critical": {
        "name": "Enterprise Fair Share",
        "description": "Ensures no traffic starvation with near-zero packet loss",
        "weights": [0.10, 0.15, 0.15, 0.25, 0.35]
    }
}
