"""
Discrete-event Network Simulation Environment computing the 5 QoS metrics:
Throughput, Delay, Jitter (RFC 3550), Packet Loss, and Fairness (Jain's Index).
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from .packet import Packet
from .traffic_generator import TrafficGenerator
from .queues import MultiQueueManager
from config import (
    NetworkConfig, NUM_CLASSES, TRAFFIC_CLASSES,
    CLASS_VOIP, CLASS_VIDEO, CLASS_IOT, CLASS_BEST_EFFORT,
    OBJ_THROUGHPUT, OBJ_DELAY, OBJ_JITTER, OBJ_LOSS, OBJ_FAIRNESS
)

class NetworkQoSEnv:
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.dt = self.config.time_step_ms / 1000.0  # seconds per step
        self.link_capacity_bps = self.config.link_capacity_mbps * 1e6
        
        self.traffic_gen = TrafficGenerator(self.config.base_arrival_rates)
        self.queue_mgr = MultiQueueManager(self.config.max_queue_depth_packets)
        
        self.current_time = 0.0
        self.step_count = 0
        
        # State tracking per class
        self.prev_delays = {c: 10.0 for c in range(NUM_CLASSES)}
        self.rfc3550_jitters = {c: 1.0 for c in range(NUM_CLASSES)}
        
        # Historical metrics for smoothing & inspection
        self.last_metrics: Dict[str, Any] = {}
        self.cumulative_delivered_bytes = 0
        self.cumulative_dropped_bytes = 0
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_time = 0.0
        self.step_count = 0
        self.traffic_gen = TrafficGenerator(self.config.base_arrival_rates)
        self.queue_mgr.clear()
        self.prev_delays = {c: 10.0 for c in range(NUM_CLASSES)}
        self.rfc3550_jitters = {c: 1.0 for c in range(NUM_CLASSES)}
        self.cumulative_delivered_bytes = 0
        self.cumulative_dropped_bytes = 0
        
        self.last_metrics = {
            "throughput_mbps": 0.0,
            "avg_delay_ms": 0.0,
            "avg_jitter_ms": 0.0,
            "packet_loss_ratio": 0.0,
            "jains_fairness": 1.0,
            "link_utilization": 0.0,
            "class_metrics": {}
        }
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one control interval (dt seconds):
        action: array of 8 floats:
          - action[0:4]: dynamic scheduling weights for 4 queues (positive)
          - action[4:8]: dynamic AQM early drop thresholds [0.3, 1.0]
        """
        self.step_count += 1
        
        # Parse action
        weights = list(action[:NUM_CLASSES])
        # Ensure positive weights
        weights = [max(0.01, float(w)) for w in weights]
        
        thresholds = list(action[NUM_CLASSES:2 * NUM_CLASSES])
        self.queue_mgr.set_drop_thresholds(thresholds)

        # 1. Traffic Generation
        arrived_packets = self.traffic_gen.generate_packets_for_interval(self.current_time, self.dt)
        arrived_count_by_class = {c: 0 for c in range(NUM_CLASSES)}
        arrived_bytes_by_class = {c: 0 for c in range(NUM_CLASSES)}
        for p in arrived_packets:
            arrived_count_by_class[p.traffic_class] += 1
            arrived_bytes_by_class[p.traffic_class] += p.size_bytes

        # 2. Enqueue arrived packets
        admitted_packets: List[Packet] = []
        dropped_packets: List[Packet] = []
        for p in arrived_packets:
            admitted = self.queue_mgr.enqueue(p, p.creation_time)
            if admitted:
                admitted_packets.append(p)
            else:
                dropped_packets.append(p)

        # 3. Dequeue and transmit packets
        transmitted_packets = self.queue_mgr.schedule_and_dequeue(
            weights=weights,
            link_capacity_bps=self.link_capacity_bps,
            dt_seconds=self.dt,
            current_time=self.current_time,
            propagation_delay_ms=self.config.propagation_delay_ms
        )

        # 4. Metrics calculation for this time step
        # Delivered bytes & throughput
        delivered_bytes_by_class = {c: 0 for c in range(NUM_CLASSES)}
        delays_by_class: Dict[int, List[float]] = {c: [] for c in range(NUM_CLASSES)}
        for p in transmitted_packets:
            delivered_bytes_by_class[p.traffic_class] += p.size_bytes
            delays_by_class[p.traffic_class].append(p.end_to_end_delay)

        total_delivered_bytes = sum(delivered_bytes_by_class.values())
        self.cumulative_delivered_bytes += total_delivered_bytes
        
        throughput_mbps = (total_delivered_bytes * 8.0) / (self.dt * 1e6)
        link_utilization = min(1.0, throughput_mbps / max(1.0, self.config.link_capacity_mbps))

        # Class-specific delay & RFC 3550 Jitter
        class_stats = {}
        all_delays: List[float] = []
        all_jitters: List[float] = []
        dropped_count_by_class = {c: 0 for c in range(NUM_CLASSES)}
        for p in dropped_packets:
            dropped_count_by_class[p.traffic_class] += 1

        for c in range(NUM_CLASSES):
            c_delays = delays_by_class[c]
            if len(c_delays) > 0:
                mean_delay = float(np.mean(c_delays))
                # RFC 3550 Jitter filter: J = J + (|D_i - D_{i-1}| - J) / 16
                for d in c_delays:
                    diff = abs(d - self.prev_delays[c])
                    self.rfc3550_jitters[c] += (diff - self.rfc3550_jitters[c]) / 16.0
                    self.prev_delays[c] = d
            else:
                # If no packets delivered, retain base propagation + slight queue estimation
                q_len = len(self.queue_mgr.queues[c])
                estimated_delay = self.config.propagation_delay_ms + (q_len * 1500 * 8.0 / self.link_capacity_bps * 1000.0)
                mean_delay = estimated_delay
            
            total_arrived = arrived_count_by_class[c]
            dropped_cnt = dropped_count_by_class[c]
            loss_ratio = (dropped_cnt / max(1, total_arrived)) if total_arrived > 0 else 0.0

            class_stats[c] = {
                "name": TRAFFIC_CLASSES[c]["name"],
                "color": TRAFFIC_CLASSES[c]["color"],
                "throughput_mbps": (delivered_bytes_by_class[c] * 8.0) / (self.dt * 1e6),
                "avg_delay_ms": round(mean_delay, 2),
                "jitter_ms": round(self.rfc3550_jitters[c], 2),
                "loss_ratio": round(loss_ratio, 4),
                "arrived_packets": total_arrived,
                "delivered_packets": len(delays_by_class[c]),
                "dropped_packets": dropped_cnt,
                "queue_depth": len(self.queue_mgr.queues[c]),
                "weight_assigned": round(weights[c] / sum(weights), 3),
                "drop_threshold": round(self.queue_mgr.drop_thresholds[c], 2)
            }
            all_delays.append(mean_delay)
            all_jitters.append(self.rfc3550_jitters[c])

        # Overall average delay & jitter
        avg_delay_ms = float(np.mean(all_delays))
        avg_jitter_ms = float(np.mean(all_jitters))

        # Overall Packet Loss Ratio
        total_arrived_all = sum(arrived_count_by_class.values())
        total_dropped_all = sum(dropped_count_by_class.values())
        overall_loss_ratio = (total_dropped_all / max(1, total_arrived_all)) if total_arrived_all > 0 else 0.0

        # Jain's Fairness Index across classes: JFI = (sum(x_i))^2 / (n * sum(x_i^2))
        # x_i = satisfaction ratio = min(1.0, delivered_bytes / max(1, arrived_bytes))
        satisfaction_ratios = []
        for c in range(NUM_CLASSES):
            arrived_b = arrived_bytes_by_class[c]
            delivered_b = delivered_bytes_by_class[c]
            if arrived_b > 0:
                sat = min(1.5, delivered_b / arrived_b)
            else:
                sat = 1.0
            satisfaction_ratios.append(sat)
        
        sum_x = sum(satisfaction_ratios)
        sum_x_sq = sum(x**2 for x in satisfaction_ratios)
        if sum_x_sq > 0:
            jains_fairness = (sum_x ** 2) / (NUM_CLASSES * sum_x_sq)
        else:
            jains_fairness = 1.0
        jains_fairness = float(np.clip(jains_fairness, 0.25, 1.0))

        # 5. Multi-Objective Vector Reward Formulation R = [R_T, R_D, R_J, R_L, R_F] in [0, 1]
        # Target bounds:
        # Throughput: normalized to link capacity
        r_throughput = float(np.clip(throughput_mbps / self.config.link_capacity_mbps, 0.0, 1.0))
        # Delay: exp(-delay / 50ms)
        r_delay = float(np.clip(np.exp(-avg_delay_ms / 60.0), 0.0, 1.0))
        # Jitter: exp(-jitter / 10ms)
        r_jitter = float(np.clip(np.exp(-avg_jitter_ms / 12.0), 0.0, 1.0))
        # Packet Loss: 1.0 - loss_ratio
        r_loss = float(np.clip(1.0 - (overall_loss_ratio * 3.0), 0.0, 1.0))
        # Fairness: Jain's Fairness Index directly in [0.25, 1.0] -> normalize to [0, 1]
        r_fairness = float(np.clip((jains_fairness - 0.25) / 0.75, 0.0, 1.0))

        vector_reward = np.array([r_throughput, r_delay, r_jitter, r_loss, r_fairness], dtype=np.float32)

        # Advance time
        self.current_time += self.dt

        self.last_metrics = {
            "step": self.step_count,
            "sim_time_sec": round(self.current_time, 2),
            "throughput_mbps": round(throughput_mbps, 2),
            "avg_delay_ms": round(avg_delay_ms, 2),
            "avg_jitter_ms": round(avg_jitter_ms, 2),
            "packet_loss_ratio": round(overall_loss_ratio, 4),
            "packet_loss_percent": round(overall_loss_ratio * 100.0, 2),
            "jains_fairness": round(jains_fairness, 3),
            "link_utilization": round(link_utilization, 3),
            "vector_reward": [round(float(v), 3) for v in vector_reward],
            "class_metrics": class_stats,
            "total_arrived": total_arrived_all,
            "total_delivered": len(transmitted_packets),
            "total_dropped": total_dropped_all,
            "recent_drops": [p.to_dict() for p in self.queue_mgr.recent_drops[-10:]]
        }

        next_state = self._get_state()
        done = False
        return next_state, vector_reward, done, self.last_metrics

    def _get_state(self) -> np.ndarray:
        """
        Constructs the 24-dimensional normalized network state vector:
        - 4 classes x 5 features = 20 dims:
          [queue_depth_ratio, arrival_rate_norm, loss_ratio, delay_norm, jitter_norm]
        - 4 global features:
          [link_utilization, total_buffer_occupancy_ratio, overall_loss_ratio, jains_fairness]
        """
        state_features: List[float] = []
        occupancies = self.queue_mgr.get_queue_occupancy_ratios()
        
        for c in range(NUM_CLASSES):
            q_occ = occupancies[c]
            # Normalizing arrival rate by 500 pps
            base_rate = self.config.base_arrival_rates.get(c, 200.0)
            mult = self.traffic_gen.burst_multipliers.get(c, 1.0)
            arr_norm = min(2.0, (base_rate * mult) / 500.0)
            
            c_metrics = self.last_metrics.get("class_metrics", {}).get(c, {})
            loss_r = c_metrics.get("loss_ratio", 0.0)
            delay_norm = min(3.0, c_metrics.get("avg_delay_ms", 15.0) / 100.0)
            jitter_norm = min(3.0, c_metrics.get("jitter_ms", 3.0) / 25.0)

            state_features.extend([q_occ, arr_norm, loss_r, delay_norm, jitter_norm])

        # Global features
        util = self.last_metrics.get("link_utilization", 0.5)
        total_q_occ = sum(occupancies.values()) / float(NUM_CLASSES)
        ov_loss = self.last_metrics.get("packet_loss_ratio", 0.0)
        fairness = self.last_metrics.get("jains_fairness", 1.0)

        state_features.extend([util, total_q_occ, ov_loss, fairness])
        return np.array(state_features, dtype=np.float32)
