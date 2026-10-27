"""
Baseline QoS Controllers for comparative benchmarking:
1. FIFO Drop-Tail
2. Static Weighted Fair Queueing (Static WFQ)
3. RED + Deficit Weighted Round Robin (RED + DWRR)
4. Single-Objective Greedy Throughput Controller (Demonstrating the problem statement)
"""
import numpy as np
from config import NUM_CLASSES, CLASS_VOIP, CLASS_VIDEO, CLASS_IOT, CLASS_BEST_EFFORT

class BaseQoSController:
    def __init__(self, name: str):
        self.name = name

    def get_action(self, state: np.ndarray, preferences: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

class FIFODropTailController(BaseQoSController):
    """
    Classic FIFO with Drop-Tail buffering:
    - Equal scheduling weights (no priority differentiation)
    - Full buffer thresholds (no early drop, causing severe bufferbloat)
    """
    def __init__(self):
        super().__init__("FIFO Drop-Tail")

    def get_action(self, state: np.ndarray, preferences: np.ndarray = None) -> np.ndarray:
        # Equal weights [0.25, 0.25, 0.25, 0.25], hard drop threshold 1.0 for all
        weights = [0.25, 0.25, 0.25, 0.25]
        thresholds = [1.0, 1.0, 1.0, 1.0]
        return np.array(weights + thresholds, dtype=np.float32)

class StaticWFQController(BaseQoSController):
    """
    Static Weighted Fair Queueing:
    - Static priority weights: Video 35%, VoIP 25%, IoT 20%, Best Effort 20%
    - Fixed thresholds at 0.90
    """
    def __init__(self):
        super().__init__("Static WFQ")

    def get_action(self, state: np.ndarray, preferences: np.ndarray = None) -> np.ndarray:
        weights = [0.25, 0.35, 0.20, 0.20]
        thresholds = [0.90, 0.90, 0.90, 0.90]
        return np.array(weights + thresholds, dtype=np.float32)

class RED_DWRRController(BaseQoSController):
    """
    Random Early Detection with Deficit Weighted Round Robin:
    - Heuristically adjusts thresholds based on observed queue occupancies.
    - Fixed class scheduling weights.
    """
    def __init__(self):
        super().__init__("RED + DWRR")

    def get_action(self, state: np.ndarray, preferences: np.ndarray = None) -> np.ndarray:
        weights = [0.30, 0.30, 0.20, 0.20]
        # Inspect queue occupancies from state [index 0, 5, 10, 15]
        thresholds = []
        for c in range(NUM_CLASSES):
            q_occ = state[c * 5]
            # When queue fills up, lower threshold to drop earlier and curb bufferbloat
            thresh = float(np.clip(0.95 - (q_occ * 0.35), 0.50, 0.95))
            thresholds.append(thresh)
        return np.array(weights + thresholds, dtype=np.float32)

class GreedySingleObjectiveController(BaseQoSController):
    """
    Greedy Throughput Maximizer:
    - Prioritizes only high-throughput flows (Best Effort & Video with large MTU packets)
    - Starves low-bandwidth latency-critical flows (VoIP and IoT)
    - Demonstrates the central problem statement: maximizing throughput destroys delay, jitter, and fairness.
    """
    def __init__(self):
        super().__init__("Greedy Throughput DL")

    def get_action(self, state: np.ndarray, preferences: np.ndarray = None) -> np.ndarray:
        # Give 90% bandwidth to Best Effort and Video
        weights = [0.05, 0.45, 0.05, 0.45]
        # Keep thresholds high to maximize raw throughput
        thresholds = [1.0, 1.0, 1.0, 1.0]
        return np.array(weights + thresholds, dtype=np.float32)
