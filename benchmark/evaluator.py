"""
Benchmarking Engine for Multi-Objective QoS Optimization.
Executes deterministic comparative trials across FIFO, Static WFQ, RED+DWRR,
Greedy Single-Objective DL, and our Multi-Objective Deep Learning Framework (MODL).
Computes exact 5-D metrics, Pareto dominance, and Hypervolume indicators.
"""
import copy
import random
import numpy as np
from typing import Dict, List, Any
from simulator.network_env import NetworkQoSEnv
from simulator.baselines import (
    FIFODropTailController,
    StaticWFQController,
    RED_DWRRController,
    GreedySingleObjectiveController
)
from models.modl_agent import MODLAgent
from models.pareto_loss import compute_hypervolume, is_pareto_dominant
from config import NetworkConfig, NUM_CLASSES, TRAFFIC_CLASSES

def run_benchmark_comparison(
    modl_agent: MODLAgent,
    preference_weights: np.ndarray = None,
    scenario: str = "congested_all",
    eval_steps: int = 80,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a side-by-side comparative simulation across all controllers
    under the exact same incoming traffic sequence.
    """
    if preference_weights is None:
        preference_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)

    controllers = [
        ("FIFO Drop-Tail", FIFODropTailController()),
        ("Static WFQ", StaticWFQController()),
        ("RED + DWRR", RED_DWRRController()),
        ("Greedy Throughput DL", GreedySingleObjectiveController()),
        ("MODL-QoS (Ours)", modl_agent)
    ]

    results: Dict[str, Dict[str, Any]] = {}
    objective_points = {}

    for name, ctrl in controllers:
        # Seed both Python and NumPy RNG for identical packet arrivals across controllers
        np.random.seed(seed)
        random.seed(seed)

        env = NetworkQoSEnv(NetworkConfig())
        env.traffic_gen.set_scenario(scenario)
        state = env.reset()

        step_throughputs = []
        step_delays = []
        step_jitters = []
        step_losses = []
        step_fairnesses = []
        step_vector_rewards = []
        class_throughput_accum = {c: 0.0 for c in range(NUM_CLASSES)}
        class_delay_accum = {c: [] for c in range(NUM_CLASSES)}

        for _ in range(eval_steps):
            if name == "MODL-QoS (Ours)":
                action = ctrl.select_action(state, preference_weights, noise_std=0.0)
            else:
                action = ctrl.get_action(state, preference_weights)

            next_state, reward_vec, _, metrics = env.step(action)
            state = next_state

            step_throughputs.append(metrics["throughput_mbps"])
            step_delays.append(metrics["avg_delay_ms"])
            step_jitters.append(metrics["avg_jitter_ms"])
            step_losses.append(metrics["packet_loss_percent"])
            step_fairnesses.append(metrics["jains_fairness"])
            step_vector_rewards.append(metrics["vector_reward"])

            for c in range(NUM_CLASSES):
                c_m = metrics["class_metrics"].get(c, {})
                class_throughput_accum[c] += c_m.get("throughput_mbps", 0.0)
                class_delay_accum[c].append(c_m.get("avg_delay_ms", 0.0))

        # Calculate averages
        mean_thru = float(np.mean(step_throughputs))
        mean_delay = float(np.mean(step_delays))
        mean_jitter = float(np.mean(step_jitters))
        mean_loss = float(np.mean(step_losses))
        mean_fairness = float(np.mean(step_fairnesses))

        # Normalized objective vector for Pareto/Hypervolume calculation [T, D, J, L, F] in [0, 1]
        norm_t = min(1.0, mean_thru / 40.0)
        norm_d = np.exp(-mean_delay / 60.0)
        norm_j = np.exp(-mean_jitter / 12.0)
        norm_l = max(0.0, 1.0 - (mean_loss / 100.0 * 3.0))
        norm_f = max(0.0, (mean_fairness - 0.25) / 0.75)
        obj_vec = np.array([norm_t, norm_d, norm_j, norm_l, norm_f])
        objective_points[name] = obj_vec

        # Breakdown per class
        class_summary = {}
        for c in range(NUM_CLASSES):
            class_summary[TRAFFIC_CLASSES[c]["name"]] = {
                "color": TRAFFIC_CLASSES[c]["color"],
                "avg_throughput_mbps": round(float(class_throughput_accum[c] / eval_steps), 2),
                "avg_delay_ms": round(float(np.mean(class_delay_accum[c])), 2)
            }

        results[name] = {
            "throughput_mbps": round(mean_thru, 2),
            "delay_ms": round(mean_delay, 2),
            "jitter_ms": round(mean_jitter, 2),
            "packet_loss_percent": round(mean_loss, 2),
            "fairness_jfi": round(mean_fairness, 3),
            "normalized_vector": [round(float(v), 3) for v in obj_vec],
            "class_summary": class_summary
        }

    # Compute Hypervolume indicator for each controller relative to [0,0,0,0,0]
    for name in results:
        pt = objective_points[name]
        hv = compute_hypervolume([pt], num_samples=2000)
        results[name]["hypervolume_indicator"] = hv

    # Calculate Pareto dominance comparison against baselines
    modl_vec = objective_points["MODL-QoS (Ours)"]
    dominance_matrix = {}
    for name, vec in objective_points.items():
        if name != "MODL-QoS (Ours)":
            dominance_matrix[f"MODL_dominates_{name}"] = is_pareto_dominant(modl_vec, vec)

    # Compute percentage improvements of MODL over FIFO Drop-Tail
    fifo_res = results["FIFO Drop-Tail"]
    modl_res = results["MODL-QoS (Ours)"]
    
    delay_imp = ((fifo_res["delay_ms"] - modl_res["delay_ms"]) / max(1.0, fifo_res["delay_ms"])) * 100.0
    jitter_imp = ((fifo_res["jitter_ms"] - modl_res["jitter_ms"]) / max(1.0, fifo_res["jitter_ms"])) * 100.0
    loss_imp = ((fifo_res["packet_loss_percent"] - modl_res["packet_loss_percent"]) / max(0.1, fifo_res["packet_loss_percent"])) * 100.0
    fairness_imp = ((modl_res["fairness_jfi"] - fifo_res["fairness_jfi"]) / max(0.1, fifo_res["fairness_jfi"])) * 100.0

    improvement_summary = {
        "delay_reduction_percent": round(delay_imp, 1),
        "jitter_reduction_percent": round(jitter_imp, 1),
        "packet_loss_reduction_percent": round(loss_imp, 1),
        "fairness_improvement_percent": round(fairness_imp, 1)
    }

    return {
        "scenario": scenario,
        "eval_steps": eval_steps,
        "preference_weights": [round(float(w), 2) for w in preference_weights],
        "controllers": results,
        "dominance_matrix": dominance_matrix,
        "improvements_over_fifo": improvement_summary
    }
