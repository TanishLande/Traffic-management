"""
Training Script for Multi-Objective Deep Learning QoS Framework.
Trains Preference-Conditioned Pareto Actor-Critic across diverse traffic scenarios
and saves optimized weights to 'modl_qos_weights.pth'.
"""
import os
import sys
import time
import numpy as np
import torch

# Ensure safe console printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from simulator.network_env import NetworkQoSEnv
from models.modl_agent import MODLAgent
from models.pareto_loss import compute_hypervolume
from config import NetworkConfig, DLConfig, PREFERENCE_PROFILES

def sample_dirichlet_preference(num_objectives: int = 5) -> np.ndarray:
    """Sample a random preference vector on the 5-D simplex (sums to 1.0)."""
    alpha = np.ones(num_objectives)
    weights = np.random.dirichlet(alpha)
    return weights.astype(np.float32)

def train(
    num_episodes: int = 80,
    steps_per_episode: int = 50,
    save_path: str = "modl_qos_weights.pth",
    seed: int = 42
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print("[*] Initializing Multi-Objective Deep Learning QoS Framework Training")
    print("=" * 70)

    env = NetworkQoSEnv(NetworkConfig())
    agent = MODLAgent(DLConfig())

    scenarios = ["Normal Traffic", "voip_rush", "video_surge", "iot_storm", "bulk_download", "congested_all"]
    pareto_archive = []

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        # 1. Randomly sample preference vector to explore full Pareto frontier
        if ep % 4 == 0:
            # Anchor to standard balanced profile periodically
            pref = np.array(PREFERENCE_PROFILES["balanced"]["weights"], dtype=np.float32)
        else:
            pref = sample_dirichlet_preference(5)

        # 2. Cycle scenarios to ensure robust generalization
        scenario = scenarios[ep % len(scenarios)]
        env.traffic_gen.set_scenario(scenario)
        state = env.reset()

        ep_rewards = []
        noise_std = max(0.05, 0.35 * (1.0 - ep / num_episodes))  # Decaying exploration noise

        for step in range(steps_per_episode):
            action = agent.select_action(state, pref, noise_std=noise_std)
            next_state, reward_vec, done, metrics = env.step(action)

            # Store in replay buffer
            agent.store_transition(state, pref, action, reward_vec, next_state, done)
            state = next_state
            ep_rewards.append(reward_vec)

            # Train network once buffer has enough samples
            if len(agent.replay_buffer) >= agent.config.batch_size:
                agent.update()

        mean_ep_r = np.mean(ep_rewards, axis=0)
        pareto_archive.append(mean_ep_r)
        if len(pareto_archive) > 100:
            pareto_archive.pop(0)

        # Logging
        if ep % 10 == 0 or ep == 1:
            hv = compute_hypervolume(pareto_archive, num_samples=1000)
            elapsed = time.time() - start_time
            print(
                f"[Episode {ep:02d}/{num_episodes}] "
                f"Scenario: {scenario:<14} | "
                f"Throughput R: {mean_ep_r[0]:.2f} | Delay R: {mean_ep_r[1]:.2f} | "
                f"Jitter R: {mean_ep_r[2]:.2f} | Loss R: {mean_ep_r[3]:.2f} | Fair R: {mean_ep_r[4]:.2f} | "
                f"5D-Hypervolume: {hv:.4f} | Time: {elapsed:.1f}s"
            )

    # Save trained model weights
    agent.save_weights(save_path)
    print("=" * 70)
    print(f"[OK] Training completed in {time.time() - start_time:.1f}s!")
    print(f"[+] Model weights successfully serialized to: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    train(num_episodes=60, steps_per_episode=40)
