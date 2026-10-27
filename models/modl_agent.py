"""
Multi-Objective Deep Reinforcement Learning Agent (MODLAgent).
Coordinates actor-critic policy updates, experience replay, preference conditioning,
and model checkpoint serialization.
"""
import os
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import MultiObjectiveActorCritic
from .pareto_loss import chebyshev_scalarization, weighted_sum_scalarization
from config import DLConfig, NUM_CLASSES, NUM_OBJECTIVES

class MODLAgent:
    def __init__(self, config: Optional[DLConfig] = None, device: str = "cpu"):
        self.config = config or DLConfig()
        self.device = torch.device(device)
        
        self.policy_net = MultiObjectiveActorCritic(
            state_dim=self.config.state_dim,
            pref_dim=NUM_OBJECTIVES,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        self.target_net = MultiObjectiveActorCritic(
            state_dim=self.config.state_dim,
            pref_dim=NUM_OBJECTIVES,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        
        # Experience Replay Buffer: (s, w, a, r_vec, s_next, done)
        self.replay_buffer = deque(maxlen=self.config.buffer_capacity)
        self.training_steps = 0
        self.loss_history: List[Dict[str, float]] = []

    def select_action(
        self,
        state: np.ndarray,
        preference: np.ndarray,
        noise_std: float = 0.0
    ) -> np.ndarray:
        """
        Selects QoS action vector [weights (4) + drop_thresholds (4)]
        conditioned on the 5-dimensional preference vector.
        """
        self.policy_net.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            p_tensor = torch.tensor(preference, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.policy_net(s_tensor, p_tensor)
            action = action.squeeze(0).cpu().numpy()

        if noise_std > 0.0:
            # Add exploration noise to weights and renormalize
            w_noise = np.random.normal(0, noise_std, size=NUM_CLASSES)
            weights = np.maximum(0.01, action[:NUM_CLASSES] + w_noise)
            weights = weights / np.sum(weights)

            # Add exploration noise to thresholds
            t_noise = np.random.normal(0, noise_std * 0.5, size=NUM_CLASSES)
            thresholds = np.clip(action[NUM_CLASSES:] + t_noise, 0.40, 0.98)
            action = np.concatenate([weights, thresholds])

        return action.astype(np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        preference: np.ndarray,
        action: np.ndarray,
        reward_vec: np.ndarray,
        next_state: np.ndarray,
        done: bool
    ):
        """Append transition tuple into replay buffer."""
        self.replay_buffer.append((
            state.astype(np.float32),
            preference.astype(np.float32),
            action.astype(np.float32),
            reward_vec.astype(np.float32),
            next_state.astype(np.float32),
            float(done)
        ))

    def update(self, batch_size: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Executes a Multi-Objective Actor-Critic update step:
        - Vector Value Critic Loss: MSE(V(s, w), r + gamma * V(s', w))
        - Pareto Actor Loss: Chebyshev scalarized expected objective return
        """
        bs = batch_size or self.config.batch_size
        if len(self.replay_buffer) < bs:
            return None

        batch = random.sample(self.replay_buffer, bs)
        states, prefs, actions, r_vecs, next_states, dones = zip(*batch)

        s_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        p_t = torch.tensor(np.array(prefs), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        r_t = torch.tensor(np.array(r_vecs), dtype=torch.float32, device=self.device)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        self.policy_net.train()

        # 1. Critic Target Computation
        with torch.no_grad():
            _, next_v = self.target_net(s_next, p_t)
            # Vector Bellman target: y_vec = r_vec + gamma * (1 - done) * V(s', w)
            target_v = r_t + (1.0 - dones_t) * self.config.gamma * next_v

        # 2. Predicted Values and Critic Loss
        pred_actions, pred_v = self.policy_net(s_t, p_t)
        critic_loss = nn.MSELoss()(pred_v, target_v)

        # 3. Actor Loss with Augmented Chebyshev Scalarization
        # Evaluate policy's generated actions to maximize scalarized expected multi-objective return
        scalarized_utility = chebyshev_scalarization(
            pred_v, p_t, rho=self.config.chebyshev_rho
        )
        actor_loss = -torch.mean(scalarized_utility)

        # Action regularization (encourage entropy in weights to avoid sudden starvation)
        pred_weights = pred_actions[:, :NUM_CLASSES]
        entropy_bonus = -torch.mean(torch.sum(pred_weights * torch.log(pred_weights + 1e-8), dim=-1))
        
        total_loss = critic_loss + actor_loss - 0.01 * entropy_bonus

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network: theta_target = tau * theta + (1 - tau) * theta_target
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )

        self.training_steps += 1
        loss_entry = {
            "step": self.training_steps,
            "total_loss": round(float(total_loss.item()), 4),
            "critic_loss": round(float(critic_loss.item()), 4),
            "actor_loss": round(float(actor_loss.item()), 4),
            "scalarized_utility": round(float(scalarized_utility.mean().item()), 4)
        }
        self.loss_history.append(loss_entry)
        if len(self.loss_history) > 200:
            self.loss_history.pop(0)

        return loss_entry

    def save_weights(self, path: str):
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps
        }, path)

    def load_weights(self, path: str) -> bool:
        """Load model checkpoint if available."""
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(ckpt["policy_net"])
                self.target_net.load_state_dict(ckpt["target_net"])
                if "optimizer" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if "training_steps" in ckpt:
                    self.training_steps = ckpt["training_steps"]
                self.policy_net.eval()
                return True
            except Exception as e:
                print(f"Warning: Failed to load weights from {path}: {e}")
                return False
        return False
