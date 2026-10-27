"""
Deep Neural Network Architectures for Multi-Objective QoS Optimization.
Preference-Conditioned Pareto Actor-Critic (PC-PAC).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import NUM_CLASSES, NUM_OBJECTIVES

class MultiObjectiveActorCritic(nn.Module):
    def __init__(self, state_dim: int = 24, pref_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.pref_dim = pref_dim
        self.total_input_dim = state_dim + pref_dim

        # Shared Feature Trunk
        self.fc_in = nn.Linear(self.total_input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Residual Representation Block
        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Multi-Head Vector Critic: predicts [V_T, V_D, V_J, V_L, V_F]
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, NUM_OBJECTIVES)
        )

        # Actor Head 1: Scheduling Weights (Softmax -> sums to 1.0)
        self.actor_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, NUM_CLASSES)
        )

        # Actor Head 2: Dynamic AQM Early Drop Thresholds [0.38, 0.98]
        self.actor_thresholds = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, NUM_CLASSES)
        )

    def extract_features(self, state: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        """Concatenate state and preference vector then extract shared representations."""
        x = torch.cat([state, pref], dim=-1)
        h = F.leaky_relu(self.ln1(self.fc_in(x)), 0.1)
        # Residual connection
        res = F.leaky_relu(self.res_fc1(h), 0.1)
        res = self.res_fc2(res)
        h = F.leaky_relu(self.ln2(h + res), 0.1)
        return h

    def forward(self, state: torch.Tensor, pref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing:
        - action: [batch, 8] (4 weights + 4 drop thresholds)
        - vector_values: [batch, 5] (critic estimates for 5 QoS objectives)
        """
        h = self.extract_features(state, pref)
        
        # Action computation
        raw_w = self.actor_weights(h)
        weights = F.softmax(raw_w, dim=-1)  # Strictly positive, sums to 1.0

        raw_thresh = self.actor_thresholds(h)
        # Sigmoid scaled to [0.38, 0.98]
        thresholds = 0.38 + 0.60 * torch.sigmoid(raw_thresh)

        action = torch.cat([weights, thresholds], dim=-1)
        vector_values = self.critic_head(h)

        return action, vector_values

    def get_action(self, state: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        """Deterministic inference for action generation."""
        action, _ = self.forward(state, pref)
        return action

    def get_values(self, state: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        """Compute vector value estimates."""
        h = self.extract_features(state, pref)
        return self.critic_head(h)
