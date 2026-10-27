"""
Unit tests for PyTorch Multi-Objective Deep Learning models,
Pareto scalarization, and Hypervolume indicator computations.
"""
import unittest
import torch
import numpy as np
from models.networks import MultiObjectiveActorCritic
from models.pareto_loss import (
    weighted_sum_scalarization,
    chebyshev_scalarization,
    compute_hypervolume,
    is_pareto_dominant
)
from models.modl_agent import MODLAgent
from config import DLConfig, NUM_OBJECTIVES, NUM_CLASSES

class TestDLArchitecture(unittest.TestCase):
    def setUp(self):
        self.config = DLConfig()
        self.agent = MODLAgent(self.config)

    def test_actor_critic_forward_dimensions(self):
        net = MultiObjectiveActorCritic(state_dim=24, pref_dim=5, hidden_dim=64)
        dummy_state = torch.randn(4, 24)
        dummy_pref = torch.tensor([[0.2]*5]*4, dtype=torch.float32)

        action, values = net(dummy_state, dummy_pref)
        self.assertEqual(action.shape, (4, 8))
        self.assertEqual(values.shape, (4, 5))

        # Check that weights sum to 1.0 (softmax)
        weights = action[:, :NUM_CLASSES]
        sums = torch.sum(weights, dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

        # Check that thresholds are strictly within [0.38, 0.98]
        thresholds = action[:, NUM_CLASSES:]
        self.assertTrue(torch.all(thresholds >= 0.38))
        self.assertTrue(torch.all(thresholds <= 0.98))

    def test_chebyshev_scalarization(self):
        rewards = torch.tensor([[0.8, 0.7, 0.9, 0.95, 0.85]], dtype=torch.float32)
        weights = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)
        scalarized = chebyshev_scalarization(rewards, weights)
        self.assertEqual(scalarized.shape, (1,))

    def test_pareto_dominance(self):
        p1 = np.array([0.9, 0.8, 0.85, 0.95, 0.90])
        p2 = np.array([0.7, 0.6, 0.70, 0.80, 0.75])
        self.assertTrue(is_pareto_dominant(p1, p2))
        self.assertFalse(is_pareto_dominant(p2, p1))

    def test_hypervolume_indicator(self):
        p1 = [np.array([0.9, 0.9, 0.9, 0.9, 0.9])]
        p2 = [np.array([0.3, 0.3, 0.3, 0.3, 0.3])]
        hv1 = compute_hypervolume(p1, num_samples=1000)
        hv2 = compute_hypervolume(p2, num_samples=1000)
        self.assertGreater(hv1, hv2)

    def test_agent_replay_and_update(self):
        state = np.zeros(24, dtype=np.float32)
        pref = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        action = np.ones(8, dtype=np.float32)
        r_vec = np.ones(5, dtype=np.float32)

        # Fill replay buffer to batch size
        for _ in range(70):
            self.agent.store_transition(state, pref, action, r_vec, state, False)

        loss_info = self.agent.update(batch_size=32)
        self.assertIsNotNone(loss_info)
        self.assertIn("total_loss", loss_info)
        self.assertIn("critic_loss", loss_info)
        self.assertIn("actor_loss", loss_info)

if __name__ == "__main__":
    unittest.main()
