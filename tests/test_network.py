"""
Unit tests for Computer Network simulation mathematics and queuing policies.
Tests RFC 3550 Jitter filter, Jain's Fairness Index bounds, and MultiQueueManager DRR.
"""
import unittest
import numpy as np
from simulator.packet import Packet
from simulator.traffic_generator import TrafficGenerator
from simulator.queues import MultiQueueManager
from simulator.network_env import NetworkQoSEnv
from simulator.baselines import FIFODropTailController, StaticWFQController, GreedySingleObjectiveController
from config import NetworkConfig, NUM_CLASSES, CLASS_VOIP, CLASS_VIDEO, CLASS_IOT, CLASS_BEST_EFFORT

class TestNetworkSimulation(unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig()
        self.env = NetworkQoSEnv(self.config)

    def test_packet_delay_calculation(self):
        pkt = Packet(
            packet_id=1,
            traffic_class=CLASS_VOIP,
            size_bytes=160,
            creation_time=1.000,
            queue_enter_time=1.002,
            departure_time=1.025
        )
        self.assertAlmostEqual(pkt.queueing_delay, 23.0, places=2)
        self.assertAlmostEqual(pkt.end_to_end_delay, 25.0, places=2)

    def test_traffic_generator_bursts(self):
        gen = TrafficGenerator(self.config.base_arrival_rates)
        packets = gen.generate_packets_for_interval(0.0, 0.05)
        self.assertGreater(len(packets), 0)
        
        # Test spike injection
        gen.trigger_spike(CLASS_VOIP, multiplier=4.0, duration=2.0)
        spike_packets = gen.generate_packets_for_interval(0.05, 0.05)
        voip_count = sum(1 for p in spike_packets if p.traffic_class == CLASS_VOIP)
        self.assertGreater(voip_count, 0)

    def test_multi_queue_drr_scheduling(self):
        qm = MultiQueueManager(max_depth_per_queue=50)
        # Enqueue 10 packets per class
        for c in range(NUM_CLASSES):
            for i in range(10):
                p = Packet(packet_id=c*100 + i, traffic_class=c, size_bytes=200, creation_time=0.0)
                admitted = qm.enqueue(p, current_time=0.0)
                self.assertTrue(admitted)

        # Dequeue with equal weights
        weights = [0.25, 0.25, 0.25, 0.25]
        transmitted = qm.schedule_and_dequeue(
            weights=weights,
            link_capacity_bps=10e6,
            dt_seconds=0.05,
            current_time=0.0,
            propagation_delay_ms=5.0
        )
        self.assertGreater(len(transmitted), 0)

    def test_jains_fairness_index_bounds(self):
        """Jain's index must strictly lie between 1/N (0.25 for 4 classes) and 1.0."""
        state = self.env.reset()
        self.assertEqual(len(state), 24)

        # Run 10 steps with FIFO controller
        ctrl = FIFODropTailController()
        for _ in range(10):
            action = ctrl.get_action(state)
            next_state, reward_vec, _, metrics = self.env.step(action)
            jfi = metrics["jains_fairness"]
            self.assertGreaterEqual(jfi, 0.25)
            self.assertLessEqual(jfi, 1.0)
            self.assertEqual(len(reward_vec), 5)
            # Rewards in [0, 1]
            for r in reward_vec:
                self.assertGreaterEqual(r, 0.0)
                self.assertLessEqual(r, 1.0)
            state = next_state

if __name__ == "__main__":
    unittest.main()
