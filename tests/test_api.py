"""
Integration tests for FastAPI REST Endpoints and Benchmark Execution.
"""
import unittest
from fastapi.testclient import TestClient
from app import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_config(self):
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("traffic_classes", data)
        self.assertIn("objectives", data)
        self.assertEqual(len(data["traffic_classes"]), 4)
        self.assertEqual(len(data["objectives"]), 5)

    def test_update_preferences_profile(self):
        res = self.client.post("/api/control/preferences", json={"profile": "gaming_urllc"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["active_profile"], "gaming_urllc")

    def test_update_controller_mode(self):
        res = self.client.post("/api/control/mode", json={"mode": "fifo"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["active_controller"], "fifo")

        # Switch back to modl
        res = self.client.post("/api/control/mode", json={"mode": "modl"})
        self.assertEqual(res.status_code, 200)

    def test_traffic_spike(self):
        res = self.client.post("/api/traffic/spike", json={"class_id": 0, "multiplier": 3.0, "duration": 4.0})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "success")

    def test_benchmark_endpoint(self):
        res = self.client.get("/api/benchmark/run?scenario=Normal Traffic&steps=20")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("controllers", data)
        self.assertIn("MODL-QoS (Ours)", data["controllers"])
        self.assertIn("FIFO Drop-Tail", data["controllers"])
        self.assertIn("improvements_over_fifo", data)

if __name__ == "__main__":
    unittest.main()
