"""
FastAPI Application Backend for Multi-Objective Deep Learning QoS Framework.
Exposes WebSocket telemetry stream, REST control API, and serves the Web Dashboard.
"""
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import numpy as np

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from config import (
    NetworkConfig, DLConfig, PREFERENCE_PROFILES,
    TRAFFIC_CLASSES, OBJECTIVES, NUM_CLASSES, NUM_OBJECTIVES
)
from simulator.network_env import NetworkQoSEnv
from simulator.baselines import (
    FIFODropTailController,
    StaticWFQController,
    RED_DWRRController,
    GreedySingleObjectiveController
)
from models.modl_agent import MODLAgent
from benchmark.evaluator import run_benchmark_comparison

# Global state instances
env = NetworkQoSEnv(NetworkConfig())
agent = MODLAgent(DLConfig())

# Load pre-trained weights if exists
WEIGHTS_PATH = "modl_qos_weights.pth"
if os.path.exists(WEIGHTS_PATH):
    loaded = agent.load_weights(WEIGHTS_PATH)
    print(f"[+] Loaded weights from {WEIGHTS_PATH}: {loaded}")
else:
    print("[*] No pre-existing weights found. Running with initialized network.")

# Active controllers dict
baseline_controllers = {
    "fifo": FIFODropTailController(),
    "wfq": StaticWFQController(),
    "red": RED_DWRRController(),
    "greedy": GreedySingleObjectiveController()
}

# Runtime control state
runtime_state = {
    "active_controller": "modl",  # "modl", "fifo", "wfq", "red", "greedy"
    "active_profile": "balanced",
    "preference_weights": [0.20, 0.20, 0.20, 0.20, 0.20],  # [T, D, J, L, F]
    "current_scenario": "Normal Traffic",
    "is_paused": False,
    "simulation_speed": 1.0,  # 1.0 = normal, 2.0 = fast, 0.5 = slow
    "latest_metrics": {}
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# Background simulation runner
async def simulation_loop():
    current_state = env.reset()
    while True:
        try:
            if not runtime_state["is_paused"]:
                # Select action based on active controller
                ctrl_key = runtime_state["active_controller"]
                pref_arr = np.array(runtime_state["preference_weights"], dtype=np.float32)

                if ctrl_key == "modl":
                    action = agent.select_action(current_state, pref_arr, noise_std=0.0)
                elif ctrl_key in baseline_controllers:
                    action = baseline_controllers[ctrl_key].get_action(current_state, pref_arr)
                else:
                    action = baseline_controllers["fifo"].get_action(current_state, pref_arr)

                next_state, reward_vec, _, metrics = env.step(action)
                current_state = next_state

                # Enrich telemetry payload
                telemetry = {
                    "type": "telemetry",
                    "timestamp": time.time(),
                    "active_controller": runtime_state["active_controller"],
                    "active_profile": runtime_state["active_profile"],
                    "preference_weights": runtime_state["preference_weights"],
                    "scenario": runtime_state["current_scenario"],
                    "metrics": metrics
                }
                runtime_state["latest_metrics"] = metrics
                await manager.broadcast(telemetry)

            # Sleep interval adjusted by simulation speed
            interval = max(0.02, (env.config.time_step_ms / 1000.0) / max(0.1, runtime_state["simulation_speed"]))
            await asyncio.sleep(interval)
        except Exception as e:
            print(f"[!] Error in simulation loop: {e}")
            await asyncio.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()

app = FastAPI(title="Multi-Objective Deep Learning QoS Optimization", lifespan=lifespan)

# Request Models
class PreferenceRequest(BaseModel):
    weights: Optional[List[float]] = None
    profile: Optional[str] = None

class ControllerModeRequest(BaseModel):
    mode: str  # "modl", "fifo", "wfq", "red", "greedy"

class ScenarioRequest(BaseModel):
    scenario: str

class TrafficSpikeRequest(BaseModel):
    class_id: int
    multiplier: float = 3.5
    duration: float = 6.0

class SimControlRequest(BaseModel):
    paused: Optional[bool] = None
    speed: Optional[float] = None

# REST API Endpoints

@app.get("/api/config")
def get_system_config():
    """Retrieve network traffic classes, QoS objectives, and default profiles."""
    return {
        "traffic_classes": TRAFFIC_CLASSES,
        "objectives": OBJECTIVES,
        "preference_profiles": PREFERENCE_PROFILES,
        "runtime_state": runtime_state
    }

@app.post("/api/control/preferences")
def update_preferences(req: PreferenceRequest):
    """Update QoS objective preference weights or apply preset profile."""
    if req.profile and req.profile in PREFERENCE_PROFILES:
        prof = PREFERENCE_PROFILES[req.profile]
        runtime_state["active_profile"] = req.profile
        runtime_state["preference_weights"] = list(prof["weights"])
    elif req.weights and len(req.weights) == NUM_OBJECTIVES:
        w_sum = sum(req.weights)
        if w_sum > 0:
            norm_w = [round(w / w_sum, 3) for w in req.weights]
            runtime_state["preference_weights"] = norm_w
            runtime_state["active_profile"] = "custom"
        else:
            raise HTTPException(status_code=400, detail="Preference weights sum must be greater than 0")
    else:
        raise HTTPException(status_code=400, detail="Invalid preference parameters")

    return {
        "status": "success",
        "active_profile": runtime_state["active_profile"],
        "preference_weights": runtime_state["preference_weights"]
    }

@app.post("/api/control/mode")
def set_controller_mode(req: ControllerModeRequest):
    """Switch active QoS scheduling algorithm."""
    valid_modes = ["modl", "fifo", "wfq", "red", "greedy"]
    if req.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Mode must be one of {valid_modes}")
    
    runtime_state["active_controller"] = req.mode
    return {"status": "success", "active_controller": runtime_state["active_controller"]}

@app.post("/api/traffic/scenario")
def set_scenario(req: ScenarioRequest):
    """Change the network traffic generation pattern."""
    env.traffic_gen.set_scenario(req.scenario)
    runtime_state["current_scenario"] = req.scenario
    return {"status": "success", "scenario": req.scenario}

@app.post("/api/traffic/spike")
def trigger_traffic_spike(req: TrafficSpikeRequest):
    """Inject an acute traffic surge on a specific class."""
    if not (0 <= req.class_id < NUM_CLASSES):
        raise HTTPException(status_code=400, detail=f"Invalid class_id {req.class_id}")
    
    env.traffic_gen.trigger_spike(req.class_id, req.multiplier, req.duration)
    return {
        "status": "success",
        "message": f"Injected {req.multiplier}x traffic spike into Class {req.class_id} for {req.duration}s"
    }

@app.post("/api/simulation/control")
def control_simulation(req: SimControlRequest):
    """Pause, resume, or adjust speed of simulation."""
    if req.paused is not None:
        runtime_state["is_paused"] = req.paused
    if req.speed is not None:
        runtime_state["simulation_speed"] = max(0.1, min(5.0, req.speed))
    return {
        "status": "success",
        "is_paused": runtime_state["is_paused"],
        "simulation_speed": runtime_state["simulation_speed"]
    }

@app.post("/api/simulation/reset")
def reset_simulation():
    """Reset simulation queues, counters, and statistics."""
    env.reset()
    return {"status": "success", "message": "Simulation environment reset successfully"}

@app.get("/api/benchmark/run")
def run_benchmark(scenario: str = "congested_all", steps: int = 60):
    """
    Execute rigorous side-by-side benchmark comparing:
    FIFO vs Static WFQ vs RED+DWRR vs Greedy Throughput DL vs Multi-Objective DL.
    """
    pref = np.array(runtime_state["preference_weights"], dtype=np.float32)
    benchmark_results = run_benchmark_comparison(
        modl_agent=agent,
        preference_weights=pref,
        scenario=scenario,
        eval_steps=min(120, max(20, steps))
    )
    return benchmark_results

@app.get("/api/training/history")
def get_training_history():
    """Return recent loss convergence records."""
    return {
        "training_steps": agent.training_steps,
        "loss_history": agent.loss_history[-100:]
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time bidirectional WebSocket stream."""
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Handle incoming WebSocket commands if client sends any
            try:
                cmd = json.loads(data)
                if cmd.get("action") == "ping":
                    await ws.send_json({"type": "pong", "time": time.time()})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("[+] Starting Multi-Objective Deep Learning QoS Optimization Server at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
