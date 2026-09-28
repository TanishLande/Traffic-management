"""
FastAPI Application — V2X Network Backend Server.
Serves the Traffic Ops Dashboard and broadcasts live simulation data
over WebSocket to connected browser clients.
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# --- V2X System Bootstrap ---
from core.v2x_models import BSMMessage, VRUMessage
from core.gateway import V2XGateway
from core.sensor_fusion import LocalDynamicMap
from core.emergency import EmergencyCoordinator
from core.signal_controller import AdaptiveSignalController
from core.simulation import SimulationEngine

# Instantiate the full V2X pipeline
gateway = V2XGateway()
ldm = LocalDynamicMap()
evp = EmergencyCoordinator()
asc = AdaptiveSignalController(ldm, evp)
engine = SimulationEngine(gateway, ldm, evp, asc)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

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
sim_paused = False
sim_speed = 1.0  # Multiplier


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()


app = FastAPI(title="AI-Powered V2X Network", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


async def simulation_loop():
    """Background async simulation loop — runs at ~20 FPS."""
    global sim_paused, sim_speed
    TICK_RATE = 0.05   # 50ms per tick = 20 ticks/sec

    while True:
        await asyncio.sleep(TICK_RATE / max(0.1, sim_speed))
        if not sim_paused:
            try:
                state = engine.tick(dt=TICK_RATE * sim_speed)
                await manager.broadcast(state)
            except Exception as e:
                print(f"[SIM ERROR] {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            await handle_command(msg)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def handle_command(msg: dict):
    """Handle scenario control commands from the dashboard."""
    global sim_paused, sim_speed
    cmd = msg.get("cmd")

    if cmd == "spawn_emergency":
        vid = engine.spawn_emergency_vehicle()
        print(f"[CMD] Emergency vehicle spawned: {vid}")

    elif cmd == "spawn_jaywalker":
        engine.spawn_jaywalker()
        print("[CMD] Jaywalker injected")

    elif cmd == "rush_hour_on":
        engine.set_rush_hour(True)
        print("[CMD] Rush hour ON")

    elif cmd == "rush_hour_off":
        engine.set_rush_hour(False)
        print("[CMD] Rush hour OFF")

    elif cmd == "pause":
        sim_paused = True
        print("[CMD] Simulation paused")

    elif cmd == "resume":
        sim_paused = False
        print("[CMD] Simulation resumed")

    elif cmd == "speed":
        sim_speed = float(msg.get("value", 1.0))
        print(f"[CMD] Speed set to {sim_speed}x")


@app.get("/api/status")
async def api_status():
    return {
        "status": "running",
        "paused": sim_paused,
        "speed": sim_speed,
        "gateway": gateway.stats(),
        "evp": evp.get_status(),
        "kpis": engine.kpis,
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI-Powered V2X Network Prototype — SIH 2026")
    print("  Server: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
