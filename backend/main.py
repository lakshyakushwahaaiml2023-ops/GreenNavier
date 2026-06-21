import os
import base64
import sys
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from shapely.geometry import mapping

# Add parent directory to path to ensure robust imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import StableFluidsSolver

# Global simulation state variables
solver = None
buildings_latlon = []
roads_latlon = []
active_connections = set()
step_counter = 0

pinn_model = None
pinn_buffer = []  # list of (state_64, residual)
current_physics_score = 0.0
current_physics_warning = False

async def simulation_worker():
    """
    Background worker that continuously steps the fluid solver every 50ms.
    Broadcasts the downsampled concentration data to all active WebSocket clients.
    """
    global step_counter, solver, active_connections, current_physics_score, current_physics_warning, pinn_model
    while True:
        try:
            # Step the physical simulation (dt=1.0)
            solver.step(dt=1.0)
            step_counter += 1
            
            # Downsample 128x128 concentration grid to 64x64 using block averaging
            conc_64 = solver.conc.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            u_64 = solver.u.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            v_64 = solver.v.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            state_64 = np.stack([conc_64, u_64, v_64], axis=0) # (3, 64, 64)
            
            # Record residual
            res = solver.residuals[-1] if solver.residuals else 0.0
            pinn_buffer.append((state_64, res))
            if len(pinn_buffer) > 2000:
                pinn_buffer.pop(0)
                
            # Run inference every 10 steps
            if step_counter % 10 == 0 and pinn_model is not None:
                try:
                    import torch
                    model_in = torch.tensor(state_64, dtype=torch.float32).unsqueeze(0)
                    pinn_model.eval()
                    with torch.no_grad():
                        pred = pinn_model(model_in)
                        current_physics_score = float(pred[0].item())
                    current_physics_warning = (current_physics_score > 0.05)
                except Exception as err:
                    print(f"Error running PINN inference: {err}")
                    
            # Trigger background training every 500 steps
            if step_counter % 500 == 0 and len(pinn_buffer) >= 50:
                try:
                    buffer_copy = list(pinn_buffer)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    from backend.pinn import train_pinn
                    asyncio.create_task(asyncio.to_thread(train_pinn, pinn_model, buffer_copy, device))
                except Exception as err:
                    print(f"Error launching PINN training: {err}")
            
            # Prepare state frame
            frame = {
                "conc": conc_64.tolist(),
                "max_conc": float(solver.conc.max()),
                "physics_residual": res,
                "step": step_counter,
                "physics_score": current_physics_score,
                "physics_warning": current_physics_warning
            }
            
            # Broadcast frame to all connected WebSocket clients
            if active_connections:
                clients = list(active_connections)
                await asyncio.gather(
                    *(client.send_json(frame) for client in clients),
                    return_exceptions=True
                )
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            
        # Step interval of 50ms
        await asyncio.sleep(0.05)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager for startup and shutdown procedures.
    """
    global solver, buildings_latlon, roads_latlon, pinn_model
    # Load OSM data masks and initialize solver
    solver = StableFluidsSolver(npz_path='data/grid_masks.npz')
    
    # Default parameters: west wind (270 degrees) at speed 0.3
    solver.wind_angle = 270.0
    solver.wind_speed = 0.3
    
    # Initialize PINN model
    import torch
    from backend.pinn import PINNErrorEstimator
    pinn_model = PINNErrorEstimator()
    if os.path.exists('data/pinn_estimator.pt'):
        try:
            pinn_model.load_state_dict(torch.load('data/pinn_estimator.pt', map_location='cpu'))
            print("Loaded pre-trained PINN error estimator weights.")
        except Exception as e:
            print(f"Error loading PINN estimator weights: {e}")
    else:
        print("No pre-trained PINN estimator found. Will train from scratch.")
        
    # Initialize geometries and set traffic intensity to low (0.1) on all roads
    try:
        from backend.osm_loader import fetch_region
        # Indore target location coords
        lat, lon, radius = 22.7533, 75.8937, 800
        buildings_latlon, roads_latlon = fetch_region(lat, lon, radius)
        num_roads = len(roads_latlon)
        solver.traffic_sources = {i: 0.1 for i in range(num_roads)}
    except Exception as e:
        print(f"Error fetching lat/lon elements on startup: {e}")
        buildings_latlon, roads_latlon = [], []
        
    # Start background task running the simulation
    app.state.simulation_task = asyncio.create_task(simulation_worker())
    print("Simulation background worker started successfully.")
    
    yield
    
    # Shutdown simulation loop
    app.state.simulation_task.cancel()
    try:
        await app.state.simulation_task
    except asyncio.CancelledError:
        pass
    print("Simulation background worker stopped.")

app = FastAPI(title="GreenNavier Air Dispersion API", lifespan=lifespan)

# Enable CORS for Three.js client connection from local/custom ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST Request schemas
class WindSettings(BaseModel):
    angle: float
    speed: float

class TrafficSettings(BaseModel):
    road_id: int
    intensity: float

class DecaySettings(BaseModel):
    decay_rate: float

class GreenCorridorSettings(BaseModel):
    cells: list[dict]

class SourceSettings(BaseModel):
    x: int
    y: int
    strength: float

class RemoveSourceSettings(BaseModel):
    x: int
    y: int

class BuildingSettings(BaseModel):
    x: int
    y: int
    w: int
    h: int
    height_m: float

@app.post("/set_wind")
def set_wind(settings: WindSettings):
    solver.wind_angle = settings.angle
    solver.wind_speed = settings.speed
    return {"message": "Wind updated successfully", "angle": settings.angle, "speed": settings.speed}

@app.post("/set_decay")
def set_decay(settings: DecaySettings):
    solver.decay_rate = settings.decay_rate
    return {"message": "Decay rate updated successfully", "decay_rate": settings.decay_rate}

@app.post("/set_green_corridor")
def set_green_corridor(settings: GreenCorridorSettings):
    # Reset green corridor mask
    solver.green_corridor_mask = np.zeros_like(solver.green_corridor_mask)
    for cell in settings.cells:
        gx, gy = cell.get("x"), cell.get("y")
        if gx is not None and gy is not None:
            if 0 <= gx < solver.grid_size and 0 <= gy < solver.grid_size:
                solver.green_corridor_mask[gy, gx] = 1.0
    return {"message": "Green corridor updated successfully", "active_cells": len(settings.cells)}

@app.post("/set_traffic")
def set_traffic(settings: TrafficSettings):
    solver.traffic_sources[settings.road_id] = settings.intensity
    return {"message": "Traffic updated successfully", "road_id": settings.road_id, "intensity": settings.intensity}

@app.post("/add_source")
def add_source(settings: SourceSettings):
    # Check if point source already exists at coordinates, if so update it, else append
    found = False
    for idx, (sx, sy, _) in enumerate(solver.point_sources):
        if sx == settings.x and sy == settings.y:
            solver.point_sources[idx] = (settings.x, settings.y, settings.strength)
            found = True
            break
    if not found:
        solver.point_sources.append((settings.x, settings.y, settings.strength))
    return {"message": "Point source added/updated", "x": settings.x, "y": settings.y, "strength": settings.strength}

@app.post("/remove_source")
def remove_source(settings: RemoveSourceSettings):
    initial_len = len(solver.point_sources)
    solver.point_sources = [
        (sx, sy, strength) for sx, sy, strength in solver.point_sources
        if not (sx == settings.x and sy == settings.y)
    ]
    if len(solver.point_sources) < initial_len:
        return {"message": "Point source removed successfully"}
    return {"message": "Point source not found"}

@app.post("/add_building")
def add_building(settings: BuildingSettings):
    # Clamp grid coordinates to 128x128 boundary
    x0 = max(0, min(127, settings.x))
    y0 = max(0, min(127, settings.y))
    x1 = max(0, min(128, settings.x + settings.w))
    y1 = max(0, min(128, settings.y + settings.h))
    
    # Set building inside obstacle mask and height map live
    solver.obstacle_mask[y0:y1, x0:x1] = 1.0
    solver.height_map[y0:y1, x0:x1] = settings.height_m
    return {"message": "Building added successfully", "coords": (x0, y0, x1, y1)}

@app.post("/reset")
def reset_simulation():
    solver.conc = np.zeros_like(solver.conc)
    return {"message": "Concentration field reset successfully"}

class ScreenshotSettings(BaseModel):
    filename: str
    dataUrl: str  # base64 data URL: "data:image/png;base64,..."

@app.post("/save_screenshot")
def save_screenshot(settings: ScreenshotSettings):
    try:
        os.makedirs('figures', exist_ok=True)
        # Strip the data URL header
        header, encoded = settings.dataUrl.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        out_path = os.path.join('figures', settings.filename)
        with open(out_path, 'wb') as f:
            f.write(img_bytes)
        return {"message": "Screenshot saved", "path": out_path}
    except Exception as e:
        return {"message": f"Error saving screenshot: {e}"}

@app.get("/map_data")
def get_map_data():
    building_features = []
    for i, (poly, h) in enumerate(buildings_latlon):
        building_features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "height": h,
                "id": i
            }
        })
    
    road_features = []
    for i, (ls, r_type, name) in enumerate(roads_latlon):
        road_features.append({
            "type": "Feature",
            "geometry": mapping(ls),
            "properties": {
                "road_type": r_type,
                "name": name,
                "id": i
            }
        })
        
    return {
        "buildings": {
            "type": "FeatureCollection",
            "features": building_features
        },
        "roads": {
            "type": "FeatureCollection",
            "features": road_features
        }
    }

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    print("WebSocket client connected.")
    
    try:
        # Send current state immediately on connect/reconnect
        conc_64 = solver.conc.reshape(64, 2, 64, 2).mean(axis=(1, 3))
        frame = {
            "conc": conc_64.tolist(),
            "max_conc": float(solver.conc.max()),
            "physics_residual": float(solver.residuals[-1]) if solver.residuals else 0.0,
            "step": step_counter,
            "physics_score": current_physics_score,
            "physics_warning": current_physics_warning
        }
        await websocket.send_json(frame)
        
        # Maintain connection to receive close/disconnect frames
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        active_connections.discard(websocket)
