import os
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

async def simulation_worker():
    """
    Background worker that continuously steps the fluid solver every 50ms.
    Broadcasts the downsampled concentration data to all active WebSocket clients.
    """
    global step_counter, solver, active_connections
    while True:
        try:
            # Step the physical simulation (dt=1.0)
            solver.step(dt=1.0)
            step_counter += 1
            
            # Downsample 128x128 concentration grid to 64x64 using block averaging
            conc_64 = solver.conc.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            
            # Prepare state frame
            frame = {
                "conc": conc_64.tolist(),
                "max_conc": float(solver.conc.max()),
                "physics_residual": float(solver.residuals[-1]) if solver.residuals else 0.0,
                "step": step_counter
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
    global solver, buildings_latlon, roads_latlon
    # Load OSM data masks and initialize solver
    solver = StableFluidsSolver(npz_path='data/grid_masks.npz')
    
    # Default parameters: west wind (270 degrees) at speed 0.3
    solver.wind_angle = 270.0
    solver.wind_speed = 0.3
    
    # Initialize geometries and set traffic intensity to medium (0.5) on all roads
    try:
        from backend.osm_loader import fetch_region
        # Indore target location coords
        lat, lon, radius = 22.7533, 75.8937, 800
        buildings_latlon, roads_latlon = fetch_region(lat, lon, radius)
        num_roads = len(roads_latlon)
        solver.traffic_sources = {i: 0.5 for i in range(num_roads)}
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
    for i, (ls, r_type) in enumerate(roads_latlon):
        road_features.append({
            "type": "Feature",
            "geometry": mapping(ls),
            "properties": {
                "road_type": r_type,
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
            "step": step_counter
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
