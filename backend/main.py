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
last_conc_ws = None

pinn_model = None
pinn_buffer = []  # list of (state_64, residual)

# User-placed elements tracking
user_buildings = {}  # id -> {x1, y1, x2, y2, height_m}
user_sources = {}    # id -> (gx, gy, strength)
base_obstacle_mask = None
base_height_map = None
current_physics_score = 0.0
current_physics_warning = False

async def simulation_worker():
    """
    Background worker that continuously steps the fluid solver every 50ms.
    Broadcasts the downsampled concentration data to all active WebSocket clients.
    """
    global step_counter, solver, active_connections, current_physics_score, current_physics_warning, pinn_model, last_conc_ws
    while True:
        try:
            # Step the physical simulation (dt=1.0) in a background thread to prevent blocking
            await asyncio.to_thread(solver.step, dt=1.0)
            step_counter += 1
            
            # Downsample 128x128 species concentration grids and velocity grids to 64x64
            co_64 = solver.conc_co.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            no_64 = solver.conc_no.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            no2_64 = solver.conc_no2.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            o3_64 = solver.conc_o3.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            u_64 = solver.u.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            v_64 = solver.v.reshape(64, 2, 64, 2).mean(axis=(1, 3))
            state_64 = np.stack([co_64, no_64, no2_64, o3_64, u_64, v_64], axis=0) # (6, 64, 64)
            
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
                    
                    async def run_training():
                        from backend.pinn import train_pinn
                        import torch
                        await asyncio.to_thread(train_pinn, buffer_copy, device)
                        # Reload the newly trained weights into the main thread's model (always kept on CPU)
                        if pinn_model is not None and os.path.exists('data/pinn_estimator.pt'):
                            try:
                                pinn_model.load_state_dict(torch.load('data/pinn_estimator.pt', map_location='cpu'))
                                print("[PINN Trainer] Swapped trained model into main simulation worker.")
                            except Exception as load_err:
                                print(f"Error loading trained PINN model: {load_err}")

                    asyncio.create_task(run_training())
                except Exception as err:
                    print(f"Error launching PINN training: {err}")
            
            # Downsample payload concentration grid using numpy slice [::2, ::2]
            conc_co_ws = solver.conc_co[::2, ::2]
            conc_no_ws = solver.conc_no[::2, ::2]
            conc_no2_ws = solver.conc_no2[::2, ::2]
            conc_o3_ws = solver.conc_o3[::2, ::2]
            
            # Only send if max concentration change of NO2 since last frame > 0.001
            should_send = True
            if last_conc_ws is not None:
                max_change = float(np.max(np.abs(conc_no2_ws - last_conc_ws)))
                if max_change <= 0.001:
                    should_send = False
            
            if should_send:
                last_conc_ws = conc_no2_ws.copy()
                b64_co = base64.b64encode(conc_co_ws.astype(np.float32).tobytes()).decode('utf-8')
                b64_no = base64.b64encode(conc_no_ws.astype(np.float32).tobytes()).decode('utf-8')
                b64_no2 = base64.b64encode(conc_no2_ws.astype(np.float32).tobytes()).decode('utf-8')
                b64_o3 = base64.b64encode(conc_o3_ws.astype(np.float32).tobytes()).decode('utf-8')
                
                # Prepare state frame
                frame = {
                    "conc": b64_no2,  # fallback for backward compatibility
                    "conc_co": b64_co,
                    "conc_no": b64_no,
                    "conc_no2": b64_no2,
                    "conc_o3": b64_o3,
                    "max_conc": float(solver.conc_no2.max()),
                    "max_conc_co": float(solver.conc_co.max()),
                    "max_conc_no": float(solver.conc_no.max()),
                    "max_conc_no2": float(solver.conc_no2.max()),
                    "max_conc_o3": float(solver.conc_o3.max()),
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
    global solver, buildings_latlon, roads_latlon, pinn_model, base_obstacle_mask, base_height_map, user_buildings, user_sources
    import backend.pinn
    backend.pinn.stop_training = False
    # Load OSM data masks and initialize solver
    solver = StableFluidsSolver(npz_path='data/grid_masks.npz')
    
    # Store base masks for custom modifications
    base_obstacle_mask = solver.obstacle_mask.copy()
    base_height_map = solver.height_map.copy()
    user_buildings = {}
    user_sources = {}
    
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
    import backend.pinn
    backend.pinn.stop_training = True
    
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

class ChemistrySettings(BaseModel):
    J_NO2: float
    C_OH: float
    C_HO2: float
    oh_model: str
    T: float

class SourceSettings(BaseModel):
    id: str
    x: int
    y: int
    strength: float

class RemoveSourceSettings(BaseModel):
    id: str

class BuildingSettings(BaseModel):
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    height_m: float

class RemoveBuildingSettings(BaseModel):
    id: str

class RegionSettings(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

@app.post("/load_region")
async def load_region(settings: RegionSettings):
    global solver, buildings_latlon, roads_latlon, step_counter, base_obstacle_mask, base_height_map, user_buildings, user_sources
    from backend.osm_loader import fetch_and_rasterize_bbox
    try:
        buildings, roads, obstacle_mask, road_mask, height_map, center_lat, center_lon, radius_m = await asyncio.to_thread(
            fetch_and_rasterize_bbox,
            settings.min_lat,
            settings.max_lat,
            settings.min_lon,
            settings.max_lon
        )
    except ValueError as val_err:
        return {"status": "error", "message": str(val_err)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load region: {str(e)}"}

    buildings_latlon = buildings
    roads_latlon = roads
    step_counter = 0

    solver = StableFluidsSolver(
        npz_path='data/grid_masks.npz',
        roads=roads,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m
    )
    
    # Store base masks for the new region
    base_obstacle_mask = solver.obstacle_mask.copy()
    base_height_map = solver.height_map.copy()
    user_buildings = {}
    user_sources = {}

    num_roads = len(roads_latlon)
    solver.traffic_sources = {i: 0.1 for i in range(num_roads)}
    solver.wind_angle = 270.0
    solver.wind_speed = 0.3

    map_data = get_map_data()

    return {
        "status": "success",
        "map_data": map_data,
        "center": [center_lat, center_lon],
        "radius_m": radius_m
    }

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

@app.post("/set_chemistry")
def set_chemistry(settings: ChemistrySettings):
    global solver
    solver.J_NO2 = settings.J_NO2
    solver.C_OH = settings.C_OH
    solver.C_HO2 = settings.C_HO2
    solver.oh_model = settings.oh_model
    solver.T = settings.T
    return {
        "message": "Chemistry parameters updated successfully",
        "J_NO2": settings.J_NO2,
        "C_OH": settings.C_OH,
        "C_HO2": settings.C_HO2,
        "oh_model": settings.oh_model,
        "T": settings.T
    }

@app.post("/set_traffic")
def set_traffic(settings: TrafficSettings):
    solver.traffic_sources[settings.road_id] = settings.intensity
    return {"message": "Traffic updated successfully", "road_id": settings.road_id, "intensity": settings.intensity}

# Helper to restamp all active user buildings onto the base masks
def rebuild_obstacle_masks():
    global solver, base_obstacle_mask, base_height_map, user_buildings
    if solver is None or base_obstacle_mask is None or base_height_map is None:
        return
    # Reset to base (OSM only)
    solver.obstacle_mask = base_obstacle_mask.copy()
    solver.height_map = base_height_map.copy()
    # Stamp user buildings
    for b in user_buildings.values():
        x0 = max(0, min(127, min(b['x1'], b['x2'])))
        x1 = max(0, min(128, max(b['x1'], b['x2'])))
        y0 = max(0, min(127, min(b['y1'], b['y2'])))
        y1 = max(0, min(128, max(b['y1'], b['y2'])))
        if x0 == x1:
            x1 = min(128, x0 + 1)
        if y0 == y1:
            y1 = min(128, y0 + 1)
        solver.obstacle_mask[y0:y1, x0:x1] = 1.0
        solver.height_map[y0:y1, x0:x1] = b['height_m']

@app.post("/add_source")
def add_source(settings: SourceSettings):
    global user_sources, solver
    user_sources[settings.id] = (settings.x, settings.y, settings.strength)
    solver.point_sources = list(user_sources.values())
    return {"message": "Point source added successfully", "id": settings.id}

@app.post("/update_source")
def update_source(settings: SourceSettings):
    global user_sources, solver
    user_sources[settings.id] = (settings.x, settings.y, settings.strength)
    solver.point_sources = list(user_sources.values())
    return {"message": "Point source updated successfully", "id": settings.id}

@app.post("/remove_source")
def remove_source(settings: RemoveSourceSettings):
    global user_sources, solver
    if settings.id in user_sources:
        del user_sources[settings.id]
        solver.point_sources = list(user_sources.values())
        return {"message": "Point source removed successfully"}
    return {"message": "Point source not found"}

@app.post("/add_building")
def add_building(settings: BuildingSettings):
    global user_buildings
    user_buildings[settings.id] = {
        "x1": settings.x1,
        "y1": settings.y1,
        "x2": settings.x2,
        "y2": settings.y2,
        "height_m": settings.height_m
    }
    rebuild_obstacle_masks()
    return {"message": "Building added successfully", "id": settings.id}

@app.post("/update_building")
def update_building(settings: BuildingSettings):
    global user_buildings
    user_buildings[settings.id] = {
        "x1": settings.x1,
        "y1": settings.y1,
        "x2": settings.x2,
        "y2": settings.y2,
        "height_m": settings.height_m
    }
    rebuild_obstacle_masks()
    return {"message": "Building updated successfully", "id": settings.id}

@app.post("/remove_building")
def remove_building(settings: RemoveBuildingSettings):
    global user_buildings
    if settings.id in user_buildings:
        del user_buildings[settings.id]
        rebuild_obstacle_masks()
        return {"message": "Building removed successfully"}
    return {"message": "Building not found"}

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
        conc_co_ws = solver.conc_co[::2, ::2]
        conc_no_ws = solver.conc_no[::2, ::2]
        conc_no2_ws = solver.conc_no2[::2, ::2]
        conc_o3_ws = solver.conc_o3[::2, ::2]
        
        b64_co = base64.b64encode(conc_co_ws.astype(np.float32).tobytes()).decode('utf-8')
        b64_no = base64.b64encode(conc_no_ws.astype(np.float32).tobytes()).decode('utf-8')
        b64_no2 = base64.b64encode(conc_no2_ws.astype(np.float32).tobytes()).decode('utf-8')
        b64_o3 = base64.b64encode(conc_o3_ws.astype(np.float32).tobytes()).decode('utf-8')
        
        frame = {
            "conc": b64_no2,
            "conc_co": b64_co,
            "conc_no": b64_no,
            "conc_no2": b64_no2,
            "conc_o3": b64_o3,
            "max_conc": float(solver.conc_no2.max()),
            "max_conc_co": float(solver.conc_co.max()),
            "max_conc_no": float(solver.conc_no.max()),
            "max_conc_no2": float(solver.conc_no2.max()),
            "max_conc_o3": float(solver.conc_o3.max()),
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
