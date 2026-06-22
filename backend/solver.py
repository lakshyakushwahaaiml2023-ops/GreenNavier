import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add parent directory to path to ensure robust imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def enforce_neumann_boundary(field, obstacle_mask):
    """
    Enforces a zero-gradient (Neumann) boundary condition at obstacle boundaries
    by setting the obstacle cell's value to the average of its active fluid neighbors.
    """
    obs = (obstacle_mask > 0.5)
    
    r_f = np.roll(field, -1, axis=1); r_f[:, -1] = field[:, -1]
    l_f = np.roll(field, 1, axis=1); l_f[:, 0] = field[:, 0]
    d_f = np.roll(field, -1, axis=0); d_f[-1, :] = field[-1, :]
    u_f = np.roll(field, 1, axis=0); u_f[0, :] = field[0, :]
    
    r_o = np.roll(obstacle_mask, -1, axis=1); r_o[:, -1] = obstacle_mask[:, -1]
    l_o = np.roll(obstacle_mask, 1, axis=1); l_o[:, 0] = obstacle_mask[:, 0]
    d_o = np.roll(obstacle_mask, -1, axis=0); d_o[-1, :] = obstacle_mask[-1, :]
    u_o = np.roll(obstacle_mask, 1, axis=0); u_o[0, :] = obstacle_mask[0, :]
    
    # Active neighbor values from fluid cells (mask <= 0.5)
    r_val = r_f * (r_o <= 0.5)
    l_val = l_f * (l_o <= 0.5)
    d_val = d_f * (d_o <= 0.5)
    u_val = u_f * (u_o <= 0.5)
    
    count = (
        (r_o <= 0.5).astype(np.float32) + 
        (l_o <= 0.5).astype(np.float32) + 
        (d_o <= 0.5).astype(np.float32) + 
        (u_o <= 0.5).astype(np.float32)
    )
    
    mask = obs & (count > 0)
    updated_field = field.copy()
    updated_field[mask] = (r_val[mask] + l_val[mask] + d_val[mask] + u_val[mask]) / count[mask]
    return updated_field

def enforce_velocity_boundary(u, v, obstacle_mask):
    """
    Enforces no-penetration (v . n = 0) and no-slip (velocity = 0 inside obstacles)
    boundary conditions at obstacle boundaries.
    """
    obs = (obstacle_mask > 0.5)
    
    # Set boundaries to False so outer edges of the grid act as open boundaries
    r_o = np.roll(obs, -1, axis=1); r_o[:, -1] = False
    l_o = np.roll(obs, 1, axis=1); l_o[:, 0] = False
    d_o = np.roll(obs, -1, axis=0); d_o[-1, :] = True  # Keep bottom closed or open? Let's keep it open as False
    u_o = np.roll(obs, 1, axis=0); u_o[0, :] = False
    # Wait, let's make d_o[-1, :] False as well!
    # Let's write it down:
    # r_o[:, -1] = False
    # l_o[:, 0] = False
    # d_o[-1, :] = False
    # u_o[0, :] = False
    # This is correct.
    r_o = np.roll(obs, -1, axis=1); r_o[:, -1] = False
    l_o = np.roll(obs, 1, axis=1); l_o[:, 0] = False
    d_o = np.roll(obs, -1, axis=0); d_o[-1, :] = False
    u_o = np.roll(obs, 1, axis=0); u_o[0, :] = False
    
    u_new = u.copy()
    v_new = v.copy()
    
    # Zero velocity inside obstacles (no-slip)
    u_new[obs] = 0.0
    v_new[obs] = 0.0
    
    # No flow into obstacles from adjacent fluid cells
    u_new[r_o & ~obs] = np.minimum(u_new[r_o & ~obs], 0.0)
    u_new[l_o & ~obs] = np.maximum(u_new[l_o & ~obs], 0.0)
    v_new[d_o & ~obs] = np.minimum(v_new[d_o & ~obs], 0.0)
    v_new[u_o & ~obs] = np.maximum(v_new[u_o & ~obs], 0.0)
    
    return u_new, v_new

class StableFluidsSolver:
    def __init__(self, npz_path='data/grid_masks.npz', grid_size=128, roads=None, center_lat=None, center_lon=None, radius_m=None):
        self.grid_size = grid_size
        
        # Load npz masks
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            self.obstacle_mask = data['obstacle_mask'].astype(np.float32)
            self.road_mask = data['road_mask'].astype(np.float32)
            self.height_map = data['height_map'].astype(np.float32)
            
            # Load metadata from NPZ if present
            self.center_lat = float(data['center_lat']) if 'center_lat' in data else center_lat
            self.center_lon = float(data['center_lon']) if 'center_lon' in data else center_lon
            self.radius_m = float(data['radius_m']) if 'radius_m' in data else radius_m
            print(f"Loaded grid masks from {npz_path}.")
        else:
            print(f"Warning: {npz_path} not found. Creating empty masks.")
            self.obstacle_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
            self.road_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
            self.height_map = np.zeros((grid_size, grid_size), dtype=np.float32)
            self.center_lat = center_lat
            self.center_lon = center_lon
            self.radius_m = radius_m
            
        # State: velocity fields (u: horizontal/cols, v: vertical/rows) and concentration
        self.u = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.v = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.conc = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Parameters (settable mid-simulation)
        self.wind_angle = 270.0  # 0=north (blowing south), 90=east (blowing west), 270=west (blowing east)
        self.wind_speed = 0.5
        self.decay_rate = 0.999  # Settable decay rate for simulation times
        self.green_corridor_mask = np.zeros((grid_size, grid_size), dtype=np.float32)  # Mask for green corridors
        self.traffic_sources = {}  # dict mapping road segment index -> intensity
        self.point_sources = []    # list of (grid_x, grid_y, strength)
        
        # History metrics
        self.residuals = []
        
        # Initialize individual road segment masks
        self.road_segments_masks = []
        self._init_road_segments(roads=roads)

    def _init_road_segments(self, roads=None):
        try:
            from backend.osm_loader import fetch_region, project_geom
            
            lat = self.center_lat if self.center_lat is not None else 22.7533
            lon = self.center_lon if self.center_lon is not None else 75.8937
            radius = self.radius_m if self.radius_m is not None else 800
            
            # If roads is not passed, fetch it
            if roads is None:
                _, roads = fetch_region(lat, lon, radius)
            
            # Recreate local coordinates projection factors
            a = 6378137.0
            e2 = 0.00669437999014
            lat_rad = np.radians(lat)
            N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
            M = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad)**2)**1.5
            
            meters_per_deg_lat = M * (np.pi / 180.0)
            meters_per_deg_lon = N * np.cos(lat_rad) * (np.pi / 180.0)
            
            min_x, max_x = -radius, radius
            min_y, max_y = -radius, radius
            
            def map_x(x):
                return int((x - min_x) / (max_x - min_x) * self.grid_size)
            def map_y(y):
                return int((max_y - y) / (max_y - min_y) * self.grid_size)

            for i, (ls, road_type, *_) in enumerate(roads):
                proj_ls = project_geom(ls, lat, lon, meters_per_deg_lat, meters_per_deg_lon)
                road_img = Image.new('L', (self.grid_size, self.grid_size), 0)
                draw = ImageDraw.Draw(road_img)
                width = 3 if road_type == 'primary' else 1
                
                if proj_ls.geom_type == 'LineString':
                    coords = [(map_x(x), map_y(y)) for x, y in proj_ls.coords]
                    if len(coords) >= 2:
                        draw.line(coords, fill=255, width=width)
                elif proj_ls.geom_type == 'MultiLineString':
                    for sub_ls in proj_ls.geoms:
                        coords = [(map_x(x), map_y(y)) for x, y in sub_ls.coords]
                        if len(coords) >= 2:
                            draw.line(coords, fill=255, width=width)
                            
                self.road_segments_masks.append(np.array(road_img) > 0)
            print(f"Initialized {len(self.road_segments_masks)} road segments from OSM database.")
        except Exception as e:
            print(f"Warning: Reconstructing road segments failed: {e}. Falling back to default global road mask.")
            self.road_segments_masks = []

    def _get_road_emission(self):
        emission = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if len(self.road_segments_masks) > 0:
            for i, mask in enumerate(self.road_segments_masks):
                intensity = self.traffic_sources.get(i, 0.0)
                if intensity > 0.0:
                    emission[mask] = np.maximum(emission[mask], intensity)
        else:
            # Fallback: use global road_mask with average of active traffic values
            if self.traffic_sources:
                avg_intensity = np.mean(list(self.traffic_sources.values()))
            else:
                avg_intensity = 0.0
            emission = self.road_mask * avg_intensity
        return emission

    def step(self, dt=1.0):
        # 1. Apply wind inflow on boundary cells
        theta_rad = np.radians(self.wind_angle)
        u_wind = -np.sin(theta_rad) * self.wind_speed
        v_wind = np.cos(theta_rad) * self.wind_speed
        
        # Inflow boundaries
        if u_wind > 0:
            self.u[:, 0] = u_wind
            self.v[:, 0] = v_wind
        elif u_wind < 0:
            self.u[:, -1] = u_wind
            self.v[:, -1] = v_wind
            
        if v_wind > 0:
            self.u[0, :] = u_wind
            self.v[0, :] = v_wind
        elif v_wind < 0:
            self.u[-1, :] = u_wind
            self.v[-1, :] = v_wind

        # Apply wind body force (atmospheric forcing) in fluid cells to drive the wind in the interior
        fluid_mask = (self.obstacle_mask <= 0.5)
        self.u[fluid_mask] = 0.9 * self.u[fluid_mask] + 0.1 * u_wind
        self.v[fluid_mask] = 0.9 * self.v[fluid_mask] + 0.1 * v_wind

        # Apply Green Corridor velocity dampening (reduce velocity by 50% in corridor cells)
        if np.any(self.green_corridor_mask > 0.5):
            corridor = (self.green_corridor_mask > 0.5)
            self.u[corridor] *= 0.5
            self.v[corridor] *= 0.5

        # Helper for boundary-clamped shifts
        def get_neighbors(field):
            r = np.empty_like(field)
            r[:, :-1] = field[:, 1:]
            r[:, -1] = field[:, -1]
            
            l = np.empty_like(field)
            l[:, 1:] = field[:, :-1]
            l[:, 0] = field[:, 0]
            
            d = np.empty_like(field)
            d[:-1, :] = field[1:, :]
            d[-1, :] = field[-1, :]
            
            u_n = np.empty_like(field)
            u_n[1:, :] = field[:-1, :]
            u_n[0, :] = field[0, :]
            
            return r, l, d, u_n

        # Bilinear semi-Lagrangian advection helper with open boundaries support
        def advect_open(field, u_field, v_field, default_inflow_value):
            h, w = field.shape
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            src_x = grid_x - u_field * dt
            src_y = grid_y - v_field * dt
            
            # Open boundaries check
            inside = (src_x >= 0.0) & (src_x <= w - 1.0) & (src_y >= 0.0) & (src_y <= h - 1.0)
            
            src_x = np.clip(src_x, 0.0, w - 1.0)
            src_y = np.clip(src_y, 0.0, h - 1.0)
            
            x0 = np.floor(src_x).astype(np.int32)
            x1 = x0 + 1
            y0 = np.floor(src_y).astype(np.int32)
            y1 = y0 + 1
            
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            
            wx = src_x - x0
            wy = src_y - y0
            
            interpolated = (
                (1.0 - wx) * (1.0 - wy) * field[y0, x0] +
                wx * (1.0 - wy) * field[y0, x1] +
                (1.0 - wx) * wy * field[y1, x0] +
                wx * wy * field[y1, x1]
            )
            
            if isinstance(default_inflow_value, np.ndarray):
                interpolated[~inside] = default_inflow_value[~inside]
            else:
                interpolated[~inside] = default_inflow_value
                
            return interpolated

        # 2. Advect velocity (inflow is wind velocity)
        u_inflow = np.full_like(self.u, u_wind)
        v_inflow = np.full_like(self.v, v_wind)
        u_advected = advect_open(self.u, self.u, self.v, u_inflow)
        v_advected = advect_open(self.v, self.u, self.v, v_inflow)
        self.u, self.v = u_advected, v_advected
        
        # 3. Diffuse velocity (implicit, viscosity=0.0001)
        visc = 0.0001
        alpha = visc * dt
        beta = 1.0 + 4.0 * alpha
        for _ in range(20):
            r_u, l_u, d_u, u_u = get_neighbors(self.u)
            self.u = (self.u + alpha * (r_u + l_u + d_u + u_u)) / beta
            
            r_v, l_v, d_v, u_v = get_neighbors(self.v)
            self.v = (self.v + alpha * (r_v + l_v + d_v + u_v)) / beta
            
        # 4. Project to divergence-free
        r_u, l_u, d_u, u_u = get_neighbors(self.u)
        r_v, l_v, d_v, u_v = get_neighbors(self.v)
        
        div = 0.5 * (r_u - l_u + d_v - u_v)
        
        p = np.zeros_like(self.u)
        for _ in range(25):
            r_p, l_p, d_p, u_p = get_neighbors(p)
            p = (r_p + l_p + d_p + u_p - div) / 4.0
            p = enforce_neumann_boundary(p, self.obstacle_mask)
            
            # Enforce Dirichlet p=0 on open outer boundary edges (only where not buildings)
            p[0, self.obstacle_mask[0, :] <= 0.5] = 0.0
            p[-1, self.obstacle_mask[-1, :] <= 0.5] = 0.0
            p[self.obstacle_mask[:, 0] <= 0.5, 0] = 0.0
            p[self.obstacle_mask[:, -1] <= 0.5, -1] = 0.0
            
        r_p, l_p, d_p, u_p = get_neighbors(p)
        self.u -= 0.5 * (r_p - l_p)
        self.v -= 0.5 * (d_p - u_p)
        
        # Compute post-projection divergence for PINN training residual
        r_u_f, l_u_f, d_u_f, u_u_f = get_neighbors(self.u)
        r_v_f, l_v_f, d_v_f, u_v_f = get_neighbors(self.v)
        final_div = 0.5 * (r_u_f - l_u_f + d_v_f - u_v_f)
        self.residuals.append(float(np.mean(np.abs(final_div))))
        
        # 5. Zero velocity inside obstacle cells and enforce no-penetration
        self.u, self.v = enforce_velocity_boundary(self.u, self.v, self.obstacle_mask)
        
        # 6. Advect concentration (clean air inflow: default_inflow_value = 0.0)
        self.conc = enforce_neumann_boundary(self.conc, self.obstacle_mask)
        self.conc = advect_open(self.conc, self.u, self.v, 0.0)
        
        # Zero concentration inside obstacles
        self.conc[self.obstacle_mask > 0.5] = 0.0

        # Apply Green Corridor absorption (absorb 35% concentration in corridor cells)
        if np.any(self.green_corridor_mask > 0.5):
            corridor = (self.green_corridor_mask > 0.5)
            self.conc[corridor] *= 0.65
        
        # 7. Add emissions (scaled down to prevent saturation and maintain high visual contrast)
        road_emission = self._get_road_emission()
        self.conc += road_emission * 0.05 * dt
        
        for gx, gy, strength in self.point_sources:
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.conc[gy, gx] += strength * dt
                
        # 8. Apply concentration decay (dilution/deposition)
        self.conc *= self.decay_rate

    def get_state(self):
        """
        Returns the current state for JSON serializations.
        """
        return {
            'u': self.u.tolist(),
            'v': self.v.tolist(),
            'conc': self.conc.tolist(),
            'obstacle_mask': self.obstacle_mask.tolist()
        }

if __name__ == '__main__':
    print("Initializing test simulation of 200 steps...")
    solver = StableFluidsSolver(npz_path='data/grid_masks.npz')
    
    # Configure test parameters
    solver.wind_angle = 270.0  # Wind from west (blowing east)
    solver.wind_speed = 0.5
    solver.point_sources = [(64, 64, 2.0)]  # Point source at center
    
    # 0.7 traffic intensity on all roads
    num_roads = len(solver.road_segments_masks)
    solver.traffic_sources = {i: 0.7 for i in range(num_roads)}
    
    frames = []
    
    # Run 200 simulation steps
    for step_idx in range(200):
        solver.step(dt=1.0)
        
        # Save concentration visualization frame every step
        val_norm = np.clip(solver.conc / 15.0, 0.0, 1.0)
        rgba = plt.cm.inferno(val_norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        
        # Overlay buildings as dark gray
        rgb[solver.obstacle_mask > 0.5] = [64, 64, 64]
        
        img = Image.fromarray(rgb)
        # Resize for better visibility in GIF
        img_resized = img.resize((256, 256), Image.Resampling.NEAREST)
        frames.append(img_resized)
        
        if (step_idx + 1) % 20 == 0:
            print(f"Step {step_idx + 1}/200 completed. Mean residual: {solver.residuals[-1]:.6f}")
            
    # Save frames as GIF
    os.makedirs('data', exist_ok=True)
    gif_path = os.path.join('data', 'concentration_simulation.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"Successfully saved concentration simulation to {gif_path}.")
