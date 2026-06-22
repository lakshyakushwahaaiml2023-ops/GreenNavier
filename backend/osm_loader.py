import os
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from PIL import Image, ImageDraw

# Configure osmnx cache
try:
    ox.settings.use_cache = True
    ox.settings.log_console = True
except AttributeError:
    ox.config(use_cache=True, log_console=True)

def safe_get_scalar(row, col_name):
    if col_name is None or col_name not in row:
        return None
    val = row[col_name]
    if val is None:
        return None
    if isinstance(val, (list, np.ndarray, pd.Series)):
        if len(val) > 0:
            val = val[0]
        else:
            return None
    if pd.isna(val):
        return None
    return val

def fetch_region(center_lat, center_lon, radius_m=800):
    """
    Downloads building footprints and road network for a circle of radius_m around the given center point.
    
    Returns:
      - buildings: list of (polygon, height_m) in EPSG:4326 (lat/lon)
      - roads: list of (linestring, road_type) in EPSG:4326 (lat/lon)
    """
    center_point = (center_lat, center_lon)
    print(f"Fetching region around center ({center_lat}, {center_lon}) with radius {radius_m}m...")

    # 1. Fetch buildings
    try:
        # Newer osmnx versions use features_from_point, older use geometries_from_point
        try:
            gdf_buildings = ox.features_from_point(center_point, tags={'building': True}, dist=radius_m)
        except AttributeError:
            gdf_buildings = ox.geometries_from_point(center_point, tags={'building': True}, dist=radius_m)
        print(f"Fetched {len(gdf_buildings)} building features.")
    except Exception as e:
        print(f"Warning: Failed to fetch buildings: {e}")
        gdf_buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # 2. Fetch roads network
    try:
        # network_type='all' fetches all roads and paths
        G = ox.graph_from_point(center_point, dist=radius_m, network_type='all')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        print(f"Fetched {len(gdf_edges)} road segments.")
    except Exception as e:
        print(f"Warning: Failed to fetch roads: {e}")
        gdf_edges = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # 3. Extract buildings: list of (polygon, height_m)
    buildings = []
    if not gdf_buildings.empty:
        levels_col = 'building:levels' if 'building:levels' in gdf_buildings.columns else None
        height_col = 'height' if 'height' in gdf_buildings.columns else None
        
        for idx, row in gdf_buildings.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            # Estimate height: building:levels * 3.5m, default 10.5m
            height = 10.5
            levels_val = safe_get_scalar(row, levels_col)
            height_val = safe_get_scalar(row, height_col)
                
            if levels_val is not None:
                try:
                    levels_num = float(str(levels_val).strip())
                    height = levels_num * 3.5
                except ValueError:
                    pass
            elif height_val is not None:
                try:
                    val_str = str(height_val).lower().replace('m', '').strip()
                    height = float(val_str)
                except ValueError:
                    pass
            
            if geom.geom_type == 'Polygon':
                buildings.append((geom, height))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    buildings.append((poly, height))

    # 4. Extract roads: list of (linestring, road_type, name)
    roads = []
    if not gdf_edges.empty:
        highway_col = 'highway' if 'highway' in gdf_edges.columns else None
        name_col = 'name' if 'name' in gdf_edges.columns else None
        for idx, row in gdf_edges.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            road_type = 'residential'
            h_val = safe_get_scalar(row, highway_col)
            if h_val is not None:
                h_val = str(h_val).lower().strip()
                
                primary_types = {'primary', 'motorway', 'trunk', 'primary_link', 'motorway_link', 'trunk_link'}
                secondary_types = {'secondary', 'tertiary', 'secondary_link', 'tertiary_link'}
                
                if h_val in primary_types:
                    road_type = 'primary'
                elif h_val in secondary_types:
                    road_type = 'secondary'
                else:
                    road_type = 'residential'
                    
            name_val = safe_get_scalar(row, name_col) if name_col else None
            if not name_val:
                name_val = f"Unnamed {road_type.capitalize()}"
            else:
                name_val = str(name_val)
                    
            if geom.geom_type == 'LineString':
                roads.append((geom, road_type, name_val))
            elif geom.geom_type == 'MultiLineString':
                for ls in geom.geoms:
                    roads.append((ls, road_type, name_val))
                    
    return buildings, roads

def project_geom(geom, lat_center, lon_center, meters_per_deg_lat, meters_per_deg_lon):
    """
    Projects a shapely geometry from WGS84 to local meters centered at (lon_center, lat_center)
    """
    def project_pt(lon, lat):
        x = (lon - lon_center) * meters_per_deg_lon
        y = (lat - lat_center) * meters_per_deg_lat
        return (x, y)
    
    if geom.geom_type == 'Polygon':
        ext = [project_pt(x, y) for x, y in geom.exterior.coords]
        ints = [[project_pt(x, y) for x, y in interior.coords] for interior in geom.interiors]
        return Polygon(ext, ints)
    elif geom.geom_type == 'MultiPolygon':
        polys = []
        for poly in geom.geoms:
            ext = [project_pt(x, y) for x, y in poly.exterior.coords]
            ints = [[project_pt(x, y) for x, y in interior.coords] for interior in poly.interiors]
            polys.append(Polygon(ext, ints))
        return MultiPolygon(polys)
    elif geom.geom_type == 'LineString':
        coords = [project_pt(x, y) for x, y in geom.coords]
        return LineString(coords)
    elif geom.geom_type == 'MultiLineString':
        lines = []
        for ls in geom.geoms:
            coords = [project_pt(x, y) for x, y in ls.coords]
            lines.append(LineString(coords))
        return MultiLineString(lines)
    return geom

def rasterize(buildings, roads, grid_size=128, center_lat=22.7533, center_lon=75.8937, radius_m=800):
    """
    Projects geometries and rasterizes them onto a grid_size x grid_size numpy grid.
    Saves masks to data/grid_masks.npz and a verification PNG.
    """
    print("Rasterizing data...")
    # Calculate accurate scaling factors at local latitude
    a = 6378137.0  # Earth equatorial radius (meters)
    e2 = 0.00669437999014  # Eccentricity squared
    
    lat_rad = np.radians(center_lat)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    M = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad)**2)**1.5
    
    meters_per_deg_lat = M * (np.pi / 180.0)
    meters_per_deg_lon = N * np.cos(lat_rad) * (np.pi / 180.0)
    
    # Project all geometries to meters relative to center
    proj_buildings = [(project_geom(poly, center_lat, center_lon, meters_per_deg_lat, meters_per_deg_lon), h) for poly, h in buildings]
    proj_roads = [(project_geom(ls, center_lat, center_lon, meters_per_deg_lat, meters_per_deg_lon), t) for ls, t, *_ in roads]
    
    # Bounding box in meters
    min_x = -radius_m
    max_x = radius_m
    min_y = -radius_m
    max_y = radius_m
    
    # Helper to map coordinates to grid pixels
    def map_x(x):
        return int((x - min_x) / (max_x - min_x) * grid_size)
    def map_y(y):
        # row 0 is top (max_y), row 127 is bottom (min_y)
        return int((max_y - y) / (max_y - min_y) * grid_size)
    
    def draw_poly(draw, poly, fill_val):
        if poly.exterior is None:
            return
        ext_coords = [(map_x(x), map_y(y)) for x, y in poly.exterior.coords]
        draw.polygon(ext_coords, fill=fill_val)
        for interior in poly.interiors:
            int_coords = [(map_x(x), map_y(y)) for x, y in interior.coords]
            draw.polygon(int_coords, fill=0)

    def draw_line(draw, geom, fill_val, width):
        if geom.geom_type == 'LineString':
            coords = [(map_x(x), map_y(y)) for x, y in geom.coords]
            if len(coords) >= 2:
                draw.line(coords, fill=fill_val, width=width)
        elif geom.geom_type == 'MultiLineString':
            for ls in geom.geoms:
                coords = [(map_x(x), map_y(y)) for x, y in ls.coords]
                if len(coords) >= 2:
                    draw.line(coords, fill=fill_val, width=width)

    # 1. Initialize arrays
    obstacle_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    height_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Draw buildings
    for poly, height in proj_buildings:
        # Create temp PIL image for this building
        temp_img = Image.new('L', (grid_size, grid_size), 0)
        draw = ImageDraw.Draw(temp_img)
        draw_poly(draw, poly, 1)
        
        mask = np.array(temp_img) > 0
        obstacle_mask[mask] = 1.0
        height_map[mask] = np.maximum(height_map[mask], height)
        
    # Draw roads
    road_img = Image.new('L', (grid_size, grid_size), 0)
    draw = ImageDraw.Draw(road_img)
    for ls, road_type in proj_roads:
        # Determine cell width
        # primary: 3 cells, secondary/residential: 1 cell
        width = 3 if road_type == 'primary' else 1
        draw_line(draw, ls, 255, width)
        
    road_mask = (np.array(road_img) > 0).astype(np.float32)
    
    # Save npz file
    os.makedirs('data', exist_ok=True)
    npz_path = os.path.join('data', 'grid_masks.npz')
    np.savez_compressed(npz_path, 
                       obstacle_mask=obstacle_mask, 
                       road_mask=road_mask, 
                       height_map=height_map,
                       center_lat=center_lat,
                       center_lon=center_lon,
                       radius_m=radius_m)
    print(f"Saved numpy masks to {npz_path}")
    
    # Save verification PNG: buildings=dark gray (64,64,64), roads=yellow (255,220,0), air=white (255,255,255)
    vis_arr = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 255 # default air = white
    
    # Dark gray for buildings
    vis_arr[obstacle_mask > 0.5] = [64, 64, 64]
    
    # Yellow for roads
    vis_arr[road_mask > 0.5] = [255, 220, 0]
    
    png_path = os.path.join('data', 'grid_masks_verification.png')
    vis_img = Image.fromarray(vis_arr)
    vis_img.save(png_path)
    print(f"Saved verification PNG to {png_path}")
    
    return obstacle_mask, road_mask, height_map

def get_bbox_meters(min_lat, max_lat, min_lon, max_lon):
    """
    Computes width and height in meters, center lat/lon, and scaling factors for a bounding box.
    """
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    a = 6378137.0  # Earth equatorial radius (meters)
    e2 = 0.00669437999014  # Eccentricity squared
    
    lat_rad = np.radians(center_lat)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    M = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad)**2)**1.5
    
    meters_per_deg_lat = M * (np.pi / 180.0)
    meters_per_deg_lon = N * np.cos(lat_rad) * (np.pi / 180.0)
    
    width_m = (max_lon - min_lon) * meters_per_deg_lon
    height_m = (max_lat - min_lat) * meters_per_deg_lat
    
    return width_m, height_m, center_lat, center_lon, meters_per_deg_lat, meters_per_deg_lon

def fetch_and_rasterize_bbox(min_lat, max_lat, min_lon, max_lon):
    """
    Fetches OSM geometry for a bounding box, pads it to a square, and rasterizes it to a 128x128 grid.
    """
    # 1. Get sizes
    width_m, height_m, center_lat, center_lon, meters_per_deg_lat, meters_per_deg_lon = get_bbox_meters(
        min_lat, max_lat, min_lon, max_lon
    )
    
    # Validate size limits (200m - 1500m across)
    if width_m < 200.0 or height_m < 200.0 or width_m > 1500.0 or height_m > 1500.0:
        raise ValueError("Please select an area between 200m and 1500m across")
        
    # Pad to square to preserve aspect ratio
    size_m = max(width_m, height_m)
    radius_m = size_m / 2.0
    
    half_size_lat = radius_m / meters_per_deg_lat
    half_size_lon = radius_m / meters_per_deg_lon
    
    pad_min_lat = center_lat - half_size_lat
    pad_max_lat = center_lat + half_size_lat
    pad_min_lon = center_lon - half_size_lon
    pad_max_lon = center_lon + half_size_lon
    
    print(f"Original select bbox meters: {width_m:.1f}m x {height_m:.1f}m")
    print(f"Padded square bbox center: ({center_lat}, {center_lon}), radius: {radius_m:.1f}m")
    
    # 2. Fetch from OSMnx (or use offline pre-cached files if available)
    # Bbox format in OSMnx 2.x is (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)
    bbox_tuple = (pad_min_lon, pad_min_lat, pad_max_lon, pad_max_lat)
    
    offline_buildings_path = 'data/indore_buildings_offline.pkl'
    offline_roads_path = 'data/indore_roads_offline.pkl'
    
    use_offline = os.path.exists(offline_buildings_path) and os.path.exists(offline_roads_path)
    
    if use_offline:
        import pickle
        from shapely.geometry import box
        print("[OSM] Offline cache detected. Loading and clipping region offline...")
        
        # Load buildings
        try:
            with open(offline_buildings_path, 'rb') as f:
                gdf_buildings_all = pickle.load(f)
            bbox_poly = box(pad_min_lon, pad_min_lat, pad_max_lon, pad_max_lat)
            gdf_buildings = gpd.clip(gdf_buildings_all, bbox_poly)
            print(f"[OSM] Loaded & clipped {len(gdf_buildings)} buildings from local cache.")
        except Exception as e:
            print(f"Warning: Failed to load offline buildings: {e}. Falling back to online.")
            use_offline = False
            
        # Load roads
        if use_offline:
            try:
                with open(offline_roads_path, 'rb') as f:
                    G_all = pickle.load(f)
                G = ox.truncate.truncate_graph_bbox(G_all, bbox_tuple)
                gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
                print(f"[OSM] Loaded & clipped {len(gdf_edges)} road segments from local cache.")
            except Exception as e:
                print(f"Warning: Failed to load offline roads: {e}. Falling back to online.")
                use_offline = False

    if not use_offline:
        # Fetch buildings
        try:
            try:
                gdf_buildings = ox.features_from_bbox(bbox=bbox_tuple, tags={'building': True})
            except TypeError:
                # Fallback for positional signature
                gdf_buildings = ox.features_from_bbox(pad_max_lat, pad_min_lat, pad_max_lon, pad_min_lon, tags={'building': True})
            print(f"Fetched {len(gdf_buildings)} building features.")
        except Exception as e:
            print(f"Warning: Failed to fetch buildings: {e}")
            gdf_buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
        # Fetch road network
        try:
            try:
                G = ox.graph_from_bbox(bbox=bbox_tuple, network_type='all')
            except TypeError:
                # Fallback for positional signature
                G = ox.graph_from_bbox(pad_max_lat, pad_min_lat, pad_max_lon, pad_min_lon, network_type='all')
            gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
            print(f"Fetched {len(gdf_edges)} road segments.")
        except Exception as e:
            print(f"Warning: Failed to fetch roads: {e}")
            gdf_edges = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    # 3. Parse buildings: list of (polygon, height_m)
    buildings = []
    if not gdf_buildings.empty:
        levels_col = 'building:levels' if 'building:levels' in gdf_buildings.columns else None
        height_col = 'height' if 'height' in gdf_buildings.columns else None
        
        for idx, row in gdf_buildings.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            height = 10.5
            levels_val = safe_get_scalar(row, levels_col)
            height_val = safe_get_scalar(row, height_col)
                
            if levels_val is not None:
                try:
                    levels_num = float(str(levels_val).strip())
                    height = levels_num * 3.5
                except ValueError:
                    pass
            elif height_val is not None:
                try:
                    val_str = str(height_val).lower().replace('m', '').strip()
                    height = float(val_str)
                except ValueError:
                    pass
            
            if geom.geom_type == 'Polygon':
                buildings.append((geom, height))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    buildings.append((poly, height))
                    
    # 4. Parse roads: list of (linestring, road_type, name)
    roads = []
    if not gdf_edges.empty:
        highway_col = 'highway' if 'highway' in gdf_edges.columns else None
        name_col = 'name' if 'name' in gdf_edges.columns else None
        for idx, row in gdf_edges.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            road_type = 'residential'
            h_val = safe_get_scalar(row, highway_col)
            if h_val is not None:
                h_val = str(h_val).lower().strip()
                
                primary_types = {'primary', 'motorway', 'trunk', 'primary_link', 'motorway_link', 'trunk_link'}
                secondary_types = {'secondary', 'tertiary', 'secondary_link', 'tertiary_link'}
                
                if h_val in primary_types:
                    road_type = 'primary'
                elif h_val in secondary_types:
                    road_type = 'secondary'
                else:
                    road_type = 'residential'
                    
            name_val = safe_get_scalar(row, name_col) if name_col else None
            if not name_val:
                name_val = f"Unnamed {road_type.capitalize()}"
            else:
                name_val = str(name_val)
                    
            if geom.geom_type == 'LineString':
                roads.append((geom, road_type, name_val))
            elif geom.geom_type == 'MultiLineString':
                for ls in geom.geoms:
                    roads.append((ls, road_type, name_val))
                    
    # 5. Rasterize and save to NPZ
    obstacle_mask, road_mask, height_map = rasterize(
        buildings, roads, grid_size=128, center_lat=center_lat, center_lon=center_lon, radius_m=radius_m
    )
    
    return buildings, roads, obstacle_mask, road_mask, height_map, center_lat, center_lon, radius_m

if __name__ == '__main__':
    # Vijay Nagar Square, Indore coordinates
    lat = 22.7533
    lon = 75.8937
    radius = 800
    
    buildings, roads = fetch_region(lat, lon, radius)
    rasterize(buildings, roads, grid_size=128, center_lat=lat, center_lon=lon, radius_m=radius)
