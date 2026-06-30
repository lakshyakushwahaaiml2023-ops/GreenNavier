# 🌿 GreenNavier — Real-Time Urban Air Dispersion Simulator

> **Physics-informed fluid simulation meets interactive 3D city visualization.**  
> Simulate, predict, and mitigate urban air pollution in real time — powered by Navier-Stokes fluid dynamics and a live-training PINN error estimator.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r165-black?style=flat&logo=three.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OSMnx](https://img.shields.io/badge/OSMnx-Real--World%20Maps-7FC97F?style=flat)

---

## 📌 What Is GreenNavier?

GreenNavier simulates how multiple coupled pollutant species disperse through a real city grid — specifically the **Indore, India** urban core — under live-configurable wind conditions, traffic loads, and diurnal chemical reactions.

It combines:
- A **128×128 Stable Fluids** Navier-Stokes solver coupled with a positive-definite mass-conserving chemistry box solver tracking CO, NO, NO₂, and O₃
- A **6-Channel PINN (Physics-Informed Neural Network)** error estimator that monitors solver consistency and flags physics violations
- A **Three.js 3D scene** built from real OpenStreetMap buildings and roads, with a real-time volumetric smoke renderer and active species selector HUD
- **Interactive demo modes** including a scripted Green Corridor intervention that reduces mean grid concentrations by ~14%

### 🆕 Advanced Interactive Features & Interventions

GreenNavier supports advanced real-time modifications directly inside the 3D viewport:
- **Interactive Bounding Box Region Selector**: An integrated fullscreen Leaflet map on startup lets you draw a custom simulation bounding box ($200\text{ m} - 1500\text{ m}$) centered anywhere in Indore. The 3D scene and fluid grids reload and rebuild dynamically.
- **100% Offline Indore Mode**: Dynamically clips and loads OpenStreetMap buildings and roads locally using pre-cached pickles, enabling offline simulation.
- **Drag-to-Draw Building Placement**: Click and drag on the ground to draw custom building footprints with grid snapping. Choose its height ($3\text{ m} - 50\text{ m}$) using a floating popup, extruding it as a distinct blue-gray building with a temporary 2-second pulsing highlight.
- **8-Handle Building Resizing & Translation**: Clicking any custom building reveals 8 drag handles (white cubes) on its base to adjust width/depth with a live dimensions label, or drag the building body to translate it on the grid.
- **Interactive Point Source Factories**: Click to place custom factory cylinders with shimmering top faces and ambient orange PointLights. Adjust emission strength via floating sliders and radius via a draggable base ring.
- **Escape Key Actions**: Instantly cancels active placement actions or deselects current objects.
- **Panel Minimize / Restore**: Collapse the controls panel down off-screen to maximize viewport space (retaining stats visibility) using a close button, and restore it via a floating gear settings button.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Three.js)                        │
│  3D city mesh · pollution heatmap · wind particles · UI panels  │
└─────────────────────┬────────────────────────┬──────────────────┘
                      │  WebSocket (frames)    │  REST (controls)
                      ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend  (Python)                    │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  StableFluidsSolver  │    │     PINNErrorEstimator        │   │
│  │  128×128 grid        │───▶│  Tiny 3-layer CNN (PyTorch)  │   │
│  │  Semi-Lagrangian     │    │  Predicts divergence error   │   │
│  │  advection + proj.   │    │  Trains every 500 steps      │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│                                                                  │
│  OSMnx · Shapely · NumPy · Pillow                               │
└─────────────────────────────────────────────────────────────────┘
                      │
                 data/grid_masks.npz
            (obstacle · road · height maps)
```

### Data Flow
1. **Startup**: Backend loads real OSM geometry for Indore 800 m radius → rasterizes buildings and 1,950 road segments onto a 128×128 grid
2. **Simulation loop**: Every 50 ms — advect velocity → diffuse → project (div-free) → advect concentration → apply road emissions → apply green corridor absorption
3. **PINN inference**: Every 10 steps, the CNN predicts the physics residual from the current (conc, u, v) state; flags a warning if score > 0.05
4. **WebSocket broadcast**: Downsampled 64×64 concentration grid + metadata sent to all connected clients
5. **Frontend render**: Three.js maps concentration to a colour texture on a ground plane, updates wind particles, and drives demo overlays

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Fluid Solver** | NumPy (custom) | Stable Fluids Navier-Stokes + positive-definite mass-conserving chemistry box model (CO, NO, NO₂, O₃) |
| **PINN Monitor** | PyTorch 2.x | 6-channel input CNN; predicts divergence residual; trained online on simulation buffer |
| **Backend API** | FastAPI + Uvicorn | WebSocket frame streaming; REST endpoints for wind/traffic/corridor/reset |
| **Map Data** | OSMnx + Shapely | Fetches and rasterizes real OpenStreetMap buildings & roads |
| **3D Renderer** | Three.js (r165) | City mesh, pollution heatmap texture, wind particle system, camera orbits |
| **Visualization** | Matplotlib + Pillow | Poster figure generation (300 DPI PNGs) |
| **Language** | Python 3.10+ / Vanilla JS | No build step required |

---

## 🌍 Real-World Impact & Benefits

### Air Quality Insights
- **Urban canyon effect**: The solver captures how pollution pools 2–3× higher in street canyons between tall buildings compared to open areas — a well-documented meteorological phenomenon
- **Rush-hour spike modelling**: Traffic intensity is controllable per road segment, allowing realistic 6 pm surge simulations
- **Green corridor mitigation**: A 15%-wide vegetated strip modelled with 80% per-step PM2.5 absorption (representative of a dense multi-row tree canopy) reduces mean domain PM2.5 by **~13.9%** at steady state

### Decision-Support for Urban Planners
- Planners can test "what if" scenarios — adding a corridor, changing traffic routing, planting trees along arterial roads — and see pollution consequences in seconds
- The 3D view with real OSM geometry makes results immediately interpretable to non-specialists

### Research Validation
- The PINN residual monitor provides **continuous physics quality assurance** — flagging when the numerical solver diverges from physical correctness, which is otherwise invisible in black-box simulations
- Mean divergence residual: `0.015` (no corridor) vs `0.014` (green corridor) — the corridor's velocity dampening measurably improves flow regularity

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Mean PM2.5 reduction — green corridor (peak) | **13.9 %** |
| Mean PM2.5 reduction — green corridor (steady-state) | **13.8 %** |
| Physics residual — no corridor | `0.01524` |
| Physics residual — with corridor | `0.01412` (7.3 % lower) |
| Solver step time | **~27 ms** per step |
| PINN-monitored adaptive (estimated) | **~23 ms** per step (~15 % faster) |
| Road segments modelled | **1,950** (real OSM, Indore) |
| Building features | **4,506** |

---

## 🚀 Quickstart

### Prerequisites

- Python **3.10+**
- `pip` (standard)
- A modern browser (Chrome / Firefox / Edge)

### 1 · Clone and set up

```bash
git clone <your-repo-url>
cd GreenNavier2/greennavier

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2 · Pre-generate grid masks (first time only)

The simulation needs a rasterized map of buildings and roads. Run:

```bash
python -c "
from backend.osm_loader import fetch_region, build_grid_masks
import numpy as np, os
buildings, roads = fetch_region(22.7533, 75.8937, 800)
masks = build_grid_masks(buildings, roads)
os.makedirs('data', exist_ok=True)
np.savez('data/grid_masks.npz', **masks)
print('Saved data/grid_masks.npz')
"
```

> **Note**: OSM data is cached in `cache/` after the first fetch — subsequent runs are instant.

### 3 · Start the backend

```bash
# From inside greennavier/
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
Loaded grid masks from data/grid_masks.npz.
Initialized 1950 road segments from OSM database.
Simulation background worker started successfully.
```

### 4 · Open the frontend

Open `frontend/index.html` directly in your browser:

```
# Windows
start frontend\index.html

# macOS
open frontend/index.html

# Or simply drag the file into Chrome/Firefox
```

The 3D city will load and the pollution heatmap will begin updating in real time.

---

## 🎮 Demo Modes

Click **"▶ Demo Mode"** in the top-right panel to run a scripted three-act sequence:

| Act | What Happens |
|---|---|
| **Moment 1 · Urban Canyon** | West wind + moderate traffic. Camera orbits to the densest building cluster. Overlay: *"PM2.5 trapped — 2.4× higher than open areas"* |
| **Moment 2 · Rush Hour Surge** | All traffic jumps to 80 %. Camera pulls back. Pollution intensifies toward red. Stats panel shows PM2.5 climbing. |
| **Moment 3 · Green Corridor** | A semi-transparent green strip appears along the main arterial. Trees absorb 35 % of passing pollution per step. Overlay shows live reduction %. |

---

## 📁 Project Structure

```
greennavier/
├── backend/
│   ├── main.py          # FastAPI app — WebSocket, REST endpoints, simulation loop
│   ├── solver.py        # StableFluidsSolver — Navier-Stokes on 128×128 grid
│   ├── pinn.py          # PINNErrorEstimator — 3-layer CNN, online training
│   └── osm_loader.py    # OSMnx fetch + rasterise to grid masks (online/offline)
├── frontend/
│   └── index.html       # Full Three.js app (self-contained, no build step)
├── data/
│   ├── grid_masks.npz   # Pre-rasterised obstacle / road / height maps
│   ├── pinn_estimator.pt  # Saved PINN weights (auto-generated at runtime)
│   ├── indore_buildings_offline.pkl  # Pre-cached Indore buildings for offline clipping
│   ├── indore_roads_offline.pkl      # Pre-cached Indore roads for offline clipping
│   └── osm_features.pkl              # Pre-cached general OSM features
├── cache/               # OSMnx HTTP cache (JSON)
├── figures/             # Generated poster figures (300 DPI PNGs)
├── generate_figures.py  # Standalone script — runs solver, saves all poster figures
├── requirements.txt
└── README.md
```

---

## 📈 Generating Poster Figures

To regenerate all four publication-quality figures:

```bash
# From inside greennavier/ with venv active
python generate_figures.py
```

This runs the solver for **500 steps × 2 configurations** (no corridor / with green corridor), triggers rush hour at step 150, and saves:

| Output | Description |
|---|---|
| `figures/pollution_comparison.png` | Time-series of mean PM2.5 with rush-hour line and PINN warning markers |
| `figures/performance_comparison.png` | Bar chart: full solver vs PINN-monitored adaptive |
| `figures/peak_no_corridor.png` | Heatmap at step 300 — no corridor |
| `figures/peak_with_corridor.png` | Heatmap at step 300 — green corridor active |

---

## 🔭 Future Scope

### Completed Implementations
- [x] **3D pollution volume rendering** — Integrated real-time WebGL volumetric fog rendering with slider-controlled density
- [x] **Time-of-day scheduling** — Added presets (morning, noon, evening, night) that automatically animate wind parameters and scale traffic emissions dynamically
- [x] **Dynamic City Scaling** — Enabled interactive region selector (Leaflet map) to select custom 3D viewports dynamically
- [x] **Multi-species transport & photochemistry** — Added CO, NO, NO₂, and O₃ concentration fields with coupled reactions and a 6-channel neural monitor

### Near-Term
- [ ] **Wind field from real sensor data** — ingest live anemometer readings from Indore Smart City API to replace the uniform wind body force

### Medium-Term
- [ ] **PINN-adaptive solver** — when the PINN flags low residual steps, skip the expensive Helmholtz projection and use the PINN's predicted correction instead (~15 % speedup already estimated)
- [ ] **Mobile-friendly UI** — replace Three.js orbit controls with touch gestures; target 30 fps on mid-range Android devices
- [ ] **Multi-city support** — parameterise OSM fetch coordinates for any city; add a city picker dropdown
- [ ] **Health impact overlay** — map PM2.5 concentration to WHO exposure risk categories with colour-coded zone overlays

### Long-Term Research
- [ ] **Coupled mesoscale model** — replace the uniform wind field with a downscaled WRF (Weather Research & Forecasting) output at 100 m resolution
- [ ] **Inverse design optimisation** — use the differentiable solver to automatically find optimal green corridor placement that minimises population-weighted exposure
- [ ] **Federated sensor assimilation** — assimilate real-time PM2.5 sensor readings (e.g., PurpleAir network) to continuously correct the simulation state

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Contributors

| Name | Role |
|---|---|
| Lakshya | Core simulation engine, PINN integration, 3D frontend |

---

> *Built as a Minor Project — Department of Artificial Inteligence and Machine Learning*  
> *Demonstrating the intersection of computational fluid dynamics, machine learning, and urban sustainability.*
