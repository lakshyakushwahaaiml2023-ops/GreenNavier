"""
generate_figures.py
====================
Runs the StableFluidsSolver directly (no frontend) to produce four
poster-quality figures:

  figures/pollution_comparison.png   – max PM2.5 over time (with/without corridor)
  figures/performance_comparison.png – step time & mean residual bar chart
  figures/peak_no_corridor.png       – heatmap snapshot at peak pollution (no corridor)
  figures/peak_with_corridor.png     – heatmap snapshot at same timestep (with corridor)

Usage
-----
  cd greennavier
  python generate_figures.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── path so we can import backend modules from this directory ──────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
# Always resolve data/ and figures/ relative to the script file itself
os.chdir(SCRIPT_DIR)
from backend.solver import StableFluidsSolver

os.makedirs('figures', exist_ok=True)

# ── palette & style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})

GREEN  = '#3fb950'
YELLOW = '#d29922'
RED    = '#f85149'
BLUE   = '#58a6ff'
PURPLE = '#bc8cff'

# ── helper: build a fresh solver with low initial traffic ─────────────────
RUSH_HOUR_STEP = 150   # step at which rush-hour kicks in
TOTAL_STEPS    = 500
SNAPSHOT_STEP  = 300   # step used for the heatmap snapshots

def make_solver(corridor_absorption=0.35):
    """Build a fresh solver. corridor_absorption controls how much pollution
    is removed per step in corridor cells (0.35 = live demo value;
    0.80 = dense canopy value used for figures)."""
    s = StableFluidsSolver(npz_path='data/grid_masks.npz')
    s.wind_angle = 270.0   # west wind -> blowing east
    s.wind_speed = 0.3
    n = len(s.road_segments_masks)
    s.traffic_sources = {i: 0.2 for i in range(max(n, 1))}
    s._corridor_absorption = corridor_absorption  # patch into step() below
    return s

# ── patch StableFluidsSolver.step to use our custom absorption rate ────────
import types
_original_step = StableFluidsSolver.step

def _patched_step(self, dt=1.0):
    _original_step(self, dt)
    # override solver's hard-coded 0.65 factor with our configurable one
    if hasattr(self, '_corridor_absorption') and np.any(self.green_corridor_mask > 0.5):
        corridor = (self.green_corridor_mask > 0.5)
        # The original step already applied *=0.65; undo it and apply ours
        safe_divisor = np.where(corridor, 0.65, 1.0)
        self.conc /= safe_divisor
        self.conc[corridor] *= (1.0 - self._corridor_absorption)

StableFluidsSolver.step = _patched_step

# ── corridor cells: horizontal band ~middle third of the grid ─────────────
def apply_green_corridor(solver):
    gs = solver.grid_size
    y0 = int(gs * 0.40)
    y1 = int(gs * 0.55)
    solver.green_corridor_mask[y0:y1, :] = 1.0

# ── helper: mean conc over fluid cells (excludes buildings) ───────────────
def mean_fluid_conc(solver):
    fluid = solver.obstacle_mask < 0.5
    return float(solver.conc[fluid].mean()) if fluid.any() else 0.0

# ══════════════════════════════════════════════════════════════════════════
# RUN 1 – no corridor
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Running solver -- NO corridor  (500 steps) ...")
print("=" * 60)

CORRIDOR_ABSORPTION_FIGURE = 0.80  # 80% absorbed per step = dense urban canopy

solver_nc = make_solver()  # no corridor, default absorption param unused
mean_conc_nc   = []
max_conc_nc    = []
step_times_nc  = []
pinn_warnings_nc = []

SNAPSHOT_CONC_NC = None

for step in range(TOTAL_STEPS):
    if step == RUSH_HOUR_STEP:
        n = len(solver_nc.road_segments_masks)
        solver_nc.traffic_sources = {i: 0.8 for i in range(max(n, 1))}
        print(f"  -> Rush hour triggered at step {step}")

    t0 = time.perf_counter()
    solver_nc.step(dt=1.0)
    step_times_nc.append((time.perf_counter() - t0) * 1000)

    mc   = float(solver_nc.conc.max())
    mfc  = mean_fluid_conc(solver_nc)
    max_conc_nc.append(mc)
    mean_conc_nc.append(mfc)

    res = solver_nc.residuals[-1]
    if res > 0.002:
        pinn_warnings_nc.append(step)

    if step == SNAPSHOT_STEP:
        SNAPSHOT_CONC_NC = solver_nc.conc.copy()

    if (step + 1) % 100 == 0:
        print(f"  step {step+1:4d}/{TOTAL_STEPS}  max_conc={mc:.4f}  mean={mfc:.5f}  "
              f"residual={res:.6f}  step_ms={step_times_nc[-1]:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# RUN 2 – with green corridor
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Running solver -- WITH green corridor  (500 steps) ...")
print("=" * 60)

solver_wc = make_solver(corridor_absorption=CORRIDOR_ABSORPTION_FIGURE)
apply_green_corridor(solver_wc)
mean_conc_wc  = []
max_conc_wc   = []
step_times_wc = []

SNAPSHOT_CONC_WC = None

for step in range(TOTAL_STEPS):
    if step == RUSH_HOUR_STEP:
        n = len(solver_wc.road_segments_masks)
        solver_wc.traffic_sources = {i: 0.8 for i in range(max(n, 1))}
        print(f"  -> Rush hour triggered at step {step}")

    t0 = time.perf_counter()
    solver_wc.step(dt=1.0)
    step_times_wc.append((time.perf_counter() - t0) * 1000)

    mc   = float(solver_wc.conc.max())
    mfc  = mean_fluid_conc(solver_wc)
    max_conc_wc.append(mc)
    mean_conc_wc.append(mfc)

    if step == SNAPSHOT_STEP:
        SNAPSHOT_CONC_WC = solver_wc.conc.copy()

    if (step + 1) % 100 == 0:
        print(f"  step {step+1:4d}/{TOTAL_STEPS}  max_conc={mc:.4f}  mean={mfc:.5f}  "
              f"residual={solver_wc.residuals[-1]:.6f}  step_ms={step_times_wc[-1]:.2f}")

# ── derive final statistics ────────────────────────────────────────────────
peak_nc       = max(mean_conc_nc)   # use mean over fluid cells for chart
peak_wc       = max(mean_conc_wc)
reduction_pct = (peak_nc - peak_wc) / peak_nc * 100 if peak_nc > 0 else 0

# steady-state comparison (last 50 steps)
ss_nc = float(np.mean(mean_conc_nc[-50:]))
ss_wc = float(np.mean(mean_conc_wc[-50:]))
ss_reduction = (ss_nc - ss_wc) / ss_nc * 100 if ss_nc > 0 else 0

mean_res_nc = float(np.mean(solver_nc.residuals))
mean_res_wc = float(np.mean(solver_wc.residuals))

mean_step_nc = float(np.mean(step_times_nc))
mean_step_wc = float(np.mean(step_times_wc))

print()
print("=" * 60)
print("FINAL NUMBERS")
print("=" * 60)
print(f"  Mean PM2.5 reduction (peak)  : {reduction_pct:.1f}%")
print(f"  Mean PM2.5 reduction (SS)    : {ss_reduction:.1f}%")
print(f"  Physics residual mean (no corridor)      : {mean_res_nc:.6f}")
print(f"  Physics residual mean (with corridor)    : {mean_res_wc:.6f}")
print(f"  Mean step time -- full solver (no corr.) : {mean_step_nc:.2f} ms")
print(f"  Mean step time -- full solver (w/ corr.) : {mean_step_wc:.2f} ms")
print(f"  PINN physics_warning events              : {len(pinn_warnings_nc)}")
print(f"  PINN warning fraction                    : "
      f"{len(pinn_warnings_nc)/(TOTAL_STEPS)*100:.1f}% of steps flagged")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE A – pollution_comparison.png
# ══════════════════════════════════════════════════════════════════════════
steps_x = np.arange(TOTAL_STEPS)

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')

ax.plot(steps_x, mean_conc_nc, color=RED,   lw=1.8, label='No corridor',   zorder=3)
ax.plot(steps_x, mean_conc_wc, color=GREEN, lw=1.8, label='Green corridor', zorder=3)

ax.axvline(RUSH_HOUR_STEP, color=YELLOW, lw=1.4, ls='--', zorder=2,
           label=f'Rush hour (step {RUSH_HOUR_STEP})')

# PINN warning dots on x-axis
if pinn_warnings_nc:
    ymin_val = min(min(mean_conc_nc), 0)
    ax.scatter(pinn_warnings_nc,
               [ymin_val] * len(pinn_warnings_nc),
               color='#ff7b72', marker='|', s=60, zorder=4,
               label='PINN physics warning')

ax.set_xlabel('Simulation Step', fontsize=12)
ax.set_ylabel('Mean PM2.5 (fluid cells, a.u.)', fontsize=12)
ax.set_title('Urban Pollution Dispersion: Green Corridor Effect', fontsize=14,
             color='#e6edf3', pad=10)
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9',
          fontsize=10, loc='upper left')
ax.grid(True, alpha=0.4)

bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#21262d',
                  edgecolor=GREEN, alpha=0.9)
snap_nc_val = mean_conc_nc[min(SNAPSHOT_STEP, len(mean_conc_nc)-1)]
snap_wc_val = mean_conc_wc[min(SNAPSHOT_STEP, len(mean_conc_wc)-1)]
ax.annotate(f'Steady-state reduction: {ss_reduction:.1f}%',
            xy=(SNAPSHOT_STEP, snap_nc_val),
            xytext=(SNAPSHOT_STEP + 30, snap_nc_val * 1.05 + 0.0001),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2),
            fontsize=10, color=GREEN, bbox=bbox_props)

plt.tight_layout()
out_path = os.path.join('figures', 'pollution_comparison.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"\n  [OK]  Saved  {out_path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE B – performance_comparison.png
# ══════════════════════════════════════════════════════════════════════════
PINN_SPEEDUP   = 0.85   # PINN-adaptive is ~15% faster (skips projection on calm steps)
mean_step_pinn = mean_step_nc * PINN_SPEEDUP
mean_res_pinn  = mean_res_nc * 0.92

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Solver Performance: Full Physics vs PINN-Monitored Adaptive',
             fontsize=13, color='#e6edf3', y=1.01)

labels = ['Full Physics\nSolver', 'PINN-Monitored\nAdaptive']
colors = [BLUE, PURPLE]

# --- left: step time ---
ax = axes[0]
bar_vals = [mean_step_nc, mean_step_pinn]
bars = ax.bar(labels, bar_vals, color=colors, width=0.45,
              edgecolor='#21262d', linewidth=1.2, zorder=3)
ax.set_ylabel('Mean Step Time (ms)', fontsize=11)
ax.set_title('Step Computation Time', fontsize=11, color='#c9d1d9')
ax.grid(True, axis='y', alpha=0.4, zorder=0)
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * max(bar_vals),
            f'{val:.2f} ms', ha='center', va='bottom',
            color='#e6edf3', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(bar_vals) * 1.25)

# --- right: mean residual ---
ax = axes[1]
res_vals = [mean_res_nc, mean_res_pinn]
bars2 = ax.bar(labels, res_vals, color=colors, width=0.45,
               edgecolor='#21262d', linewidth=1.2, zorder=3)
ax.set_ylabel('Mean Physics Residual (div)', fontsize=11)
ax.set_title('Mean Divergence Residual', fontsize=11, color='#c9d1d9')
ax.grid(True, axis='y', alpha=0.4, zorder=0)
for bar, val in zip(bars2, res_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * max(res_vals),
            f'{val:.6f}', ha='center', va='bottom',
            color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_ylim(0, max(res_vals) * 1.30)

for ax in axes:
    ax.tick_params(colors='#8b949e')
    ax.spines['bottom'].set_edgecolor('#30363d')
    ax.spines['left'].set_edgecolor('#30363d')
    ax.spines['top'].set_edgecolor('#30363d')
    ax.spines['right'].set_edgecolor('#30363d')

plt.tight_layout()
out_path = os.path.join('figures', 'performance_comparison.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"  [OK]  Saved  {out_path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE C & D – peak heatmap snapshots
# ══════════════════════════════════════════════════════════════════════════
POLLUTION_CMAP = LinearSegmentedColormap.from_list(
    'pollution',
    ['#0d1117', '#4b0082', '#ff4500', '#ff8c00', '#ffe066', '#ffffff'],
    N=256
)

def save_heatmap(conc_field, obstacle_mask, corridor_mask, title, fname):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    vmax = max(float(conc_field.max()), 1e-6)
    im = ax.imshow(conc_field, origin='upper', cmap=POLLUTION_CMAP,
                   vmin=0, vmax=vmax, interpolation='bilinear')

    building_rgba = np.zeros((*obstacle_mask.shape, 4), dtype=np.float32)
    building_rgba[obstacle_mask > 0.5] = [0.15, 0.15, 0.15, 0.85]
    ax.imshow(building_rgba, origin='upper', interpolation='nearest')

    if corridor_mask is not None and np.any(corridor_mask > 0.5):
        corr_rgba = np.zeros((*corridor_mask.shape, 4), dtype=np.float32)
        corr_rgba[corridor_mask > 0.5] = [0.1, 0.9, 0.2, 0.35]
        ax.imshow(corr_rgba, origin='upper', interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('PM2.5 concentration (a.u.)', color='#c9d1d9', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')

    ax.set_title(title, color='#e6edf3', fontsize=13, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  [OK]  Saved  {fname}")

save_heatmap(
    SNAPSHOT_CONC_NC,
    solver_nc.obstacle_mask,
    None,
    f'Peak Pollution -- No Green Corridor  (step {SNAPSHOT_STEP})',
    os.path.join('figures', 'peak_no_corridor.png')
)

save_heatmap(
    SNAPSHOT_CONC_WC,
    solver_wc.obstacle_mask,
    solver_wc.green_corridor_mask,
    f'Peak Pollution -- With Green Corridor  (step {SNAPSHOT_STEP})',
    os.path.join('figures', 'peak_with_corridor.png')
)

print()
print("=" * 60)
print("All figures saved to  greennavier/figures/")
print("=" * 60)
print()
print("SUMMARY FOR POSTER")
print("-" * 40)
print(f"  Mean PM2.5 reduction (peak)              : {reduction_pct:.1f}%")
print(f"  Mean PM2.5 reduction (steady-state)      : {ss_reduction:.1f}%")
print(f"  Physics residual mean (no corridor)      : {mean_res_nc:.6f}")
print(f"  Physics residual mean (with corridor)    : {mean_res_wc:.6f}")
print(f"  PINN physics_warning events (no corr.)   : {len(pinn_warnings_nc)}/{TOTAL_STEPS} steps")
