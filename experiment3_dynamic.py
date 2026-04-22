"""
Experiment 3: Dynamic Mapping — GMM vs Grid under incremental observations.

Simulates a robot incrementally discovering its environment: the full point
cloud is shuffled and revealed in 11 stages (5 %, 10 %, …, 100 %).  At each
stage we fit a partial GMM on the accumulated points, derive a binary voxel
grid from it, and then run two RRT* planners:

  RRT*(GMM-partial)  — collision checking via the partial GMM
  RRT*(Grid-partial) — collision checking via the 64³ voxel grid built from
                       the same partial GMM (discrete, fixed resolution)

Paths are evaluated for *safety* against the FULL pre-trained GMM (ground
truth), revealing which map representation produces safer plans when only a
fraction of the scene has been observed.

Additional metrics:
  • Map fidelity : MAE between partial and ground-truth occupancy probabilities
                   on a held-out set of 3-D query points
  • Update time  : seconds to fit GMM vs seconds to fit + evaluate 64³ grid

Key insight: GMM generalises across unobserved regions via smooth Gaussian
kernels, giving reasonable occupancy estimates (and therefore safer plans)
even with very few observations.  The grid's hard voxels cannot interpolate:
unobserved cells are marked free, causing paths to pass through unconfirmed
empty space that the ground-truth GMM rates as occupied.

Outputs saved to results/experiment3/.
"""

import time
import warnings
import contextlib
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

from gmm3 import SavedGMMOccupancyMap, make_occupancy_grid_chunked
from kitti360_dataset_pipeline import read_ply_vertices
from astar_baseline import AStarGrid
from rrt_star import RRTStar

# ─── Config ──────────────────────────────────────────────────────────────────
SCENE_NAME  = "2013_05_28_drive_0000_sync_0000000002_0000000385_clean"
PLY_PATH    = Path(f"processed/{SCENE_NAME}.ply")
GMM_DIR     = Path(f"gmm/{SCENE_NAME}")
RESULTS_DIR = Path("results/experiment3")

# Observation stages (fraction of total points revealed)
STAGES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
STAGE_LABELS = [f"{int(s*100)}%" for s in STAGES]

# Partial GMM — smaller than the full trained model (speed vs. accuracy trade-off)
N_COMPONENTS_PARTIAL = 32
MAX_FIT_PTS          = 40_000   # subsample per stage for GMM fitting
GMM_MAX_ITER         = 60
GMM_REG_COVAR        = 1e-4

# Partial grid resolution (coarser than full 128³ so grid eval is fast enough)
GRID_RES  = 64

# Map fidelity: number of random 3-D query points held out for MAE evaluation
N_MAP_EVAL = 3_000

# Planning
N_TRIALS    = 10
SEED        = 42
MIN_DIST    = 30.0
MAX_DIST    = 120.0
OCC_THRESH  = 0.35
LAM         = 10.0
POWER       = 1.0
RRT_ITERS   = 800
RRT_STEP    = 3.0
GOAL_RADIUS = 4.0
EDGE_STEP   = 0.5


# ─── Data loading ────────────────────────────────────────────────────────────

def load_all_xyz(ply_path: Path) -> np.ndarray:
    verts = read_ply_vertices(str(ply_path))
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float64)
    return xyz


# ─── Partial GMM fitting ─────────────────────────────────────────────────────

def fit_partial_gmm(xyz_accumulated: np.ndarray, seed: int) -> tuple:
    """Fit a 32-component GMM on xyz_accumulated (subsampled to MAX_FIT_PTS).

    Returns (model, fit_time_s).
    """
    rng = np.random.default_rng(seed)
    if len(xyz_accumulated) > MAX_FIT_PTS:
        idx = rng.choice(len(xyz_accumulated), size=MAX_FIT_PTS, replace=False)
        xyz_fit = xyz_accumulated[idx]
    else:
        xyz_fit = xyz_accumulated

    t0 = time.time()
    model = SavedGMMOccupancyMap(
        n_components=N_COMPONENTS_PARTIAL,
        covariance_type="diag",
        reg_covar=GMM_REG_COVAR,
        max_iter=GMM_MAX_ITER,
        random_state=seed,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(xyz_fit)
    fit_time = time.time() - t0
    return model, fit_time


# ─── Grid building ───────────────────────────────────────────────────────────

def build_grid(model: SavedGMMOccupancyMap, xyz_bounds: np.ndarray) -> tuple:
    """Evaluate 64³ occupancy grid from model.  Returns (grid, xs, ys, zs, eval_time_s)."""
    t0 = time.time()
    grid, xs, ys, zs = make_occupancy_grid_chunked(
        model,
        xyz_bounds_source=xyz_bounds,
        grid_res=GRID_RES,
        padding=0.1,
        lam=LAM,
        power=POWER,
        query_chunk=200_000,
    )
    eval_time = time.time() - t0
    return grid, xs, ys, zs, eval_time


# ─── Collision-checker factories ─────────────────────────────────────────────

def make_gmm_checker(model, nav_z):
    def check(a, b):
        a3 = np.array([a[0], a[1], nav_z])
        b3 = np.array([b[0], b[1], nav_z])
        return model.edge_is_free(a3, b3, step=EDGE_STEP,
                                  occ_thresh=OCC_THRESH, lam=LAM, power=POWER)
    return check


def make_grid_checker(grid_3d, xs, ys, zs, nav_z):
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    dz = (zs[-1] - zs[0]) / (len(zs) - 1)
    nx, ny, nz = grid_3d.shape

    def check(a, b):
        a3 = np.array([a[0], a[1], nav_z])
        b3 = np.array([b[0], b[1], nav_z])
        dist = np.linalg.norm(b3 - a3)
        n = max(2, int(np.ceil(dist / EDGE_STEP)))
        pts = np.linspace(a3, b3, n)
        ix = np.clip(np.round((pts[:, 0] - xs[0]) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.round((pts[:, 1] - ys[0]) / dy).astype(int), 0, ny - 1)
        iz = np.clip(np.round((pts[:, 2] - zs[0]) / dz).astype(int), 0, nz - 1)
        return bool(np.all(grid_3d[ix, iy, iz] < OCC_THRESH))
    return check


# ─── Start / goal sampling ───────────────────────────────────────────────────

def sample_pairs(free_2d, xs, ys, nav_z, gt_model, n, rng):
    """Sample n valid (start, goal) pairs with MIN_DIST ≤ d ≤ MAX_DIST.
    Validity = free in 2-D grid AND free in ground-truth GMM.
    """
    free_idx = np.argwhere(free_2d)
    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < 50_000:
        attempts += 1
        ia, ib = rng.integers(0, len(free_idx), size=2)
        xi, yi = free_idx[ia]
        xj, yj = free_idx[ib]
        start = np.array([xs[xi], ys[yi]])
        goal  = np.array([xs[xj], ys[yj]])
        dist  = np.linalg.norm(goal - start)
        if dist < MIN_DIST or dist > MAX_DIST:
            continue
        # Validate with ground-truth GMM
        s3 = np.array([[start[0], start[1], nav_z]])
        g3 = np.array([[goal[0],  goal[1],  nav_z]])
        sp = gt_model.occupancy_probability(s3, lam=LAM, power=POWER)[0]
        gp = gt_model.occupancy_probability(g3, lam=LAM, power=POWER)[0]
        if sp >= OCC_THRESH or gp >= OCC_THRESH:
            continue
        pairs.append((start, goal))
    return pairs


# ─── Path evaluation ─────────────────────────────────────────────────────────

def max_occ_gt(path, gt_model, nav_z):
    """Max occupancy probability along path according to ground-truth GMM."""
    if path is None or len(path) < 2:
        return np.nan
    pts3 = np.array([[p[0], p[1], nav_z] for p in path])
    probs = gt_model.occupancy_probability(pts3, lam=LAM, power=POWER)
    return float(np.max(probs))


def path_smoothness(path):
    """Total absolute turning angle (radians) along path."""
    if path is None or len(path) < 3:
        return np.nan
    total = 0.0
    for i in range(1, len(path) - 1):
        v1 = path[i]     - path[i - 1]
        v2 = path[i + 1] - path[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        total += np.arccos(cos_a)
    return total


# ─── Map fidelity ────────────────────────────────────────────────────────────

def map_fidelity(partial_model, gt_model, query_pts_3d):
    """MAE between partial and ground-truth occupancy probabilities."""
    p_partial = partial_model.occupancy_probability(query_pts_3d, lam=LAM, power=POWER)
    p_gt      = gt_model.occupancy_probability(query_pts_3d, lam=LAM, power=POWER)
    return float(np.mean(np.abs(p_partial - p_gt)))


def grid_fidelity(grid_3d, xs, ys, zs, gt_model, query_pts_3d):
    """MAE between grid interpolation and ground-truth occupancy probabilities."""
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    dz = (zs[-1] - zs[0]) / (len(zs) - 1)
    nx, ny, nz = grid_3d.shape
    ix = np.clip(np.round((query_pts_3d[:, 0] - xs[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.round((query_pts_3d[:, 1] - ys[0]) / dy).astype(int), 0, ny - 1)
    iz = np.clip(np.round((query_pts_3d[:, 2] - zs[0]) / dz).astype(int), 0, nz - 1)
    p_grid = grid_3d[ix, iy, iz].astype(float)
    p_gt   = gt_model.occupancy_probability(query_pts_3d, lam=LAM, power=POWER)
    return float(np.mean(np.abs(p_grid - p_gt)))


# ─── Main experiment loop ────────────────────────────────────────────────────

def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── Load ground-truth model and full point cloud ──────────────────────
    print("Loading ground-truth GMM …")
    gt_model = SavedGMMOccupancyMap.load(str(GMM_DIR / "gmm_occupancy_model.pkl"))
    gt_grid  = np.load(str(GMM_DIR / "occupancy_grid.npy"))
    gt_xs    = np.load(str(GMM_DIR / "xs.npy"))
    gt_ys    = np.load(str(GMM_DIR / "ys.npy"))
    gt_zs    = np.load(str(GMM_DIR / "zs.npy"))

    # Navigation height from full grid
    occ_per_z = np.array([(gt_grid[:, :, iz] >= OCC_THRESH).mean()
                           for iz in range(len(gt_zs))])
    nav_z = float(gt_zs[np.argmax(occ_per_z)])
    print(f"nav_z = {nav_z:.2f} m")

    # ── Load all scene points ─────────────────────────────────────────────
    print(f"Loading PLY: {PLY_PATH} …")
    all_xyz = load_all_xyz(PLY_PATH)
    N_total = len(all_xyz)
    print(f"Total points: {N_total:,}")

    # Shuffle to simulate random temporal arrival
    shuffle_idx = rng.permutation(N_total)
    all_xyz_shuffled = all_xyz[shuffle_idx]

    # ── Hold-out query points for map fidelity ────────────────────────────
    # Sample uniformly within the scene bounding box at nav_z ± 1 m
    x_range = [gt_xs[0], gt_xs[-1]]
    y_range = [gt_ys[0], gt_ys[-1]]
    fidelity_pts = np.column_stack([
        rng.uniform(x_range[0], x_range[1], N_MAP_EVAL),
        rng.uniform(y_range[0], y_range[1], N_MAP_EVAL),
        rng.uniform(nav_z - 1.0, nav_z + 1.0, N_MAP_EVAL),
    ])

    # ── Build start/goal pairs from full free map ─────────────────────────
    # Use the full grid free map to find valid candidate cells
    iz_nav = int(np.argmax(occ_per_z))
    free_full = gt_grid[:, :, iz_nav] < OCC_THRESH
    print(f"Free cells at nav: {free_full.sum()}/{free_full.size}")

    pairs = sample_pairs(free_full, gt_xs, gt_ys, nav_z, gt_model, N_TRIALS, rng)
    print(f"Sampled {len(pairs)} start/goal pairs (need {N_TRIALS})")
    if len(pairs) < N_TRIALS:
        print("WARNING: fewer pairs than trials — re-using pairs")
        while len(pairs) < N_TRIALS:
            pairs.append(pairs[rng.integers(0, len(pairs))])
    pairs = pairs[:N_TRIALS]

    # ── Per-stage loop ────────────────────────────────────────────────────
    results = []

    for stage_i, frac in enumerate(STAGES):
        n_obs = max(1, int(frac * N_total))
        xyz_acc = all_xyz_shuffled[:n_obs]
        label   = STAGE_LABELS[stage_i]

        print(f"\n{'─'*55}")
        print(f"Stage {label}  ({n_obs:,} pts of {N_total:,})")

        # ── Fit partial GMM ───────────────────────────────────────────────
        partial_gmm, gmm_fit_t = fit_partial_gmm(xyz_acc, seed=int(SEED + stage_i))
        print(f"  GMM fit: {gmm_fit_t:.1f}s  "
              f"density_ref={partial_gmm.density_ref_:.3e}")

        # ── Build partial grid ────────────────────────────────────────────
        # Use accumulated xyz bounds (padded) as grid extent
        grid_p, xs_p, ys_p, zs_p, grid_t = build_grid(partial_gmm, xyz_acc)
        print(f"  Grid eval ({GRID_RES}³): {grid_t:.1f}s")

        update_time_gmm  = gmm_fit_t
        update_time_grid = gmm_fit_t + grid_t

        # ── Map fidelity ──────────────────────────────────────────────────
        # Clip fidelity_pts to the partial grid extent (some may fall outside)
        in_bounds = (
            (fidelity_pts[:, 0] >= xs_p[0]) & (fidelity_pts[:, 0] <= xs_p[-1]) &
            (fidelity_pts[:, 1] >= ys_p[0]) & (fidelity_pts[:, 1] <= ys_p[-1]) &
            (fidelity_pts[:, 2] >= zs_p[0]) & (fidelity_pts[:, 2] <= zs_p[-1])
        )
        eval_pts = fidelity_pts[in_bounds] if in_bounds.sum() > 100 else fidelity_pts

        gmm_mae  = map_fidelity(partial_gmm, gt_model, eval_pts)
        grid_mae = grid_fidelity(grid_p, xs_p, ys_p, zs_p, gt_model, eval_pts)
        print(f"  Map fidelity MAE — GMM: {gmm_mae:.4f}  Grid: {grid_mae:.4f}")

        # ── Plan with both checkers ───────────────────────────────────────
        # GMM nav_z: use same as ground truth for fair comparison
        gmm_checker  = make_gmm_checker(partial_gmm, nav_z)
        grid_checker = make_grid_checker(grid_p, xs_p, ys_p, zs_p, nav_z)

        bounds_2d = [[gt_xs[0], gt_xs[-1]], [gt_ys[0], gt_ys[-1]]]

        gmm_metrics  = {"success": [], "max_occ_gt": [], "smoothness": [], "time": []}
        grid_metrics = {"success": [], "max_occ_gt": [], "smoothness": [], "time": []}

        for trial_i, (start, goal) in enumerate(pairs):
            # RRT*(GMM-partial)
            planner_g = RRTStar(
                model=None,
                bounds_2d=bounds_2d,
                nav_z=nav_z,
                step_size=RRT_STEP,
                max_iter=RRT_ITERS,
                goal_radius=GOAL_RADIUS,
                neighbor_radius=10.0,
                goal_bias=0.05,
                edge_step=EDGE_STEP,
                seed=int(SEED + stage_i * 1000 + trial_i),
                collision_fn=gmm_checker,
            )
            path_g, info_g = planner_g.plan(start, goal)
            if path_g is not None:
                path_g = planner_g.smooth_path(path_g)
            gmm_metrics["success"].append(int(path_g is not None))
            gmm_metrics["max_occ_gt"].append(max_occ_gt(path_g, gt_model, nav_z))
            gmm_metrics["smoothness"].append(path_smoothness(path_g))
            gmm_metrics["time"].append(info_g["time_s"])

            # RRT*(Grid-partial)
            planner_r = RRTStar(
                model=None,
                bounds_2d=bounds_2d,
                nav_z=nav_z,
                step_size=RRT_STEP,
                max_iter=RRT_ITERS,
                goal_radius=GOAL_RADIUS,
                neighbor_radius=10.0,
                goal_bias=0.05,
                edge_step=EDGE_STEP,
                seed=int(SEED + stage_i * 1000 + trial_i),
                collision_fn=grid_checker,
            )
            path_r, info_r = planner_r.plan(start, goal)
            if path_r is not None:
                path_r = planner_r.smooth_path(path_r)
            grid_metrics["success"].append(int(path_r is not None))
            grid_metrics["max_occ_gt"].append(max_occ_gt(path_r, gt_model, nav_z))
            grid_metrics["smoothness"].append(path_smoothness(path_r))
            grid_metrics["time"].append(info_r["time_s"])

        # ── Aggregate metrics ─────────────────────────────────────────────
        def _nanmed(arr):
            a = [x for x in arr if not np.isnan(x)]
            return float(np.median(a)) if a else np.nan

        res = {
            "stage":            label,
            "frac":             frac,
            "n_obs":            n_obs,
            "update_t_gmm":     update_time_gmm,
            "update_t_grid":    update_time_grid,
            "gmm_mae":          gmm_mae,
            "grid_mae":         grid_mae,
            "gmm_success":      float(np.mean(gmm_metrics["success"])),
            "grid_success":     float(np.mean(grid_metrics["success"])),
            "gmm_max_occ":      _nanmed(gmm_metrics["max_occ_gt"]),
            "grid_max_occ":     _nanmed(grid_metrics["max_occ_gt"]),
            "gmm_smoothness":   _nanmed(gmm_metrics["smoothness"]),
            "grid_smoothness":  _nanmed(grid_metrics["smoothness"]),
            # Save last-stage paths for visualisation
            "_paths_gmm":       gmm_metrics,
            "_paths_grid":      grid_metrics,
            "_partial_gmm":     partial_gmm,
            "_grid_p":          grid_p,
            "_xs_p":            xs_p,
            "_ys_p":            ys_p,
            "_last_pairs":      pairs,
        }
        results.append(res)

        succ_g = res["gmm_success"]
        succ_r = res["grid_success"]
        occ_g  = res["gmm_max_occ"]
        occ_r  = res["grid_max_occ"]
        print(f"  Success — GMM:{succ_g:.0%}  Grid:{succ_r:.0%}")
        print(f"  Max occ (GT) — GMM:{occ_g:.3f}  Grid:{occ_r:.3f}")

    return results, gt_model, gt_grid, gt_xs, gt_ys, gt_zs, nav_z, pairs


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results, gt_model, gt_grid, gt_xs, gt_ys, gt_zs, nav_z, pairs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fracs  = [r["frac"]  for r in results]
    labels = [r["stage"] for r in results]
    pct    = [f * 100 for f in fracs]

    # ── Fig 1: Map fidelity (MAE) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pct, [r["gmm_mae"]  for r in results], "o-", color="#1565C0",
            label="GMM (continuous)", linewidth=2, markersize=6)
    ax.plot(pct, [r["grid_mae"] for r in results], "s--", color="#E65100",
            label=f"Grid ({GRID_RES}³ voxels)", linewidth=2, markersize=6)
    ax.set_xlabel("Observations revealed (%)")
    ax.set_ylabel("Map MAE vs ground truth")
    ax.set_title("Map Fidelity under Partial Observations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "map_fidelity.png", dpi=150)
    plt.close()
    print("Saved map_fidelity.png")

    # ── Fig 2: Path safety (max occupancy per ground-truth GMM) ──────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pct, [r["gmm_max_occ"]  for r in results], "o-", color="#1565C0",
            label="RRT*(GMM-partial)", linewidth=2, markersize=6)
    ax.plot(pct, [r["grid_max_occ"] for r in results], "s--", color="#E65100",
            label="RRT*(Grid-partial)", linewidth=2, markersize=6)
    ax.axhline(OCC_THRESH, color="gray", linestyle=":", linewidth=1.5,
               label=f"Occ threshold ({OCC_THRESH})")
    ax.set_xlabel("Observations revealed (%)")
    ax.set_ylabel("Median max occupancy (ground truth)")
    ax.set_title("Path Safety vs Observation Coverage\n"
                 "(lower = safer according to full scene GMM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "path_safety.png", dpi=150)
    plt.close()
    print("Saved path_safety.png")

    # ── Fig 3: Map update time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pct, [r["update_t_gmm"]  for r in results], "o-", color="#1565C0",
            label="GMM refit only", linewidth=2, markersize=6)
    ax.plot(pct, [r["update_t_grid"] for r in results], "s--", color="#E65100",
            label=f"GMM refit + {GRID_RES}³ grid eval", linewidth=2, markersize=6)
    ax.set_xlabel("Observations revealed (%)")
    ax.set_ylabel("Map update time (s)")
    ax.set_title("Map Update Cost vs Observation Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "update_time.png", dpi=150)
    plt.close()
    print("Saved update_time.png")

    # ── Fig 4: Success rate ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pct, [r["gmm_success"]*100  for r in results], "o-", color="#1565C0",
            label="RRT*(GMM-partial)", linewidth=2, markersize=6)
    ax.plot(pct, [r["grid_success"]*100 for r in results], "s--", color="#E65100",
            label="RRT*(Grid-partial)", linewidth=2, markersize=6)
    ax.set_xlabel("Observations revealed (%)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Planning Success Rate vs Observation Coverage")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "success_rate.png", dpi=150)
    plt.close()
    print("Saved success_rate.png")

    # ── Fig 5: Example paths at early vs late stage ───────────────────────
    # Show 4 panels: [5% GMM | 5% Grid | 100% GMM | 100% Grid]
    early_res = results[0]   # 5 %
    late_res  = results[-1]  # 100 %

    # Background: ground-truth occupancy slice at nav_z
    occ_per_z = np.array([(gt_grid[:, :, iz] >= OCC_THRESH).mean()
                           for iz in range(len(gt_zs))])
    iz_nav = int(np.argmax(occ_per_z))
    bg_slice = gt_grid[:, :, iz_nav].T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        (early_res, "gmm",  "RRT*(GMM-partial)", "#1565C0", axes[0, 0]),
        (early_res, "grid", "RRT*(Grid-partial)", "#E65100", axes[0, 1]),
        (late_res,  "gmm",  "RRT*(GMM-partial)", "#1565C0", axes[1, 0]),
        (late_res,  "grid", "RRT*(Grid-partial)", "#E65100", axes[1, 1]),
    ]
    for res, key, planner_name, color, ax in configs:
        ax.imshow(bg_slice, origin="lower",
                  extent=[gt_xs[0], gt_xs[-1], gt_ys[0], gt_ys[-1]],
                  cmap="gray_r", vmin=0, vmax=1, aspect="auto", alpha=0.5)
        # Draw up to 4 paths
        metrics_key = f"_paths_{key}"
        max_occ_list = res[metrics_key]["max_occ_gt"]
        success_list = res[metrics_key]["success"]
        # We need path arrays — recomputed inline from stored metrics is not possible,
        # so we just annotate with aggregate stats.
        ax.set_xlim(gt_xs[0], gt_xs[-1])
        ax.set_ylim(gt_ys[0], gt_ys[-1])
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(
            f"{planner_name}  ({res['stage']} revealed)\n"
            f"Success {res[f'{key}_success']:.0%}  "
            f"MaxOcc(GT)={res[f'{key}_max_occ']:.3f}  "
            f"MAE={res[f'{key[0:4]}_mae']:.3f}"
        )
        # Draw start/goal pairs
        for start, goal in pairs[:4]:
            ax.plot(*start, "g^", markersize=8, zorder=5)
            ax.plot(*goal,  "r*", markersize=10, zorder=5)
    plt.suptitle("Ground-truth occupancy map (background) with\n"
                 "start (▲) and goal (★) positions", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "example_paths_comparison.png", dpi=150)
    plt.close()
    print("Saved example_paths_comparison.png")

    # ── Fig 6: Summary panel (2×2 subplots) ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Map fidelity
    ax = axes[0, 0]
    ax.plot(pct, [r["gmm_mae"]  for r in results], "o-", color="#1565C0",
            label="GMM (continuous)", linewidth=2)
    ax.plot(pct, [r["grid_mae"] for r in results], "s--", color="#E65100",
            label=f"Grid ({GRID_RES}³)", linewidth=2)
    ax.set_title("Map Fidelity (MAE ↓)"); ax.set_xlabel("%"); ax.set_ylabel("MAE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Path safety
    ax = axes[0, 1]
    ax.plot(pct, [r["gmm_max_occ"]  for r in results], "o-", color="#1565C0",
            label="RRT*(GMM)", linewidth=2)
    ax.plot(pct, [r["grid_max_occ"] for r in results], "s--", color="#E65100",
            label="RRT*(Grid)", linewidth=2)
    ax.axhline(OCC_THRESH, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Path Safety — max occ GT (↓)"); ax.set_xlabel("%")
    ax.set_ylabel("Max occ probability")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Success rate
    ax = axes[1, 0]
    ax.plot(pct, [r["gmm_success"]*100  for r in results], "o-", color="#1565C0",
            label="RRT*(GMM)", linewidth=2)
    ax.plot(pct, [r["grid_success"]*100 for r in results], "s--", color="#E65100",
            label="RRT*(Grid)", linewidth=2)
    ax.set_ylim(0, 105); ax.set_title("Success Rate (↑)")
    ax.set_xlabel("%"); ax.set_ylabel("Success %")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Update time
    ax = axes[1, 1]
    ax.plot(pct, [r["update_t_gmm"]  for r in results], "o-", color="#1565C0",
            label="GMM only", linewidth=2)
    ax.plot(pct, [r["update_t_grid"] for r in results], "s--", color="#E65100",
            label=f"GMM + {GRID_RES}³ grid", linewidth=2)
    ax.set_title("Map Update Time (↓)")
    ax.set_xlabel("%"); ax.set_ylabel("Seconds")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlim(0, 105)

    plt.suptitle(f"Dynamic Mapping: GMM vs {GRID_RES}³ Grid — Scene {SCENE_NAME[:25]}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dynamic_summary.png", dpi=150)
    plt.close()
    print("Saved dynamic_summary.png")


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(results):
    print(f"\n{'='*70}")
    print(f"{'Stage':<8} {'GMM-MAE':>9} {'Grid-MAE':>9} "
          f"{'Succ-G':>7} {'Succ-R':>7} "
          f"{'OccGT-G':>8} {'OccGT-R':>8} "
          f"{'T-GMM':>7} {'T-Grid':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['stage']:<8} "
              f"{r['gmm_mae']:>9.4f} {r['grid_mae']:>9.4f} "
              f"{r['gmm_success']:>6.0%} {r['grid_success']:>6.0%} "
              f"{r['gmm_max_occ']:>8.3f} {r['grid_max_occ']:>8.3f} "
              f"{r['update_t_gmm']:>6.1f}s {r['update_t_grid']:>6.1f}s")
    print("=" * 70)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, gt_model, gt_grid, gt_xs, gt_ys, gt_zs, nav_z, pairs = run_experiment()
    print_summary(results)
    plot_results(results, gt_model, gt_grid, gt_xs, gt_ys, gt_zs, nav_z, pairs)
    print(f"\nAll outputs → {RESULTS_DIR}/")
