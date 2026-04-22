"""
Experiment 1: GMM-RRT* vs A* on occupancy grid.

Uses the pre-trained GMM model and pre-computed occupancy grid for one scene.
Compares both planners across 30 random start/goal trials on:
  - planning time
  - path length (raw and shortcut-smoothed)
  - max occupancy probability along path (safety proxy)
  - path smoothness (total heading change)
"""

import os
import warnings
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn version mismatch

from gmm3 import SavedGMMOccupancyMap
from rrt_star import RRTStar, smooth_path
from astar_baseline import AStarGrid

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
GMM_DIR = "gmm/2013_05_28_drive_0000_sync_0000000002_0000000385_clean"
RESULTS_DIR = "results/experiment1"

# ------------------------------------------------------------------
# Experiment parameters
# ------------------------------------------------------------------
N_TRIALS = 30
SEED = 42
MIN_DIST = 30.0   # metres
MAX_DIST = 120.0  # metres
OCC_THRESH = 0.35
LAM = 10.0
POWER = 1.0

RRT_STEP = 3.0
RRT_ITERS = 3000
RRT_GOAL_RADIUS = 4.0
RRT_NEIGHBOR_RADIUS = 10.0
RRT_GOAL_BIAS = 0.05
RRT_EDGE_STEP = 0.3


# ==================================================================
# Metric helpers
# ==================================================================

def path_length(path):
    """Euclidean path length in world metres."""
    if path is None or len(path) < 2:
        return 0.0
    pts = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def path_smoothness(path):
    """Sum of absolute heading-change angles (radians) — lower is smoother."""
    if path is None or len(path) < 3:
        return 0.0
    pts = np.array(path)
    vecs = np.diff(pts, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    unit = vecs / norms
    cos_a = np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(cos_a)))


def max_occupancy_along_path(model, path, nav_z, n_samples=300):
    """Maximum GMM occupancy probability sampled along the path."""
    if path is None or len(path) < 2:
        return 1.0
    pts = np.array(path)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        return 0.0
    t = np.linspace(0, total, n_samples)
    xy = np.column_stack([np.interp(t, cum, pts[:, d]) for d in range(pts.shape[1])])
    xyz = np.column_stack([xy, np.full(n_samples, nav_z)])
    occ = model.occupancy_probability(xyz, lam=LAM, power=POWER)
    return float(np.max(occ))


def mean_clearance_along_path(model, path, nav_z, n_samples=300):
    """Mean (1 − occupancy) along path — higher means more clearance."""
    if path is None or len(path) < 2:
        return 0.0
    pts = np.array(path)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        return 1.0
    t = np.linspace(0, total, n_samples)
    xy = np.column_stack([np.interp(t, cum, pts[:, d]) for d in range(pts.shape[1])])
    xyz = np.column_stack([xy, np.full(n_samples, nav_z)])
    occ = model.occupancy_probability(xyz, lam=LAM, power=POWER)
    return float(np.mean(1.0 - occ))


# ==================================================================
# Start / goal sampling
# ==================================================================

def sample_pairs(free_2d, xs, ys, nav_z, model, n, min_dist, max_dist, rng):
    """Sample n (start, goal) pairs that are free in both the grid and GMM."""
    free_cells = np.argwhere(free_2d)
    pairs = []
    attempts = 0
    max_attempts = n * 500

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        i1, i2 = rng.integers(0, len(free_cells), size=2)
        ix1, iy1 = free_cells[i1]
        ix2, iy2 = free_cells[i2]
        x1, y1 = float(xs[ix1]), float(ys[iy1])
        x2, y2 = float(xs[ix2]), float(ys[iy2])
        d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if d < min_dist or d > max_dist:
            continue
        # Double-check with GMM (grid and GMM may disagree near boundaries)
        pts = np.array([[x1, y1, nav_z], [x2, y2, nav_z]])
        occ = model.occupancy_probability(pts, lam=LAM, power=POWER)
        if occ[0] >= OCC_THRESH or occ[1] >= OCC_THRESH:
            continue
        pairs.append(([x1, y1], [x2, y2]))

    if len(pairs) < n:
        print(f"  Warning: only {len(pairs)}/{n} valid pairs after {attempts} attempts")
    return pairs


# ==================================================================
# Plotting
# ==================================================================

def plot_metrics(results, save_dir):
    """Box plots comparing RRT* and A* on key metrics."""
    def _vals(key, planner):
        return [r[key] for r in results if r[f"{planner}_success"] and r[key] is not None]

    metrics = [
        ("raw_len",         "Raw path length (m)",          False),
        ("smooth_len",      "Smoothed path length (m)",     False),
        ("time_s",          "Planning time (s)",            True),
        ("max_occ",         "Max occupancy along path",     False),
        ("smoothness_raw",  "Raw smoothness (rad)",         False),
        ("smoothness_smo",  "Smoothed smoothness (rad)",    False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    colors = {"RRT*": "#2196F3", "A*": "#FF9800"}

    for ax, (key, label, log_scale) in zip(axes, metrics):
        rrt_vals = _vals(key, "rrt")
        ast_vals = _vals(key, "astar")
        bp = ax.boxplot(
            [rrt_vals, ast_vals],
            labels=["RRT*", "A*"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], [colors["RRT*"], colors["A*"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(label, fontsize=10)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians
        for i, vals in enumerate([rrt_vals, ast_vals], 1):
            if vals:
                med = float(np.median(vals))
                ax.text(i, med, f" {med:.2f}", va="center", fontsize=8)

    fig.suptitle("Experiment 1: GMM-RRT* vs A* Grid Baseline\n"
                 f"({sum(r['rrt_success'] for r in results)} RRT* / "
                 f"{sum(r['astar_success'] for r in results)} A* successes "
                 f"out of {len(results)} trials)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_paths(results, occ_slice, xs, ys, save_dir, max_figs=3):
    """Overlay RRT* and A* paths on the occupancy map for select trials."""
    successes = [r for r in results if r["rrt_success"] and r["astar_success"]][:max_figs]
    if not successes:
        print("  No jointly-successful trials to plot paths for.")
        return

    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for k, r in enumerate(successes):
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(
            occ_slice.T, origin="lower", extent=extent,
            cmap="Greys", vmin=0, vmax=1, alpha=0.6,
        )

        def _plot(path, color, label, ls="-"):
            if path is None:
                return
            pts = np.array(path)
            ax.plot(pts[:, 0], pts[:, 1], ls, color=color, lw=1.5, label=label, alpha=0.85)

        _plot(r["rrt_path_raw"],    "#2196F3", "RRT* raw",      "-")
        _plot(r["rrt_path_smooth"], "#0D47A1", "RRT* smoothed", "--")
        _plot(r["astar_path_raw"],    "#FF9800", "A* raw",      "-")
        _plot(r["astar_path_smooth"], "#E65100", "A* smoothed", "--")

        start, goal = np.array(r["start"]), np.array(r["goal"])
        ax.scatter(*start, s=120, c="green",  zorder=5, marker="o", label="Start")
        ax.scatter(*goal,  s=120, c="red",    zorder=5, marker="*", label="Goal")

        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(
            f"Trial {r['trial']}  |  dist={r['straight_dist']:.0f}m\n"
            f"RRT* len={r['smooth_len']:.0f}m ({r['time_s_rrt']:.1f}s)  "
            f"A* len={r['astar_smooth_len']:.0f}m ({r['time_s_astar']:.3f}s)",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
        plt.tight_layout()
        path = os.path.join(save_dir, f"path_trial_{r['trial']:02d}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def print_summary(results):
    """Print a summary table to stdout."""
    def _stats(rrt_key, ast_key, planner):
        key = rrt_key if planner == "rrt" else ast_key
        success_key = "rrt_success" if planner == "rrt" else "astar_success"
        vals = [r[key] for r in results if r[success_key] and r[key] is not None]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})"

    def _time_stats(rrt_key, ast_key, planner):
        # Include ALL trials for time (not just successes) — time is always measured
        key = rrt_key if planner == "rrt" else ast_key
        vals = [r[key] for r in results if r[key] is not None]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})"

    print("\n" + "=" * 72)
    print(f"{'Metric':<30} {'RRT* (GMM)':>20}   {'A* (Grid)':>20}")
    print("-" * 72)

    # (label, rrt_key, astar_key, use_time_stats)
    rows = [
        ("Success rate",         None,           None,               False),
        ("Plan time — all (s)",  "time_s",       "time_s_astar",     True),
        ("Raw path len (m)",     "raw_len",       "astar_raw_len",   False),
        ("Smooth path len (m)",  "smooth_len",    "astar_smooth_len",False),
        ("Max occupancy",        "max_occ",       "astar_max_occ",   False),
        ("Mean clearance",       "mean_clr",      "astar_mean_clr",  False),
        ("Smoothness raw (rad)", "smoothness_raw","astar_smoothness_raw",False),
        ("Smoothness smo (rad)", "smoothness_smo","astar_smoothness_smo",False),
    ]
    for label, rk, ak, use_time in rows:
        if rk is None:
            rrt_s = f"{sum(r['rrt_success'] for r in results)}/{len(results)}"
            ast_s = f"{sum(r['astar_success'] for r in results)}/{len(results)}"
            print(f"  {label:<28} {rrt_s:>20}   {ast_s:>20}")
        else:
            fn = _time_stats if use_time else _stats
            r_str = fn(rk, ak, "rrt")
            a_str = fn(rk, ak, "astar")
            print(f"  {label:<28} {r_str:>20}   {a_str:>20}")
    print("=" * 72 + "\n")


# ==================================================================
# Main experiment loop
# ==================================================================

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- Load model and grid ----------------------------------------
    print("Loading GMM model …")
    model = SavedGMMOccupancyMap.load(f"{GMM_DIR}/gmm_occupancy_model.pkl")
    print(f"  {model.n_components} components, density_ref={model.density_ref_:.4e}")

    print("Loading occupancy grid …")
    grid = np.load(f"{GMM_DIR}/occupancy_grid.npy")
    xs   = np.load(f"{GMM_DIR}/xs.npy")
    ys   = np.load(f"{GMM_DIR}/ys.npy")
    zs   = np.load(f"{GMM_DIR}/zs.npy")

    # ---- Build planners ---------------------------------------------
    astar = AStarGrid(grid, xs, ys, zs, occ_thresh=OCC_THRESH)
    nav_z  = astar.nav_z
    free_2d = astar.free_2d
    print(f"  Nav z={nav_z:.3f}m (iz={astar.best_iz}), "
          f"free={free_2d.mean()*100:.1f}%")

    bounds_2d = [[float(xs[0]), float(xs[-1])], [float(ys[0]), float(ys[-1])]]
    rrt = RRTStar(
        model=model,
        bounds_2d=bounds_2d,
        nav_z=nav_z,
        step_size=RRT_STEP,
        max_iter=RRT_ITERS,
        goal_radius=RRT_GOAL_RADIUS,
        neighbor_radius=RRT_NEIGHBOR_RADIUS,
        goal_bias=RRT_GOAL_BIAS,
        occ_thresh=OCC_THRESH,
        lam=LAM,
        power=POWER,
        edge_step=RRT_EDGE_STEP,
        seed=SEED,
    )

    # Collision checker shared by A* smoothing
    def collision_free_2d(a, b):
        return model.edge_is_free(
            np.array([a[0], a[1], nav_z]),
            np.array([b[0], b[1], nav_z]),
            step=RRT_EDGE_STEP, occ_thresh=OCC_THRESH, lam=LAM, power=POWER,
        )

    # ---- Sample start / goal pairs ----------------------------------
    print(f"Sampling {N_TRIALS} (start, goal) pairs …")
    pairs = sample_pairs(free_2d, xs, ys, nav_z, model,
                         N_TRIALS, MIN_DIST, MAX_DIST, rng)

    # ---- Run trials -------------------------------------------------
    results = []
    occ_slice = grid[:, :, astar.best_iz]  # for plotting

    for i, (start, goal) in enumerate(pairs):
        straight = float(np.linalg.norm(np.array(goal) - np.array(start)))
        print(f"\nTrial {i+1:2d}/{len(pairs)}  dist={straight:.1f}m  "
              f"start=({start[0]:.1f},{start[1]:.1f})  "
              f"goal=({goal[0]:.1f},{goal[1]:.1f})")

        # ---- RRT* ---------------------------------------------------
        t0 = time.time()
        rrt_path, rrt_info = rrt.plan(start, goal)
        rrt_smooth = rrt.smooth_path(rrt_path) if rrt_path else None
        rrt_time = time.time() - t0

        raw_rrt_len = path_length(rrt_path)
        smo_rrt_len = path_length(rrt_smooth)
        max_occ_rrt = max_occupancy_along_path(model, rrt_smooth or rrt_path, nav_z)
        clr_rrt     = mean_clearance_along_path(model, rrt_smooth or rrt_path, nav_z)
        smo_rrt_raw = path_smoothness(rrt_path)
        smo_rrt_smo = path_smoothness(rrt_smooth)

        print(f"  RRT*  {'✓' if rrt_info['success'] else '✗'}  "
              f"t={rrt_time:.1f}s  raw={raw_rrt_len:.1f}m  "
              f"smooth={smo_rrt_len:.1f}m  max_occ={max_occ_rrt:.3f}")

        # ---- A* -----------------------------------------------------
        t0 = time.time()
        astar_path, astar_info = astar.plan(start, goal)
        astar_smooth = smooth_path(astar_path, collision_free_2d) if astar_path else None
        astar_time = time.time() - t0

        raw_ast_len = path_length(astar_path)
        smo_ast_len = path_length(astar_smooth)
        max_occ_ast = max_occupancy_along_path(model, astar_smooth or astar_path, nav_z)
        clr_ast     = mean_clearance_along_path(model, astar_smooth or astar_path, nav_z)
        smo_ast_raw = path_smoothness(astar_path)
        smo_ast_smo = path_smoothness(astar_smooth)

        print(f"  A*    {'✓' if astar_info['success'] else '✗'}  "
              f"t={astar_time:.3f}s  raw={raw_ast_len:.1f}m  "
              f"smooth={smo_ast_len:.1f}m  max_occ={max_occ_ast:.3f}")

        results.append({
            "trial":           i + 1,
            "start":           start,
            "goal":            goal,
            "straight_dist":   straight,
            # RRT*
            "rrt_success":     rrt_info["success"],
            "time_s":          rrt_time,
            "time_s_rrt":      rrt_time,
            "raw_len":         raw_rrt_len if rrt_info["success"] else None,
            "smooth_len":      smo_rrt_len if rrt_info["success"] else None,
            "max_occ":         max_occ_rrt if rrt_info["success"] else None,
            "mean_clr":        clr_rrt     if rrt_info["success"] else None,
            "smoothness_raw":  smo_rrt_raw if rrt_info["success"] else None,
            "smoothness_smo":  smo_rrt_smo if rrt_info["success"] else None,
            "rrt_nodes":       rrt_info.get("nodes_expanded"),
            "rrt_path_raw":    rrt_path,
            "rrt_path_smooth": rrt_smooth,
            # A*
            "astar_success":   astar_info["success"],
            "time_s_astar":    astar_time,
            "astar_raw_len":   raw_ast_len if astar_info["success"] else None,
            "astar_smooth_len":smo_ast_len if astar_info["success"] else None,
            "astar_max_occ":   max_occ_ast if astar_info["success"] else None,
            "astar_mean_clr":  clr_ast     if astar_info["success"] else None,
            "astar_smoothness_raw": smo_ast_raw if astar_info["success"] else None,
            "astar_smoothness_smo": smo_ast_smo if astar_info["success"] else None,
            "astar_nodes":     astar_info.get("nodes_expanded"),
            "astar_path_raw":  astar_path,
            "astar_path_smooth": astar_smooth,
        })

    # ---- Summary and plots ------------------------------------------
    print_summary(results)

    print("Saving plots …")
    # Rebuild metric-aligned dicts for the generic plotter
    plot_results_list = []
    for r in results:
        plot_results_list.append({
            "trial":          r["trial"],
            "start":          r["start"],
            "goal":           r["goal"],
            "straight_dist":  r["straight_dist"],
            "rrt_success":    r["rrt_success"],
            "astar_success":  r["astar_success"],
            # RRT* columns expected by plot_metrics
            "raw_len":        r["raw_len"],
            "smooth_len":     r["smooth_len"],
            "time_s":         r["time_s"],
            "max_occ":        r["max_occ"],
            "mean_clr":       r["mean_clr"],
            "smoothness_raw": r["smoothness_raw"],
            "smoothness_smo": r["smoothness_smo"],
            # A* columns
            "astar_raw_len":        r["astar_raw_len"],
            "astar_smooth_len":     r["astar_smooth_len"],
            "time_s_astar":         r["time_s_astar"],
            "astar_max_occ":        r["astar_max_occ"],
            "astar_mean_clr":       r["astar_mean_clr"],
            "astar_smoothness_raw": r["astar_smoothness_raw"],
            "astar_smoothness_smo": r["astar_smoothness_smo"],
        })

    # Custom box-plot that handles both planners' separate columns
    _plot_both_metrics(plot_results_list, RESULTS_DIR)
    plot_paths(results, occ_slice, xs, ys, RESULTS_DIR, max_figs=3)
    print(f"\nAll outputs saved to {RESULTS_DIR}/")


def _plot_both_metrics(results, save_dir):
    """Box plots with one group per metric, RRT* vs A* side by side."""
    metrics = [
        ("raw_len",         "astar_raw_len",        "Raw path length (m)"),
        ("smooth_len",      "astar_smooth_len",      "Smoothed path length (m)"),
        ("time_s",          "time_s_astar",          "Planning time (s)"),
        ("max_occ",         "astar_max_occ",         "Max occupancy (lower=safer)"),
        ("smoothness_raw",  "astar_smoothness_raw",  "Raw smoothness (rad)"),
        ("smoothness_smo",  "astar_smoothness_smo",  "Smoothed smoothness (rad)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # time keys: include all trials regardless of success
    time_keys = {"time_s", "time_s_astar"}

    for ax, (rrt_key, ast_key, label) in zip(axes, metrics):
        if rrt_key in time_keys:
            rrt_vals = [r[rrt_key] for r in results if r[rrt_key] is not None]
            ast_vals = [r[ast_key] for r in results if r[ast_key] is not None]
        else:
            rrt_vals = [r[rrt_key] for r in results if r["rrt_success"] and r[rrt_key] is not None]
            ast_vals = [r[ast_key] for r in results if r["astar_success"] and r[ast_key] is not None]

        bp = ax.boxplot(
            [rrt_vals, ast_vals],
            labels=["RRT*\n(GMM)", "A*\n(Grid)"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        colors = ["#2196F3", "#FF9800"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if "time" in rrt_key:
            ax.set_yscale("log")

        for i, vals in enumerate([rrt_vals, ast_vals], 1):
            if vals:
                med = float(np.median(vals))
                ax.text(i + 0.28, med, f"{med:.3g}", va="center", fontsize=8)

    n_rrt  = sum(r["rrt_success"]   for r in results)
    n_astar = sum(r["astar_success"] for r in results)
    fig.suptitle(
        f"Experiment 1 — GMM-RRT* vs A* Grid Baseline\n"
        f"RRT* success: {n_rrt}/{len(results)}   A* success: {n_astar}/{len(results)}",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "metrics_boxplot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    run()
