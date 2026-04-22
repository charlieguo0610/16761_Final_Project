"""
Experiment 2: Multi-scene 2×2 factorial — {RRT*, A*} × {GMM, Grid}

Four planners per scene:
  RRT*(GMM)  — RRT* with continuous GMM collision checking (lam=10)
  RRT*(Grid) — RRT* with precomputed 128³ grid collision checking
  A*(GMM)    — A* on a fresh 2-D free map evaluated from GMM (lam=10)
  A*(Grid)   — A* on the precomputed 128³ grid free map (lam=20 at training)

Train scenes (params tuned on 0000a): 0000a, 0000b, 0002, 0003, 0004, 0005, 0006, 0007
Test  scenes (held-out, never tuned): 0009, 0010

Outputs saved to results/experiment2/.
"""

import os
import warnings
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from gmm3 import SavedGMMOccupancyMap
from rrt_star import RRTStar, smooth_path
from astar_baseline import AStarGrid
from experiment1 import (
    path_length, path_smoothness,
    max_occupancy_along_path, mean_clearance_along_path,
    sample_pairs,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
GMM_ROOT    = "gmm"
RESULTS_DIR = "results/experiment2"

N_TRIALS   = 20
SEED       = 42
MIN_DIST   = 30.0
MAX_DIST   = 120.0

OCC_THRESH = 0.35
LAM        = 10.0
POWER      = 1.0

RRT_STEP            = 3.0
RRT_ITERS           = 3000
RRT_GOAL_RADIUS     = 4.0
RRT_NEIGHBOR_RADIUS = 10.0
RRT_GOAL_BIAS       = 0.05
RRT_EDGE_STEP       = 0.3

SCENES = [
    ("0000a", "2013_05_28_drive_0000_sync_0000000002_0000000385_clean", "train"),
    ("0000b", "2013_05_28_drive_0000_sync_0000000834_0000001286_clean", "train"),
    ("0002",  "2013_05_28_drive_0002_sync_0000004391_0000004625_clean", "train"),
    ("0003",  "2013_05_28_drive_0003_sync_0000000002_0000000282_clean", "train"),
    ("0004",  "2013_05_28_drive_0004_sync_0000002897_0000003118_clean", "train"),
    ("0005",  "2013_05_28_drive_0005_sync_0000000002_0000000357_clean", "train"),
    ("0006",  "2013_05_28_drive_0006_sync_0000000002_0000000403_clean", "train"),
    ("0007",  "2013_05_28_drive_0007_sync_0000000002_0000000125_clean", "train"),
    ("0009",  "2013_05_28_drive_0009_sync_0000000002_0000000292_clean", "test"),
    ("0010",  "2013_05_28_drive_0010_sync_0000000002_0000000208_clean", "test"),
]

# Planner keys and display names
PLANNERS = ["rrt_gmm", "rrt_grid", "ast_gmm", "ast_grid"]
PLANNER_LABELS = {
    "rrt_gmm":  "RRT*(GMM)",
    "rrt_grid": "RRT*(Grid)",
    "ast_gmm":  "A*(GMM)",
    "ast_grid": "A*(Grid)",
}
PLANNER_COLORS = {
    "rrt_gmm":  "#1565C0",   # dark blue
    "rrt_grid": "#64B5F6",   # light blue
    "ast_gmm":  "#E65100",   # dark orange
    "ast_grid": "#FFCC02",   # amber
}


# ------------------------------------------------------------------
# Collision-checker factories
# ------------------------------------------------------------------

def make_gmm_checker(model, nav_z, occ_thresh=OCC_THRESH,
                     lam=LAM, power=POWER, step=RRT_EDGE_STEP):
    """Return (a_2d, b_2d) -> bool using the GMM model."""
    def check(a, b):
        a3 = np.array([a[0], a[1], nav_z])
        b3 = np.array([b[0], b[1], nav_z])
        return model.edge_is_free(a3, b3, step=step,
                                  occ_thresh=occ_thresh, lam=lam, power=power)
    return check


def make_grid_checker(grid_3d, xs, ys, zs, nav_z,
                      occ_thresh=OCC_THRESH, step=RRT_EDGE_STEP):
    """Return (a_2d, b_2d) -> bool using the precomputed 128³ grid."""
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    dz = (zs[-1] - zs[0]) / (len(zs) - 1)
    nx, ny, nz = grid_3d.shape

    def check(a, b):
        a3 = np.array([a[0], a[1], nav_z])
        b3 = np.array([b[0], b[1], nav_z])
        dist = np.linalg.norm(b3 - a3)
        n = max(2, int(np.ceil(dist / step)))
        pts = np.linspace(a3, b3, n)
        ix = np.clip(np.round((pts[:, 0] - xs[0]) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.round((pts[:, 1] - ys[0]) / dy).astype(int), 0, ny - 1)
        iz = np.clip(np.round((pts[:, 2] - zs[0]) / dz).astype(int), 0, nz - 1)
        return bool(np.all(grid_3d[ix, iy, iz] < occ_thresh))

    return check


def build_gmm_free_map(model, xs, ys, nav_z,
                       occ_thresh=OCC_THRESH, lam=LAM, power=POWER):
    """
    Evaluate GMM at every (x, y) cell of the 128×128 nav grid.
    Returns a (nx, ny) bool array (True = free) — same shape as AStarGrid.free_2d.
    Uses lam=10 (experiment params), NOT the lam=20 used when building occupancy_grid.npy.
    """
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, nav_z)])
    occ = model.occupancy_probability(pts, lam=lam, power=power)
    return (occ.reshape(len(xs), len(ys)) < occ_thresh)


# ------------------------------------------------------------------
# Metric helpers (trial-level)
# ------------------------------------------------------------------

def _trial_metrics(model, nav_z, path_raw, path_smo, success):
    if not success:
        return {k: None for k in
                ("raw_len", "smo_len", "max_occ", "clr", "smo_raw", "smo_smo")}
    best = path_smo if path_smo else path_raw
    return {
        "raw_len":  path_length(path_raw),
        "smo_len":  path_length(path_smo),
        "max_occ":  max_occupancy_along_path(model, best, nav_z),
        "clr":      mean_clearance_along_path(model, best, nav_z),
        "smo_raw":  path_smoothness(path_raw),
        "smo_smo":  path_smoothness(path_smo),
    }


# ------------------------------------------------------------------
# Per-scene runner
# ------------------------------------------------------------------

def run_scene(label, gmm_dir, split, rng_seed):
    gmm_path = os.path.join(GMM_ROOT, gmm_dir)

    print(f"\n{'─'*65}")
    print(f"Scene {label}  [{split.upper()}]  {gmm_dir}")

    model = SavedGMMOccupancyMap.load(os.path.join(gmm_path, "gmm_occupancy_model.pkl"))
    grid  = np.load(os.path.join(gmm_path, "occupancy_grid.npy"))
    xs    = np.load(os.path.join(gmm_path, "xs.npy"))
    ys    = np.load(os.path.join(gmm_path, "ys.npy"))
    zs    = np.load(os.path.join(gmm_path, "zs.npy"))

    # A*(Grid) — uses precomputed grid (lam=20 at training time)
    astar_grid = AStarGrid(grid, xs, ys, zs, occ_thresh=OCC_THRESH)
    nav_z      = astar_grid.nav_z
    free_pct   = astar_grid.free_2d.mean() * 100

    # A*(GMM) — same grid geometry but fresh occupancy from GMM at lam=10
    astar_gmm = AStarGrid(grid, xs, ys, zs, occ_thresh=OCC_THRESH)
    astar_gmm.free_2d = build_gmm_free_map(model, xs, ys, nav_z)
    gmm_free_pct = astar_gmm.free_2d.mean() * 100

    print(f"  nav_z={nav_z:.1f}m  "
          f"free(grid)={free_pct:.1f}%  free(GMM@lam10)={gmm_free_pct:.1f}%  "
          f"extent={xs[-1]-xs[0]:.0f}×{ys[-1]-ys[0]:.0f}m")

    bounds_2d      = [[float(xs[0]), float(xs[-1])], [float(ys[0]), float(ys[-1])]]
    gmm_checker    = make_gmm_checker(model, nav_z)
    grid_checker   = make_grid_checker(grid, xs, ys, zs, nav_z)

    # RRT*(GMM)
    rrt_gmm = RRTStar(
        model=model, bounds_2d=bounds_2d, nav_z=nav_z,
        step_size=RRT_STEP, max_iter=RRT_ITERS,
        goal_radius=RRT_GOAL_RADIUS, neighbor_radius=RRT_NEIGHBOR_RADIUS,
        goal_bias=RRT_GOAL_BIAS, occ_thresh=OCC_THRESH,
        lam=LAM, power=POWER, edge_step=RRT_EDGE_STEP, seed=rng_seed,
    )
    # RRT*(Grid)
    rrt_grid = RRTStar(
        model=None, bounds_2d=bounds_2d, nav_z=nav_z,
        step_size=RRT_STEP, max_iter=RRT_ITERS,
        goal_radius=RRT_GOAL_RADIUS, neighbor_radius=RRT_NEIGHBOR_RADIUS,
        goal_bias=RRT_GOAL_BIAS, occ_thresh=OCC_THRESH,
        lam=LAM, power=POWER, edge_step=RRT_EDGE_STEP, seed=rng_seed,
        collision_fn=grid_checker,
    )

    # Sample valid start/goal pairs (use grid free map for sampling)
    rng   = np.random.default_rng(rng_seed)
    pairs = sample_pairs(astar_grid.free_2d, xs, ys, nav_z, model,
                         N_TRIALS, MIN_DIST, MAX_DIST, rng)

    rows = []
    for i, (start, goal) in enumerate(pairs):
        straight = float(np.linalg.norm(np.array(goal) - start))
        row = {"trial": i + 1, "straight": straight,
               "start": start, "goal": goal}

        for key, planner, smoothfn in [
            ("rrt_gmm",  rrt_gmm,   rrt_gmm.smooth_path),
            ("rrt_grid", rrt_grid,  rrt_grid.smooth_path),
            ("ast_gmm",  astar_gmm, lambda p: smooth_path(p, gmm_checker)),
            ("ast_grid", astar_grid,lambda p: smooth_path(p, grid_checker)),
        ]:
            path_raw, info = planner.plan(start, goal)
            path_smo = smoothfn(path_raw) if path_raw else None
            row[f"{key}_ok"] = info["success"]
            row[f"{key}_t"]  = info["time_s"]
            row[f"{key}_path_smo"] = path_smo
            m = _trial_metrics(model, nav_z, path_raw, path_smo, info["success"])
            for mk, mv in m.items():
                row[f"{key}_{mk}"] = mv

        rows.append(row)
        print(f"  [{i+1:2d}] "
              f"RRT*(GMM) {'✓' if row['rrt_gmm_ok'] else '✗'} {row['rrt_gmm_t']:.1f}s | "
              f"RRT*(Grid) {'✓' if row['rrt_grid_ok'] else '✗'} {row['rrt_grid_t']:.1f}s | "
              f"A*(GMM) {'✓' if row['ast_gmm_ok'] else '✗'} {row['ast_gmm_t']:.3f}s | "
              f"A*(Grid) {'✓' if row['ast_grid_ok'] else '✗'} {row['ast_grid_t']:.3f}s  "
              f"d={straight:.0f}m")

    def _med(key):
        # key format: "{planner_prefix}_{metric}", planner prefix = first 2 parts
        planner = "_".join(key.split("_")[:2])
        ok_key  = f"{planner}_ok"
        vals    = [r[key] for r in rows if r[ok_key] and r[key] is not None]
        return float(np.median(vals)) if vals else float("nan")

    def _t_med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.median(vals)) if vals else float("nan")

    summary = {
        "label": label, "split": split,
        "free_pct": free_pct, "gmm_free_pct": gmm_free_pct,
        "n_trials": len(rows),
        "nav_z": nav_z,
        "x_span": float(xs[-1] - xs[0]),
        "y_span": float(ys[-1] - ys[0]),
        "_rows": rows,
        "_grid_slice": grid[:, :, astar_grid.best_iz],
        "_xs": xs, "_ys": ys,
    }
    for p in PLANNERS:
        summary[f"{p}_success"] = sum(r[f"{p}_ok"] for r in rows)
        summary[f"{p}_t_med"]   = _t_med(f"{p}_t")
        for mk in ("raw_len", "smo_len", "max_occ", "clr", "smo_raw", "smo_smo"):
            summary[f"{p}_{mk}_med"] = _med(f"{p}_{mk}")

    # Derived advantage: smoothness of A*(Grid) raw vs RRT*(GMM) raw
    summary["smo_advantage_gmm"]  = (summary["ast_grid_smo_raw_med"] /
                                     max(summary["rrt_gmm_smo_raw_med"], 1e-6))
    summary["smo_advantage_grid"] = (summary["ast_grid_smo_raw_med"] /
                                     max(summary["rrt_grid_smo_raw_med"], 1e-6))
    return summary


# ------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------

def print_summary(summaries):
    metrics = [
        ("Success",       "{p}_success",   "{s[n_trials]}",  lambda v, r: f"{int(v):2d}/{r}"),
        ("Time med (s)",  "{p}_t_med",     "",               lambda v, _: f"{v:.3f}"),
        ("Smo raw (rad)", "{p}_smo_raw_med","",              lambda v, _: f"{v:.2f}"),
        ("Smo smo (rad)", "{p}_smo_smo_med","",              lambda v, _: f"{v:.2f}"),
        ("Max occ",       "{p}_max_occ_med","",              lambda v, _: f"{v:.3f}"),
        ("Clearance",     "{p}_clr_med",   "",               lambda v, _: f"{v:.3f}"),
        ("Len smo (m)",   "{p}_smo_len_med","",              lambda v, _: f"{v:.1f}"),
    ]

    col_w = 12
    header = f"{'Scene':<7} {'Split':<6} {'Free%':>5}  " + \
             "  ".join(f"{PLANNER_LABELS[p]:>{col_w}}" for p in PLANNERS)

    for label, key_tmpl, ref_tmpl, fmt in metrics:
        print(f"\n── {label} " + "─" * (len(header) - len(label) - 4))
        print(header)
        print("─" * len(header))
        for s in summaries:
            ref = str(s["n_trials"])
            vals = []
            for p in PLANNERS:
                key = key_tmpl.format(p=p)
                v = s[key]
                vals.append(fmt(v, ref).rjust(col_w))
            print(f"{s['label']:<7} {s['split']:<6} {s['free_pct']:>4.1f}%  " +
                  "  ".join(vals))

    # Aggregate train vs test
    print(f"\n{'='*70}")
    for split in ("train", "test"):
        sub = [s for s in summaries if s["split"] == split]
        if not sub:
            continue
        print(f"\n  {split.upper()} ({len(sub)} scenes) — median across scenes:")
        for p in PLANNERS:
            sr  = np.mean([s[f"{p}_success"] / s["n_trials"] for s in sub]) * 100
            t   = np.median([s[f"{p}_t_med"] for s in sub])
            smo = np.median([s[f"{p}_smo_raw_med"] for s in sub])
            occ = np.median([s[f"{p}_max_occ_med"] for s in sub])
            print(f"    {PLANNER_LABELS[p]:<14} success={sr:.0f}%  "
                  f"time={t:.2f}s  smo_raw={smo:.2f}rad  max_occ={occ:.3f}")
    print(f"{'='*70}\n")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _bar4(ax, labels, vals_by_planner, ylabel, title, test_labels=None):
    """Grouped bar chart with 4 planners per scene."""
    n = len(labels)
    x = np.arange(n)
    w = 0.18
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for i, p in enumerate(PLANNERS):
        ax.bar(x + offsets[i], vals_by_planner[p], w,
               label=PLANNER_LABELS[p],
               color=PLANNER_COLORS[p], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    if test_labels:
        for xi, lbl in enumerate(labels):
            if lbl in test_labels:
                ax.axvspan(xi - 0.5, xi + 0.5, color="gold", alpha=0.12, zorder=0)


def plot_all(summaries, save_dir):
    labels     = [s["label"] for s in summaries]
    test_lbls  = {s["label"] for s in summaries if s["split"] == "test"}

    def _vbp(metric_key):
        return {p: [s[f"{p}_{metric_key}"] for s in summaries] for p in PLANNERS}

    # ── Figure 1: smoothness + occupancy ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    _bar4(axes[0], labels, _vbp("smo_raw_med"),
          "Median raw smoothness (rad)", "Raw path smoothness (lower = better)",
          test_lbls)
    _bar4(axes[1], labels, _vbp("smo_smo_med"),
          "Median smoothed smoothness (rad)", "Smoothed path smoothness",
          test_lbls)
    _bar4(axes[2], labels, _vbp("max_occ_med"),
          "Median max occupancy", "Max occupancy along path (lower = safer)",
          test_lbls)
    _bar4(axes[3], labels, _vbp("clr_med"),
          "Median mean clearance", "Mean clearance (higher = safer)",
          test_lbls)

    fig.suptitle("Experiment 2 — 2×2 Factorial: {RRT*, A*} × {GMM, Grid}\n"
                 "(gold = held-out test scenes)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(save_dir, "smoothness_and_safety.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved {out}")

    # ── Figure 2: success rate + time + path length ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sr_vals = {p: [s[f"{p}_success"] / s["n_trials"] * 100 for s in summaries]
               for p in PLANNERS}
    _bar4(axes[0], labels, sr_vals,
          "Success rate (%)", "Planning success rate", test_lbls)
    axes[0].set_ylim(0, 115)

    t_vals = {p: [s[f"{p}_t_med"] for s in summaries] for p in PLANNERS}
    _bar4(axes[1], labels, t_vals,
          "Median planning time (s)", "Planning time", test_lbls)
    axes[1].set_yscale("log")

    len_vals = {p: [s[f"{p}_smo_len_med"] for s in summaries] for p in PLANNERS}
    _bar4(axes[2], labels, len_vals,
          "Median smoothed path length (m)", "Path length after shortcut-smoothing",
          test_lbls)

    fig.suptitle("Experiment 2 — Success, Time, Path Length", fontsize=12)
    plt.tight_layout()
    out = os.path.join(save_dir, "success_time_length.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved {out}")

    # ── Figure 3: scatter — free% vs smoothness advantage ────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    free_pcts = [s["free_pct"] for s in summaries]

    pairs_to_plot = [
        ("smo_advantage_gmm",  "A*(Grid) / RRT*(GMM) smoothness",
         "Algorithm advantage: A*(Grid) raw / RRT*(GMM) raw"),
        ("smo_advantage_grid", "A*(Grid) / RRT*(Grid) smoothness",
         "Representation effect: same A* ratio with Grid-RRT*"),
    ]
    for ax, (key, ylabel, title) in zip(axes, pairs_to_plot):
        for s, fp in zip(summaries, free_pcts):
            c = "red" if s["split"] == "test" else "steelblue"
            m = "D"  if s["split"] == "test" else "o"
            ax.scatter(fp, s[key], c=c, marker=m, s=80, zorder=3)
            ax.annotate(s["label"], (fp, s[key]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.axhline(1, color="gray", ls="--", lw=1)
        ax.set_xlabel("Free space % in nav slice", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],marker="o",color="w",markerfacecolor="steelblue",
                  markersize=9, label="Train"),
           Line2D([0],[0],marker="D",color="w",markerfacecolor="red",
                  markersize=9, label="Test (held-out)")]
    axes[0].legend(handles=leg, fontsize=8)

    fig.suptitle("Experiment 2 — Does scene complexity drive GMM advantage?",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(save_dir, "complexity_vs_advantage.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved {out}")

    # ── Figure 4: algorithm × representation heatmap (mean across scenes) ──
    splits = {"train": [s for s in summaries if s["split"] == "train"],
              "test":  [s for s in summaries if s["split"] == "test"],
              "all":   summaries}

    for split_name, sub in splits.items():
        if not sub:
            continue
        metrics_heat = [
            ("smo_raw_med",  "Raw smoothness (rad)↓"),
            ("max_occ_med",  "Max occupancy↓"),
            ("clr_med",      "Clearance↑"),
            ("smo_len_med",  "Path length (m)↓"),
        ]
        data = np.array([[np.nanmedian([s[f"{p}_{mk}"] for s in sub])
                          for p in PLANNERS]
                         for mk, _ in metrics_heat])

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(PLANNERS)))
        ax.set_xticklabels([PLANNER_LABELS[p] for p in PLANNERS], fontsize=10)
        ax.set_yticks(range(len(metrics_heat)))
        ax.set_yticklabels([lbl for _, lbl in metrics_heat], fontsize=10)
        plt.colorbar(im, ax=ax)
        for ri in range(len(metrics_heat)):
            for ci in range(len(PLANNERS)):
                ax.text(ci, ri, f"{data[ri, ci]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if data[ri, ci] > data[ri].mean() else "black")
        ax.set_title(f"Median metrics — {split_name} scenes  "
                     f"(n={len(sub)} scenes × {N_TRIALS} trials)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(save_dir, f"heatmap_{split_name}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Saved {out}")

    # ── Figure 5: example paths for 2 scenes ─────────────────────────
    interesting = sorted(summaries,
                         key=lambda s: s["smo_advantage_gmm"], reverse=True)[:2]
    for s in interesting:
        rows = s["_rows"]
        occ  = s["_grid_slice"]
        xs, ys = s["_xs"], s["_ys"]
        trial = next((r for r in rows
                      if all(r[f"{p}_ok"] for p in PLANNERS)), None)
        if trial is None:
            continue

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(occ.T, origin="lower",
                  extent=[xs[0], xs[-1], ys[0], ys[-1]],
                  cmap="Greys", vmin=0, vmax=1, alpha=0.55)

        ls_map = {"rrt_gmm": "-", "rrt_grid": "--", "ast_gmm": "-.", "ast_grid": ":"}
        for p in PLANNERS:
            path = trial[f"{p}_path_smo"]
            if path:
                pts = np.array(path)
                ax.plot(pts[:, 0], pts[:, 1], ls_map[p],
                        color=PLANNER_COLORS[p], lw=2.5,
                        label=PLANNER_LABELS[p], alpha=0.9)

        st = np.array(trial["start"]); go = np.array(trial["goal"])
        ax.scatter(*st, s=150, c="green", zorder=6, marker="o", label="Start")
        ax.scatter(*go, s=150, c="red",   zorder=6, marker="*", label="Goal")

        smo_vals = "  ".join(
            f"{PLANNER_LABELS[p]}:{trial.get(f'{p}_smo_raw', 0) or 0:.1f}"
            for p in PLANNERS
        )
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(
            f"Scene {s['label']} [{s['split'].upper()}]  free={s['free_pct']:.1f}%\n"
            f"Raw smoothness (rad): {smo_vals}",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        out = os.path.join(save_dir, f"paths_scene_{s['label']}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Saved {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summaries = []

    for label, gmm_dir, split in SCENES:
        s = run_scene(label, gmm_dir, split, rng_seed=SEED)
        summaries.append(s)

    print_summary(summaries)

    print("Saving plots …")
    plot_all(summaries, RESULTS_DIR)
    print(f"\nAll outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
