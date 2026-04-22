"""
Batch GMM training for all processed scenes.

- Skips scenes that already have a trained model + grid.
- Loads PLY files with the pure-numpy reader (no open3d required).
- Trains in parallel across available CPU cores.
- Saves: gmm_occupancy_model.pkl, occupancy_grid.npy, xs/ys/zs.npy,
         and an xy-slice PNG for quick visual inspection.
"""

import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
PROCESSED_DIR = Path("processed")
GMM_DIR       = Path("gmm")

N_COMPONENTS   = 64
COV_TYPE       = "diag"
MAX_ITER       = 100
REG_COVAR      = 1e-4
MAX_FIT_POINTS = 200_000
RANDOM_STATE   = 0
GRID_RES       = 128
LAM            = 20.0
POWER          = 1.0
OCC_THRESH     = 0.35

# Use at most this many parallel workers
MAX_WORKERS = 4


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _load_xyz(ply_path: str, max_points: int, rng_seed: int) -> np.ndarray:
    """Read PLY → (N,3) float64, random-subsampled to max_points."""
    from kitti360_dataset_pipeline import read_ply_vertices
    verts = read_ply_vertices(ply_path)
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float64)
    if len(xyz) > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def _save_slice_png(grid_prob, xs, ys, zs, out_path: str):
    """Save the xy occupancy slice at the most-occupied z level."""
    occ_per_z = np.array([(grid_prob[:, :, iz] >= OCC_THRESH).mean()
                           for iz in range(len(zs))])
    iz = int(np.argmax(occ_per_z))
    sl = grid_prob[:, :, iz]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sl.T, origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap="hot_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Occupancy probability")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"xy occupancy slice  z={zs[iz]:.2f}m  (iz={iz})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------
# Per-scene training (runs in worker process)
# -----------------------------------------------------------------

def train_scene(args):
    """Train one GMM scene. Returns (scene_name, success, elapsed_s)."""
    ply_path, out_dir = args

    # Limit internal threading to avoid CPU oversubscription
    os.environ["OMP_NUM_THREADS"]      = "2"
    os.environ["MKL_NUM_THREADS"]      = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"

    import warnings; warnings.filterwarnings("ignore")

    from gmm3 import SavedGMMOccupancyMap, make_occupancy_grid_chunked

    scene_name = Path(ply_path).stem
    t0 = time.time()

    try:
        os.makedirs(out_dir, exist_ok=True)

        # ---- Load ------------------------------------------------
        print(f"[{scene_name}] loading ...", flush=True)
        xyz = _load_xyz(ply_path, MAX_FIT_POINTS, RANDOM_STATE)
        x0, x1 = xyz[:, 0].min(), xyz[:, 0].max()
        y0, y1 = xyz[:, 1].min(), xyz[:, 1].max()
        z0, z1 = xyz[:, 2].min(), xyz[:, 2].max()
        print(f"[{scene_name}] {len(xyz):,} pts  "
              f"x=[{x0:.0f},{x1:.0f}]  y=[{y0:.0f},{y1:.0f}]  "
              f"z=[{z0:.1f},{z1:.1f}]", flush=True)

        # ---- Fit GMM (suppress per-iteration sklearn output) -----
        print(f"[{scene_name}] fitting GMM ({N_COMPONENTS} components) ...", flush=True)
        model = SavedGMMOccupancyMap(
            n_components=N_COMPONENTS,
            covariance_type=COV_TYPE,
            reg_covar=REG_COVAR,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(xyz)
        print(f"[{scene_name}] GMM fitted  density_ref={model.density_ref_:.3e}", flush=True)

        # ---- Save model -----------------------------------------
        model_path = os.path.join(out_dir, "gmm_occupancy_model.pkl")
        model.save(model_path)

        # ---- Occupancy grid -------------------------------------
        print(f"[{scene_name}] building {GRID_RES}³ occupancy grid ...", flush=True)
        grid_prob, xs, ys, zs = make_occupancy_grid_chunked(
            model,
            xyz_bounds_source=xyz,
            grid_res=GRID_RES,
            padding=0.1,
            lam=LAM,
            power=POWER,
            query_chunk=200_000,
        )
        np.save(os.path.join(out_dir, "occupancy_grid.npy"), grid_prob.astype(np.float32))
        np.save(os.path.join(out_dir, "xs.npy"), xs)
        np.save(os.path.join(out_dir, "ys.npy"), ys)
        np.save(os.path.join(out_dir, "zs.npy"), zs)

        # ---- Quick visualisation --------------------------------
        _save_slice_png(grid_prob, xs, ys, zs,
                        os.path.join(out_dir, "xy_slice_nav.png"))

        elapsed = time.time() - t0
        print(f"[{scene_name}] DONE in {elapsed:.1f}s  →  {out_dir}", flush=True)
        return scene_name, True, elapsed

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"[{scene_name}] ERROR after {elapsed:.1f}s: {exc}", flush=True)
        traceback.print_exc()
        return scene_name, False, elapsed


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def _is_trained(out_dir: Path) -> bool:
    return (out_dir / "gmm_occupancy_model.pkl").exists() and \
           (out_dir / "occupancy_grid.npy").exists()


def main():
    all_scenes = sorted(PROCESSED_DIR.glob("*.ply"))

    todo, done = [], []
    for ply in all_scenes:
        out_dir = GMM_DIR / ply.stem
        if _is_trained(out_dir):
            done.append(ply.stem)
        else:
            todo.append((str(ply), str(out_dir)))

    print(f"Scenes total : {len(all_scenes)}")
    print(f"Already done : {len(done)}")
    for s in done:
        print(f"  SKIP  {s}")
    print(f"To train     : {len(todo)}")
    for p, _ in todo:
        print(f"  TODO  {Path(p).stem}")

    if not todo:
        print("Nothing to train.")
        return

    n_workers = min(MAX_WORKERS, len(todo), mp.cpu_count())
    print(f"\nLaunching {n_workers} parallel worker(s) for {len(todo)} scene(s) …\n")

    t_global = time.time()

    if n_workers <= 1:
        results = [train_scene(a) for a in todo]
    else:
        # spawn avoids fork+sklearn threading issues on Linux
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(train_scene, todo)

    total_s = time.time() - t_global
    n_ok = sum(ok for _, ok, _ in results)

    print(f"\n{'='*55}")
    print(f"Finished  {n_ok}/{len(todo)} scenes in {total_s/60:.1f} min")
    for name, ok, t in sorted(results, key=lambda r: r[2]):
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag}  {name}  ({t:.1f}s)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
