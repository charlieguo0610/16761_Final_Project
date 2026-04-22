# GMM Occupancy Mapping for Mobile Robot Navigation
**CMU 16-761 Mobile Robot Planning — Spring 2026**

We represent urban LiDAR scenes as Gaussian Mixture Models (GMMs) and use them as continuous collision checkers for RRT\* motion planning. The core question: does a probabilistic continuous map produce safer, smoother paths than a standard discrete voxel grid?

---

## Setup

```bash
conda create -n mobile-robots python=3.11
conda activate mobile-robots
pip install -r requirements.txt
```

---

## How it works

Each scene (from KITTI-360) is fit as a 64-component diagonal-covariance GMM on up to 200k LiDAR points. The GMM assigns a continuous occupancy probability to any 3D point. RRT\* uses this directly as a collision checker; A\* uses a thresholded 128³ voxel grid derived from the same GMM.

```
gmm3.py                  # GMM occupancy model
rrt_star.py              # RRT* with GMM collision checking
astar_baseline.py        # A* on voxel grid
train_gmms.py            # Batch-train all 10 scenes
experiment1.py           # Exp 1: RRT*(GMM) vs A*(Grid), single scene
experiment2_multiscene.py  # Exp 2: 4-planner factorial, 10 scenes
experiment3_dynamic.py   # Exp 3: GMM vs Grid under partial observations
processed/               # 10 PLY scenes (Git LFS, 374 MB)
gmm/                     # Trained models + occupancy grids
results/                 # Plots and outputs
```

---

## Experiments

### Experiment 1 — RRT\*(GMM) vs A\*(Grid), single scene, 30 trials

| Metric | RRT\*(GMM) | A\*(Grid) |
|---|---|---|
| Success rate | 80% | 100% |
| Planning time | ~2.6 s | ~0.13 s |
| Raw path smoothness | **1.9 rad** | 9.9 rad |
| Smoothed smoothness | **0.78 rad** | 1.28 rad |
| Max occupancy | **0.165** | 0.201 |

GMM paths are **5× smoother** and **18% further from obstacles**. After shortcut smoothing, path lengths converge (~102 m each) — GMM adds safety without sacrificing efficiency.

```bash
python experiment1.py   # → results/experiment1/
```

---

### Experiment 2 — 2×2 factorial across 10 scenes

Four planners: `{RRT*, A*} × {GMM checker, Grid checker}`. Evaluated on 8 train scenes and 2 held-out test scenes (0009, 0010) without retuning any parameters.

| Split | Scenes |
|---|---|
| Train | 0000a, 0000b, 0002, 0003, 0004, 0005, 0006, 0007 |
| Test (held-out) | **0009, 0010** |

RRT\*(GMM) produces the smoothest paths (median 1.4 rad) across all scenes. The advantage holds on held-out scenes, confirming it doesn't require per-scene tuning.

```bash
python train_gmms.py            # fit GMMs for all 10 scenes (~5–10 min)
python experiment2_multiscene.py  # → results/experiment2/
```

---

### Experiment 3 — Dynamic mapping under partial observations

The point cloud is shuffled and revealed in 11 stages (5%→100%), simulating a robot exploring an unknown environment. At each stage, both a partial GMM and a 64³ voxel grid are built from the accumulated points. Two RRT\* planners compete — one using the GMM, one using the grid — and paths are evaluated for safety against the full pre-trained GMM (ground truth).

**Key findings:**
- GMM achieves 10–20% lower map error (MAE) at every coverage level
- Grid-planned paths score 4–6× higher occupancy against ground truth at mid-coverage — the grid's discrete cells create false-free regions where no point landed, causing the planner to route into hazardous areas
- GMM updates faster: it skips the 64³ grid re-evaluation step (~0.5 s saved per update)

```bash
python experiment3_dynamic.py   # → results/experiment3/
```

---

## Dataset

10 scenes from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) (CC BY-NC-SA 3.0), voxel-downsampled to 0.10 m. Total: 13.6M points, 374 MB.

| # | Sequence | Points | Free% |
|--:|----------|-------:|------:|
| 1 | 0000a | 1,537,065 | 81.6% |
| 2 | 0000b | 1,369,037 | 71.5% |
| 3 | 0002 | 1,399,570 | 81.4% |
| 4 | 0003 | 1,236,101 | 86.8% |
| 5 | 0004 | 1,357,738 | 82.2% |
| 6 | 0005 | 1,475,695 | 86.7% |
| 7 | 0006 | 1,500,124 | 94.8% |
| 8 | 0007 | 1,193,993 | 93.8% |
| 9 *(test)* | 0009 | 1,446,272 | 86.5% |
| 10 *(test)* | 0010 | 1,106,672 | 85.2% |

```bash
git lfs pull   # if PLY files appear as pointer files
```

---

## License

KITTI-360 data: **CC BY-NC-SA 3.0**.  Code: CMU 16-761 coursework.
