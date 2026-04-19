# 16-761 Final Project — KITTI-360 Static Scene Dataset

LiDAR point-cloud scenes for GMM-based occupancy mapping and RRT motion planning.

## Dataset

10 static-scene windows selected from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) (CC BY-NC-SA 3.0), covering 9 driving sequences in Karlsruhe, Germany. Each PLY was produced from the official fused accumulated point clouds, filtered to visible points only, and voxel-downsampled to **0.10 m**.

| # | Sequence | Frame window | Points | Size |
|--:|----------|--------------|-------:|-----:|
| 1 | 0000 | 0000000002 – 0000000385 | 1,537,065 | 41 MB |
| 2 | 0000 | 0000000834 – 0000001286 | 1,369,037 | 37 MB |
| 3 | 0002 | 0000004391 – 0000004625 | 1,399,570 | 37 MB |
| 4 | 0003 | 0000000002 – 0000000282 | 1,236,101 | 33 MB |
| 5 | 0004 | 0000002897 – 0000003118 | 1,357,738 | 36 MB |
| 6 | 0005 | 0000000002 – 0000000357 | 1,475,695 | 39 MB |
| 7 | 0006 | 0000000002 – 0000000403 | 1,500,124 | 40 MB |
| 8 | 0007 | 0000000002 – 0000000125 | 1,193,993 | 32 MB |
| 9 | 0009 | 0000000002 – 0000000292 | 1,446,272 | 39 MB |
| 10 | 0010 | 0000000002 – 0000000208 | 1,106,672 | 30 MB |

**Total: 13,622,267 points, 374 MB** (binary little-endian PLY, stored via Git LFS).

### PLY fields

`x y z red green blue semantic instance visible confidence` — world-coordinate frame.

## Git LFS

The 10 processed PLY files (374 MB total) are stored with [Git LFS](https://git-lfs.com/). After cloning, you must pull the actual data:

```bash
# Install Git LFS (one-time)
# macOS: brew install git-lfs
# Ubuntu: sudo apt install git-lfs
git lfs install

# Clone (LFS pointers are fetched automatically with recent git versions)
git clone https://github.com/charlieguo0610/16761_Final_Project.git
cd 16761_Final_Project

# If PLY files show as small pointer files, pull the real data:
git lfs pull
```

Without `git lfs install`, the PLY files will be ~130-byte pointer files instead of the actual point clouds.

## Pipeline

```
kitti360_dataset_pipeline.py          # download / extract / filter / export
kitti360_batch_example.json           # manifest for the 10 scenes above
kitti360_urls.example.json            # template for download URLs
```

### Quick start

```bash
# Reproduce the 10 processed scenes (requires the 3 official KITTI-360 zips)
python kitti360_dataset_pipeline.py batch-export \
  --manifest kitti360_batch_example.json \
  --root ./KITTI-360 \
  --out-dir ./processed \
  --voxel-size 0.10 \
  --visible-only
```

See `KITTI360_dataset_script_README.txt` for the full workflow.

### GMM

Run 
```
python gmm3.py --ply_path PLY_PATH --out_dir OUT_DIR
```
to run GMM and visualize the occupancy map. 3D occupancy probability can be loaded from `occupancy_grid.npy`. To load from existing gmm model and query occupancy probability at positions in the scene, use code snipet below: 

```
    # Example re-load + query points
    q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.5],
        [3.0, 1.0, 1.2],
    ], dtype=np.float64)

    probs = load_model_and_query(
        "gmm/2013_05_28_drive_0000_sync_0000000002_0000000385_clean/gmm_occupancy_model.pkl",
         q,
        lam=10.0,
        power=1.0,
    )
    print("Reloaded model query:", probs)
```

## License

KITTI-360 data is released under **CC BY-NC-SA 3.0** by the Autonomous Vision Group (CVPR 2022). Code in this repo is for academic coursework (CMU 16-761).
