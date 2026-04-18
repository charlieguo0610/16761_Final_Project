
#!/usr/bin/env python3
"""
KITTI-360 dataset prep script for LiDAR -> GMM -> RRT experiments.

What this script does
---------------------
1) Download official KITTI-360 archives from user-provided official URLs or copy
   already-downloaded archives into a local archive folder.
2) Extract only the pieces you need (calibration, poses, semantics, raw Velodyne).
3) Re-export official fused static/dynamic windows as clean PLY files
   (optional crop/downsample/filter).
4) Accumulate raw Velodyne .bin scans into a world-frame PLY using KITTI-360
   calibration and poses.

Why two paths?
--------------
- If you want the fastest route for static-scene mapping, use the official fused
  static windows under data_3d_semantics/.../static/*.ply.
- If you want to build your own windows from raw scans, use build-from-raw.

Notes
-----
- KITTI-360 requires registration / intended-use declaration on the official site.
  This script assumes you are using official archives/URLs you are allowed to use.
- The script uses only Python stdlib + numpy.
- Output PLY files are written in binary little-endian format.

Example
-------
# 1) Put official URLs (or local paths) into urls.json
python kitti360_dataset_pipeline.py download \
    --manifest urls.json \
    --root ./KITTI-360

# 2) List available fused static windows for sequence 0000
python kitti360_dataset_pipeline.py list-windows \
    --root ./KITTI-360 \
    --sequence 0000

# 3) Export one fused static window as cropped/downsampled PLY
python kitti360_dataset_pipeline.py prepare-fused \
    --root ./KITTI-360 \
    --sequence 0000 \
    --window 0000000000_0000000240 \
    --out ./processed/0000_0000000000_0000000240_clean.ply \
    --crop "-20,80,-30,30,-2,8" \
    --voxel-size 0.10 \
    --visible-only

# 4) Build your own accumulated world-frame PLY from raw scans
python kitti360_dataset_pipeline.py build-from-raw \
    --root ./KITTI-360 \
    --sequence 0000 \
    --start 0 \
    --end 240 \
    --stride 5 \
    --out ./processed/0000_0000000000_0000000240_raw_accum.ply \
    --crop "-20,80,-30,30,-2,8" \
    --voxel-size 0.10

# 5) Batch-export 10 scenes in one shot
python kitti360_dataset_pipeline.py batch-export \
    --manifest kitti360_batch_example.json \
    --root ./KITTI-360 \
    --out-dir ./processed \
    --voxel-size 0.10 \
    --visible-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Constants / utilities
# -----------------------------

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}

DTYPE_TO_PLY_TYPE = {
    ("i", 1): "char",
    ("u", 1): "uchar",
    ("i", 2): "short",
    ("u", 2): "ushort",
    ("i", 4): "int",
    ("u", 4): "uint",
    ("f", 4): "float",
    ("f", 8): "double",
}

DEFAULT_ARCHIVE_NAMES = {
    "calibration": ["calibration.zip"],
    "poses": ["data_poses.zip"],
    "semantics": ["data_3d_semantics.zip", "data_3d_semantics_train.zip"],
    "raw_velodyne": None,  # sequence-dependent
}


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def seq_to_name(seq: str) -> str:
    seq = str(seq)
    if seq.startswith("2013_05_28_drive_") and seq.endswith("_sync"):
        return seq
    return f"2013_05_28_drive_{int(seq):04d}_sync"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def parse_crop(crop: Optional[str]) -> Optional[Tuple[float, float, float, float, float, float]]:
    if crop is None:
        return None
    parts = [float(x.strip()) for x in crop.split(",")]
    if len(parts) != 6:
        raise ValueError("--crop must be xmin,xmax,ymin,ymax,zmin,zmax")
    xmin, xmax, ymin, ymax, zmin, zmax = parts
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError("Invalid crop bounds")
    return xmin, xmax, ymin, ymax, zmin, zmax


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def find_existing_file(search_roots: Sequence[Path], candidate_names: Sequence[str]) -> Optional[Path]:
    for root in search_roots:
        for name in candidate_names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


# -----------------------------
# Download / extract
# -----------------------------

def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """
    Accepted JSON shapes:
    1) {"calibration.zip": "https://...", "data_poses.zip": "..."}
    2) {"archives": {"calibration.zip": "https://...", ...}}
    3) {"calibration": "https://...", "poses": "..."}  # keys arbitrary, script uses filename basename
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "archives" in data and isinstance(data["archives"], dict):
        data = data["archives"]

    flat: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(v, str):
            raise ValueError(f"Manifest value for key {k!r} must be a string URL/path")
        filename = Path(k).name if k.endswith(".zip") else Path(v).name
        if not filename.endswith(".zip"):
            # fallback if key is semantic name like "poses"
            filename = Path(v).name
        if not filename.endswith(".zip"):
            raise ValueError(f"Could not infer .zip filename for key={k!r}, value={v!r}")
        flat[filename] = v
    return flat


def download_or_copy(src: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        print(f"[skip] {dst} already exists")
        return

    ensure_dir(dst.parent)

    # local file copy
    local_candidate = Path(src)
    if local_candidate.exists():
        print(f"[copy] {local_candidate} -> {dst}")
        shutil.copy2(local_candidate, dst)
        return

    if src.startswith("file://"):
        local_path = Path(src[7:])
        print(f"[copy] {local_path} -> {dst}")
        shutil.copy2(local_path, dst)
        return

    if not (src.startswith("http://") or src.startswith("https://")):
        raise FileNotFoundError(f"Source not found and not an HTTP(S) URL: {src}")

    print(f"[download] {src} -> {dst}")
    req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total is not None else None
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = 100.0 * downloaded / total_size
                print(f"\r  {human_size(downloaded)} / {human_size(total_size)} ({pct:5.1f}%)", end="")
            else:
                print(f"\r  {human_size(downloaded)}", end="")
        print("")


def extract_zip(zip_path: Path, dst_root: Path, member_prefixes: Optional[Sequence[str]] = None, force: bool = False) -> None:
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if member_prefixes:
            wanted = []
            for m in members:
                if any(m.startswith(prefix) for prefix in member_prefixes):
                    wanted.append(m)
            if not wanted:
                raise RuntimeError(f"No zip members matched prefixes {member_prefixes} in {zip_path}")
            members = wanted

        for member in members:
            target = dst_root / member
            if target.exists() and not force:
                continue
            if member.endswith("/"):
                ensure_dir(target)
                continue
            ensure_dir(target.parent)
            with zf.open(member, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def maybe_extract_component(root: Path, component: str, seq_name: Optional[str] = None, force: bool = False) -> None:
    archives_dir = root / "archives"
    search_roots = [archives_dir, root, Path(".")]

    if component == "calibration":
        if (root / "calibration").exists():
            return
        archive = find_existing_file(search_roots, DEFAULT_ARCHIVE_NAMES["calibration"])
        if archive is None:
            raise FileNotFoundError("Could not find calibration.zip in root/archives or root")
        extract_zip(archive, root, force=force)
        return

    if component == "poses":
        if seq_name is None:
            raise ValueError("seq_name required for poses")
        if (root / "data_poses" / seq_name).exists():
            return
        archive = find_existing_file(search_roots, DEFAULT_ARCHIVE_NAMES["poses"])
        if archive is None:
            raise FileNotFoundError("Could not find data_poses.zip in root/archives or root")
        prefixes = [f"data_poses/{seq_name}/"]
        extract_zip(archive, root, member_prefixes=prefixes, force=force)
        return

    if component == "semantics":
        if seq_name is None:
            raise ValueError("seq_name required for semantics")
        if (root / "data_3d_semantics" / "train" / seq_name).exists() or (root / "data_3d_semantics" / "test" / seq_name).exists():
            return
        archive = find_existing_file(search_roots, DEFAULT_ARCHIVE_NAMES["semantics"])
        if archive is None:
            raise FileNotFoundError("Could not find data_3d_semantics.zip in root/archives or root")
        prefixes = [
            f"data_3d_semantics/train/{seq_name}/",
            f"data_3d_semantics/test/{seq_name}/",
        ]
        extract_zip(archive, root, member_prefixes=prefixes, force=force)
        return

    if component == "raw_velodyne":
        if seq_name is None:
            raise ValueError("seq_name required for raw_velodyne")
        if (root / "data_3d_raw" / seq_name / "velodyne_points" / "data").exists():
            return
        archive_name = f"{seq_name}_velodyne.zip"
        archive = find_existing_file(search_roots, [archive_name])
        if archive is None:
            raise FileNotFoundError(f"Could not find {archive_name} in root/archives or root")
        extract_zip(archive, root, force=force)
        return

    raise ValueError(f"Unknown component: {component}")


# -----------------------------
# Calibration / poses
# -----------------------------

def _parse_all_floats(text: str) -> List[float]:
    return [float(x) for x in FLOAT_RE.findall(text)]


def load_cam_to_velo(calib_path: Path) -> np.ndarray:
    """
    Returns 4x4 T_cam0_to_velo from calib_cam_to_velo.txt.
    """
    text = calib_path.read_text(encoding="utf-8")
    vals = _parse_all_floats(text)
    if len(vals) < 12:
        raise RuntimeError(f"Expected at least 12 floats in {calib_path}, got {len(vals)}")
    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = np.array(vals[:12], dtype=np.float64).reshape(3, 4)
    return M


def load_R_rect_00(perspective_path: Path) -> np.ndarray:
    text = perspective_path.read_text(encoding="utf-8").splitlines()
    for line in text:
        if line.startswith("R_rect_00:"):
            vals = [float(x) for x in line.split()[1:]]
            if len(vals) != 9:
                raise RuntimeError(f"Expected 9 numbers for R_rect_00 in {perspective_path}")
            R = np.eye(4, dtype=np.float64)
            R[:3, :3] = np.array(vals, dtype=np.float64).reshape(3, 3)
            return R
    raise RuntimeError(f"Could not find R_rect_00 in {perspective_path}")


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two 4x4 SE(3) transforms at parameter t in [0, 1]."""
    q0 = _rotation_matrix_to_quaternion(T0[:3, :3])
    q1 = _rotation_matrix_to_quaternion(T1[:3, :3])
    qi = _quaternion_slerp(q0, q1, t)
    ti = (1.0 - t) * T0[:3, 3] + t * T1[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quaternion_to_rotation_matrix(qi)
    T[:3, 3] = ti
    return T


@dataclass
class FramePoses:
    frames: np.ndarray   # (N,)
    mats: np.ndarray     # (N,4,4)

    def get(self, frame: int) -> np.ndarray:
        idx = np.searchsorted(self.frames, frame)
        if idx < len(self.frames) and int(self.frames[idx]) == int(frame):
            return self.mats[idx]
        if idx == 0:
            return self.mats[0]
        if idx >= len(self.frames):
            return self.mats[-1]
        i0, i1 = idx - 1, idx
        f0, f1 = int(self.frames[i0]), int(self.frames[i1])
        alpha = (frame - f0) / (f1 - f0) if f1 != f0 else 0.0
        return se3_interpolate(self.mats[i0], self.mats[i1], alpha)


def load_frame_poses(pose_file: Path, matrix_rows: int, matrix_cols: int) -> FramePoses:
    arr = np.loadtxt(pose_file)
    if arr.ndim == 1:
        arr = arr[None, :]
    frames = arr[:, 0].astype(np.int64)
    mats = arr[:, 1:].reshape(-1, matrix_rows, matrix_cols)
    if matrix_rows == 3 and matrix_cols == 4:
        mats_h = np.tile(np.eye(4, dtype=np.float64), (mats.shape[0], 1, 1))
        mats_h[:, :3, :4] = mats
        mats = mats_h
    elif matrix_rows == 4 and matrix_cols == 4:
        pass
    else:
        raise ValueError("Unsupported matrix shape")
    return FramePoses(frames=frames, mats=mats.astype(np.float64))


# -----------------------------
# PLY IO
# -----------------------------

def read_ply_vertices(ply_path: Path) -> np.ndarray:
    """
    Reads a vertex-only PLY file into a structured numpy array.
    Supports ascii and binary_little_endian PLY with scalar vertex properties.
    """
    with open(ply_path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {ply_path}")
            s = line.decode("ascii", errors="strict").strip()
            header_lines.append(s)
            if s == "end_header":
                break

        if header_lines[0] != "ply":
            raise RuntimeError(f"{ply_path} is not a PLY file")

        fmt = None
        vertex_count = None
        props: List[Tuple[str, str]] = []
        current_element = None

        for line in header_lines[1:]:
            if not line or line.startswith("comment"):
                continue
            toks = line.split()
            if toks[0] == "format":
                fmt = toks[1]
            elif toks[0] == "element":
                current_element = toks[1]
                if current_element == "vertex":
                    vertex_count = int(toks[2])
            elif toks[0] == "property" and current_element == "vertex":
                if toks[1] == "list":
                    raise RuntimeError("This reader does not support list properties")
                ply_type, name = toks[1], toks[2]
                if ply_type not in PLY_TYPE_TO_DTYPE:
                    raise RuntimeError(f"Unsupported PLY property type: {ply_type}")
                props.append((name, PLY_TYPE_TO_DTYPE[ply_type]))

        if fmt is None or vertex_count is None or not props:
            raise RuntimeError(f"Malformed PLY header in {ply_path}")

        dtype = np.dtype(props)

        if fmt == "binary_little_endian":
            dtype = dtype.newbyteorder("<")
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            return data
        if fmt == "ascii":
            rows: List[Tuple] = []
            for _ in range(vertex_count):
                line = f.readline().decode("ascii").strip()
                vals = line.split()
                if len(vals) != len(props):
                    raise RuntimeError(f"ASCII PLY row has {len(vals)} values but expected {len(props)}")
                parsed = []
                for raw, (_, dt) in zip(vals, props):
                    kind = np.dtype(dt).kind
                    if kind in ("i", "u"):
                        parsed.append(int(raw))
                    elif kind == "f":
                        parsed.append(float(raw))
                    else:
                        raise RuntimeError(f"Unsupported dtype kind in ASCII PLY: {dt}")
                rows.append(tuple(parsed))
            data = np.array(rows, dtype=dtype)
            return data

        raise RuntimeError(f"Unsupported PLY format: {fmt}")


def write_ply_vertices(ply_path: Path, arr: np.ndarray, comments: Optional[Sequence[str]] = None) -> None:
    if arr.dtype.names is None:
        raise ValueError("write_ply_vertices expects a structured numpy array")

    ensure_dir(ply_path.parent)
    with open(ply_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        if comments:
            for c in comments:
                f.write(f"comment {c}\n".encode("ascii"))
        f.write(f"element vertex {len(arr)}\n".encode("ascii"))
        for name in arr.dtype.names:
            field_dtype = arr.dtype.fields[name][0]
            field_dtype = field_dtype.newbyteorder("=")
            key = (field_dtype.kind, field_dtype.itemsize)
            if key not in DTYPE_TO_PLY_TYPE:
                raise RuntimeError(f"Cannot map numpy dtype {field_dtype} for field {name!r} to PLY type")
            ply_type = DTYPE_TO_PLY_TYPE[key]
            f.write(f"property {ply_type} {name}\n".encode("ascii"))
        f.write(b"end_header\n")

        out = arr
        if out.dtype.byteorder not in ("<", "="):
            out = out.astype(out.dtype.newbyteorder("<"))
        else:
            out = out.astype(out.dtype.newbyteorder("<"), copy=False)
        out.tofile(f)


def select_fields(arr: np.ndarray, field_names: Sequence[str]) -> np.ndarray:
    dtype = [(name, arr.dtype.fields[name][0]) for name in field_names]
    out = np.empty(arr.shape, dtype=dtype)
    for name in field_names:
        out[name] = arr[name]
    return out


# -----------------------------
# Filtering / downsampling
# -----------------------------

def xyz_from_structured(arr: np.ndarray) -> np.ndarray:
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def crop_mask_xyz(xyz: np.ndarray, crop: Optional[Tuple[float, float, float, float, float, float]]) -> np.ndarray:
    if crop is None:
        return np.ones(len(xyz), dtype=bool)
    xmin, xmax, ymin, ymax, zmin, zmax = crop
    return (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
        (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
        (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )


def voxel_downsample_structured(arr: np.ndarray, voxel_size: Optional[float]) -> np.ndarray:
    if voxel_size is None or voxel_size <= 0:
        return arr
    xyz = xyz_from_structured(arr)
    vox = np.floor(xyz / float(voxel_size)).astype(np.int64)
    key = np.ascontiguousarray(vox).view(np.dtype((np.void, vox.dtype.itemsize * vox.shape[1])))
    _, keep_idx = np.unique(key, return_index=True)
    keep_idx = np.sort(keep_idx)
    return arr[keep_idx]


def apply_structured_filters(
    arr: np.ndarray,
    crop: Optional[Tuple[float, float, float, float, float, float]] = None,
    visible_only: bool = False,
    confidence_min: Optional[float] = None,
    keep_semantic_ids: Optional[Sequence[int]] = None,
    drop_semantic_ids: Optional[Sequence[int]] = None,
    voxel_size: Optional[float] = None,
) -> np.ndarray:
    mask = np.ones(len(arr), dtype=bool)

    xyz = xyz_from_structured(arr)
    mask &= crop_mask_xyz(xyz, crop)

    if visible_only and "isVisible" in arr.dtype.names:
        mask &= (arr["isVisible"] > 0)

    if confidence_min is not None and "confidence" in arr.dtype.names:
        mask &= (arr["confidence"] >= float(confidence_min))

    if keep_semantic_ids is not None and "semanticID" in arr.dtype.names:
        ks = np.array(list(keep_semantic_ids), dtype=np.int32)
        mask &= np.isin(arr["semanticID"], ks)

    if drop_semantic_ids is not None and "semanticID" in arr.dtype.names:
        ds = np.array(list(drop_semantic_ids), dtype=np.int32)
        mask &= ~np.isin(arr["semanticID"], ds)

    out = arr[mask]
    out = voxel_downsample_structured(out, voxel_size)
    return out


# -----------------------------
# Commands
# -----------------------------

def cmd_download(args: argparse.Namespace) -> None:
    root = Path(args.root)
    archives_dir = root / "archives"
    ensure_dir(archives_dir)

    manifest = load_manifest(Path(args.manifest))
    for filename, src in manifest.items():
        dst = archives_dir / filename
        download_or_copy(src, dst, force=args.force)

    print("\nDone. Archives are in:", archives_dir)


def _find_static_dir(root: Path, seq_name: str, dynamic: bool = False) -> Path:
    split_dirs = [
        root / "data_3d_semantics" / "train" / seq_name,
        root / "data_3d_semantics" / "test" / seq_name,
    ]
    sub = "dynamic" if dynamic else "static"
    for d in split_dirs:
        candidate = d / sub
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find semantics {sub}/ directory for {seq_name}")


def cmd_list_windows(args: argparse.Namespace) -> None:
    root = Path(args.root)
    seq_name = seq_to_name(args.sequence)
    maybe_extract_component(root, "semantics", seq_name=seq_name, force=args.force)
    static_dir = _find_static_dir(root, seq_name, dynamic=args.dynamic)
    files = sorted(static_dir.glob("*.ply"))
    if not files:
        raise RuntimeError(f"No PLY windows found in {static_dir}")
    print(f"# {static_dir}")
    for p in files:
        print(p.stem)


def cmd_prepare_fused(args: argparse.Namespace) -> None:
    root = Path(args.root)
    seq_name = seq_to_name(args.sequence)
    maybe_extract_component(root, "semantics", seq_name=seq_name, force=args.force)

    target_dir = _find_static_dir(root, seq_name, dynamic=args.dynamic)
    all_files = sorted(target_dir.glob("*.ply"))
    if not all_files:
        raise RuntimeError(f"No PLY windows found in {target_dir}")

    if args.all_windows:
        selected = all_files
    else:
        if not args.window:
            print("Available windows:")
            for p in all_files:
                print(" ", p.stem)
            print("\nPass --window <start_end> or use --all-windows")
            return
        wanted = args.window[:-4] if args.window.endswith(".ply") else args.window
        selected = [p for p in all_files if p.stem == wanted]
        if not selected:
            raise FileNotFoundError(f"Window {wanted} not found in {target_dir}")

    crop = parse_crop(args.crop)
    keep_ids = parse_int_list(args.keep_semantic_ids)
    drop_ids = parse_int_list(args.drop_semantic_ids)
    out_base = Path(args.out)

    for idx, ply_path in enumerate(selected):
        print(f"[read] {ply_path}")
        arr = read_ply_vertices(ply_path)
        print(f"  points before: {len(arr):,}")

        arr = apply_structured_filters(
            arr,
            crop=crop,
            visible_only=args.visible_only,
            confidence_min=args.confidence_min,
            keep_semantic_ids=keep_ids,
            drop_semantic_ids=drop_ids,
            voxel_size=args.voxel_size,
        )
        print(f"  points after : {len(arr):,}")

        if args.xyz_only:
            arr = select_fields(arr, ["x", "y", "z"])
        elif args.xyzrgb_only:
            fields = [n for n in ["x", "y", "z", "red", "green", "blue"] if n in arr.dtype.names]
            arr = select_fields(arr, fields)

        if args.all_windows:
            ensure_dir(out_base)
            out_path = out_base / f"{ply_path.stem}_processed.ply"
        else:
            out_path = out_base
            if out_path.suffix.lower() != ".ply":
                ensure_dir(out_path)
                out_path = out_path / f"{ply_path.stem}_processed.ply"

        write_ply_vertices(
            out_path,
            arr,
            comments=[
                "Source: KITTI-360 fused semantics window",
                f"Sequence: {seq_name}",
                f"Window: {ply_path.stem}",
            ],
        )
        print(f"[write] {out_path}")


def build_world_frame_from_raw(
    root: Path,
    seq_name: str,
    start: Optional[int],
    end: Optional[int],
    stride: int,
    crop: Optional[Tuple[float, float, float, float, float, float]],
    voxel_size: Optional[float],
    point_dim: int,
    min_range: Optional[float],
    max_range: Optional[float],
    keep_every_nth_point: int,
    add_frame_field: bool,
) -> np.ndarray:
    maybe_extract_component(root, "calibration")
    maybe_extract_component(root, "poses", seq_name=seq_name)
    maybe_extract_component(root, "raw_velodyne", seq_name=seq_name)

    raw_dir = root / "data_3d_raw" / seq_name / "velodyne_points" / "data"
    pose_file = root / "data_poses" / seq_name / "cam0_to_world.txt"
    calib_cam_to_velo = root / "calibration" / "calib_cam_to_velo.txt"

    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if not pose_file.exists():
        raise FileNotFoundError(pose_file)
    if not calib_cam_to_velo.exists():
        raise FileNotFoundError(calib_cam_to_velo)

    cam0_to_world = load_frame_poses(pose_file, matrix_rows=4, matrix_cols=4)
    T_cam0_to_velo = load_cam_to_velo(calib_cam_to_velo)
    # cam0_to_world.txt = T_{world <- cam0_unrectified}
    # calib_cam_to_velo.txt = T_{velo <- cam0_unrectified}
    # Both reference the same unrectified cam0 frame, so R_rect is NOT needed:
    #   T_{world <- velo} = cam0_to_world @ inv(cam_to_velo)
    T_velo_to_cam0 = np.linalg.inv(T_cam0_to_velo)

    bin_files = sorted(raw_dir.glob("*.bin"))
    frames = [int(p.stem) for p in bin_files]

    filtered: List[Tuple[int, Path]] = []
    for frame, p in zip(frames, bin_files):
        if start is not None and frame < start:
            continue
        if end is not None and frame > end:
            continue
        filtered.append((frame, p))

    if stride > 1:
        filtered = filtered[::stride]

    if not filtered:
        raise RuntimeError("No raw frames matched the requested range")

    xyz_blocks: List[np.ndarray] = []
    intensity_blocks: List[np.ndarray] = []
    frame_blocks: List[np.ndarray] = []

    for i, (frame, path) in enumerate(filtered, start=1):
        raw = np.fromfile(path, dtype=np.float32)
        if point_dim <= 0 or raw.size % point_dim != 0:
            raise RuntimeError(f"{path} has {raw.size} float32 values which is not divisible by point_dim={point_dim}")
        pts = raw.reshape(-1, point_dim)
        xyz = pts[:, :3].astype(np.float64)
        intensity = pts[:, 3].astype(np.float32) if point_dim >= 4 else np.zeros(len(pts), dtype=np.float32)

        # Optional thinning before transformation (useful for huge windows)
        if keep_every_nth_point > 1:
            keep = np.arange(0, len(xyz), keep_every_nth_point)
            xyz = xyz[keep]
            intensity = intensity[keep]

        ranges = np.linalg.norm(xyz, axis=1)
        mask = np.isfinite(xyz).all(axis=1)
        if min_range is not None:
            mask &= (ranges >= float(min_range))
        if max_range is not None:
            mask &= (ranges <= float(max_range))
        xyz = xyz[mask]
        intensity = intensity[mask]

        T_cam0_to_world_i = cam0_to_world.get(frame)
        T_velo_to_world = T_cam0_to_world_i @ T_velo_to_cam0

        xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
        xyz_world = (T_velo_to_world @ xyz_h.T).T[:, :3]

        if crop is not None:
            xyz_mask = crop_mask_xyz(xyz_world, crop)
            xyz_world = xyz_world[xyz_mask]
            intensity = intensity[xyz_mask]

        xyz_blocks.append(xyz_world.astype(np.float32))
        intensity_blocks.append(intensity.astype(np.float32))
        if add_frame_field:
            frame_blocks.append(np.full(len(xyz_world), frame, dtype=np.int32))

        if i % 20 == 0 or i == len(filtered):
            print(f"  processed {i}/{len(filtered)} frames")

    if not xyz_blocks:
        raise RuntimeError("No points left after filtering")

    xyz_all = np.concatenate(xyz_blocks, axis=0)
    intensity_all = np.concatenate(intensity_blocks, axis=0)
    frame_all = np.concatenate(frame_blocks, axis=0) if add_frame_field else None

    if add_frame_field:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4"), ("frame", "i4")]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")]
    arr = np.empty(len(xyz_all), dtype=dtype)
    arr["x"] = xyz_all[:, 0]
    arr["y"] = xyz_all[:, 1]
    arr["z"] = xyz_all[:, 2]
    arr["intensity"] = intensity_all
    if add_frame_field and frame_all is not None:
        arr["frame"] = frame_all

    arr = voxel_downsample_structured(arr, voxel_size)
    return arr


def cmd_build_from_raw(args: argparse.Namespace) -> None:
    root = Path(args.root)
    seq_name = seq_to_name(args.sequence)
    crop = parse_crop(args.crop)

    arr = build_world_frame_from_raw(
        root=root,
        seq_name=seq_name,
        start=args.start,
        end=args.end,
        stride=args.stride,
        crop=crop,
        voxel_size=args.voxel_size,
        point_dim=args.point_dim,
        min_range=args.min_range,
        max_range=args.max_range,
        keep_every_nth_point=args.keep_every_nth_point,
        add_frame_field=args.add_frame_field,
    )

    print(f"points after accumulation/filtering/downsampling: {len(arr):,}")
    out = Path(args.out)
    write_ply_vertices(
        out,
        arr,
        comments=[
            "Source: KITTI-360 raw Velodyne accumulation",
            f"Sequence: {seq_name}",
            f"Frames: {args.start if args.start is not None else 'min'}..{args.end if args.end is not None else 'max'}",
            f"Stride: {args.stride}",
        ],
    )
    print(f"[write] {out}")


def cmd_batch_export(args: argparse.Namespace) -> None:
    """Export multiple fused static windows listed in a JSON manifest."""
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    with open(args.manifest, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    if not isinstance(scenes, list):
        raise ValueError("Batch manifest must be a JSON array of scene specs")

    global_crop = parse_crop(args.crop)
    total = len(scenes)

    for i, spec in enumerate(scenes, start=1):
        seq = str(spec.get("sequence", ""))
        window = str(spec.get("window", ""))
        if not seq or not window:
            eprint(f"  [skip] entry {i}: missing 'sequence' or 'window'")
            continue

        scene_crop_str = spec.get("crop", args.crop)
        scene_crop = parse_crop(scene_crop_str) if scene_crop_str else global_crop
        scene_voxel = float(spec["voxel_size"]) if "voxel_size" in spec else args.voxel_size

        seq_name = seq_to_name(seq)
        wanted = window[:-4] if window.endswith(".ply") else window
        print(f"\n=== [{i}/{total}] {seq_name} / {wanted} ===")

        try:
            maybe_extract_component(root, "semantics", seq_name=seq_name)
            target_dir = _find_static_dir(root, seq_name, dynamic=args.dynamic)
        except FileNotFoundError as e:
            eprint(f"  [skip] {e}")
            continue

        ply_files = [p for p in sorted(target_dir.glob("*.ply")) if p.stem == wanted]
        if not ply_files:
            eprint(f"  [skip] window {wanted} not found in {target_dir}")
            continue

        arr = read_ply_vertices(ply_files[0])
        print(f"  points before: {len(arr):,}")

        keep_ids = parse_int_list(spec.get("keep_semantic_ids"))
        drop_ids = parse_int_list(spec.get("drop_semantic_ids"))

        arr = apply_structured_filters(
            arr,
            crop=scene_crop,
            visible_only=args.visible_only,
            confidence_min=args.confidence_min,
            keep_semantic_ids=keep_ids,
            drop_semantic_ids=drop_ids,
            voxel_size=scene_voxel,
        )
        print(f"  points after : {len(arr):,}")

        if args.xyz_only:
            arr = select_fields(arr, ["x", "y", "z"])
        elif args.xyzrgb_only:
            fields = [n for n in ["x", "y", "z", "red", "green", "blue"] if n in arr.dtype.names]
            arr = select_fields(arr, fields)

        out_path = out_dir / f"{seq_name}_{wanted}_clean.ply"
        write_ply_vertices(out_path, arr, comments=[
            "Source: KITTI-360 fused static window",
            f"Sequence: {seq_name}",
            f"Window: {wanted}",
        ])
        print(f"  [write] {out_path}")

    print(f"\nDone. Batch-exported {total} scenes -> {out_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download / extract / convert KITTI-360 data into PLY for mapping + navigation experiments."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    s = sub.add_parser("download", help="Download/copy official KITTI-360 zip archives listed in a JSON manifest.")
    s.add_argument("--manifest", required=True, help="JSON manifest of filename -> URL/path")
    s.add_argument("--root", required=True, help="KITTI-360 root directory")
    s.add_argument("--force", action="store_true", help="Overwrite existing archives")
    s.set_defaults(func=cmd_download)

    # list-windows
    s = sub.add_parser("list-windows", help="List fused static/dynamic PLY windows available for a sequence.")
    s.add_argument("--root", required=True, help="KITTI-360 root directory")
    s.add_argument("--sequence", required=True, help="Sequence ID like 0000 or full sequence name")
    s.add_argument("--dynamic", action="store_true", help="List dynamic windows instead of static")
    s.add_argument("--force", action="store_true", help="Force extraction if needed")
    s.set_defaults(func=cmd_list_windows)

    # prepare-fused
    s = sub.add_parser("prepare-fused", help="Read official fused static/dynamic PLY, filter/crop/downsample, and write a clean PLY.")
    s.add_argument("--root", required=True, help="KITTI-360 root directory")
    s.add_argument("--sequence", required=True, help="Sequence ID like 0000 or full sequence name")
    s.add_argument("--window", help="Window name like 0000000000_0000000240 (omit .ply is okay)")
    s.add_argument("--all-windows", action="store_true", help="Process all PLY windows in the sequence")
    s.add_argument("--dynamic", action="store_true", help="Use dynamic windows instead of static")
    s.add_argument("--out", required=True, help="Output .ply path, or output directory if --all-windows")
    s.add_argument("--crop", help="xmin,xmax,ymin,ymax,zmin,zmax")
    s.add_argument("--voxel-size", type=float, default=None, help="Voxel size in meters for downsampling")
    s.add_argument("--visible-only", action="store_true", help="Keep only points with isVisible > 0 when available")
    s.add_argument("--confidence-min", type=float, default=None, help="Keep only points with confidence >= value when available")
    s.add_argument("--keep-semantic-ids", default=None, help="Comma-separated semantic IDs to keep")
    s.add_argument("--drop-semantic-ids", default=None, help="Comma-separated semantic IDs to drop")
    s.add_argument("--xyz-only", action="store_true", help="Write only x,y,z fields")
    s.add_argument("--xyzrgb-only", action="store_true", help="Write only x,y,z,red,green,blue fields if available")
    s.add_argument("--force", action="store_true", help="Force extraction if needed")
    s.set_defaults(func=cmd_prepare_fused)

    # build-from-raw
    s = sub.add_parser("build-from-raw", help="Accumulate raw Velodyne scans into a world-frame PLY.")
    s.add_argument("--root", required=True, help="KITTI-360 root directory")
    s.add_argument("--sequence", required=True, help="Sequence ID like 0000 or full sequence name")
    s.add_argument("--start", type=int, default=None, help="First frame ID to include")
    s.add_argument("--end", type=int, default=None, help="Last frame ID to include")
    s.add_argument("--stride", type=int, default=1, help="Keep every Nth raw frame")
    s.add_argument("--out", required=True, help="Output .ply path")
    s.add_argument("--crop", help="xmin,xmax,ymin,ymax,zmin,zmax")
    s.add_argument("--voxel-size", type=float, default=None, help="Voxel size in meters for downsampling")
    s.add_argument("--point-dim", type=int, default=4, help="Number of float32 values per raw Velodyne point (default: 4)")
    s.add_argument("--min-range", type=float, default=None, help="Discard points closer than this distance (m)")
    s.add_argument("--max-range", type=float, default=None, help="Discard points farther than this distance (m)")
    s.add_argument("--keep-every-nth-point", type=int, default=1, help="Thin each raw scan before transformation")
    s.add_argument("--add-frame-field", action="store_true", help="Store source frame index as an extra PLY property")
    s.set_defaults(func=cmd_build_from_raw)

    # batch-export
    s = sub.add_parser("batch-export", help="Export multiple fused static windows listed in a JSON manifest.")
    s.add_argument("--manifest", required=True, help="JSON array of {sequence, window, ...} scene specs")
    s.add_argument("--root", required=True, help="KITTI-360 root directory")
    s.add_argument("--out-dir", required=True, help="Output directory for clean PLY files")
    s.add_argument("--crop", default=None, help="Default xmin,xmax,ymin,ymax,zmin,zmax (per-scene override in manifest)")
    s.add_argument("--voxel-size", type=float, default=None, help="Default voxel size (per-scene override in manifest)")
    s.add_argument("--visible-only", action="store_true", help="Keep only visible points")
    s.add_argument("--confidence-min", type=float, default=None, help="Min confidence")
    s.add_argument("--dynamic", action="store_true", help="Use dynamic windows instead of static")
    s.add_argument("--xyz-only", action="store_true", help="Write only x,y,z fields")
    s.add_argument("--xyzrgb-only", action="store_true", help="Write only x,y,z,red,green,blue fields")
    s.set_defaults(func=cmd_batch_export)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
