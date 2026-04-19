import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

try:
    import open3d as o3d
except ImportError:
    o3d = None


class SavedGMMOccupancyMap:
    def __init__(
        self,
        n_components=64,
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=100,
        random_state=0,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state

        self.gmm = None
        self.density_ref_ = None
        self.fitted_ = False

    def fit(self, X, density_ref_percentile=95):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must have shape (N, 3)")

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            init_params="kmeans",
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm.fit(X)
        self.fitted_ = True

        n_ref = min(len(X), 50000)
        dens = self.density(X[:n_ref])
        self.density_ref_ = float(np.percentile(dens, density_ref_percentile))

     
        return self

    def density(self, X_query, chunk_size=200000):
        if not self.fitted_ or self.gmm is None:
            raise RuntimeError("Model is not fitted or loaded.")

        X_query = np.asarray(X_query, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query[None, :]
        if X_query.ndim != 2 or X_query.shape[1] != 3:
            raise ValueError("X_query must have shape (Q, 3) or (3,)")

        outs = []
        for i in range(0, len(X_query), chunk_size):
            chunk = X_query[i:i + chunk_size]
            logp = self.gmm.score_samples(chunk)
            outs.append(np.exp(logp))
        return np.concatenate(outs, axis=0)

    def occupancy_probability(self, X_query, lam=10.0, power=1.0, chunk_size=200000):
        s = self.density(X_query, chunk_size=chunk_size)
        if self.density_ref_ is not None:
            s = s / (self.density_ref_ + 1e-12)
        s = np.clip(s, 0.0, None)
        return 1.0 - np.exp(-lam * (s ** power))

    def shrink_covariances(self, factor=0.5):
        if not self.fitted_ or self.gmm is None:
            raise RuntimeError("Model is not fitted or loaded.")
        if factor <= 0:
            raise ValueError("factor must be > 0")

        if self.covariance_type == "diag":
            self.gmm.covariances_ *= factor ** 2
            self.gmm.precisions_cholesky_ = 1.0 / np.sqrt(self.gmm.covariances_)
        elif self.covariance_type == "spherical":
            self.gmm.covariances_ *= factor ** 2
            self.gmm.precisions_cholesky_ = 1.0 / np.sqrt(self.gmm.covariances_)
        else:
            raise NotImplementedError(
                "shrink_covariances currently supports diag and spherical covariance types."
            )

    def is_occupied(self, x, occ_thresh=0.35, lam=10.0, power=1.0):
        p = self.occupancy_probability(x, lam=lam, power=power)[0]
        return bool(p >= occ_thresh)

    def edge_is_free(self, a, b, step=0.2, occ_thresh=0.35, lam=10.0, power=1.0):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dist = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(dist / step)))
        pts = np.linspace(a, b, n)
        occ = self.occupancy_probability(pts, lam=lam, power=power)
        return bool(np.all(occ < occ_thresh))

    def save(self, path):
        if not self.fitted_ or self.gmm is None:
            raise RuntimeError("Cannot save an unfitted model.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved model to: {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path} did not contain a {cls.__name__}")
        return obj


def load_and_downsample_ply(
    ply_path,
    voxel_size=0.05,
    max_fit_points=200000,
    random_state=0,
):
    if o3d is None:
        raise ImportError("Please install open3d: pip install open3d")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Original points: {len(pcd.points)}")

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"After voxel downsampling: {len(pcd.points)}")

    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = np.asarray(pcd.colors, dtype=np.float64) if len(pcd.colors) > 0 else None

    if len(xyz) > max_fit_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(xyz), size=max_fit_points, replace=False)
        xyz_fit = xyz[idx]
        print(f"Using random subset for fit: {len(xyz_fit)}")
    else:
        xyz_fit = xyz
        print(f"Using all downsampled points for fit: {len(xyz_fit)}")

    return xyz_fit, xyz, rgb


def make_occupancy_grid_chunked(
    occ_map,
    xyz_bounds_source,
    grid_res=64,
    padding=0.1,
    lam=20.0,
    power=2.0,
    query_chunk=200000,
):
    mins = xyz_bounds_source.min(axis=0)
    maxs = xyz_bounds_source.max(axis=0)
    extent = maxs - mins

    mins = mins - padding * extent
    maxs = maxs + padding * extent

    xs = np.linspace(mins[0], maxs[0], grid_res)
    ys = np.linspace(mins[1], maxs[1], grid_res)
    zs = np.linspace(mins[2], maxs[2], grid_res)

    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

    prob = occ_map.occupancy_probability(
        query,
        lam=lam,
        power=power,
        chunk_size=query_chunk,
    )
    grid_prob = prob.reshape(grid_res, grid_res, grid_res)
    return grid_prob, xs, ys, zs


def save_slice_visualization(grid_prob, xs, ys, zs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ix = len(xs) // 2
    iy = len(ys) // 2
    iz = len(zs) // 2

    slices = [
        ("yz_slice_xmid.png", grid_prob[ix, :, :].T, [ys[0], ys[-1], zs[0], zs[-1]], "y", "z"),
        ("xz_slice_ymid.png", grid_prob[:, iy, :].T, [xs[0], xs[-1], zs[0], zs[-1]], "x", "z"),
        ("xy_slice_zmid.png", grid_prob[:, :, iz].T, [xs[0], xs[-1], ys[0], ys[-1]], "x", "y"),
    ]

    for fname, img, extent, xlabel, ylabel in slices:
        plt.figure(figsize=(7, 6))
        plt.imshow(img, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Occupancy probability")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(fname[:-4])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def save_input_pointcloud_preview(xyz, rgb, out_png, max_points=100000):
    N = len(xyz)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        xyz_vis = xyz[idx]
        rgb_vis = rgb[idx] if rgb is not None else None
    else:
        xyz_vis = xyz
        rgb_vis = rgb

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if rgb_vis is not None:
        ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2], c=rgb_vis, s=0.5)
    else:
        ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2], s=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Input point cloud preview")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_3d_occupancy_pointcloud(grid_prob, xs, ys, zs, out_ply, thresh=0.3):
    if o3d is None:
        raise ImportError("open3d is required to save the occupancy point cloud.")

    mask = grid_prob >= thresh
    idx = np.argwhere(mask)
    if len(idx) == 0:
        print(f"No occupied voxels above threshold {thresh}")
        return

    pts = np.stack([
        xs[idx[:, 0]],
        ys[idx[:, 1]],
        zs[idx[:, 2]],
    ], axis=1)

    vals = grid_prob[mask]
    colors = np.stack([vals, np.zeros_like(vals), 1.0 - vals], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved occupancy point cloud to {out_ply}")


def run_pipeline(
    ply_path,
    out_dir="gmm_occ_output",
    model_pkl_path=None,
    voxel_size=0.05,
    max_fit_points=1000000,
    n_components=64,
    covariance_type="diag",
    reg_covar=1e-4,
    max_iter=100,
    shrink_factor=None,
    grid_res=128,
    lam=20.0,
    power=1.0,
    occ_thresh=0.3,
    random_state=0,
):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading + downsampling PLY...")
    xyz_fit, xyz_vis, rgb_vis = load_and_downsample_ply(
        ply_path=ply_path,
        voxel_size=voxel_size,
        max_fit_points=max_fit_points,
        random_state=random_state,
    )

    save_input_pointcloud_preview(
        xyz_vis,
        rgb_vis,
        os.path.join(out_dir, "input_pointcloud.png"),
    )

    print("Fitting GMM occupancy model...")
    occ_map = SavedGMMOccupancyMap(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(xyz_fit)

    if shrink_factor is not None:
        print(f"Shrinking covariances by factor {shrink_factor}...")
        occ_map.shrink_covariances(factor=shrink_factor)

    if model_pkl_path is None:
        model_pkl_path = os.path.join(out_dir, "gmm_occupancy_model.pkl")
    occ_map.save(model_pkl_path)

    print("Evaluating occupancy grid...")
    grid_prob, xs, ys, zs = make_occupancy_grid_chunked(
        occ_map,
        xyz_bounds_source=xyz_vis,
        grid_res=grid_res,
        padding=0.1,
        lam=lam,
        power=power,
        query_chunk=200000,
    )

    np.save(os.path.join(out_dir, "occupancy_grid.npy"), grid_prob)
    np.save(os.path.join(out_dir, "xs.npy"), xs)
    np.save(os.path.join(out_dir, "ys.npy"), ys)
    np.save(os.path.join(out_dir, "zs.npy"), zs)

    print("Saving occupancy slice visualizations...")
    save_slice_visualization(grid_prob, xs, ys, zs, out_dir)

    if o3d is not None:
        print("Saving 3D occupancy point cloud...")
        save_3d_occupancy_pointcloud(
            grid_prob,
            xs,
            ys,
            zs,
            os.path.join(out_dir, "occupancy_points.ply"),
            thresh=occ_thresh,
        )

    print(f"Done. Outputs saved to: {out_dir}")
    return occ_map, grid_prob, xs, ys, zs


def load_model_and_query(model_pkl_path, query_points, lam=20.0, power=2.0):
    occ_map = SavedGMMOccupancyMap.load(model_pkl_path)
    probs = occ_map.occupancy_probability(query_points, lam=lam, power=power)
    return probs


if __name__ == "__main__":
    # ply_path = "/home/chuhanc/1-Data/16761_Final_Project/processed/2013_05_28_drive_0000_sync_0000000002_0000000385_clean.ply"
    # output_dir = "gmm/2013_05_28_drive_0000_sync_0000000002_0000000385_clean"
    
    import argparse
    parser = argparse.ArgumentParser(description="Fit GMM occupancy map from a PLY point cloud.")

    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="Path to input .ply point cloud",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    args = parser.parse_args()

    occ_map, grid_prob, xs, ys, zs = run_pipeline(
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        model_pkl_path=f"{args.out_dir}/gmm_occupancy_model.pkl",
        voxel_size=0.05,
        max_fit_points=1000000,
        n_components=64,
        covariance_type="diag",
        reg_covar=1e-4,
        max_iter=100,
        shrink_factor=None,   # set None to disable sharpening
        grid_res=128,
        lam=10.0,
        power=1.0,
        occ_thresh=0.3,
        random_state=0,
    )

    # # Example re-load + query
    # q = np.array([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 2.0, 0.5],
    #     [3.0, 1.0, 1.2],
    # ], dtype=np.float64)

    # probs = load_model_and_query(
    #     "gmm_occ_new/gmm_occupancy_model.pkl",
    #     q,
    #     lam=10.0,
    #     power=1.0,
    # )
    # print("Reloaded model query:", probs)