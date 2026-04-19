
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

try:
    import open3d as o3d
except ImportError:
    o3d = None


def load_and_downsample_ply(
    ply_path,
    voxel_size=0.05,
    max_fit_points=200000,
    random_state=0,
):
    """
    Load .ply, voxel downsample, then optionally random subsample for fitting.
    Returns:
        xyz_fit: points used for GMM fitting
        xyz_vis: downsampled points for visualization/bounds
        rgb_vis: optional colors for visualization
    """
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
        print(f"Using random subset for GMM fit: {len(xyz_fit)}")
    else:
        xyz_fit = xyz
        print(f"Using all downsampled points for GMM fit: {len(xyz_fit)}")

    return xyz_fit, xyz, rgb


class FastGMMOccupancyMap:
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

    def fit(self, X):
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
        return self

    def density(self, X_query, chunk_size=200000):
        """
        Returns mixture density p(x).
        Uses score_samples = log p(x), then exponentiates.
        """
        X_query = np.asarray(X_query, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query[None, :]

        outs = []
        for i in range(0, len(X_query), chunk_size):
            chunk = X_query[i:i + chunk_size]
            logp = self.gmm.score_samples(chunk)
            outs.append(np.exp(logp))
        return np.concatenate(outs, axis=0)

    def occupancy_probability(self, X_query, lam=20.0, chunk_size=200000):
        s = self.density(X_query, chunk_size=chunk_size)
        return 1.0 - np.exp(-lam * s)


def make_occupancy_grid_chunked(
    occ_map,
    xyz_bounds_source,
    grid_res=64,
    padding=0.1,
    lam=20.0,
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


def run_fast_pipeline(
    ply_path,
    out_dir="gmm_occ_fast",
    voxel_size=0.05,
    max_fit_points=200000,
    n_components=128,
    covariance_type="diag",
    reg_covar=1e-4,
    max_iter=100,
    grid_res=128,
    lam=500.0,
    random_state=0,
):
    os.makedirs(out_dir, exist_ok=True)

    xyz_fit, xyz_vis, rgb_vis = load_and_downsample_ply(
        ply_path,
        voxel_size=voxel_size,
        max_fit_points=max_fit_points,
        random_state=random_state,
    )

    occ_map = FastGMMOccupancyMap(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(xyz_fit)

    grid_prob, xs, ys, zs = make_occupancy_grid_chunked(
        occ_map,
        xyz_bounds_source=xyz_vis,
        grid_res=grid_res,
        padding=0.1,
        lam=lam,
        query_chunk=200000,
    )

    np.save(os.path.join(out_dir, "occupancy_grid.npy"), grid_prob)
    np.save(os.path.join(out_dir, "xs.npy"), xs)
    np.save(os.path.join(out_dir, "ys.npy"), ys)
    np.save(os.path.join(out_dir, "zs.npy"), zs)

    save_slice_visualization(grid_prob, xs, ys, zs, out_dir)

    print(f"Saved results to {out_dir}")
    return occ_map, grid_prob, xs, ys, zs

if __name__ == "__main__":
    # Example:
    # python script.py
    ply_path = "/home/chuhanc/1-Data/16761_Final_Project/processed/2013_05_28_drive_0000_sync_0000000002_0000000385_clean.ply"

    occ_map, grid_prob, xs, ys, zs = run_fast_pipeline(
        ply_path=ply_path,
        out_dir="gmm_occ_fast",
        voxel_size=0.03,         # try 0.03 to 0.10
        max_fit_points=200000,   # 100k-300k is a good range
        n_components=128,         # try 32, 64, 96
        covariance_type="diag",  # much faster than "full"
        reg_covar=1e-4,
        max_iter=100,
        grid_res=128,             # start small; 48 or 64
        lam=500.0,
    )