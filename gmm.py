import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.cluster import KMeans

try:
    import open3d as o3d
except ImportError:
    o3d = None


def log_gaussian_pdf(X, mu, Sigma, eps=1e-6):
    """
    X: (N, D)
    mu: (D,)
    Sigma: (D, D)
    returns: (N,) log N(X | mu, Sigma)
    """
    D = X.shape[1]
    Sigma = Sigma + eps * np.eye(D)
    L = np.linalg.cholesky(Sigma)
    diff = X - mu
    sol = np.linalg.solve(L, diff.T)  # (D, N)
    maha = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet + maha)


def load_ply_xyz_rgb(ply_path):
    """
    Load XYZ and RGB from a .ply file.
    Returns:
        xyz: (N,3) float64
        rgb: (N,3) float64 in [0,1] or None
    """
    if o3d is None:
        raise ImportError(
            "open3d is required to read PLY files easily. Install with: pip install open3d"
        )

    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float64)

    rgb = None
    if len(pcd.colors) > 0:
        rgb = np.asarray(pcd.colors, dtype=np.float64)

    return xyz, rgb


class GMMOccupancyMap:
    def __init__(
        self,
        n_components=32,
        max_iters=100,
        tol=1e-4,
        reg_covar=1e-4,
        random_state=0,
    ):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None
        self.fitted_ = False

    def _initialize_kmeans(self, X):
        N, D = X.shape
        km = KMeans(
            n_clusters=self.n_components,
            init="k-means++",
            n_init=5,
            random_state=self.random_state,
        )
        labels = km.fit_predict(X)

        weights = np.zeros(self.n_components, dtype=np.float64)
        means = np.zeros((self.n_components, D), dtype=np.float64)
        covs = np.zeros((self.n_components, D, D), dtype=np.float64)

        rng = np.random.default_rng(self.random_state)

        for k in range(self.n_components):
            pts = X[labels == k]
            if len(pts) == 0:
                means[k] = X[rng.integers(0, len(X))]
                covs[k] = np.eye(D)
                weights[k] = 1.0 / self.n_components
                continue

            means[k] = pts.mean(axis=0)
            centered = pts - means[k]
            cov = centered.T @ centered / max(len(pts), 1)
            cov += self.reg_covar * np.eye(D)
            covs[k] = cov
            weights[k] = len(pts) / N

        weights /= weights.sum()
        return weights, means, covs

    def fit(self, X):
        """
        X: (N,3) point positions
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        assert D == 3, "Expected 3D positions"

        weights, means, covs = self._initialize_kmeans(X)
        prev_ll = -np.inf

        for it in range(self.max_iters):
            print(f"EM iter {it}...")
            # E-step
            log_probs = np.zeros((N, self.n_components), dtype=np.float64)
            for k in range(self.n_components):
                log_probs[:, k] = np.log(weights[k] + 1e-12) + log_gaussian_pdf(X, means[k], covs[k])

            log_norm = logsumexp(log_probs, axis=1, keepdims=True)
            responsibilities = np.exp(log_probs - log_norm)
            ll = np.sum(log_norm)

            # M-step
            Nk = responsibilities.sum(axis=0) + 1e-12
            weights = Nk / N
            means = (responsibilities.T @ X) / Nk[:, None]

            new_covs = np.zeros_like(covs)
            for k in range(self.n_components):
                diff = X - means[k]
                weighted = responsibilities[:, k][:, None] * diff
                cov = weighted.T @ diff / Nk[k]
                cov += self.reg_covar * np.eye(D)
                new_covs[k] = cov
            covs = new_covs

            if np.abs(ll - prev_ll) < self.tol:
                print(f"EM converged at iter {it}, loglik={ll:.4f}")
                break
            prev_ll = ll

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covs
        self.precisions_ = np.linalg.inv(covs)
        self.fitted_ = True
        return self

    def density(self, X_query, mahalanobis_gate=None):
        """
        Continuous score s(x) = sum_k pi_k N(x | mu_k, Sigma_k)
        X_query: (Q,3) or (3,)
        """
        assert self.fitted_
        X_query = np.asarray(X_query, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query[None, :]

        Q = X_query.shape[0]
        out = np.zeros(Q, dtype=np.float64)

        for k in range(self.n_components):
            mu = self.means_[k]
            Sigma = self.covariances_[k]
            Prec = self.precisions_[k]

            diff = X_query - mu
            d2 = np.einsum("qi,ij,qj->q", diff, Prec, diff)

            if mahalanobis_gate is not None:
                mask = d2 <= mahalanobis_gate ** 2
                if not np.any(mask):
                    continue
                vals = np.zeros(Q, dtype=np.float64)
                vals[mask] = np.exp(log_gaussian_pdf(X_query[mask], mu, Sigma))
            else:
                vals = np.exp(log_gaussian_pdf(X_query, mu, Sigma))

            out += self.weights_[k] * vals

        return out

    def occupancy_probability(self, X_query, lam=10.0, mahalanobis_gate=3.0):
        """
        P_occ(x) = 1 - exp(-lam * s(x))
        """
        s = self.density(X_query, mahalanobis_gate=mahalanobis_gate)
        return 1.0 - np.exp(-lam * s)


def make_occupancy_grid(gmm_map, xyz, grid_res=64, padding=0.1, lam=10.0, mahalanobis_gate=3.0):
    """
    Build a dense occupancy volume over the bounding box of the point cloud.

    Returns:
        grid_prob: (nx, ny, nz)
        xs, ys, zs: coordinate arrays
    """
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = maxs - mins

    mins = mins - padding * extent
    maxs = maxs + padding * extent

    xs = np.linspace(mins[0], maxs[0], grid_res)
    ys = np.linspace(mins[1], maxs[1], grid_res)
    zs = np.linspace(mins[2], maxs[2], grid_res)

    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)

    prob = gmm_map.occupancy_probability(
        query, lam=lam, mahalanobis_gate=mahalanobis_gate
    )
    grid_prob = prob.reshape(grid_res, grid_res, grid_res)

    return grid_prob, xs, ys, zs


def save_slice_visualization(grid_prob, xs, ys, zs, out_dir):
    """
    Save three mid-plane occupancy slices.
    """
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
        plt.title(fname.replace(".png", ""))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def save_3d_occupancy_pointcloud(grid_prob, xs, ys, zs, out_ply, thresh=0.2):
    """
    Save voxels with occupancy >= thresh as a point cloud PLY.
    """
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
    colors = np.stack([vals, np.zeros_like(vals), 1.0 - vals], axis=1)  # red-high, blue-low

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved occupancy point cloud to {out_ply}")


def save_input_pointcloud_preview(xyz, rgb, out_png, max_points=100000):
    """
    Save a simple 3D scatter preview of the input cloud.
    """
    N = len(xyz)
    if N > max_points:
        ids = np.random.choice(N, size=max_points, replace=False)
        xyz_vis = xyz[ids]
        rgb_vis = rgb[ids] if rgb is not None else None
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
    ax.set_title("Input PLY point cloud")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_pipeline(
    ply_path,
    out_dir="gmm_occ_output",
    n_components=32,
    max_iters=100,
    tol=1e-4,
    reg_covar=1e-4,
    grid_res=64,
    lam=10.0,
    mahalanobis_gate=3.0,
    occ_thresh=0.3,
):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading PLY...")
    xyz, rgb = load_ply_xyz_rgb(ply_path)
    print(f"Loaded {len(xyz)} points")

    save_input_pointcloud_preview(
        xyz, rgb, os.path.join(out_dir, "input_pointcloud.png")
    )

    print("Fitting GMM...")
    gmm_map = GMMOccupancyMap(
        n_components=n_components,
        max_iters=max_iters,
        tol=tol,
        reg_covar=reg_covar,
        random_state=0,
    ).fit(xyz)

    print("Evaluating occupancy grid...")
    grid_prob, xs, ys, zs = make_occupancy_grid(
        gmm_map,
        xyz,
        grid_res=grid_res,
        padding=0.1,
        lam=lam,
        mahalanobis_gate=mahalanobis_gate,
    )

    np.save(os.path.join(out_dir, "occupancy_grid.npy"), grid_prob)
    np.save(os.path.join(out_dir, "xs.npy"), xs)
    np.save(os.path.join(out_dir, "ys.npy"), ys)
    np.save(os.path.join(out_dir, "zs.npy"), zs)

    print("Saving slice visualizations...")
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
    return gmm_map, grid_prob, xs, ys, zs


if __name__ == "__main__":
    # Example:
    # python script.py
    ply_path = "/home/chuhanc/1-Data/16761_Final_Project/processed/2013_05_28_drive_0000_sync_0000000002_0000000385_clean.ply"

    run_pipeline(
        ply_path=ply_path,
        out_dir="gmm_occ_output50",
        n_components=48,
        max_iters=50,
        tol=1e-4,
        reg_covar=1e-3,
        grid_res=72,
        lam=20.0,
        mahalanobis_gate=3.0,
        occ_thresh=0.25,
    )