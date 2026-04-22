"""RRT* planner using GMM occupancy model for continuous collision checking."""

import time
import numpy as np
from scipy.spatial import KDTree


def smooth_path(path, is_edge_free):
    """Shortcut path: greedily skip waypoints when direct connection is collision-free."""
    if path is None or len(path) < 3:
        return path
    path = [np.asarray(p) for p in path]
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if is_edge_free(path[i], path[j]):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


class RRTStar:
    """
    RRT* in 2D (x, y) with 3D collision checking at a fixed navigation height.

    Uses SavedGMMOccupancyMap.edge_is_free() for continuous obstacle avoidance.
    """

    def __init__(
        self,
        model,
        bounds_2d,
        nav_z,
        step_size=3.0,
        max_iter=3000,
        goal_radius=4.0,
        neighbor_radius=10.0,
        goal_bias=0.05,
        occ_thresh=0.35,
        lam=10.0,
        power=1.0,
        edge_step=0.3,
        seed=None,
        collision_fn=None,
    ):
        """
        Args:
            model:           SavedGMMOccupancyMap instance (or None if collision_fn given).
            bounds_2d:       [[xmin, xmax], [ymin, ymax]] in world metres.
            nav_z:           Fixed z height (metres) for 2-D navigation plane.
            step_size:       Max extension distance per iteration (metres).
            max_iter:        Maximum RRT* iterations.
            goal_radius:     Distance threshold to declare goal reached (metres).
            neighbor_radius: Ball radius for RRT* rewiring (metres).
            goal_bias:       Probability of sampling the goal directly.
            occ_thresh:      Occupancy probability above which a point is occupied.
            lam, power:      Parameters forwarded to occupancy_probability().
            edge_step:       Sampling interval along edges for collision check (metres).
            seed:            Random seed for reproducibility.
            collision_fn:    Optional callable (a_2d, b_2d) -> bool that replaces the
                             default GMM-based edge checker. When provided, model may
                             be None.
        """
        self.model = model
        self.bounds = np.array(bounds_2d, dtype=float)  # (2, 2): [[xlo,xhi],[ylo,yhi]]
        self.nav_z = float(nav_z)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.neighbor_radius = neighbor_radius
        self.goal_bias = goal_bias
        self.occ_thresh = occ_thresh
        self.lam = lam
        self.power = power
        self.edge_step = edge_step
        self.rng = np.random.default_rng(seed)
        self._collision_fn = collision_fn  # overrides GMM if set

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_3d(self, pt_2d):
        return np.array([pt_2d[0], pt_2d[1], self.nav_z], dtype=np.float64)

    def _collision_free(self, a, b):
        if self._collision_fn is not None:
            return self._collision_fn(a, b)
        return self.model.edge_is_free(
            self._to_3d(a), self._to_3d(b),
            step=self.edge_step,
            occ_thresh=self.occ_thresh,
            lam=self.lam,
            power=self.power,
        )

    def _sample(self, goal):
        if self.rng.random() < self.goal_bias:
            return goal.copy()
        return np.array([
            self.rng.uniform(self.bounds[0, 0], self.bounds[0, 1]),
            self.rng.uniform(self.bounds[1, 0], self.bounds[1, 1]),
        ])

    def _steer(self, from_pt, to_pt):
        diff = to_pt - from_pt
        d = np.linalg.norm(diff)
        if d <= self.step_size:
            return to_pt.copy()
        return from_pt + self.step_size * diff / d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, start, goal):
        """
        Plan a path from start to goal.

        Args:
            start: (x, y) start position in world metres.
            goal:  (x, y) goal position in world metres.

        Returns:
            path: list of np.array([x, y]) waypoints, or None if planning failed.
            info: dict — keys: success, time_s, nodes_expanded, path_cost.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        t0 = time.time()

        # Check start / goal reachability (single-point edge collapses to a check)
        if not self._collision_free(start, start):
            return None, {"success": False, "time_s": 0.0, "nodes_expanded": 0,
                          "error": "start is occupied"}
        if not self._collision_free(goal, goal):
            return None, {"success": False, "time_s": 0.0, "nodes_expanded": 0,
                          "error": "goal is occupied"}

        nodes = [start.copy()]   # list of np.array [x,y]
        parents = [-1]           # parent index for each node
        costs = [0.0]            # cost-from-start for each node

        goal_node = -1
        best_goal_cost = float("inf")

        # Rebuild KDTree every REBUILD_EVERY new nodes
        REBUILD_EVERY = 20
        tree = KDTree(np.array(nodes))
        nodes_since_rebuild = 0

        for _ in range(self.max_iter):
            if nodes_since_rebuild >= REBUILD_EVERY:
                tree = KDTree(np.array(nodes))
                nodes_since_rebuild = 0

            x_rand = self._sample(goal)

            # Nearest neighbour
            _, idx_nn = tree.query(x_rand)
            x_nearest = nodes[idx_nn]

            # Steer
            x_new = self._steer(x_nearest, x_rand)

            if not self._collision_free(x_nearest, x_new):
                continue

            # Nodes within rewiring ball
            near_idxs = tree.query_ball_point(x_new, self.neighbor_radius)

            # Choose best parent among near nodes
            best_parent = idx_nn
            best_cost = costs[idx_nn] + np.linalg.norm(x_new - x_nearest)

            for i in near_idxs:
                c = costs[i] + np.linalg.norm(x_new - nodes[i])
                if c < best_cost and self._collision_free(nodes[i], x_new):
                    best_cost = c
                    best_parent = i

            # Add node
            new_idx = len(nodes)
            nodes.append(x_new.copy())
            parents.append(best_parent)
            costs.append(best_cost)
            nodes_since_rebuild += 1

            # Rewire: redirect near neighbours through new node if cheaper
            for i in near_idxs:
                c = best_cost + np.linalg.norm(nodes[i] - x_new)
                if c < costs[i] and self._collision_free(x_new, nodes[i]):
                    parents[i] = new_idx
                    costs[i] = c

            # Check goal arrival
            if np.linalg.norm(x_new - goal) <= self.goal_radius:
                if best_cost < best_goal_cost:
                    best_goal_cost = best_cost
                    goal_node = new_idx

        elapsed = time.time() - t0
        info = {
            "success": goal_node != -1,
            "time_s": elapsed,
            "nodes_expanded": len(nodes),
            "path_cost": best_goal_cost if goal_node != -1 else None,
        }

        if goal_node == -1:
            return None, info

        # Walk parent pointers to reconstruct path
        path = []
        idx = goal_node
        while idx != -1:
            path.append(nodes[idx].copy())
            idx = parents[idx]
        path.reverse()

        # Append exact goal if the final node is not already there
        if np.linalg.norm(path[-1] - goal) > 0.01 and self._collision_free(path[-1], goal):
            path.append(goal.copy())

        return path, info

    def smooth_path(self, path):
        """Shortcut-smooth a path using this planner's collision checker."""
        return smooth_path(path, self._collision_free)
