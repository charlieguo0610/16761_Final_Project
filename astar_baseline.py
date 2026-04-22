"""A* planner on a precomputed voxel occupancy grid (baseline comparison)."""

import heapq
import time
import numpy as np


class AStarGrid:
    """
    A* planner on a 2-D slice of a 3-D occupancy grid.

    Automatically selects the z-slice with the highest obstacle density as
    the navigation plane (typically vehicle-body height in KITTI-360).
    """

    def __init__(self, occupancy_grid, xs, ys, zs, occ_thresh=0.35):
        """
        Args:
            occupancy_grid: (nx, ny, nz) float32 occupancy probabilities [0, 1].
            xs, ys, zs:     1-D coordinate arrays for each axis (world metres).
            occ_thresh:     Cells with probability >= this are treated as obstacles.
        """
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.occ_thresh = occ_thresh

        # Pick z-slice with most obstacle cells as the 2-D navigation plane
        occ_per_z = np.array(
            [(occupancy_grid[:, :, iz] >= occ_thresh).mean() for iz in range(len(zs))]
        )
        self.best_iz = int(np.argmax(occ_per_z))
        self.nav_z = float(zs[self.best_iz])

        # Binary free-space map: True = free, shape (nx, ny)
        self.free_2d = occupancy_grid[:, :, self.best_iz] < occ_thresh

        # World-space cell dimensions
        self.dx = float((xs[-1] - xs[0]) / (len(xs) - 1))
        self.dy = float((ys[-1] - ys[0]) / (len(ys) - 1))

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        ix = int(np.clip(round((x - self.xs[0]) / self.dx), 0, len(self.xs) - 1))
        iy = int(np.clip(round((y - self.ys[0]) / self.dy), 0, len(self.ys) - 1))
        return ix, iy

    def grid_to_world(self, ix, iy):
        return float(self.xs[ix]), float(self.ys[iy])

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, start_xy, goal_xy):
        """
        Plan a shortest path from start to goal on the 2-D occupancy grid.

        Args:
            start_xy: (x, y) start in world metres.
            goal_xy:  (x, y) goal in world metres.

        Returns:
            path: list of [x, y] world-coordinate waypoints, or None if no path.
            info: dict — keys: success, time_s, nodes_expanded, path_cost_world.
        """
        t0 = time.time()
        sx, sy = self.world_to_grid(*start_xy)
        gx, gy = self.world_to_grid(*goal_xy)
        nx, ny = self.free_2d.shape

        def _fail(reason):
            return None, {"success": False, "time_s": time.time() - t0,
                          "nodes_expanded": 0, "error": reason}

        if not self.free_2d[sx, sy]:
            return _fail("start is in occupied cell")
        if not self.free_2d[gx, gy]:
            return _fail("goal is in occupied cell")

        # Diagonal-distance heuristic (admissible for 8-connectivity)
        def h(x, y):
            ddx, ddy = abs(x - gx), abs(y - gy)
            return (ddx + ddy) + (np.sqrt(2) - 2) * min(ddx, ddy)

        # 8-connected neighbours: (di, dj, move_cost)
        NEIGHBORS = [
            (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
            ( 0, -1, 1.0),                       ( 0, 1, 1.0),
            ( 1, -1, np.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, np.sqrt(2)),
        ]

        open_heap = [(h(sx, sy), 0.0, sx, sy)]
        g_score = {(sx, sy): 0.0}
        came_from = {}
        closed = set()
        nodes_expanded = 0

        while open_heap:
            f, g, x, y = heapq.heappop(open_heap)

            if (x, y) in closed:
                continue
            closed.add((x, y))
            nodes_expanded += 1

            if x == gx and y == gy:
                # Reconstruct world-coordinate path
                path = []
                cur = (x, y)
                while cur in came_from:
                    path.append(list(self.grid_to_world(*cur)))
                    cur = came_from[cur]
                path.append(list(self.grid_to_world(sx, sy)))
                path.reverse()

                # g is in grid units; scale by average cell size for world metres
                avg_cell = (self.dx + self.dy) / 2.0
                return path, {
                    "success": True,
                    "time_s": time.time() - t0,
                    "nodes_expanded": nodes_expanded,
                    "path_cost_world": g * avg_cell,
                }

            for di, dj, cost in NEIGHBORS:
                ni, nj = x + di, y + dj
                if 0 <= ni < nx and 0 <= nj < ny and self.free_2d[ni, nj]:
                    if (ni, nj) in closed:
                        continue
                    ng = g + cost
                    if ng < g_score.get((ni, nj), float("inf")):
                        g_score[(ni, nj)] = ng
                        came_from[(ni, nj)] = (x, y)
                        heapq.heappush(open_heap, (ng + h(ni, nj), ng, ni, nj))

        return None, {"success": False, "time_s": time.time() - t0,
                      "nodes_expanded": nodes_expanded, "error": "no path found"}
