#!/usr/bin/env python3
"""
Mask unprojection and graph-selection helpers shared by the perception pipeline.

  - unproject_mask_points : YOLO mask + depth + camera pose -> world-frame points
                            (median-depth outlier rejection, even-stride subsample
                            to MAX_GNG_POINTS), paper Eq. (1) and Sec. V-A.
  - make_gng              : factory for the surface-graph fitter. GNG_IMPL=paper
                            (default) is the DBL-GNG of Siow et al. 2024
                            (dbl_gng_paper.py); legacy / raw select the ablation
                            baselines (earlier batch GNG with edge aging /
                            raw mask points, no graph).
  - GngOverlay and the draw/tint/build helpers are optional visualisation utilities
    (class-name keyed graphs; the paper pipeline in perception.py fits one graph
    per detected INSTANCE instead).
"""
import hashlib
import colorsys
import os
import time

import cv2
import numpy as np

from dbl_gng import DBLGrowingNeuralGas, deterministic_subsample
from dbl_gng_paper import PaperDBLGNG


class RawPoints:
    """Ablation 'no graph': the (<=250) unprojected mask points themselves are
    handed to the planner as if they were nodes (no learning, no edges)."""
    last_info = {}

    def fit(self, points):
        pts = np.asarray(points, dtype=np.float64)
        return {i: p for i, p in enumerate(pts)}, {}


def make_gng(seed=None):
    """GNG_IMPL=paper (default, DBL-GNG per Siow et al. 2024) or legacy (the
    earlier batch-update GNG, kept as the ablation baseline). GNG_SEED
    seeds the paper implementation's random start points; GNG_NODES,
    GNG_ETA, GNG_CUT override max nodes / start points / cut percentile
    for ablations."""
    impl = os.environ.get("GNG_IMPL", "paper")
    if impl == "raw":
        return RawPoints()
    if impl == "legacy":
        return DBLGrowingNeuralGas()
    if seed is None:
        seed = int(os.environ.get("GNG_SEED", "0"))
    return PaperDBLGNG(max_nodes=int(os.environ.get("GNG_NODES", "30")),
                       n_start=int(os.environ.get("GNG_ETA", "4")),
                       cut_pct=float(os.environ.get("GNG_CUT", "15")),
                       seed=seed)

REFIT_INTERVAL_S = 1.2
MAX_GNG_POINTS = 250
MIN_MASK_PIXELS = 60
NODE_SPHERE_RADIUS = 0.008  # meters
GRAPH_COLOR = (0.0, 1.0, 0.0)  # green, as requested


_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def color_for_class(name, s=0.75, v=0.95):
    """Stable (same class -> same color, every run) distinct color per
    object class, hashed from the class name rather than detection index so
    it doesn't shuffle when YOLO's detection order changes frame to frame.
    The hash is spread across hue via the golden-ratio-conjugate trick
    (hue_n = frac(hash * phi)) rather than plain hash-mod-360, since two
    unrelated class names hashing to nearby degrees would otherwise
    occasionally land on near-identical colors (seen with plain mod-360 on
    this project's own object set: two classes both landing near cyan)."""
    h32 = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    hue = ((h32 * _GOLDEN_RATIO_CONJUGATE) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return (r, g, b)


def filter_out_class(results, model_names, exclude_substr):
    """Ultralytics Results, filtered down to detections whose class name
    does NOT contain exclude_substr (case-insensitive) -- e.g. drop every
    "dining table" detection before it's drawn or masked anywhere. Shared
    by GngOverlay (depth/point-cloud masking) and the RGB annotator so both
    apply the exact same exclusion rule from one place."""
    if results.boxes is None or len(results.boxes) == 0:
        return results
    classes = results.boxes.cls.cpu().numpy().astype(int)
    keep = [i for i, c in enumerate(classes) if exclude_substr not in model_names[int(c)].lower()]
    if len(keep) == len(classes):
        return results
    return results[keep]


def _cv_from_mj():
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


OUTLIER_THRESH_M = 0.05  # matches the user's existing grasp_ws ROS node's default


def unproject_mask_points(mask, depth, cam_pos, cam_mat, fx, fy, cx, cy,
                           depth_min, depth_max, max_points=MAX_GNG_POINTS,
                           outlier_thresh=OUTLIER_THRESH_M):
    """3D world-frame points for the pixels inside a boolean mask.

    A YOLO instance mask can leak a few pixels past the true object edge
    (segmentation isn't pixel-perfect, especially on this synthetic-render
    domain), and any leaked pixel that lands on a neighboring object or the
    table sits at a very different depth from the rest of the mask -- that's
    enough to blow up a min/max extent estimate. Reject-by-median-depth is
    the same simple filter this project's own grasp_ws ROS node
    (dbl_gng_segmentor.py) already uses ahead of its GNG fit: keep only
    pixels within outlier_thresh of the mask's median depth, before GNG
    ever sees them -- no PCA/RANSAC involved, just one order statistic."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return np.zeros((0, 3))
    z = depth[vs, us].astype(np.float64)
    valid = (z > depth_min) & (z < depth_max)
    us, vs, z = us[valid], vs[valid], z[valid]
    if len(us) == 0:
        return np.zeros((0, 3))
    med_z = np.median(z)
    fg = np.abs(z - med_z) < outlier_thresh
    us, vs, z = us[fg], vs[fg], z[fg]
    if len(us) == 0:
        return np.zeros((0, 3))
    idx = deterministic_subsample(len(us), max_points)
    us, vs, z = us[idx], vs[idx], z[idx]
    x = (us.astype(np.float64) - cx) / fx * z
    y = (vs.astype(np.float64) - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1)
    R_world_cv = cam_mat @ _cv_from_mj()
    return pts_cam @ R_world_cv.T + cam_pos


def world_to_pixel(points_world, cam_pos, cam_mat, fx, fy, cx, cy):
    """Inverse of unproject_mask_points: world points -> (u, v) pixel coords
    (and camera-frame depth), points behind the camera get z<=0."""
    R_world_cv = cam_mat @ _cv_from_mj()
    pts_cam = (points_world - cam_pos) @ R_world_cv
    z = pts_cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy
    return u, v, z


class GngOverlay:
    def __init__(self, fx, fy, cx, cy, depth_min, depth_max, exclude_substr="table"):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.depth_min, self.depth_max = depth_min, depth_max
        self.exclude_substr = exclude_substr
        self._graphs = {}  # class_name -> dict(positions, edges, t)
        self.gng = make_gng()

    def masks_for_frame(self, yolo_results, model_names):
        """Returns [(class_name, bool_mask_2d, color_rgb01), ...] for every
        detection this frame whose class doesn't match exclude_substr."""
        out = []
        results = filter_out_class(yolo_results, model_names, self.exclude_substr)
        if results.masks is None:
            return out
        masks = results.masks.data.cpu().numpy().astype(bool)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for mask, cls_id in zip(masks, classes):
            name = model_names[int(cls_id)]
            if mask.sum() < MIN_MASK_PIXELS:
                continue
            out.append((name, mask, color_for_class(name)))
        return out

    def update_graphs(self, mask_infos, depth, cam_pos, cam_mat):
        """Refit any object's DBL-GNG graph whose timer has elapsed. Objects
        not seen this frame simply keep their last graph (aged out only by
        not being refreshed -- there's no need to actively clear them for
        this visualization)."""
        now = time.time()
        for name, mask, _color in mask_infos:
            entry = self._graphs.get(name)
            if entry is not None and (now - entry["t"]) < REFIT_INTERVAL_S:
                continue
            pts = unproject_mask_points(mask, depth, cam_pos, cam_mat,
                                         self.fx, self.fy, self.cx, self.cy,
                                         self.depth_min, self.depth_max)
            if len(pts) < 2:
                continue
            positions, edges = self.gng.fit(pts)
            self._graphs[name] = dict(positions=positions, edges=edges, t=now)

    def active_graphs(self, mask_infos):
        """(positions, edges) pairs for the classes visible this frame."""
        names = {name for name, _m, _c in mask_infos}
        return [self._graphs[n] for n in names if n in self._graphs]

    def graph_for(self, class_name):
        return self._graphs.get(class_name)

    # ------------------------------------------------------------------
    # Point cloud (3D) overlay
    # ------------------------------------------------------------------
    def tint_point_colors(self, colors_flat, us_grid, vs_grid, mask_infos, blend=0.6):
        """colors_flat: (H*W_strided, 3) float [0,1] array matching the
        us_grid/vs_grid sampling grid used to build the scene point cloud
        (Open3D point-cloud view). Blends each object's distinct color
        into the points that fall inside its mask."""
        if not mask_infos:
            return colors_flat
        out = colors_flat.copy()
        for _name, mask, color in mask_infos:
            sel = mask[vs_grid, us_grid].reshape(-1)
            if not sel.any():
                continue
            out[sel] = (1 - blend) * out[sel] + blend * np.array(color)
        return out

    def build_graph_geometry(self, graphs):
        """Merged green sphere mesh (nodes) + LineSet (edges) in world
        frame, or (None, None) if there's nothing to show."""
        all_positions = []
        all_edges = []
        offset = 0
        for g in graphs:
            positions, edges = g["positions"], g["edges"]
            if not positions:
                continue
            ids = list(positions.keys())
            id_to_idx = {nid: offset + i for i, nid in enumerate(ids)}
            all_positions.extend(positions[nid] for nid in ids)
            for e in edges:
                a, b = tuple(e)
                if a in id_to_idx and b in id_to_idx:
                    all_edges.append((id_to_idx[a], id_to_idx[b]))
            offset += len(ids)

        if not all_positions:
            return None, None

        import open3d as o3d  # optional dependency, only for the Open3D point-cloud view
        mesh = o3d.geometry.TriangleMesh()
        for p in all_positions:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=NODE_SPHERE_RADIUS, resolution=6)
            sphere.translate(p)
            mesh += sphere
        mesh.paint_uniform_color(GRAPH_COLOR)
        mesh.compute_vertex_normals()

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.array(all_positions))
        if all_edges:
            lineset.lines = o3d.utility.Vector2iVector(np.array(all_edges))
            lineset.colors = o3d.utility.Vector3dVector(
                np.tile(GRAPH_COLOR, (len(all_edges), 1)))
        return mesh, lineset

    # ------------------------------------------------------------------
    # Depth view (2D) overlay
    # ------------------------------------------------------------------
    def draw_on_depth_image(self, depth_color_bgr, mask_infos, graphs, cam_pos, cam_mat):
        """Draws per-object mask contours (distinct colors) and the green
        GNG graph (projected to pixel space) onto a BGR depth colormap
        image, in place, and returns it."""
        for _name, mask, color in mask_infos:
            bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(depth_color_bgr, contours, -1, bgr, 2)

        green_bgr = (0, 255, 0)
        for g in graphs:
            positions, edges = g["positions"], g["edges"]
            if not positions:
                continue
            ids = list(positions.keys())
            pts_world = np.array([positions[i] for i in ids])
            u, v, z = world_to_pixel(pts_world, cam_pos, cam_mat,
                                      self.fx, self.fy, self.cx, self.cy)
            px = {nid: (int(round(u[i])), int(round(v[i])))
                  for i, nid in enumerate(ids) if z[i] > 0}
            for e in edges:
                a, b = tuple(e)
                if a in px and b in px:
                    cv2.line(depth_color_bgr, px[a], px[b], green_bgr, 1, cv2.LINE_AA)
            for p in px.values():
                cv2.circle(depth_color_bgr, p, 3, green_bgr, -1, cv2.LINE_AA)
        return depth_color_bgr
