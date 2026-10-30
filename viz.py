#!/usr/bin/env python3
"""
Visual overlays for the pose evaluation (viewer window + MP4):
  - Overlay: DBL-GNG graphs (nodes = spheres, edges = capsules) and the light-
    blue grasp-region boxes as 3D decor geoms, applied to any mjvScene (the
    passive viewer's user_scn and the video renderer's scene).
  - make_inset: the head-camera frame with the YOLO26 instance masks / boxes /
    labels and the same graphs + regions projected into the image.
"""
import cv2
import mujoco
import numpy as np

from gng_overlay import world_to_pixel

OBJ_COLORS = {            # RGB 0-255, one per benchmark object
    "cracker_box": (230, 60, 60), "mustard_bottle": (240, 200, 40), "mug": (200, 80, 200),
    "banana": (80, 220, 120), "tennis_ball": (60, 200, 240),
}
UNMATCHED = (170, 170, 170)
REGION_RGB = (140, 220, 255)   # light blue, as in the paper's figures
_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


class Overlay:
    """items: list of dict(obj, positions {id: xyz}, edges {frozenset: w}, region (lo, hi) | None, target bool)."""

    def __init__(self):
        self.items = []
        self.viewer = None

    def set(self, items):
        self.items = items
        self._push()

    def clear(self):
        self.items = []
        self._push()

    def _push(self):
        v = self.viewer
        if v is not None and v.is_running():
            with v.lock():
                v.user_scn.ngeom = 0
                self.apply(v.user_scn)

    @staticmethod
    def _next(scn):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        return g

    def _line(self, scn, p0, p1, width, rgba):
        g = self._next(scn)
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0, float), np.asarray(p1, float))

    def apply(self, scn):
        for it in self.items:
            rgb = np.array(OBJ_COLORS.get(it["obj"], UNMATCHED)) / 255.0
            node_rgba = list(rgb) + [1.0]
            ids = list(it["positions"])
            for nid in ids:
                g = self._next(scn)
                if g is None:
                    return
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0045, 0, 0]), it["positions"][nid],
                                    np.eye(3).flatten(), np.asarray(node_rgba, dtype=np.float32))
            for e in it["edges"]:
                a, b = tuple(e)
                if a in it["positions"] and b in it["positions"]:
                    self._line(scn, it["positions"][a], it["positions"][b], 0.0016, node_rgba)
            if it.get("region") is not None and it.get("target"):
                c = _corners(*it["region"])
                for a, b in _BOX_EDGES:
                    self._line(scn, c[a], c[b], 0.0022, list(np.array(REGION_RGB) / 255.0) + [1.0])


def _draw_graphs(img, items, cam_pos, cam_mat, intr, node_r=3):
    """Projects every item's DBL-GNG graph (nodes in the object colour, white edges) and, for the
    target item, the light-blue grasp-region box into the image (in place)."""
    fx, fy, cx, cy = intr
    for it in items:
        col = OBJ_COLORS.get(it["obj"], UNMATCHED)
        ids = list(it["positions"])
        P = np.array([it["positions"][i] for i in ids])
        u, v, z = world_to_pixel(P, cam_pos, cam_mat, fx, fy, cx, cy)
        pix = {nid: (int(round(u[k])), int(round(v[k]))) for k, nid in enumerate(ids) if z[k] > 0}
        for e in it["edges"]:
            a, b = tuple(e)
            if a in pix and b in pix:
                cv2.line(img, pix[a], pix[b], (255, 255, 255), 1, cv2.LINE_AA)
        for p in pix.values():
            cv2.circle(img, p, node_r, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, node_r, (0, 0, 0), 1, cv2.LINE_AA)
        if it.get("region") is not None and it.get("target"):
            c = _corners(*it["region"])
            uu, vv, zz = world_to_pixel(c, cam_pos, cam_mat, fx, fy, cx, cy)
            pts = [(int(round(uu[k])), int(round(vv[k]))) if zz[k] > 0 else None for k in range(8)]
            for a, b in _BOX_EDGES:
                if pts[a] is not None and pts[b] is not None:
                    cv2.line(img, pts[a], pts[b], REGION_RGB, 2, cv2.LINE_AA)


def make_inset(rgb, instances, items, cam_pos, cam_mat, intr, size=(480, 360)):
    """rgb: HxWx3 uint8 (RGB). Returns the annotated frame resized to size."""
    img = rgb.copy()
    h, w = img.shape[:2]
    for inst in instances:
        col = OBJ_COLORS.get(inst["obj"], UNMATCHED) if inst["obj"] else UNMATCHED
        m = inst.get("mask")
        if m is not None and m.shape == img.shape[:2]:
            img[m] = (0.55 * img[m] + 0.45 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2, cv2.LINE_AA)
        b = inst.get("box")
        if b is not None:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
            lab = f"{inst['det_class']} {inst['conf']:.2f}"
            cv2.putText(img, lab, (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lab, (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    _draw_graphs(img, items, cam_pos, cam_mat, intr)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def depth_colormap(depth, depth_min=0.2, depth_max=2.5):
    """Metric depth (m) -> RGB colormap, closer = warmer (same convention as the live demo windows)."""
    d = np.clip(depth, depth_min, depth_max)
    d = ((d - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(255 - d, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def make_depth_view(depth, instances, items, cam_pos, cam_mat, intr, size=(640, 480), depth_range=(0.2, 2.5)):
    """Depth colormap with the YOLO mask contours, the DBL-GNG graphs and the target's grasp region."""
    img = depth_colormap(depth, *depth_range)
    for inst in instances:
        col = OBJ_COLORS.get(inst["obj"], UNMATCHED) if inst["obj"] else UNMATCHED
        m = inst.get("mask")
        if m is not None and m.shape == img.shape[:2]:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2, cv2.LINE_AA)
    _draw_graphs(img, items, cam_pos, cam_mat, intr)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if size != img.shape[1::-1] else img
