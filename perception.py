#!/usr/bin/env python3
"""
Per-instance perception for the pose-level evaluation: render the head
camera, run YOLO26 instance segmentation, unproject each instance's mask, fit
a fresh graph per instance, and match instances to the 5 benchmark objects.

Why not GngOverlay.update_graphs: it keys graphs by detector CLASS NAME, so
two instances sharing a label (e.g. a cracker box mislabelled "bottle" next
to the real mustard bottle) overwrite each other. Here every instance is fit
separately.

Instance -> benchmark-object association uses the simulator's ground-truth
body positions (nearest body to the graph centre, accepted within
MATCH_MAX_DIST); among several instances matching one object the
highest-confidence one wins. This is a simulation-side bookkeeping step, not
part of the perception-to-grasp pipeline: grasp parameters come only from the
matched graph's nodes.
"""
import time

import mujoco
import numpy as np

import sim_core as sc
from gng_overlay import (MIN_MASK_PIXELS, filter_out_class, make_gng,
                         unproject_mask_points)

MATCH_MAX_DIST = 0.12  # m, horizontal (xy) distance: mesh origins sit at different heights


class Perceiver:
    def __init__(self, sim, segmenter, conf_min=0.25, gng_seed=0):
        self.sim, self.segmenter, self.conf_min = sim, segmenter, conf_min
        self.gng_seed = gng_seed
        m = sim.model
        self.rr = mujoco.Renderer(m, height=sc.IMG_H, width=sc.IMG_W)
        self.rd = mujoco.Renderer(m, height=sc.IMG_H, width=sc.IMG_W)
        self.rd.enable_depth_rendering()
        self.cam_id = m.camera(sc.CAM_NAME).id
        self.intr = sc.get_intrinsics(m, sc.CAM_NAME, sc.IMG_W, sc.IMG_H)
        self.last_rgb = None
        self.last_depth = None
        self.last_timing = {}

    def render(self):
        d = self.sim.data
        t = time.perf_counter()
        self.rr.update_scene(d, camera=sc.CAM_NAME)
        rgb = self.rr.render().copy()
        self.rd.update_scene(d, camera=sc.CAM_NAME)
        depth = self.rd.render().copy()
        self.last_rgb = rgb
        self.last_depth = depth
        self.last_cam = (d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy())
        self.last_timing["render_ms"] = (time.perf_counter() - t) * 1e3
        return rgb, depth, d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def detect(self, depth_noise_std=0.0, rng=None):
        """Returns {object_name: detection dict} for every benchmark object
        matched this frame, plus the list of all instances (for logging)."""
        rgb, depth, cam_pos, cam_mat = self.render()
        if depth_noise_std > 0 and rng is not None:
            depth = depth + rng.normal(0.0, depth_noise_std, depth.shape)
            self.last_depth = depth   # the (noisy) depth that the pipeline actually uses
        t = time.perf_counter()
        res = self.segmenter.predict(rgb)
        self.last_timing["yolo_ms"] = (time.perf_counter() - t) * 1e3
        instances, gng_ms = [], 0.0
        if res is not None and res.masks is not None:
            res = filter_out_class(res, self.segmenter.model.names, "table")
        if res is not None and res.masks is not None and res.boxes is not None:
            masks = res.masks.data.cpu().numpy().astype(bool)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            fx, fy, cx, cy = self.intr
            for mask, cls_id, conf, box in zip(masks, classes, confs, boxes):
                if conf < self.conf_min or mask.sum() < MIN_MASK_PIXELS:
                    continue
                pts = unproject_mask_points(mask, depth, cam_pos, cam_mat, fx, fy, cx, cy,
                                            sc.DEPTH_MIN, sc.DEPTH_MAX)
                if len(pts) < 3:
                    continue
                gng = make_gng(seed=self.gng_seed)
                t = time.perf_counter()
                positions, edges = gng.fit(pts)
                dt = (time.perf_counter() - t) * 1e3
                gng_ms += dt
                if not positions:
                    continue
                centre = np.mean(list(positions.values()), axis=0)
                obj = min(sc.OBJECTS_BY_NAME,
                          key=lambda n: np.linalg.norm(self.sim.data.body(n).xpos[:2] - centre[:2]))
                dist = float(np.linalg.norm(self.sim.data.body(obj).xpos[:2] - centre[:2]))
                instances.append(dict(
                    det_class=self.segmenter.model.names[int(cls_id)], conf=float(conf),
                    mask_px=int(mask.sum()), n_points=len(pts), mask=mask, box=box, positions=positions,
                    edges=edges, obj=obj if dist < MATCH_MAX_DIST else None,
                    match_dist=dist, gng_ms=dt, gng_info=dict(getattr(gng, "last_info", {}))))
        self.last_timing["gng_ms"] = gng_ms
        best = {}
        for inst in instances:
            o = inst["obj"]
            if o is not None and (o not in best or inst["conf"] > best[o]["conf"]):
                best[o] = inst
        return best, instances
