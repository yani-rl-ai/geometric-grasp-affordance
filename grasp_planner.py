#!/usr/bin/env python3
"""
Geometric grasp-candidate planner, driven entirely by DBL-GNG node
positions -- no PCA, no RANSAC.

Rationale: DBL-GNG nodes already sit on the object's surface (they're the
fitted topology of its own 3D points), so there's no need to fit a plane or
recover principal axes to reason about size -- the axis-aligned min/max
extent of the node positions is already a cheap, direct estimate of how
wide the object is along each world axis. Since every object in this scene
rests upright in a fixed, known convention (not arbitrarily rotated), the
world axes ARE the object's natural axes, so an axis-aligned extent is a
legitimate substitute for PCA's principal axes here (it would NOT be for an
arbitrarily-oriented object -- that's the real tradeoff of skipping PCA).

Decision rule per object, given the gripper's known max opening:
  - TOP-DOWN approach: gripper closes along whichever of (extent_x, extent_y)
    is narrower. Feasible if that width <= gripper_max_width.
  - SIDE approach: gripper closes along extent_z (vertical height), moving
    in horizontally. Feasible if extent_z <= gripper_max_width.
A tall, narrow-footprint box (e.g. a cracker box) only has a feasible TOP
approach (its height exceeds the gripper, its footprint doesn't) -- matching
the observation that motivated this: grabbing it "from the front" needs the
gripper to span its full height, which it can't, while the top only needs
it to span the box's narrow footprint. A short, wide object (e.g. a mug
whose footprint barely exceeds the gripper) can instead be grabbed from the
SIDE, spanning its height instead. Where both are feasible, TOP is
preferred (this gripper's fingers are short -- established separately while
tuning the scripted pick sequence -- so a top approach has an easier time
staying clear of the table).
"""
import os

import numpy as np
import cv2

from gng_overlay import world_to_pixel

GRIPPER_MAX_WIDTH = 0.098  # meters -- true actuator limit is 0.05m/finger *
                            # 2 = 0.10m (scene.xml ctrlrange), minus a 2mm
                            # safety margin (same philosophy as the squeeze
                            # margins elsewhere in this project). Was 0.09
                            # before the bias correction below existed; the
                            # mug's own tuned true width (0.092m, see
                            # sim_core.OBJECTS) already exceeds the old
                            # value, so 0.09 was already slightly too
                            # conservative for this gripper, not just for
                            # the corrected estimate.
FINGER_REACH = 0.077      # meters -- matches the parallel-jaw fingers in scene.xml
GRASP_COLOR = (0.55, 0.85, 1.0)  # light blue, as requested

# DBL-GNG systematically UNDERESTIMATES an object's true cross-section
# width: its ~30 nodes settle a little inward of the true surface (the
# fitted topology tracks where points *cluster*, and points thin out right
# at the silhouette edge). Measured empirically by comparing this planner's
# raw candidate width against the independently hand-tuned true half-widths
# in sim_core.OBJECTS, across 4 differently-sized, differently-shaped
# objects (box, bottle, mug, ball):
#     object          true_w  raw_w   true_w - raw_w
#     cracker_box     0.0720  0.0568      0.0152
#     mustard_bottle  0.0580  0.0439      0.0141
#     mug             0.0920  0.0799      0.0121
#     tennis_ball     0.0640  0.0453      0.0187
# The *ratio* varies a lot across these (0.71-0.87 of true width) -- object
# size doesn't predict it -- but the *absolute* shortfall is almost
# constant (0.012-0.019m, averaging ~0.015m). That points to a fixed-size
# cause (roughly one node's worth of inward pull, largely independent of
# the object it's fitted to) rather than a proportional one, so the
# correction here is additive, not multiplicative.
# Re-calibrated on the P0 (front) pose only, with jitter + depth noise, for the
# DBL-GNG implementation of dbl_gng_paper.py (plateau 8-12 mm: 28-29/30 grasps
# at the frozen depth). The 15 mm above was fit to the earlier GNG variant.
GNG_WIDTH_BIAS = float(os.environ.get("GNG_WIDTH_BIAS", 0.010))  # meters (env override for ablations)

_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
              (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _local_cross_section_width(pts, close_axis, long_axis, min_nodes=4, curvature_ratio=0.85):
    """Width along close_axis, measured only from nodes near the object's
    own center along long_axis, instead of the full bounding-box extent.

    A straight prism/cylinder's cross-section doesn't change along its
    length, so this returns effectively the same number as the full
    extent there. But a curved or tapered object (e.g. a banana) sweeps
    across close_axis as you move along long_axis -- the FULL extent then
    measures the curve's overall spread, not the true local diameter at
    the point the gripper will actually close around (near the object's
    own center, since that's where grasp_object_via_candidate aims the
    approach). Purely geometric (node positions only), no PCA/RANSAC.

    Returns None (caller falls back to the full extent) when there aren't
    enough nodes in the local slice to trust it, or when the local
    estimate isn't meaningfully narrower than the full extent (avoids
    trading real precision for sampling noise on objects that aren't
    actually curved)."""
    long_vals = pts[:, long_axis]
    close_vals = pts[:, close_axis]
    full_extent = close_vals.max() - close_vals.min()
    long_extent = long_vals.max() - long_vals.min()
    center = np.median(long_vals)
    # Band sized off long_extent (how far the shape runs), not full_extent
    # (how wide it looks along close_axis) -- the latter is exactly the
    # curvature-inflated number this function exists to correct, so sizing
    # the neighborhood off of it would be circular (a curvier object would
    # get an even wider band, defeating the localization).
    band = max(long_extent * 0.2, 0.02)
    sel = np.abs(long_vals - center) <= band
    if sel.sum() < min_nodes:
        return None
    local_extent = close_vals[sel].max() - close_vals[sel].min()
    if local_extent >= curvature_ratio * full_extent:
        return None
    return local_extent


def _local_close_center(pts, close_axis, long_axis, min_nodes=4):
    """Mid-point along close_axis of the nodes in the same band around the
    median along long_axis that _local_cross_section_width uses. For a curved
    object (banana) this lands on the object's middle instead of on the
    bounding-box centre, which a sweep of the curve shifts sideways. Returns
    None when the band holds fewer than min_nodes nodes."""
    long_vals, close_vals = pts[:, long_axis], pts[:, close_axis]
    band = max((long_vals.max() - long_vals.min()) * 0.2, 0.02)
    sel = np.abs(long_vals - np.median(long_vals)) <= band
    if sel.sum() < min_nodes:
        return None
    return 0.5 * (close_vals[sel].min() + close_vals[sel].max())


def compute_grasp_candidate(positions, gripper_max_width=GRIPPER_MAX_WIDTH,
                             finger_reach=FINGER_REACH, prefer_approach=None):
    """positions: dict {node_id: (3,) world point}, from a DBL-GNG fit.
    Returns a dict describing the chosen approach, or None if there aren't
    enough nodes to reason about.

    prefer_approach: None (default) keeps the normal "top" preference
    (Section: module docstring). Pass "side" to prefer a side candidate
    when one is feasible, without changing feasibility itself -- for
    testing/demonstrating the side-approach execution path on an object
    that would otherwise always pick "top" (every object in this scene
    currently does, since none of their top-down widths exceed the
    gripper's capacity)."""
    if not positions or len(positions) < 2:
        return None
    pts = np.array(list(positions.values()))
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    extent = maxs - mins

    candidates = []
    top_close_axis = 0 if extent[0] <= extent[1] else 1
    top_long_axis = 1 - top_close_axis
    top_width = _local_cross_section_width(pts, top_close_axis, top_long_axis)
    if top_width is None:
        top_width = extent[top_close_axis]
    top_width += GNG_WIDTH_BIAS
    if top_width <= gripper_max_width:
        candidates.append(dict(approach="top", close_axis=top_close_axis, width=top_width))
    side_width = extent[2] + GNG_WIDTH_BIAS
    if side_width <= gripper_max_width:
        approach_axis = 0 if extent[0] >= extent[1] else 1
        candidates.append(dict(approach="side", close_axis=2,
                                approach_axis=approach_axis, width=side_width))

    result = dict(mins=mins, maxs=maxs, extent=extent, gripper_max_width=gripper_max_width)
    if not candidates:
        result["feasible"] = False
        return result

    # Prefer "top" (short-fingered gripper reaches better from above, see
    # module docstring) unless prefer_approach overrides it; among
    # equal-preference options prefer the narrowest fit for a more secure
    # grip.
    default_pref = prefer_approach or "top"
    candidates.sort(key=lambda c: (0 if c["approach"] == default_pref else 1, c["width"]))
    best = candidates[0]
    result.update(best)
    result["feasible"] = True

    if best["approach"] == "top":
        z_lo = maxs[2] - min(finger_reach, extent[2])
        region_min = np.array([mins[0], mins[1], z_lo])
        region_max = maxs.copy()
    else:
        region_min, region_max = mins.copy(), maxs.copy()
    result["region_min"] = region_min
    result["region_max"] = region_max
    result["center"] = (region_min + region_max) / 2
    # GRASP_CENTER=local (post-hoc variant, default off): centre the top-down
    # grasp on the local band's mid-point along the closing axis.
    if best["approach"] == "top" and os.environ.get("GRASP_CENTER", "bbox") == "local":
        lc = _local_close_center(pts, best["close_axis"], 1 - best["close_axis"])
        if lc is not None:
            result["center"] = result["center"].copy()
            result["center"][best["close_axis"]] = lc
    return result


def describe(name, candidate):
    if candidate is None:
        return f"{name}: no GNG nodes yet"
    if not candidate["feasible"]:
        e = candidate["extent"]
        return (f"{name}: NOT graspable by this gripper -- extent=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f})m, "
                f"gripper max={candidate['gripper_max_width']:.3f}m")
    return (f"{name}: {candidate['approach'].upper()} grasp, closing width="
            f"{candidate['width']:.3f}m <= gripper max {candidate['gripper_max_width']:.3f}m")


# ------------------------------------------------------------------
# Visualization (light blue grasp-area highlight)
# ------------------------------------------------------------------
def build_region_wireframes(candidates):
    """Combined light-blue wireframe box (Open3D LineSet) for every
    feasible candidate, or None if there's nothing to show."""
    feasible = [c for c in candidates if c is not None and c.get("feasible")]
    if not feasible:
        return None
    all_points, all_lines = [], []
    offset = 0
    for c in feasible:
        corners = _box_corners(c["region_min"], c["region_max"])
        all_points.extend(corners)
        all_lines.extend((a + offset, b + offset) for a, b in _BOX_EDGES)
        offset += 8
    import open3d as o3d  # optional dependency, only for the Open3D point-cloud view
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(all_points))
    ls.lines = o3d.utility.Vector2iVector(np.array(all_lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile(GRASP_COLOR, (len(all_lines), 1)))
    return ls


def tint_points_in_regions(colors_flat, pts_world_flat, candidates, blend=0.7):
    feasible = [c for c in candidates if c is not None and c.get("feasible")]
    if not feasible:
        return colors_flat
    out = colors_flat.copy()
    for c in feasible:
        inside = np.all((pts_world_flat >= c["region_min"]) & (pts_world_flat <= c["region_max"]), axis=1)
        if inside.any():
            out[inside] = (1 - blend) * out[inside] + blend * np.array(GRASP_COLOR)
    return out


def draw_regions_on_depth(depth_bgr, candidates, cam_pos, cam_mat, fx, fy, cx, cy):
    feasible = [c for c in candidates if c is not None and c.get("feasible")]
    if not feasible:
        return depth_bgr
    light_blue_bgr = (255, 220, 140)  # BGR
    for c in feasible:
        corners = _box_corners(c["region_min"], c["region_max"])
        u, v, z = world_to_pixel(corners, cam_pos, cam_mat, fx, fy, cx, cy)
        pts2d = [(int(round(u[i])), int(round(v[i]))) if z[i] > 0 else None for i in range(8)]
        for a, b in _BOX_EDGES:
            if pts2d[a] is not None and pts2d[b] is not None:
                cv2.line(depth_bgr, pts2d[a], pts2d[b], light_blue_bgr, 2, cv2.LINE_AA)
    return depth_bgr
