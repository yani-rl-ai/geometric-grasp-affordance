#!/usr/bin/env python3
"""
Perception-level numbers quoted in the paper (Sec. VI-A, Table V): detection rate, region-centre error and the
closing-width estimates of the cracker box and the tennis ball against the true widths.

The reference extents come from the simulator geometry, not from hand-tuned constants: all vertices of the
visible geoms of each object (mesh vertices; box/capsule corners) are transformed to the world frame and the
axis-aligned extent is taken (written to results/true_extents.json).

    python analyze_perception_metrics.py                                   # paper runs, both seeds pooled
    python analyze_perception_metrics.py --runs pose_eval_runs/main        # your own re-run
"""
import argparse
import csv
import itertools
import json
import os

import mujoco
import numpy as np

import sim_core as sc

OBJS = list(sc.OBJECTS_BY_NAME)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def true_extents():
    m = mujoco.MjModel.from_xml_path(sc.MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = {}
    for name in OBJS:
        bid = m.body(name).id
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_rgba[g][3] <= 0:   # skip invisible collision proxies
                continue
            R, P, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                pts.append(P + v @ R.T)
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, hl = m.geom_size[g][:2]
                pts += [(P + R @ (np.array([s[0] * r, s[1] * r, s[2] * (hl + r)])))[None]
                        for s in itertools.product((-1, 1), repeat=3)]
            else:
                pts += [(P + R @ (np.array(s) * m.geom_size[g]))[None] for s in itertools.product((-1, 1), repeat=3)]
        A = np.vstack(pts)
        out[name] = A.max(0) - A.min(0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["results/paper_runs/main", "results/paper_runs/confirm_main_seed2"])
    args = ap.parse_args()
    true = true_extents()
    os.makedirs("results", exist_ok=True)
    json.dump({k: v.tolist() for k, v in true.items()}, open("results/true_extents.json", "w"), indent=1)
    print("True visible-geometry extents x,y,z [mm]:",
          {k: np.round(v * 1e3).astype(int).tolist() for k, v in true.items()})
    rows = []
    for r in args.runs:
        rows += list(csv.DictReader(open(os.path.join(r, "trials.csv"))))
    print(f"\n{len(rows)} trials from {args.runs}\n")
    det = [r for r in rows if r["detected"] == "True"]
    lo, hi = wilson(len(det), len(rows))
    print(f"Detection of the target: {len(det)}/{len(rows)} = {100 * len(det) / len(rows):.1f}%  (Wilson 95% CI {lo:.0f}-{hi:.0f}%)")
    print("per object:", {o: f"{sum(r['detected'] == 'True' for r in rows if r['object'] == o)}/{sum(r['object'] == o for r in rows)}" for o in OBJS})
    print("per placement:", {p: f"{sum(r['detected'] == 'True' for r in rows if r['scenario'] == p)}/{sum(r['scenario'] == p for r in rows)}"
                             for p in sc.PLACEMENT_ORDER})
    print("\nRegion-centre error vs simulator object position [mm] (detected trials)      mean   p95    n")
    for o in OBJS:
        e = np.array([float(r["xy_err_mm"]) for r in det if r["object"] == o])
        print(f"  {o:15s}{e.mean():34.1f}{np.percentile(e, 95):7.1f}{len(e):5d}")
    e = np.array([float(r["xy_err_mm"]) for r in det])
    print(f"  {'overall':15s}{e.mean():34.1f}{np.percentile(e, 95):7.1f}{len(e):5d}   (median {np.median(e):.1f})")
    print("\nClosing width used by the planner (includes the +b bias) vs true extent along the chosen closing axis [mm]")
    print("  (the reference is only valid where the planner uses the full extent -- cracker box, tennis ball; for mug,"
          " bottle and banana the local-slice rule may select a narrower slice, so their errors are indicative only)")
    for o in OBJS:
        est, ref = [], []
        for r in det:
            if r["object"] != o:
                continue
            ax = 0 if float(r["ext_x_m"]) <= float(r["ext_y_m"]) else 1
            est.append(float(r["width_m"]) * 1e3)
            ref.append(true[o][ax] * 1e3)
        est, ref = np.array(est), np.array(ref)
        print(f"  {o:15s} estimate {est.mean():5.1f} +- {est.std():4.1f}   true {ref.mean():5.1f}   error {np.mean(est - ref):+6.1f} +- {np.std(est - ref):4.1f}")


if __name__ == "__main__":
    main()
