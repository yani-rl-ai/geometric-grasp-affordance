#!/usr/bin/env python3
"""
Searches candidate robot placements on circles around the object cluster for
placements meeting the inclusion criteria of the pose-level evaluation (paper
Sec. V-B): the base touches neither table nor objects, and YOLO26s-seg segments
all five objects (each matched to its benchmark object by horizontal distance).
The robot always yaw-faces the object centroid. 36 candidates are tested
(radius 0.70 / 0.80 m x every 20 deg); P0..P3 in sim_core.PLACEMENTS were chosen
from the ones that qualify at --conf 0.05. At the default YOLO confidence of 0.25
only 3 of the 36 qualify (the cracker box is rarely segmented), which is why the
paper uses 0.05 and does not use class labels.

Output: scenario_search_conf<conf>.csv (all candidates, incl. rejected ones).
Usage:  python scenario_search.py [--conf 0.05]
"""
import argparse
import csv
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import sim_core as sc
from perception import Perceiver
from yolo_segmenter import YoloSegmenter

RADII = (0.70, 0.80)
ANGLES_DEG = list(range(0, 360, 20))  # angle of the robot position around the centroid (0 = +x side)


def robot_contacts(sim):
    """Contacts between any robot body and the table / benchmark objects (should be none)."""
    m, d = sim.model, sim.data
    root = m.body("robot0:base_link").id
    robot_bodies = set()
    for b in range(m.nbody):
        p = b
        while p != 0 and p != root:
            p = m.body_parentid[p]
        if p == root:
            robot_bodies.add(b)
    hits = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in robot_bodies) != (b2 in robot_bodies):
            other = b2 if b1 in robot_bodies else b1
            name = m.body(other).name
            if name == "table" or name in sc.OBJECTS_BY_NAME:
                mine = b1 if b1 in robot_bodies else b2
                hits.append((m.body(mine).name, name, float(c.dist)))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold (paper: 0.05)")
    args = ap.parse_args()
    seg = YoloSegmenter(args.conf)
    base = sc.Sim(scenario="P0")
    C = np.array([base.model.body(n).pos[:2] for n in sc.OBJECTS_BY_NAME]).mean(0)
    rows = []
    for r in RADII:
        for a in ANGLES_DEG:
            phi = np.radians(a)
            xy = C + r * np.array([np.cos(phi), np.sin(phi)])
            yaw = phi + np.pi            # facing the centroid
            sim = sc.Sim(scenario=dict(xy=(float(xy[0]), float(xy[1])), yaw=float(yaw)))
            sim.step(300)
            contacts = robot_contacts(sim)
            per = Perceiver(sim, seg, conf_min=args.conf)
            best, _ = per.detect()
            confs = {o: round(best[o]["conf"], 2) if o in best else 0.0 for o in sc.OBJECTS_BY_NAME}
            row = dict(radius=r, angle_deg=a, x=round(xy[0], 3), y=round(xy[1], 3), yaw_deg=round(np.degrees(yaw) % 360, 1),
                       contacts=len(contacts), n_detected=len(best), qualifies=(len(best) == 5 and not contacts),
                       **{f"conf_{o}": c for o, c in confs.items()})
            rows.append(row)
            print(row, flush=True)
            del per
    out = f"scenario_search_conf{args.conf:g}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['qualifies'] for r in rows)} of {len(rows)} candidates qualify -> {out}")


if __name__ == "__main__":
    main()
