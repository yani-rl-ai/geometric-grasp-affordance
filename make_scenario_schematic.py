#!/usr/bin/env python3
"""Top-down schematic of the four robot placements P0..P3 around the table (docs/img/placements.png).
Needs matplotlib (requirements-optional.txt).   python make_scenario_schematic.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle

import sim_core as sc

LABELS = {"P0": "P0 front\n(calibration)", "P1": "P1 back", "P2": "P2 side (+y)", "P3": "P3 front-diagonal"}
COLORS = {"cracker_box": "#c0392b", "mustard_bottle": "#f1c40f", "mug": "#8e1b1b", "banana": "#f4d03f", "tennis_ball": "#c8e84a"}
SIZES = {"cracker_box": (0.072, 0.164), "mustard_bottle": (0.067, 0.067), "mug": (0.093, 0.093),
         "banana": (0.04, 0.178), "tennis_ball": (0.067, 0.067)}   # footprints (m)


def main():
    m = mujoco.MjModel.from_xml_path(sc.MODEL_PATH)
    tab = m.body("table")
    table_c = tab.pos[:2]
    half = m.geom("table_top").size[:2]
    obj_xy = {n: m.body(n).pos[:2] for n in sc.OBJECTS_BY_NAME}
    centroid = np.mean(list(obj_xy.values()), axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.add_patch(Rectangle(table_c - half, 2 * half[0], 2 * half[1], fc="#e8b27a", ec="#8a5a2b", lw=2, zorder=1))
    for n, (x, y) in obj_xy.items():
        w, h = SIZES[n]
        if n in ("tennis_ball", "mustard_bottle", "mug"):
            ax.add_patch(Circle((x, y), w / 2, fc=COLORS[n], ec="k", zorder=3))
        else:
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=COLORS[n], ec="k", zorder=3))
        ax.annotate(n, (x, y), xytext=(x + 0.045, y + 0.03), fontsize=7, zorder=5)
    ax.plot(*centroid, "k+", ms=12, zorder=5)
    for name, spec in sc.PLACEMENTS.items():
        (x, y), yaw = spec["xy"], spec["yaw"]
        f = np.array([np.cos(yaw), np.sin(yaw)])
        ax.add_patch(Circle((x, y), 0.28, fc="#aed6f1", ec="#1b4f72", lw=1.5, alpha=0.85, zorder=4))
        ax.add_patch(FancyArrow(x, y, 0.30 * f[0], 0.30 * f[1], width=0.04, fc="#1b4f72", ec="none", zorder=6))
        ax.plot([x, centroid[0]], [y, centroid[1]], ls="--", c="gray", lw=0.6, zorder=2)
        ax.text(x - 0.36 * f[0], y - 0.36 * f[1], LABELS[name], ha="center", va="center", fontsize=8, weight="bold", zorder=7)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 2.2)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Robot placements P0-P3 (blue disc = Fetch base, arrow = facing direction)\n"
                 "table and objects are fixed; dashed line = direction to the object centroid", fontsize=10)
    ax.grid(alpha=0.2)
    os.makedirs("docs/img", exist_ok=True)
    plt.tight_layout()
    plt.savefig("docs/img/placements.png", dpi=110)
    print("written docs/img/placements.png")


if __name__ == "__main__":
    main()
