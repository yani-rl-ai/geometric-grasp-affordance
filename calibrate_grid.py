#!/usr/bin/env python3
"""
P0-only calibration of the execution parameters the review criticised as
hand-tuned: grip-site depth below the perceived top (GRASP_DEPTH_M), finger
servo stiffness (GRASP_FINGER_KP -> grip force) and the global GNG width bias,
with NO per-object squeeze overrides (nominal margin for every object). Same
jitter/noise seeds for every setting (common random numbers); P1-P3 unused.

Usage: MUJOCO_GL=egl python calibrate_grid.py --bias 0.015 [--xy truth]
"""
import argparse
import csv
import os
import tempfile
import types

import grasp_planner
import sim_core as sc
from perception import Perceiver
import run_pose_eval as rp
from calibrate_p0 import Null

DEPTHS = [0.005, 0.015, 0.025, 0.035]
KPS = [30000, 60000, 120000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--xy", default="perception", choices=["perception", "truth"])
    ap.add_argument("--depths", nargs="+", type=float, default=DEPTHS)
    ap.add_argument("--kps", nargs="+", type=float, default=KPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.environ["GRASP_XY_SOURCE"] = args.xy
    grasp_planner.GNG_WIDTH_BIAS = args.bias
    sc.PERCEPTION_SQUEEZE_OVERRIDE.clear()
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "frames"))
    ns = types.SimpleNamespace(out=tmp, trials=args.trials)
    seg = rp.YoloSegmenter(rp.YOLO_CONF)
    out = f"calib_grid_b{args.bias:.3f}_{args.xy}{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bias", "xy", "kp", "depth", "object", "trial", "success", "reason", "width_m", "xy_err_mm"])
        for kp in args.kps:
            os.environ["GRASP_FINGER_KP"] = str(kp)
            sim = sc.Sim(scenario=rp.POSES["P0"])
            per = Perceiver(sim, seg, conf_min=rp.YOLO_CONF)
            for depth in args.depths:
                os.environ["GRASP_DEPTH_M"] = str(depth)
                tot = {}
                for oi, obj in enumerate(sc.OBJECTS_BY_NAME):
                    ok = 0
                    for t in range(1, args.trials + 1):
                        try:
                            r = rp.run_trial(sim, per, Null(), "P0", obj, t, 700000 + oi * 100 + t, ns, None)
                        except Exception as e:
                            r = dict(success=False, reason=f"exception:{e}")
                        ok += bool(r["success"])
                        w.writerow([args.bias, args.xy, kp, depth, obj, t, int(bool(r["success"])), r.get("reason", ""),
                                    r.get("width_m", ""), r.get("xy_err_mm", "")])
                        f.flush()
                    tot[obj] = ok
                print(f"b={args.bias} xy={args.xy} kp={kp} depth={depth}: {tot} total={sum(tot.values())}/{5 * args.trials}", flush=True)


if __name__ == "__main__":
    main()
