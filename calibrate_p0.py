#!/usr/bin/env python3
"""
Calibration sweep on the P0 (front) pose ONLY: for one global width bias b,
every object is tried at each squeeze margin m with the same jitter/noise seeds
(common random numbers, seeds disjoint from the final evaluation's), headless.
Output rows go to calib_p0_b<bias>.csv; calibrate_pick.py chooses (b, m_object)
from them. P1-P3 are never used for calibration.

Usage: MUJOCO_GL=egl python calibrate_p0.py --bias 0.015 --trials 6
"""
import argparse
import csv
import os
import tempfile
import types

import numpy as np

import grasp_planner
import sim_core as sc
from perception import Perceiver
import run_pose_eval as rp

MARGINS = [-0.010, -0.006, -0.002, 0.002, 0.006, 0.010, 0.015, 0.020, 0.026]


class Null:
    lines, banner = ["", ""], None
    def hold(self, s): pass
    def tick(self): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--objects", nargs="+", default=list(sc.OBJECTS_BY_NAME))
    ap.add_argument("--margins", nargs="+", type=float, default=MARGINS)
    args = ap.parse_args()
    grasp_planner.GNG_WIDTH_BIAS = args.bias
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "frames"))
    ns = types.SimpleNamespace(out=tmp, trials=args.trials)
    seg = rp.YoloSegmenter(rp.YOLO_CONF)
    sim = sc.Sim(scenario=rp.POSES["P0"])
    per = Perceiver(sim, seg, conf_min=rp.YOLO_CONF)
    rec = Null()
    out = f"calib_p0_b{args.bias:.3f}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bias", "object", "margin", "trial", "success", "reason", "width_m", "xy_err_mm", "z_hold_min", "det_conf"])
        for oi, obj in enumerate(args.objects):
            for m in args.margins:
                sc.PERCEPTION_SQUEEZE_OVERRIDE.clear()
                sc.PERCEPTION_SQUEEZE_OVERRIDE[obj] = m
                ok = 0
                for t in range(1, args.trials + 1):
                    seed = 700000 + oi * 100 + t
                    try:
                        r = rp.run_trial(sim, per, rec, "P0", obj, t, seed, ns, None)
                    except Exception as e:
                        r = dict(success=False, reason=f"exception:{e}")
                    ok += bool(r["success"])
                    w.writerow([args.bias, obj, m, t, int(bool(r["success"])), r.get("reason", ""), r.get("width_m", ""),
                                r.get("xy_err_mm", ""), r.get("z_hold_min", ""), r.get("det_conf", "")])
                    f.flush()
                print(f"b={args.bias} {obj:15s} m={m:+.3f}: {ok}/{args.trials}", flush=True)


if __name__ == "__main__":
    main()
