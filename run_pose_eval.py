#!/usr/bin/env python3
"""
Pose-level grasp evaluation: the Fetch robot faces the (fixed) table/objects
from 4 placements (P0..P3, see POSES), and for each placement grasps every
object TRIALS times. Every trial starts from a fresh simulation state
(Sim.reset: objects at spawn poses, arm at ready pose, head re-aimed) with
  - the robot base jittered by U(+-JITTER_XY) m and U(+-JITTER_YAW_DEG) deg,
  - Gaussian depth noise (DEPTH_NOISE_STD m) on the rendered depth,
  - a per-trial seed (base seed, scenario, object, trial) that also seeds the
    DBL-GNG start points.
All seeds / jitters / noise are logged, so each trial is reproducible.

Perception -> grasp is the full pipeline (YOLO26 mask -> DBL-GNG -> planner ->
executor); the grasp xy comes from the perceived candidate unless
GRASP_XY_SOURCE=truth (oracle ablation). No head-pan search: if the target is
not detected, the trial is a detection failure.

Success: the object's body origin is >= LIFT_M above where it rested before
the grasp, and stays so for HOLD_S seconds while the gripper holds at transit
height. (The instantaneous check the earlier benchmarks used is logged too.)

Runs with the MuJoCo viewer window open unless --headless is given. One MP4
(1280x720, 30 fps of simulated time, external camera + head-camera inset) is
written per scenario, plus trials.csv / trials.jsonl / config.json.

Usage:
    python run_pose_eval.py --scenarios P0 P1 P2 P3 --trials 5 --tag main
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import cv2
import mujoco
import numpy as np

import grasp_planner
import sim_core as sc
from perception import Perceiver
from yolo_segmenter import YoloSegmenter
from viz import Overlay, make_inset

POSES = sc.PLACEMENTS   # P0 front (calibration), P1 back, P2 +y side, P3 front-diagonal; see sim_core.py
YOLO_CONF = 0.05
JITTER_XY = 0.015
JITTER_YAW_DEG = 2.0
DEPTH_NOISE_STD = 0.0015
LIFT_M = 0.05
HOLD_S = 2.0
FPS = 30
VID_W, VID_H = 1280, 720

CSV_FIELDS = [
    "scenario", "object", "trial", "seed", "xy_source", "gng_impl", "jit_dx_m", "jit_dy_m", "jit_dyaw_deg",
    "head_pan", "head_tilt", "detected", "det_class", "det_conf", "mask_px", "n_points", "n_instances",
    "gng_nodes", "gng_epochs", "ext_x_m", "ext_y_m", "ext_z_m", "approach", "width_m",
    "center_x", "center_y", "truth_x", "truth_y", "xy_err_mm",
    "z0", "z_after_lift", "z_hold_min", "lift_ok_instant", "lift_ok_hold", "finger_contact",
    "success", "reason", "render_ms", "yolo_ms", "gng_ms", "plan_ms", "wall_s",
]


def video_camera(sim):
    """Free camera looking at the table perpendicular to the robot->objects line,
    so robot, table and objects are all visible in every scenario."""
    m = sim.model
    objs = np.array([m.body(n).pos[:2] for n in sc.OBJECTS_BY_NAME])
    c = objs.mean(0)
    base = m.body("robot0:base_link").pos[:2]
    d = c - base
    az = np.degrees(np.arctan2(d[1], d[0])) + 90.0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    a = np.radians(az)
    right = np.array([np.sin(a), -np.cos(a)])
    look = c + 0.25 * (base - c) + 0.32 * right      # table near the left/centre, clear of the top-right inset
    cam.lookat[:] = [look[0], look[1], 0.78]
    cam.distance = 1.65
    cam.azimuth = az
    cam.elevation = -33.0
    return cam


class Recorder:
    """Writes composite frames (external camera + head-camera inset + text)
    to an H.264 MP4 through ffmpeg, one frame per 1/FPS s of simulated time."""

    def __init__(self, sim, path, perceiver):
        self.sim, self.per, self.path = sim, perceiver, path
        self.r = None
        self.proc = None
        if path:   # path=None: nothing is composed/written (run_live_demo.py without --record)
            self.r = mujoco.Renderer(sim.model, height=VID_H, width=VID_W)
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{VID_W}x{VID_H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)
        self.dt = 1.0 / FPS
        self.t = 0.0
        self.next_t = 0.0
        self.lines = ["", ""]
        self.banner = None
        self.n_frames = 0
        self.overlay = Overlay()
        self.inset = None
        self.cam = video_camera(sim)

    def tick(self):
        self.t += self.sim.model.opt.timestep
        if self.t >= self.next_t:
            self.next_t += self.dt
            self.frame()

    def frame(self):
        self.r.update_scene(self.sim.data, camera=self.cam)
        self.overlay.apply(self.r.scene)
        img = self.r.render().copy()
        head = self.inset if self.inset is not None else (
            cv2.resize(self.per.last_rgb, (480, 360)) if self.per.last_rgb is not None else None)
        if head is not None:
            img[10:370, VID_W - 490:VID_W - 10] = head
            cv2.rectangle(img, (VID_W - 491, 9), (VID_W - 9, 371), (255, 255, 255), 2)
            cv2.putText(img, "head camera: YOLO26-seg + DBL-GNG", (VID_W - 486, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "head camera: YOLO26-seg + DBL-GNG", (VID_W - 486, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        for i, txt in enumerate(self.lines):
            cv2.putText(img, txt, (14, 32 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (14, 32 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        if self.banner:
            txt, ok = self.banner
            col = (60, 200, 60) if ok else (230, 60, 60)
            cv2.putText(img, txt, (14, VID_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, txt, (14, VID_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
        if self.proc is not None:
            self.proc.stdin.write(img.tobytes())
        self.n_frames += 1

    def hold(self, seconds):
        for _ in range(int(seconds * FPS)):
            self.frame()

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


def finger_contact(sim, body_name):
    m, d = sim.model, sim.data
    bid = m.body(body_name).id
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if a == bid and "gripper_finger_link" in m.body(b).name:
                return True
    return False


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def run_trial(sim, per, rec, scen, obj, trial, seed, args, viewer_sync):
    t_wall = time.time()
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-JITTER_XY, JITTER_XY, 2)
    dyaw = np.radians(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG))
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(scenario=scen, object=obj, trial=trial, seed=seed,
               xy_source=os.environ.get("GRASP_XY_SOURCE", "perception"),
               gng_impl=os.environ.get("GNG_IMPL", "paper"),
               jit_dx_m=round(float(dx), 4), jit_dy_m=round(float(dy), 4), jit_dyaw_deg=round(float(np.degrees(dyaw)), 3),
               detected=False, success=False)
    tag = f"{scen} | {obj} | trial {trial}/{args.trials}"
    rec.banner = None
    rec.lines = [tag, "reset simulation"]
    ov = getattr(rec, "overlay", None)
    if ov is not None:
        ov.clear()
        rec.inset = None
    if getattr(rec, "on_trial_start", None):     # optional hook (run_live_demo.py)
        rec.on_trial_start()
    sim.reset(dx, dy, dyaw)
    sim.step(200)
    for name in sc.OBJECTS_BY_NAME:   # exact spawn pose, then a short settle (as in the earlier benchmarks)
        bid = sim.model.body(name).id
        jid = sim.model.body_jntadr[bid]
        qa, va = sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid]
        sim.data.qpos[qa:qa + 3] = sim.model.body_pos[bid]
        sim.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        sim.data.qvel[va:va + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)
    sim.step(100)
    row["head_pan"] = round(float(sim.data.joint("robot0:head_pan_joint").qpos[0]), 4)
    row["head_tilt"] = round(float(sim.data.joint("robot0:head_tilt_joint").qpos[0]), 4)
    z0 = float(sim.object_pos(obj)[2])
    row["z0"] = round(z0, 4)

    rec.lines[1] = "perception: YOLO26 + DBL-GNG"
    per.gng_seed = seed % (2 ** 31)
    best, inst = per.detect(depth_noise_std=DEPTH_NOISE_STD, rng=rng)
    row.update(render_ms=round(per.last_timing.get("render_ms", 0), 2), yolo_ms=round(per.last_timing.get("yolo_ms", 0), 2),
               gng_ms=round(per.last_timing.get("gng_ms", 0), 2), n_instances=len(inst))
    cv2.imwrite(os.path.join(args.out, "frames", f"{scen}_{obj}_{trial}.png"), cv2.cvtColor(per.last_rgb, cv2.COLOR_RGB2BGR))
    if ov is not None:
        items = []
        for o, dd in best.items():
            c = grasp_planner.compute_grasp_candidate(dd["positions"])
            reg = (c["region_min"], c["region_max"]) if c is not None and c.get("feasible") else None
            items.append(dict(obj=o, positions=dd["positions"], edges=dd["edges"], region=reg, target=(o == obj)))
        ov.set(items)
        rec.inset = make_inset(per.last_rgb, inst, items, per.last_cam[0], per.last_cam[1], per.intr)
        if getattr(rec, "on_perception", None):   # optional hook (run_live_demo.py)
            rec.on_perception(per, inst, items)
    rec.hold(0.5)
    if obj not in best:
        row["reason"] = "not_detected"
        return finish(row, rec, t_wall, "FAILED: object not detected", False)
    d = best[obj]
    Q = np.array(list(d["positions"].values()))
    ext = Q.max(0) - Q.min(0)
    row.update(detected=True, det_class=d["det_class"], det_conf=round(d["conf"], 3), mask_px=d["mask_px"],
               n_points=d["n_points"], gng_nodes=len(Q), gng_epochs=d["gng_info"].get("epochs", ""),
               ext_x_m=round(float(ext[0]), 4), ext_y_m=round(float(ext[1]), 4), ext_z_m=round(float(ext[2]), 4))

    rec.lines[1] = "planning grasp region"
    t = time.perf_counter()
    cand = grasp_planner.compute_grasp_candidate(d["positions"])
    row["plan_ms"] = round((time.perf_counter() - t) * 1e3, 3)
    if cand is None or not cand.get("feasible"):
        row["reason"] = "no_feasible_candidate"
        return finish(row, rec, t_wall, "FAILED: no feasible grasp region", False)
    truth_xy = sim.object_pos(obj)[:2]
    row.update(approach=cand["approach"], width_m=round(float(cand["width"]), 4),
               center_x=round(float(cand["center"][0]), 4), center_y=round(float(cand["center"][1]), 4),
               truth_x=round(float(truth_xy[0]), 4), truth_y=round(float(truth_xy[1]), 4),
               xy_err_mm=round(float(np.linalg.norm(cand["center"][:2] - truth_xy) * 1e3), 1))

    rec.lines[1] = f"grasp ({cand['approach']}, width {cand['width'] * 1e3:.0f} mm)"
    held = sc.grasp_object_via_candidate(sim, obj, cand)
    z_lift = float(sim.object_pos(obj)[2])
    row["z_after_lift"] = round(z_lift, 4)
    row["lift_ok_instant"] = bool(held.get("success", False))
    rec.lines[1] = f"holding {HOLD_S:.0f} s"
    zs = []
    for k in range(int(HOLD_S / sim.model.opt.timestep / 50)):
        sim.step(50)
        zs.append(float(sim.object_pos(obj)[2]))
    row["z_hold_min"] = round(min(zs), 4) if zs else round(z_lift, 4)
    row["finger_contact"] = bool(finger_contact(sim, obj))
    ok = bool(held.get("success", False)) and (min(zs) - z0 >= LIFT_M) and row["finger_contact"]
    row["lift_ok_hold"] = bool(min(zs) - z0 >= LIFT_M)
    row["reason"] = "ok" if ok else ("dropped_or_not_lifted" if not row["lift_ok_hold"] else "no_finger_contact")
    return finish(row, rec, t_wall, "SUCCESS" if ok else "FAILED: " + row["reason"], ok)


def finish(row, rec, t_wall, banner, ok):
    row["success"] = bool(ok)
    row["wall_s"] = round(time.time() - t_wall, 1)
    rec.banner = (f"{row['object']}: {banner}", ok)
    rec.hold(1.0)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(POSES), choices=list(POSES))
    ap.add_argument("--objects", nargs="+", default=list(sc.OBJECTS_BY_NAME), choices=list(sc.OBJECTS_BY_NAME))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--headless", action="store_true", help="no viewer window (default: viewer shown)")
    ap.add_argument("--no-pace", action="store_true", help="do not slow the viewer to real time")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or os.path.join("pose_eval_runs", f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}")
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    cfg = dict(argv=sys.argv, poses={k: POSES[k] for k in args.scenarios}, yolo_conf=YOLO_CONF, jitter_xy_m=JITTER_XY,
               jitter_yaw_deg=JITTER_YAW_DEG, depth_noise_std_m=DEPTH_NOISE_STD, lift_m=LIFT_M, hold_s=HOLD_S,
               trials=args.trials, seed=args.seed, env={k: os.environ.get(k) for k in
                                                         ("GNG_IMPL", "GNG_NODES", "GNG_ETA", "GNG_CUT", "GRASP_XY_SOURCE", "GRASP_DEPTH_M",
                                                          "GRASP_FINGER_KP", "GNG_WIDTH_BIAS", "LEGACY_OVERRIDES")},
               gng_width_bias=grasp_planner.GNG_WIDTH_BIAS, squeeze_override=sc.PERCEPTION_SQUEEZE_OVERRIDE,
               squeeze_margin=sc.SQUEEZE_MARGIN,
               source_sha256={f: sha(f) for f in ("sim_core.py", "grasp_planner.py", "dbl_gng_paper.py", "dbl_gng.py", "yolo_segmenter.py", "viz.py",
                                                  "gng_overlay.py", "perception.py", "run_pose_eval.py", "scene.xml")})
    cfg_name = "config_" + "_".join(args.scenarios) + ".json"
    with open(os.path.join(args.out, cfg_name), "w") as f:
        json.dump(cfg, f, indent=1, default=str)

    seg = YoloSegmenter(YOLO_CONF)
    assert seg.enabled, "YOLO failed to load"
    csv_path = os.path.join(args.out, "trials.csv")
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")     # append: one process per scenario shares one run directory
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()
    jsonl = open(os.path.join(args.out, "trials.jsonl"), "a")
    tally = {}

    for si, scen in enumerate(args.scenarios):
        sim = sc.Sim(scenario=POSES[scen])
        per = Perceiver(sim, seg, conf_min=YOLO_CONF)
        rec = Recorder(sim, os.path.join(args.out, f"video_{scen}.mp4"), per) if not args.no_video else None
        viewer = None
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            vc = video_camera(sim)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = vc.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = vc.distance, vc.azimuth, vc.elevation
            if rec is not None:
                rec.overlay.viewer = viewer
        wall0, sim0 = time.time(), [0.0]
        counter = [0]

        def hook(s):
            if rec is not None:
                rec.tick()
            counter[0] += 1
            if viewer is not None and counter[0] % 8 == 0:
                if viewer.is_running():
                    viewer.sync()
                sim0[0] += 8 * s.model.opt.timestep
                if not args.no_pace:
                    ahead = sim0[0] - (time.time() - wall0)
                    if ahead > 0:
                        time.sleep(min(ahead, 0.05))
        sim.step_hook = hook
        if rec is None:
            class _Null:
                lines, banner = ["", ""], None
                overlay = Overlay()
                inset = None
                def tick(self): pass
                def hold(self, s): pass
                def close(self): pass
            rec = _Null()
            if viewer is not None:
                rec.overlay.viewer = viewer

        for obj in args.objects:
            oi = list(sc.OBJECTS_BY_NAME).index(obj)   # index in the FULL object list: subset runs reuse the paper's seeds
            for trial in range(1, args.trials + 1):
                seed = args.seed * 100000 + list(POSES).index(scen) * 1000 + oi * 10 + trial   # distinct per pose even with one process per pose
                try:
                    row = run_trial(sim, per, rec, scen, obj, trial, seed, args, None)
                except Exception as e:  # keep the sweep going; the failure is logged
                    row = dict.fromkeys(CSV_FIELDS, "")
                    row.update(scenario=scen, object=obj, trial=trial, seed=seed, success=False,
                               reason=f"exception: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                writer.writerow(row)
                csv_f.flush()
                jsonl.write(json.dumps(row) + "\n")
                jsonl.flush()
                k = (scen, obj)
                ok, n = tally.get(k, (0, 0))
                tally[k] = (ok + bool(row["success"]), n + 1)
                print(f"[{scen}] {obj:15s} trial {trial}/{args.trials}: "
                      f"{'OK ' if row['success'] else 'FAIL'} {row['reason']:<22s} "
                      f"approach={row.get('approach','-')} xy_err={row.get('xy_err_mm','-')}mm "
                      f"({row.get('wall_s', '')}s)", flush=True)
        rec.close()
        if viewer is not None:
            viewer.close()
        per.rr.close()
        per.rd.close()
        if getattr(rec, "r", None) is not None:
            rec.r.close()
        print(f"== {scen} done: " + ", ".join(f"{o}={tally[(scen, o)][0]}/{tally[(scen, o)][1]}" for o in args.objects), flush=True)
    csv_f.close()
    jsonl.close()
    tot_ok = sum(v[0] for v in tally.values())
    tot_n = sum(v[1] for v in tally.values())
    print(f"TOTAL {tot_ok}/{tot_n}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
