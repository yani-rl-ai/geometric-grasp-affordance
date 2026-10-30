#!/usr/bin/env python3
"""
Live demo of the paper pipeline: the Fetch robot grasps the five tabletop objects while
the simulation, the RGB camera, the depth camera and the DBL-GNG graphs are all visible.

Windows
  * "MuJoCo"                         3D viewer. The fitted DBL-GNG graphs (nodes + edges, one colour per object)
                                     and the light-blue grasp-region box of the current target are drawn into the scene.
  * "RGB camera | YOLO26-seg + DBL-GNG"   what the pipeline perceived: instance masks, boxes, COCO labels (not used
                                     downstream), graph projected into the image, grasp region of the target.
  * "Depth camera | DBL-GNG"         the depth image the graphs were fitted to, with the mask outlines, graphs and region.
  * "Head camera (live)"             the live RGB and depth stream of the head camera (shows the arm moving in).
The two annotated windows show a SNAPSHOT taken at perception time (they stay valid while the object is lifted);
the live window keeps streaming.

Every trial is exactly run_pose_eval.run_trial() -- the code that produced the paper's numbers -- including the
per-trial base jitter, depth noise and seeds. With --seed 1 (or 2) each demo trial uses the same seed as the
matching trial in results/paper_runs/ and the console prints the logged outcome of that trial for comparison.

Examples
    python run_live_demo.py                                  # P0, every object once
    python run_live_demo.py --pose P2 --objects banana mug --trials 2
    python run_live_demo.py --pose P1 --loop                 # until you close a window / press q or Esc
    python run_live_demo.py --record demo_P0.mp4             # also writes a 1280x720 video (needs ffmpeg)
    python run_live_demo.py --save-windows shots             # also saves the annotated RGB / depth images as PNGs

Keys (in any OpenCV window): q or Esc = quit.  Closing the MuJoCo window also quits.
Windows can be dragged/resized freely; they start side by side at the top-left of the screen.
"""
import argparse
import csv
import os
import time
import types
from datetime import datetime

os.environ.setdefault("MUJOCO_GL", "egl")      # offscreen camera rendering; must be set before mujoco is imported

import cv2
import mujoco
import numpy as np

import run_pose_eval as rp
import sim_core as sc
from perception import Perceiver
from viz import depth_colormap, make_depth_view, make_inset
from yolo_segmenter import YoloSegmenter

WIN_RGB = "RGB camera | YOLO26-seg + DBL-GNG"
WIN_DEPTH = "Depth camera | DBL-GNG"
WIN_LIVE = "Head camera (live): RGB | depth"
LIVE_HZ = 15


class DemoQuit(Exception):
    pass


def put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class LiveDisplay(rp.Recorder):
    """Same interface as run_pose_eval.Recorder (lines, banner, overlay, inset, tick, hold, close), but it
    shows the camera windows instead of (or in addition to, with --record) writing a video."""

    def __init__(self, sim, per, record_path=None, show_windows=True, pace=True, save_dir=None):
        super().__init__(sim, record_path, per)
        self.show_windows = show_windows
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.pace = pace
        self.dt = 1.0 / (rp.FPS if record_path else LIVE_HZ)
        self.snap_rgb = None          # annotated perception snapshots (RGB images, 640x480)
        self.snap_depth = None
        self.quit = False
        if show_windows:
            flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            for name, (x, y, w, h) in ((WIN_RGB, (10, 40, 640, 480)), (WIN_DEPTH, (660, 40, 640, 480)),
                                       (WIN_LIVE, (10, 560, 640, 240))):
                cv2.namedWindow(name, flags)
                cv2.resizeWindow(name, w, h)
                cv2.moveWindow(name, x, y)

    # --- hooks called by run_pose_eval.run_trial -------------------------------------------------
    def on_trial_start(self):
        self.snap_rgb = self.snap_depth = None

    def on_perception(self, per, instances, items):
        cam_pos, cam_mat = per.last_cam
        self.snap_rgb = make_inset(per.last_rgb, instances, items, cam_pos, cam_mat, per.intr, size=(640, 480))
        self.snap_depth = make_depth_view(per.last_depth, instances, items, cam_pos, cam_mat, per.intr)
        if self.save_dir:    # the two annotated perception images, e.g. for documentation
            tag = "".join(c if c.isalnum() else "_" for c in self.lines[0]).strip("_")
            cv2.imwrite(os.path.join(self.save_dir, f"{tag}_rgb.png"), cv2.cvtColor(self.snap_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.save_dir, f"{tag}_depth.png"), cv2.cvtColor(self.snap_depth, cv2.COLOR_RGB2BGR))

    # --- frame generation ------------------------------------------------------------------------
    def _live_frames(self):
        d = self.sim.data
        self.per.rr.update_scene(d, camera=sc.CAM_NAME)
        rgb = self.per.rr.render().copy()
        self.per.rd.update_scene(d, camera=sc.CAM_NAME)
        depth = self.per.rd.render().copy()
        return rgb, depth

    def _decorate(self, img):
        for i, txt in enumerate(self.lines):
            put(img, txt, (10, 24 + 24 * i), 0.62)
        if self.banner:
            txt, ok = self.banner
            put(img, txt, (10, img.shape[0] - 14), 0.8, (60, 220, 60) if ok else (70, 70, 255), 2)   # BGR
        return img

    def _show(self):
        rgb, depth = self._live_frames()
        if self.snap_rgb is not None:
            a = cv2.cvtColor(self.snap_rgb, cv2.COLOR_RGB2BGR)
            b = cv2.cvtColor(self.snap_depth, cv2.COLOR_RGB2BGR)
            put(a, "perception snapshot", (10, 470), 0.5, (200, 200, 200))
            put(b, "perception snapshot  (warm = near, 0.2-2.5 m)", (10, 470), 0.5, (200, 200, 200))
        else:
            a = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            b = cv2.cvtColor(depth_colormap(depth), cv2.COLOR_RGB2BGR)
            put(a, "live (no perception yet)", (10, 470), 0.5, (200, 200, 200))
            put(b, "live (no perception yet)", (10, 470), 0.5, (200, 200, 200))
        cv2.imshow(WIN_RGB, self._decorate(a))
        cv2.imshow(WIN_DEPTH, self._decorate(b))
        live = np.hstack([cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 240)),
                          cv2.resize(cv2.cvtColor(depth_colormap(depth), cv2.COLOR_RGB2BGR), (320, 240))])
        cv2.imshow(WIN_LIVE, live)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.quit = True

    def frame(self):
        if self.show_windows:
            self._show()
        if self.proc is not None:
            super().frame()
        if self.quit:
            raise DemoQuit()

    def hold(self, seconds):
        for _ in range(int(seconds * rp.FPS)):
            t0 = time.time()
            self.frame()
            if self.pace and self.proc is None:
                time.sleep(max(0.0, 1.0 / rp.FPS - (time.time() - t0)))

    def close(self):
        super().close()
        if self.show_windows:
            cv2.destroyAllWindows()


def logged_outcome(seed):
    """Outcome of the trial with this seed in the paper logs (results/paper_runs/{main,confirm_main_seed2})."""
    here = os.path.dirname(os.path.abspath(__file__))
    for run in ("main", "confirm_main_seed2"):
        path = os.path.join(here, "results", "paper_runs", run, "trials.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["seed"] == str(seed):
                    return f"paper log '{run}': {'OK' if r['success'] == 'True' else 'FAIL'} ({r['reason']})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="P0", choices=list(rp.POSES), help="robot placement (default P0)")
    ap.add_argument("--objects", nargs="+", default=list(sc.OBJECTS_BY_NAME), choices=list(sc.OBJECTS_BY_NAME))
    ap.add_argument("--trials", type=int, default=1, help="trials per object (default 1)")
    ap.add_argument("--seed", type=int, default=1, help="1 or 2 reproduces trials of the paper runs (default 1)")
    ap.add_argument("--loop", action="store_true", help="repeat the object list until you quit")
    ap.add_argument("--no-pace", action="store_true", help="do not slow the simulation down to real time")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the MuJoCo 3D viewer window")
    ap.add_argument("--viewer-ui", action="store_true", help="show the MuJoCo viewer side panels (hidden by default)")
    ap.add_argument("--no-windows", action="store_true", help="do not open the OpenCV camera windows")
    ap.add_argument("--record", default=None, metavar="FILE.mp4", help="also record a 1280x720 video (needs ffmpeg)")
    ap.add_argument("--save-windows", default=None, metavar="DIR",
                    help="save the annotated RGB and depth perception images of every trial as PNGs in DIR")
    ap.add_argument("--out", default=None, help="log directory (default demo_runs/<timestamp>)")
    args = ap.parse_args()

    out = args.out or os.path.join("demo_runs", f"{datetime.now():%Y%m%d_%H%M%S}_{args.pose}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    ns = types.SimpleNamespace(out=out, trials=args.trials)

    # Order matters: every EGL renderer (YOLO/depth renderers in Perceiver, the optional video renderer) must be
    # created BEFORE the passive MuJoCo viewer is launched, otherwise "Failed to make the EGL context current".
    seg = YoloSegmenter(rp.YOLO_CONF)
    sim = sc.Sim(scenario=rp.POSES[args.pose])
    per = Perceiver(sim, seg, conf_min=rp.YOLO_CONF)
    rec = LiveDisplay(sim, per, record_path=args.record, show_windows=not args.no_windows, pace=not args.no_pace,
                      save_dir=args.save_windows)

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data,
                                              show_left_ui=args.viewer_ui, show_right_ui=args.viewer_ui)
        vc = rp.video_camera(sim)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = vc.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = vc.distance, vc.azimuth, vc.elevation
        rec.overlay.viewer = viewer

    wall0, sim_t, n = time.time(), [0.0], [0]

    def hook(s):
        rec.tick()
        n[0] += 1
        if viewer is not None and n[0] % 8 == 0:
            if not viewer.is_running():
                raise DemoQuit()
            viewer.sync()
            sim_t[0] += 8 * s.model.opt.timestep
            if not args.no_pace:
                ahead = sim_t[0] - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.05))
    sim.step_hook = hook

    csv_f = open(os.path.join(out, "trials.csv"), "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=rp.CSV_FIELDS)
    writer.writeheader()
    pose_idx = list(rp.POSES).index(args.pose)
    all_objs = list(sc.OBJECTS_BY_NAME)
    tally = {}
    print(f"Placement {args.pose}; objects {args.objects}; {args.trials} trial(s) each; seed {args.seed}. "
          f"Logs: {out}. Press q/Esc in a camera window or close the MuJoCo window to stop.", flush=True)
    try:
        sweep = 0
        while True:
            for obj in args.objects:
                for t in range(1, args.trials + 1):
                    trial = sweep * args.trials + t
                    # same seed formula as run_pose_eval.main (object index = position in the full object list)
                    seed = args.seed * 100000 + pose_idx * 1000 + all_objs.index(obj) * 10 + trial
                    try:
                        row = rp.run_trial(sim, per, rec, args.pose, obj, trial, seed, ns, None)
                    except DemoQuit:
                        raise
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        row = dict.fromkeys(rp.CSV_FIELDS, "")
                        row.update(scenario=args.pose, object=obj, trial=trial, seed=seed, success=False,
                                   reason=f"exception: {type(e).__name__}: {e}")
                    writer.writerow(row)
                    csv_f.flush()
                    ok, tot = tally.get(obj, (0, 0))
                    tally[obj] = (ok + bool(row["success"]), tot + 1)
                    ref = logged_outcome(seed)
                    print(f"[{args.pose}] {obj:15s} trial {trial}: {'OK  ' if row['success'] else 'FAIL'} {row['reason']:<22s} "
                          f"xy_err={row.get('xy_err_mm', '-')}mm  seed={seed}" + (f"   | {ref}" if ref else ""), flush=True)
            sweep += 1
            if not args.loop:
                break
    except (DemoQuit, KeyboardInterrupt):
        print("Stopped by user.", flush=True)
    finally:
        csv_f.close()
        try:
            rec.close()
        except Exception:
            pass
        if viewer is not None:
            viewer.close()
    if tally:
        print("Summary: " + ", ".join(f"{o} {a}/{b}" for o, (a, b) in tally.items()), flush=True)


if __name__ == "__main__":
    main()
