#!/usr/bin/env python3
"""
Self-test of the installation. Run this first:   python check_setup.py

Checks (each prints PASS / WARN / FAIL with a hint):
  1. Python and package versions (mujoco, numpy, opencv, torch, ultralytics)
  2. GPU for YOLO26 (CUDA) -- WARN only, CPU works but is slower
  3. Display for the live demo windows -- WARN only (headless evaluation does not need it)
  4. The MuJoCo scene loads, robot placements P0..P3 can be applied, physics steps
  5. Offscreen (EGL) rendering of the head RGB-D camera
  6. The DBL-GNG implementation fits synthetic points correctly (deterministic for a seed)
  7. The YOLO26s-seg weights load (downloaded automatically on first use) and segment the five objects
  8. ffmpeg is present (only needed for --record / the video recorder) -- WARN only

Options:  --skip-yolo   skip step 7 (e.g. offline before the weights are downloaded)
Exit status 0 = no FAIL.
"""
import argparse
import importlib
import os
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status:4s}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def check_versions():
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        report("FAIL", "python >= 3.9", "found " + sys.version.split()[0])
    for mod in ("numpy", "cv2", "mujoco", "torch", "ultralytics"):
        try:
            m = importlib.import_module(mod)
            report("PASS", f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:
            report("FAIL", f"import {mod}", f"{type(e).__name__}: {e}  ->  pip install -r requirements.txt")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            report("PASS", "CUDA GPU for YOLO26", torch.cuda.get_device_name(0))
        else:
            report("WARN", "CUDA GPU for YOLO26", "not available: YOLO runs on the CPU (slower, results unaffected)")
    except Exception as e:
        report("WARN", "CUDA GPU for YOLO26", str(e))


def check_display():
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        report("WARN", "display", "DISPLAY is not set: run_live_demo.py needs a desktop session "
                                   "(run_pose_eval.py --headless does not need one, see docs/TUTORIAL.md section 4.3)")
    else:
        report("PASS", "display", os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform)


def check_scene_and_render():
    try:
        import numpy as np
        import mujoco
        import sim_core as sc
        for name in sc.PLACEMENT_ORDER:
            sim = sc.Sim(scenario=name)
            sim.step(20)
        report("PASS", "scene.xml loads; placements P0..P3 apply; physics steps")
    except Exception as e:
        report("FAIL", "MuJoCo scene", f"{type(e).__name__}: {e}")
        return None
    try:
        sim = sc.Sim(scenario="P0")
        sim.step(300)
        r_rgb = mujoco.Renderer(sim.model, height=sc.IMG_H, width=sc.IMG_W)
        r_d = mujoco.Renderer(sim.model, height=sc.IMG_H, width=sc.IMG_W)
        r_d.enable_depth_rendering()
        r_rgb.update_scene(sim.data, camera=sc.CAM_NAME)
        rgb = r_rgb.render().copy()
        r_d.update_scene(sim.data, camera=sc.CAM_NAME)
        depth = r_d.render().copy()
        ok = rgb.std() > 5 and np.isfinite(depth).all() and 0.2 < float(np.median(depth)) < 3.0
        report("PASS" if ok else "FAIL", f"offscreen RGB-D rendering (MUJOCO_GL={os.environ['MUJOCO_GL']})",
               f"rgb {rgb.shape}, median depth {np.median(depth):.2f} m")
        return sim, rgb
    except Exception as e:
        report("FAIL", "offscreen rendering", f"{type(e).__name__}: {e}\n"
               "       Try MUJOCO_GL=glfw (needs a display) or MUJOCO_GL=osmesa (apt install libosmesa6-dev), "
               "and make sure libegl1 / libgl1 are installed (see docs/TUTORIAL.md, Troubleshooting).")
        return None


def check_dbl_gng():
    try:
        import numpy as np
        from dbl_gng_paper import PaperDBLGNG
        rng = np.random.default_rng(0)
        # points on three visible faces of a 72 x 164 x 213 mm box (like a cracker box seen by the head camera)
        n = 250
        sx, sy, sz = 0.072, 0.164, 0.213
        f = rng.choice(3, size=n, p=[0.4, 0.35, 0.25])
        u, v = rng.uniform(-0.5, 0.5, (2, n))
        P = np.where(f[:, None] == 0, np.stack([u * sx, v * sy, np.full(n, sz / 2)], 1),
                     np.where(f[:, None] == 1, np.stack([u * sx, np.full(n, -sy / 2), v * sz], 1),
                              np.stack([np.full(n, sx / 2), u * sy, v * sz], 1)))
        g1 = PaperDBLGNG(max_nodes=30, n_start=4, seed=0)
        pos, edges = g1.fit(P)
        pos2, _ = PaperDBLGNG(max_nodes=30, n_start=4, seed=0).fit(P)
        Q = np.array(list(pos.values()))
        same = np.allclose(Q, np.array(list(pos2.values())))
        ext, true = Q.max(0) - Q.min(0), P.max(0) - P.min(0)
        ok = (len(pos) == 30 and len(edges) > 20 and same and g1.last_info["epochs"] == 18
              and np.all(ext > 0.7 * true) and np.all(Q.min(0) >= P.min(0) - 1e-9) and np.all(Q.max(0) <= P.max(0) + 1e-9))
        report("PASS" if ok else "FAIL", "DBL-GNG fit on synthetic box points",
               f"{len(pos)} nodes, {len(edges)} edges, {g1.last_info['epochs']} epochs, "
               f"node extent {np.round(ext * 1e3).astype(int).tolist()} mm of {np.round(true * 1e3).astype(int).tolist()} mm, deterministic={same}")
    except Exception as e:
        report("FAIL", "DBL-GNG", f"{type(e).__name__}: {e}")


def check_yolo(sim_rgb):
    if sim_rgb is None:
        report("WARN", "YOLO26 segmentation", "skipped (no rendered frame)")
        return
    try:
        from yolo_segmenter import YoloSegmenter
        seg = YoloSegmenter(0.05)
        res = seg.predict(sim_rgb[1])
        n = 0 if res.masks is None else len(res.masks.data)
        labels = [seg.model.names[int(c)] for c in res.boxes.cls]
        report("PASS" if n >= 4 else "WARN", "YOLO26s-seg weights load and segment the scene",
               f"{n} instances: {labels}" + ("" if n >= 4 else "  (expected about 5)"))
    except Exception as e:
        report("FAIL", "YOLO26 weights", f"{e}")


def check_ffmpeg():
    if shutil.which("ffmpeg"):
        report("PASS", "ffmpeg (optional, for --record)", shutil.which("ffmpeg"))
    else:
        report("WARN", "ffmpeg (optional, for --record)", "not found: sudo apt install ffmpeg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()
    check_versions()
    check_gpu()
    check_display()
    out = check_scene_and_render()
    check_dbl_gng()
    if not args.skip_yolo:
        check_yolo(out)
    check_ffmpeg()
    nf, nw = results.count("FAIL"), results.count("WARN")
    print(f"\n{results.count('PASS')} passed, {nw} warnings, {nf} failed.")
    if nf == 0:
        print("Setup looks good. Next:  python run_live_demo.py")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
