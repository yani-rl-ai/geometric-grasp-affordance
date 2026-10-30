# Tutorial: install, run the live demo, reproduce the paper

This tutorial takes you from a clean Ubuntu machine to (1) a live simulation in which you can **see the RGB camera,
the depth camera and the DBL-GNG graphs while the robot grasps**, and (2) a re-run of the experiments of the paper.

* [0. Quick start (5 commands)](#0-quick-start)
* [1. Requirements](#1-requirements)
* [2. Installation](#2-installation)
* [3. Verify the installation](#3-verify-the-installation)
* [4. Run the live demo](#4-run-the-live-demo-rgb--depth--dbl-gng--simulation)
* [5. Understand what you see](#5-understand-what-you-see)
* [6. Reproduce the paper](#6-reproduce-the-paper)
* [7. Configuration reference](#7-configuration-reference)
* [8. Use the pipeline from your own Python code](#8-use-the-pipeline-from-your-own-python-code)
* [9. Troubleshooting](#9-troubleshooting)
* [10. Repository layout](#10-repository-layout)

---

## 0. Quick start

On Ubuntu 22.04 with a desktop session (details and alternatives below):

```bash
sudo apt update && sudo apt install -y git python3.10 python3.10-venv libegl1 libgl1 ffmpeg
git clone https://github.com/yani-rl-ai/geometric-grasp-affordance.git && cd geometric-grasp-affordance
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
python check_setup.py            # every line should say PASS (WARN is fine for the optional items)
python run_live_demo.py          # robot, RGB camera, depth camera and DBL-GNG graphs on screen
```

The first `run_live_demo.py` downloads the 22 MB YOLO26 weights automatically (internet needed once).
Stop the demo with `q` / `Esc` in a camera window or by closing the MuJoCo window.

---

## 1. Requirements

| | Needed | What the paper experiments used |
|---|---|---|
| OS | Linux x86-64 (Ubuntu 22.04 recommended) | Ubuntu 22.04.5 LTS, kernel 6.8, X11 session |
| Python | 3.10 (tested); 3.9 and 3.11 should work | 3.10.12 |
| GPU | Optional but recommended: NVIDIA GPU with a recent driver for YOLO26 (CUDA). Without a GPU YOLO runs on the CPU (slower; results are not affected, the live demo is just less smooth). | RTX 3080 Laptop (8 GB), driver 595.71.05 |
| CPU | any modern CPU. DBL-GNG always runs on the CPU, single thread (NumPy) | Intel Core i9-11980HK |
| RAM / disk | 8 GB RAM; about 6-8 GB of disk for the Python environment (PyTorch with CUDA libraries dominates), 50 MB for the repository | 32 GB RAM |
| Display | For `run_live_demo.py`: a desktop session (X11, or Wayland via XWayland). Not needed for `run_pose_eval.py --headless`. | |
| Software | MuJoCo 3.2.7, PyTorch 2.14.0, Ultralytics 8.4.144, OpenCV 4.10, NumPy 1.26.4 (pinned in `requirements.txt`) | same |

Not tested: Windows, macOS (the MuJoCo passive viewer needs `mjpython` there), Wayland-only setups, WSL2.
Please open an issue if you get one of them working.

---

## 2. Installation

### 2.1 System packages (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3-pip libegl1 libgl1 ffmpeg
```

* `python3.10-venv` is required: on a fresh Ubuntu, `python3 -m venv` fails with *"ensurepip is not available"* without it (on other Ubuntu releases the package is `python3-venv`).
* `libegl1` / `libgl1` are used for offscreen (EGL) rendering of the camera images and by OpenCV.
* `ffmpeg` is only needed to record videos (`--record`, and the `--video` option of `scripts/reproduce_paper.sh`).
* On a machine with an NVIDIA GPU, install the proprietary NVIDIA driver (Ubuntu: *Software & Updates -> Additional Drivers*, or
  `sudo ubuntu-drivers autoinstall`) and reboot. The driver also provides the EGL library MuJoCo uses.

### 2.2 Get the code

```bash
git clone https://github.com/yani-rl-ai/geometric-grasp-affordance.git
cd geometric-grasp-affordance
```

Every command in this tutorial is run from this folder.

### 2.3 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # run this in every new terminal
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `mujoco`, `numpy`, `opencv-python`, `ultralytics` and `torch`. The default PyPI `torch` wheel for Linux ships
CUDA 13 libraries and needs a recent NVIDIA driver. If `check_setup.py` reports *"CUDA GPU ... not available"* on a machine that has
an NVIDIA GPU, your driver is probably too old for that wheel: install a PyTorch build that matches your driver first
(command generator: <https://pytorch.org/get-started/locally/>), change the `torch==` line of `requirements.txt` to the version you installed, and then run
`pip install -r requirements.txt`. Otherwise just continue: the CPU works.

Optional extras (only for `make_scenario_schematic.py` and the Open3D helpers in `gng_overlay.py` / `grasp_planner.py`):
`pip install -r requirements-optional.txt`.

### 2.4 YOLO26 weights

Nothing to do: the first time the pipeline starts, Ultralytics downloads `yolo26s-seg.pt` (22.4 MB, from the Ultralytics assets release)
next to `yolo_segmenter.py`. Offline? Download the file on another machine
(<https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt>), copy it next to `yolo_segmenter.py`, or point to it with
`export YOLO_WEIGHTS=/path/to/yolo26s-seg.pt`. (The weights are not part of this repository.)

---

## 3. Verify the installation

```bash
python check_setup.py
```

It checks the package versions, CUDA, the display, that the scene loads and all four robot placements apply, offscreen RGB-D rendering,
a DBL-GNG fit on synthetic points, the YOLO26 weights (and that they segment the scene) and `ffmpeg`. Example of a good result:

```
Python 3.10.12
[PASS] import numpy -- 1.26.4
[PASS] import cv2 -- 4.10.0
[PASS] import mujoco -- 3.2.7
[PASS] import torch -- 2.14.0+cu130
[PASS] import ultralytics -- 8.4.144
[PASS] CUDA GPU for YOLO26 -- NVIDIA GeForce RTX 3080 Laptop GPU
[PASS] display -- :1
[PASS] scene.xml loads; placements P0..P3 apply; physics steps
[PASS] offscreen RGB-D rendering (MUJOCO_GL=egl) -- rgb (480, 640, 3), median depth 0.84 m
[PASS] DBL-GNG fit on synthetic box points -- 30 nodes, 43 edges, 18 epochs, node extent [59, 154, 198] mm of [72, 163, 212] mm, deterministic=True
[PASS] YOLO26s-seg weights load and segment the scene -- 8 instances: [...]
[PASS] ffmpeg (optional, for --record) -- /usr/bin/ffmpeg
Setup looks good. Next:  python run_live_demo.py
```

`WARN` lines (no GPU, no display, no ffmpeg) do not stop you from running the headless evaluation. A `FAIL` line comes with a hint;
see also [Troubleshooting](#9-troubleshooting). `python check_setup.py --skip-yolo` skips the weights check (e.g. before the first download).

---

## 4. Run the live demo (RGB + depth + DBL-GNG + simulation)

```bash
python run_live_demo.py
```

This starts the Fetch robot at placement **P0**, and lets it grasp the cracker box, mustard bottle, mug, banana and tennis ball one after
another (about 6-7 s per object, paced to real time). **Four windows** appear:

| Window | What it shows |
|---|---|
| **MuJoCo : fetch_grasp_scene** | The 3D simulation (drag with the left mouse button to orbit, right button to pan, wheel to zoom; the view does not affect the experiment). The fitted **DBL-GNG graphs** (spheres = nodes, thin lines = edges, one colour per object) and the **light-blue grasp-region box** of the object being grasped are drawn into the scene. |
| **RGB camera \| YOLO26-seg + DBL-GNG** | The head-camera RGB image at *perception time*: YOLO26 instance masks, boxes and COCO labels, the DBL-GNG graph of every object projected into the image, and the grasp region of the target. |
| **Depth camera \| DBL-GNG** | The depth image the graphs were fitted to (colour map: warm = near, 0.2-2.5 m) with the mask outlines, the graphs and the grasp region. |
| **Head camera (live)** | The live RGB and depth stream of the head camera, side by side; use it to watch the arm enter the camera view. |

The two annotated windows show a *snapshot* taken at perception time, so they stay valid while the object is lifted; the live window keeps
streaming. Example (placement P0, banana; the same layout you get on screen):

![live demo](img/live_demo.png)

Windows start side by side at the top-left of the screen and can be moved and resized freely.

### 4.1 One trial, step by step

The top-left text of the camera windows tells you where you are:

1. `reset simulation` - the simulation is reset (objects back at their spawn poses, arm at the ready pose), the robot base is jittered by up to
   +-15 mm and +-2 deg, and the head is re-aimed.
2. `perception: YOLO26 + DBL-GNG` - the head camera is rendered, Gaussian depth noise (sigma 1.5 mm) is added, YOLO26 segments the scene, and one
   DBL-GNG graph is fitted per segmented object. The snapshot windows and the 3D graphs appear.
3. `planning grasp region` - the geometric planner computes closing width, approach (top-down) and the region box from the graph nodes.
4. `grasp (top, width NN mm)` - the arm moves above the target, descends, closes the gripper and lifts to the transit height.
5. `holding 2 s` - the object has to stay lifted for 2 s.
6. A green `SUCCESS` or red `FAILED: <reason>` banner. The success criterion is the paper's: object reference point >= 5 cm above its resting
   height, still >= 5 cm after 2.0 s of holding, finger-object contact at the end.

The console prints one line per trial:

```
[P0] mug             trial 1: OK   ok                     xy_err=11.4mm  seed=100021   | paper log 'main': OK (ok)
```

`xy_err` is the distance between the perceived grasp-region centre and the simulator's true object position (evaluation only; the robot uses the
perceived one). `seed` fixes the jitter, depth noise and DBL-GNG start points. With the default `--seed 1` (or `--seed 2`) **the demo trial uses
the same seed as the corresponding trial of the paper runs**, and the console shows the logged outcome of that trial after the `|`.

### 4.2 Options

```bash
python run_live_demo.py --pose P2                       # another robot placement: P0 front, P1 back, P2 +y side, P3 front-diagonal
python run_live_demo.py --objects banana mug --trials 3 # a subset of objects, three trials each
python run_live_demo.py --pose P1 --loop                # repeat forever until you quit
python run_live_demo.py --seed 2                        # reproduce trials of the seed-2 replicate instead
python run_live_demo.py --no-pace                       # run as fast as possible instead of real time
python run_live_demo.py --record demo_P0.mp4            # also write a 1280x720 30 fps video (needs ffmpeg)
python run_live_demo.py --save-windows shots            # also save the annotated RGB/depth images of every trial as PNG
python run_live_demo.py --viewer-ui                     # show the MuJoCo viewer's side panels (hidden by default)
python run_live_demo.py --no-viewer                     # only the OpenCV camera windows
python run_live_demo.py --no-windows                    # only the MuJoCo window
```

Logs of the demo (`trials.csv`, `trials.jsonl`, and the perception frames) go to `demo_runs/<timestamp>_<pose>/`.
Environment variables switch the pipeline variants used in the paper's ablations (e.g. `GNG_IMPL=legacy python run_live_demo.py`);
see [7. Configuration reference](#7-configuration-reference).

### 4.3 Running without a monitor

`run_live_demo.py` needs a display. On a server without a monitor, run the same trials without windows and record a video of the experiment instead:

```bash
python run_pose_eval.py --scenarios P0 --trials 1 --headless --no-pace --out pose_eval_runs/video_P0
```

(`--no-video` is *not* given, so `video_P0.mp4` is written: external camera view with the DBL-GNG graphs, the grasp region and an inset of the head camera
with the YOLO26 masks and graphs.) Use `--objects` to choose objects; their trials reuse the paper's seeds.

---

## 5. Understand what you see

* **Colours.** Each of the five objects has a fixed colour (cracker box red, mustard bottle yellow, mug magenta, banana green, tennis ball cyan);
  an unmatched instance is grey. Nodes are drawn in the object colour, edges in white (camera windows) or in the object colour (3D scene).
* **COCO labels.** YOLO26 is a COCO detector: it has no *cracker box* class and calls it "book", "bottle" or "bowl", often with low confidence (the
  pipeline therefore uses confidence >= 0.05). **The labels are not used anywhere**; only the instance masks enter the pipeline. Instances are
  matched to the five benchmark objects by their position, as evaluation bookkeeping.
* **Graph.** The nodes are the 3D points that DBL-GNG fitted to the object's visible surface (at most 30). The planner uses only the node
  *coordinates* (axis-aligned extents, local closing width), not the edges.
* **Light-blue box.** The candidate grasp region (paper Eq. 7): the node bounds, limited vertically to the finger reach below the top. It is a geometric
  candidate, not a certified contact set.
* **Width underestimation.** The nodes lie slightly inside the visible surface and the median-depth filter trims tall objects, so the estimated closing
  width is smaller than the true width (paper Sec. VI-A); this is why the pipeline adds a small bias `b = 10 mm`.

---

## 6. Reproduce the paper

Everything the paper reports is in `results/` (per-trial logs of every run) and can be regenerated.

### 6.1 Look at the published numbers (no simulation needed)

```bash
python make_results_tables.py            # rewrites results/results_tables.md from results/paper_runs/*/trials.csv
python analyze_perception_metrics.py     # detection, region-centre error, closing-width error (paper Table V and Sec. VI-A)
python print_table1_params.py            # prints every value of Table I straight from the code
python analyze_pose_eval.py results/paper_runs/main results/paper_runs/confirm_main_seed2   # per-run summary with Wilson intervals
```

| Paper item | Where it comes from |
|---|---|
| Table I (DBL-GNG parameters) | `python print_table1_params.py`; code in `dbl_gng_paper.py`, `gng_overlay.py` |
| Table II-IV (executor, YOLO26, machine) | `sim_core.py`, `yolo_segmenter.py`, `run_pose_eval.py` (`config_P*.json` of each run records the values used) |
| Table V (detection and region-centre error) | `analyze_perception_metrics.py`, table R5 in `results/results_tables.md` |
| Table VI (successes per object x placement, both seeds) | table R3 in `results/results_tables.md` (runs `main` + `confirm_main_seed2`) |
| Table VII (components) | table R6 rows A (frozen), D (raw points), E (no bias, post hoc) and R7 |
| Table VIII (latency) | table R8 (`results/paper_runs/profile_150`) |
| Placements (Sec. V-B) | `docs/img/placements.png`, `sim_core.PLACEMENTS`, `scenario_search.py` |

### 6.2 Re-run the experiments

First a quick check (about 1 minute): one placement, one trial per object, no windows.

```bash
python run_pose_eval.py --scenarios P0 --trials 1 --headless --no-video --no-pace --out pose_eval_runs/smoke
python analyze_pose_eval.py pose_eval_runs/smoke
```

Then the paper's principal evaluation (4 placements x 5 objects x 5 trials = 100 trials; about 2-3 minutes on the paper machine, mostly start-up time of the four processes):

```bash
scripts/reproduce_paper.sh main                 # seed 1, headless; writes pose_eval_runs/main/
python analyze_pose_eval.py pose_eval_runs/main
```

The paper collected its data **with the MuJoCo viewer open** (and recorded the `main` run on video). To do the same:
`scripts/reproduce_paper.sh --viewer --video main` (needs a desktop and ffmpeg, runs in real time, about 3-4 minutes per placement).

All steps of `scripts/reproduce_paper.sh` (run `scripts/reproduce_paper.sh --help`):

| Step | What | Seed | Run directories |
|---|---|---|---|
| `main` | principal evaluation | 1 | `main` |
| `confirm` | replicate with new seeds (frozen configuration and the no-bias variant) | 2 | `confirm_main_seed2`, `confirm_nobias_seed2` |
| `ablation` | oracle position, earlier batch GNG, raw points, no width bias, earlier configuration | 1 | `abl_oracle_xy`, `abl_legacy_gng`, `abl_raw_points`, `abl_no_bias`, `abl_previous_config` |
| `posthoc` | local-slice grasp centre, finger gain 45000 (chosen after seeing the results; reported separately) | 1 | `posthoc_local_center`, `posthoc_kp45000` |
| `profile` | latency: 150 trials (30 per object) at P0, no viewer, no video | 3 | `profile_150` |
| `calibrate` | the P0-only calibration grids (long) | 700000+ | `calib_grid_*.csv` in the current folder |
| `tables` | regenerate the tables from `--out` | | `results_tables.md` |
| `all` | main, confirm, ablation, posthoc, profile, tables | | |

Run steps one after another, not in parallel (several YOLO processes can exhaust an 8 GB GPU).
Compare your tables with the published ones:

```bash
python make_results_tables.py --runs-dir pose_eval_runs --out pose_eval_runs/results_tables.md
diff pose_eval_runs/results_tables.md results/results_tables.md      # differences are expected only where your run differs
```

### 6.3 What to expect

* **Same machine, same software.** We re-ran the `main` run with the cleaned-up code of this repository on the paper machine: all 100 trials had the
  same success/failure outcome and the same detection result as the logged run (81/100); closing widths differed by 0.12 mm on average (largest 2.9 mm).
* **Other hardware or software versions.** For a given seed the simulation and the DBL-GNG fit are deterministic, but YOLO26 masks can differ slightly between GPUs,
  drivers and PyTorch versions, which moves the estimated widths by a few millimetres and can flip borderline trials (mostly the banana and the cracker box).
  Compare aggregates, not single trials. The two seeds of the paper gave 81/100 and 86/100 for the same configuration, so expect a spread of that size.
* **Latency** depends on your CPU and GPU, and it must be measured with `profile` (no viewer, no video). Timings in the logs of runs made with the viewer or the video
  recorder are inflated because they share the GPU.
* **The first YOLO call** takes a second or more (CUDA initialisation); the paper's latency table excludes it.

### 6.4 Re-selecting the placements and the parameters

```bash
python scenario_search.py              # tests 36 candidate placements at conf 0.05 (14 qualified in our check, P1-P3 among them;
                                       # borderline candidates depend on the machine); writes scenario_search_conf0.05.csv
python scenario_search.py --conf 0.25  # only 3 of the 36 qualify at YOLO's default confidence (cracker box rarely segmented)
scripts/reproduce_paper.sh calibrate   # width bias x grasp depth x finger gain at P0 (seeds disjoint from the evaluation)
```

---

## 7. Configuration reference

### 7.1 `run_pose_eval.py`

| Option | Meaning |
|---|---|
| `--scenarios P0 P1 ...` | placements to run (default: all four). **Use one process per placement** when the viewer or video is on (see Troubleshooting) |
| `--objects ...` | subset of `cracker_box mustard_bottle mug banana tennis_ball` |
| `--trials N` | trials per object and placement (default 5) |
| `--seed S` | evaluation seed; the per-trial seed is `S*100000 + placement_index*1000 + object_index*10 + trial`, with the object index taken in the order cracker_box, mustard_bottle, mug, banana, tennis_ball (so subset runs and `run_live_demo.py` reuse the paper's seeds) |
| `--out DIR` | run directory (`trials.csv`, `trials.jsonl`, `config_*.json`, `frames/`, videos); runs for several placements can share a directory |
| `--headless` | no MuJoCo viewer window |
| `--no-pace` | do not slow the viewer down to real time |
| `--no-video` | do not record the 1280x720 video |
| `--tag NAME` | name of the run directory when `--out` is not given |

`trials.csv` has one row per trial: scenario, object, trial, seed, jitter, head pan/tilt, `detected`, detector class/confidence, mask pixels, number of points and nodes,
epochs, node extents, approach, width, perceived centre, true position, `xy_err_mm`, heights before/after lifting and during the hold, the instantaneous and the
sustained lift criteria, finger contact, `success`, `reason`, and per-stage timings.

### 7.2 Environment variables (pipeline variants)

| Variable | Default | Meaning | Paper ablation |
|---|---|---|---|
| `MUJOCO_GL` | `egl` | offscreen rendering backend (`egl`, `glfw`, `osmesa`) | |
| `GNG_IMPL` | `paper` | `paper` = DBL-GNG of Siow et al. (`dbl_gng_paper.py`); `legacy` = earlier batch GNG with edge aging (`dbl_gng.py`); `raw` = raw mask points, no graph | C, D |
| `GNG_NODES`, `GNG_ETA`, `GNG_CUT` | 30, 4, 15 | max nodes M, start points eta, edge-cut percentile | |
| `GNG_SEED` | 0 | graph seed when not set by the evaluation | |
| `GNG_WIDTH_BIAS` | 0.010 | additive width bias b (m) | `0.0` = E, D |
| `GRASP_XY_SOURCE` | `perception` | `truth` uses the simulator's object position (oracle) | B |
| `GRASP_DEPTH_M` | 0.055 | grip site below the perceived top (m) | F uses 0.005 |
| `GRASP_FINGER_KP` | 30000 | finger servo gain | H uses 45000 |
| `GRASP_CENTER` | `bbox` | `local` centres the top-down grasp on the local slice | G |
| `LEGACY_OVERRIDES` | unset | `1` re-enables the earlier per-object squeeze overrides | F |
| `YOLO_WEIGHTS` | next to `yolo_segmenter.py` | path of `yolo26s-seg.pt` | |
| `GRASP_SCENARIO` | `P0` | placement used by `sim_core.Sim()` when none is given | |

Ablation F is `GNG_IMPL=legacy GRASP_XY_SOURCE=truth GRASP_DEPTH_M=0.005 GNG_WIDTH_BIAS=0.015 LEGACY_OVERRIDES=1`
(the configuration before this study). For example, ablation E in the live demo: `GNG_WIDTH_BIAS=0.0 python run_live_demo.py`.

---

## 8. Use the pipeline from your own Python code

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sim_core as sc, grasp_planner
from perception import Perceiver
from yolo_segmenter import YoloSegmenter

sim = sc.Sim("P2")                 # placements "P0".."P3" or dict(xy=(x, y), yaw=radians)
sim.step(300)                      # let the head actuators settle
per = Perceiver(sim, YoloSegmenter(0.05), conf_min=0.05)    # renders RGB-D, runs YOLO26 and one DBL-GNG per instance
best, instances = per.detect()     # best: {object name: detection with 'positions' (graph nodes), 'edges', 'conf', ...}
cand = grasp_planner.compute_grasp_candidate(best["mug"]["positions"])
print(cand["approach"], cand["width"], cand["region_min"], cand["region_max"])   # top 0.092 [0.952 0.409 0.723] [1.045 0.491 0.8]
held = sc.grasp_object_via_candidate(sim, "mug", cand)      # execute (top-down); held["success"] is the instantaneous lift check
```

The sustained-lift success criterion, jitter, noise and seeding of the paper are in `run_pose_eval.run_trial()`.
**Create every `mujoco.Renderer` (the `Perceiver` creates two) before you launch `mujoco.viewer.launch_passive`**, otherwise EGL fails
(`Failed to make the EGL context current`).

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `The virtual environment was not created successfully because ensurepip is not available` | `sudo apt install python3.10-venv`, delete `.venv`, create it again |
| `check_setup.py`: *CUDA GPU ... not available* | No NVIDIA GPU, or the driver is too old for the CUDA 13 wheel. YOLO then runs on the CPU (works, slower). To use the GPU install a PyTorch build matching your driver (section 2.3) |
| `RuntimeError: Failed to make the EGL context current` | (a) your own script created a `mujoco.Renderer` *after* the passive viewer - create all renderers first (section 8). (b) You ran several placements in one process together with the viewer/video: use one process per placement (`scripts/reproduce_paper.sh` does). (c) No EGL on the machine: install the NVIDIA driver and `libegl1`; or try `MUJOCO_GL=osmesa` (software, slow; `sudo apt install libosmesa6-dev`) |
| A long `EGLError ... EGL_NOT_INITIALIZED` traceback *after* the summary line, at exit | Harmless teardown noise of MuJoCo's EGL renderer; the run already finished and its logs are complete |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | OpenCV's window backend lacks a system library: `sudo apt install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0` |
| `cv2.error: ... The function is not implemented. Rebuild the library with ... GTK` | `opencv-python-headless` is installed next to `opencv-python`: `pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python==4.10.0.84` |
| No windows appear / `DISPLAY` not set | The live demo needs a desktop session. Over SSH use `ssh -X` (slow) or run headless (section 4.3). On Wayland sessions (XWayland) it should work; X11 is what we tested |
| `Xlib: extension "NV-GLX" missing on display` | Harmless message of the viewer on some NVIDIA setups |
| `CUDA error: out of memory` | Run one YOLO process at a time (an 8 GB GPU ran out of memory with several runs in parallel). Or force the CPU: `CUDA_VISIBLE_DEVICES="" python ...` |
| Weights cannot be downloaded (offline) | Section 2.4: download `yolo26s-seg.pt` elsewhere and put it next to `yolo_segmenter.py` or set `YOLO_WEIGHTS` |
| The demo is slow or jerky | Use a GPU; close other GPU programs; `--no-pace` removes the real-time limit; `--no-viewer` or `--no-windows` reduces the work |
| Your numbers differ slightly from the paper | Expected on other hardware/software (section 6.3). Compare aggregates and use both seeds |
| `ModuleNotFoundError: open3d` | Only needed for the optional Open3D helpers: `pip install -r requirements-optional.txt` |
| macOS / Windows | Not tested. On macOS the MuJoCo passive viewer needs `mjpython`; `run_pose_eval.py --headless` is the most likely to work |

---

## 10. Repository layout

```
run_live_demo.py        live demo: viewer + RGB / depth / live camera windows (this tutorial, section 4)
run_pose_eval.py        the paper's evaluation protocol (4 placements, jitter, noise, seeds, success criterion, video recorder)
sim_core.py             scene, robot placements P0-P3, per-trial reset, Cartesian control, grasp execution
perception.py           render RGB-D, YOLO26 masks, unprojection, one DBL-GNG per instance, object matching
dbl_gng_paper.py        DBL-GNG of Siow et al. 2024 (Algorithms 1-6), NumPy / CPU
dbl_gng.py              earlier batch GNG with edge aging (ablation baseline "C" only)
grasp_planner.py        geometric planner: extents, local closing width, approach decision, grasp region (paper Sec. IV)
gng_overlay.py          mask unprojection, graph factory make_gng(), optional visualisation helpers
yolo_segmenter.py       YOLO26s-seg wrapper (weights downloaded on first use)
viz.py                  overlays: 3D graphs and regions in the MuJoCo scene, annotated RGB and depth images
check_setup.py          installation self-test
analyze_pose_eval.py    per-run success table with Wilson intervals
analyze_perception_metrics.py   detection, centre error and width error against the simulator geometry
make_results_tables.py  regenerates results/results_tables.md (Tables V-VIII) from the logs
print_table1_params.py  prints the Table I values from the code
calibrate_p0.py, calibrate_grid.py   P0-only calibration sweeps
scenario_search.py      search over 36 candidate placements
make_scenario_schematic.py   draws docs/img/placements.png
scripts/reproduce_paper.sh   all experiment steps of section 6.2
scene.xml, assets/      MuJoCo scene, Fetch model, YCB objects (see docs/ASSETS.md)
results/paper_runs/     per-trial logs and run configurations of every run in the paper
results/calibration/    the P0 calibration sweeps
results/results_tables.md   tables generated from those logs
docs/                   this tutorial, images, asset notes
```

The `config_P*.json` file in each run directory records the parameters and the SHA-256 of the source files that produced it. The hashes in
`results/paper_runs/` belong to the code at the time of the run; the code in this repository was cleaned up afterwards (see the check in section 6.3).
