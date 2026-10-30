#!/usr/bin/env bash
# Re-runs the experiments of the paper. Every step writes per-trial logs to $OUT/<run>/trials.csv
# (same layout as results/paper_runs/), one Python process per placement (see docs/TUTORIAL.md, Troubleshooting).
#
#   scripts/reproduce_paper.sh [options] STEP [STEP ...]
#
# STEPS
#   main        principal evaluation, seed 1                          (paper Table VI, 100 trials)
#   confirm     replicate with new seeds: frozen config + no-bias     (seed 2; Table VI pools main + seed 2)
#   ablation    oracle position, earlier batch GNG, raw points, no bias, earlier configuration  (Table VII, seed 1)
#   posthoc     post-hoc variants: local-slice grasp centre, finger gain 45000                    (seed 1)
#   profile     latency profiling, 150 trials = 30 per object at P0, headless (Table VIII)
#   calibrate   P0-only calibration grids (width bias x grasp depth x finger gain); long
#   tables      regenerate the result tables from $OUT with make_results_tables.py
#   all         main confirm ablation posthoc profile tables
#
# OPTIONS (before the steps)
#   --out DIR     output directory (default: pose_eval_runs)
#   --viewer      open the MuJoCo viewer during the runs (the paper's data collection was done like this;
#                 slower, paced to real time).  Default: --headless --no-pace (about 4 min per 100 trials)
#   --video       also record 1280x720 videos in the 'main' step (needs ffmpeg; implies --viewer pacing)
#
# Environment: activate your virtual environment first (e.g. `source .venv/bin/activate`); MUJOCO_GL defaults to egl.
# Run the steps one after another, not in parallel: parallel YOLO processes can run out of GPU memory.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"

OUT=pose_eval_runs
MODE="--headless --no-pace"
VIDEO="--no-video"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --viewer) MODE=""; shift ;;
    --video) VIDEO=""; MODE=""; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { sed -n '2,25p' "$0"; exit 1; }

POSES="P0 P1 P2 P3"

# run_set <run-name> <seed> <extra run_pose_eval args...>   (environment variables are inherited)
run_set () {
  local name="$1" seed="$2"; shift 2
  for P in $POSES; do
    echo "== $name  seed $seed  placement $P"
    python run_pose_eval.py --scenarios "$P" --out "$OUT/$name" --seed "$seed" "$@"
  done
}

step_main ()    { run_set main 1 $MODE $VIDEO; }
step_confirm () {
  run_set confirm_main_seed2 2 $MODE --no-video
  GNG_WIDTH_BIAS=0.0 run_set confirm_nobias_seed2 2 $MODE --no-video
}
step_ablation () {
  GRASP_XY_SOURCE=truth run_set abl_oracle_xy 1 $MODE --no-video                    # B  ground-truth grasp position
  GNG_IMPL=legacy       run_set abl_legacy_gng 1 $MODE --no-video                   # C  earlier batch GNG (edge aging)
  GNG_IMPL=raw GNG_WIDTH_BIAS=0.0 run_set abl_raw_points 1 $MODE --no-video         # D  raw mask points, no graph
  GNG_WIDTH_BIAS=0.0    run_set abl_no_bias 1 $MODE --no-video                      # E  no width bias (post hoc)
  GNG_IMPL=legacy GRASP_XY_SOURCE=truth GRASP_DEPTH_M=0.005 GNG_WIDTH_BIAS=0.015 LEGACY_OVERRIDES=1 \
                        run_set abl_previous_config 1 $MODE --no-video              # F  earlier configuration
}
step_posthoc () {
  GRASP_CENTER=local    run_set posthoc_local_center 1 $MODE --no-video             # G
  GRASP_FINGER_KP=45000 run_set posthoc_kp45000 1 $MODE --no-video                  # H
}
step_profile () {
  python run_pose_eval.py --scenarios P0 --trials 30 --seed 3 --out "$OUT/profile_150" --headless --no-video --no-pace
}
step_calibrate () {
  # P0 only, seeds 700000+ (disjoint from the evaluation seeds). Results: calib_grid_*.csv in the current directory.
  for b in 0.005 0.008 0.010 0.012 0.015 0.020; do
    python calibrate_grid.py --bias "$b"                                             # depth 5-35 mm x kp 30k/60k/120k
    python calibrate_grid.py --bias "$b" --depths 0.045 0.055 0.065 --kps 30000 --tag _deep   # deeper grasps
  done
}
step_tables ()  { python make_results_tables.py --runs-dir "$OUT" --out "$OUT/results_tables.md"; }

for s in "$@"; do
  case "$s" in
    main|confirm|ablation|posthoc|profile|calibrate|tables) "step_$s" ;;
    all) step_main; step_confirm; step_ablation; step_posthoc; step_profile; step_tables ;;
    *) echo "unknown step: $s" >&2; exit 1 ;;
  esac
done
