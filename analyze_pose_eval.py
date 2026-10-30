#!/usr/bin/env python3
"""Summarises pose_eval_runs/<run>/trials.csv: success per scenario x object with
Wilson 95% CIs, detection rate, failure reasons, xy error and stage latencies.
Usage: python analyze_pose_eval.py pose_eval_runs/<run> [more runs...]"""
import collections
import csv
import sys

import numpy as np

OBJS = ["cracker_box", "mustard_bottle", "mug", "banana", "tennis_ball"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f})" if n else "-"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    for run in sys.argv[1:]:
        rows = list(csv.DictReader(open(f"{run}/trials.csv")))
        print(f"\n===== {run}  ({len(rows)} trials)")
        scen = sorted({r["scenario"] for r in rows})
        print(f"{'':16s}" + "".join(f"{s:>28s}" for s in scen) + f"{'all poses':>28s}")
        for o in OBJS + ["ALL"]:
            line = f"{o:16s}"
            tk = tn = 0
            for s in scen:
                sel = [r for r in rows if r["scenario"] == s and (o == "ALL" or r["object"] == o)]
                k = sum(r["success"] == "True" for r in sel)
                tk += k
                tn += len(sel)
                line += f"{fmt(k, len(sel)):>28s}"
            print(line + f"{fmt(tk, tn):>28s}")
        det = [r for r in rows if r["detected"] == "True"]
        print(f"detected: {len(det)}/{len(rows)};  success | detected: {fmt(sum(r['success'] == 'True' for r in det), len(det))}")
        inst = sum(r["lift_ok_instant"] == "True" for r in rows)
        print(f"instantaneous-lift criterion (earlier benchmarks): {fmt(inst, len(rows))}  vs hold-2s criterion above")
        print("failure reasons:", dict(collections.Counter(r["reason"] for r in rows if r["success"] != "True")))
        xe = [f(r["xy_err_mm"]) for r in rows if f(r["xy_err_mm"]) is not None]
        if xe:
            print(f"perceived-centre xy error vs truth: mean {np.mean(xe):.1f} mm, median {np.median(xe):.1f}, p95 {np.percentile(xe, 95):.1f}")
        for key, name in (("render_ms", "RGB-D render"), ("yolo_ms", "YOLO26-seg"), ("plan_ms", "planner")):
            v = [f(r[key]) for r in rows if f(r[key]) is not None]
            if v:
                print(f"  {name:14s} mean {np.mean(v):7.2f} ms  (n={len(v)})")
        g = [(f(r["gng_ms"]), f(r["n_instances"])) for r in rows if f(r["gng_ms"]) and f(r["n_instances"])]
        if g:
            print(f"  GNG fit        mean {np.mean([a / b for a, b in g]):7.2f} ms per instance  (frame total {np.mean([a for a, b in g]):.1f} ms)")


if __name__ == "__main__":
    main()
