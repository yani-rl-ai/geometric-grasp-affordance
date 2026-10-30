#!/usr/bin/env python3
"""Prints every value reported in Table I straight from the code, with its source location, so the table can be audited.
Usage: python print_table1_params.py"""
import inspect
import os

import dbl_gng_paper as dp
import gng_overlay as go

g = go.make_gng(seed=0)                      # exactly what the experiments instantiate (GNG_IMPL=paper default)
sig = inspect.signature(dp.PaperDBLGNG.__init__)
print("class:", type(g).__name__, " file:", os.path.basename(dp.__file__))
rows = [
    ("Maximum nodes M", g.max_nodes, "make_gng(): GNG_NODES default 30"),
    ("Start points eta", g.n_start, "make_gng(): GNG_ETA default 4"),
    ("Winner rate alpha", g.alpha, "PaperDBLGNG.__init__ default"),
    ("Neighbor rate beta", g.beta, "PaperDBLGNG.__init__ default"),
    ("New-node error discount rho", g.rho, "PaperDBLGNG.__init__ default"),
    ("Global error discount delta", g.delta, "PaperDBLGNG.__init__ default"),
    ("Growth percentile", g.grow_pct, "PaperDBLGNG.__init__ default"),
    ("Edge-cut percentile", g.cut_pct, "make_gng(): GNG_CUT default 15"),
    ("Inactive-node removal probability", g.inactive_prob, "PaperDBLGNG.__init__ default"),
    ("epsilon", g.eps, "PaperDBLGNG.__init__ default"),
    ("Extra epochs after M reached", g.extra_epochs, "PaperDBLGNG.__init__ default"),
    ("Epoch cap", g.max_epochs, "PaperDBLGNG.__init__ default"),
    ("Edge cut enabled", g.do_cut, "PaperDBLGNG.__init__ default"),
    ("Maximum input points", go.MAX_GNG_POINTS, "gng_overlay.MAX_GNG_POINTS"),
    ("Minimum mask pixels", go.MIN_MASK_PIXELS, "gng_overlay.MIN_MASK_PIXELS"),
    ("Median-depth rejection (m)", go.OUTLIER_THRESH_M, "gng_overlay.OUTLIER_THRESH_M"),
]
for name, val, src in rows:
    print(f"{name:36s} {str(val):>8s}   <- {src}")
print("env overrides currently set:", {k: v for k, v in os.environ.items() if k.startswith(("GNG_", "GRASP_"))} or "none")
