"""Commitment *timing*, the mediator the referee round found missing.

Two referees independently observed that terminal commitment accuracy does not
track reward closure: progress closes 70% of the oracle premium on reward but
only ~34% of the oracle's commitment gap, and Filter+ear gains six times the
reward its commitment delta predicts under the oracle's own exchange rate. The
stated mechanism is about *when* the supporter knows, not whether it ends up
correct -- the team bonus accrues only while both agents occupy the site
together -- yet no timing metric appears anywhere in the paper.

This computes, per condition:
  t_half     first timestep whose commitment accuracy reaches halfway between
             chance (1/3) and that condition's own terminal accuracy
  early      mean commitment accuracy over the first 15 steps (pre-pickup,
             where the value-of-information weight is at full strength)
  final      accuracy at the last step

Runs entirely on CPU so it does not contend with training on the GPU.

Usage: python commit_timing.py
"""
import glob
import json
import os
import re

import numpy as np

from commit_curve import commit_curve

CHANCE = 1.0 / 3.0
EARLY = 15

CONDS = [
    ("baseline", "results_scout3/baseline_s*"),
    ("oracle", "results_scout3/oracle_s*"),
    ("progress", "results_scout3/progress_s*"),
    ("filter", "results_scout3/filter_s*"),
    ("learned", "results_scout3/learned_s*"),
    ("IPL+RSA", "results_ipl/iplprag06_s*"),
]


def t_half(curve):
    """First timestep reaching halfway from chance to this run's own plateau."""
    plateau = curve[-5:].mean()
    if plateau <= CHANCE:
        return np.nan
    target = (CHANCE + plateau) / 2.0
    idx = np.argmax(curve >= target)
    return float(idx + 1) if curve[idx] >= target else np.nan


def main():
    rows = []
    for name, pat in CONDS:
        runs = [d for d in sorted(glob.glob(pat))
                if os.path.exists(os.path.join(d, "model.pt"))]
        if not runs:
            print("  (no checkpoints for %s)" % name)
            continue
        curves = np.stack([commit_curve(d) for d in runs])
        th = np.array([t_half(c) for c in curves])
        early = curves[:, :EARLY].mean(axis=1)
        final = curves[:, -1]
        rows.append((name, len(runs), th, early, final))
        print("  rolled out %-9s n=%d" % (name, len(runs)), flush=True)

    def se(v):
        v = v[~np.isnan(v)]
        return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")

    print("\n%-10s %3s  %-16s %-16s %-16s"
          % ("condition", "n", "t_half (steps)", "early acc (<=15)", "final acc"))
    for name, n, th, early, final in rows:
        print("%-10s %3d  %6.1f +- %-7.1f %6.3f +- %-7.3f %6.3f +- %.3f"
              % (name, n, np.nanmean(th), se(th),
                 early.mean(), se(early), final.mean(), se(final)))

    # the mediation question: does timing track reward closure better than
    # terminal accuracy does?
    d = {r[0]: r for r in rows}
    if "baseline" in d and "oracle" in d:
        b, o = d["baseline"], d["oracle"]
        print("\nfraction of the oracle's gap closed, by measure:")
        print("%-10s %12s %12s %12s" % ("condition", "t_half", "early acc", "final acc"))
        for name, _, th, early, final in rows:
            if name in ("baseline", "oracle"):
                continue
            def frac(v_c, v_b, v_o):
                den = np.nanmean(v_o) - np.nanmean(v_b)
                return (np.nanmean(v_c) - np.nanmean(v_b)) / den if abs(den) > 1e-9 else float("nan")
            print("%-10s %11.0f%% %11.0f%% %11.0f%%"
                  % (name, 100 * frac(th, b[2], o[2]),
                     100 * frac(early, b[3], o[3]),
                     100 * frac(final, b[4], o[4])))

    out = {r[0]: {"n": r[1], "t_half": float(np.nanmean(r[2])),
                  "early": float(r[3].mean()), "final": float(r[4].mean())}
           for r in rows}
    with open("commit_timing.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote commit_timing.json")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
