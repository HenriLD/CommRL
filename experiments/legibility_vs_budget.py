"""Does pragmatic reasoning still produce more LEGIBLE behavior at convergence?

Task reward is an instrument here, not the object of study. The claim is about
legibility, and the baseline may well converge to a coordinated solution of its
own -- but via an arbitrary convention its co-adapted partner has learned,
rather than a grounded signal any competent observer could read.

probe_acc is the right discriminator: a FIXED geometric decoder, external to
every training condition and never co-trained with anyone. A grounded signal
raises it; a private convention does not.

Reports probe accuracy and probe cross-entropy against training budget, so the
legibility claim can be evaluated separately from the task-reward claim.
"""
import glob
import json
import math
import os

import numpy as np

PER = 10
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load(cond, key, root="results_conv"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, cond + "_s*", "history.json"))):
        h = json.load(open(p))["history"]
        if key not in h[0]:
            return None
        runs.append([e[key] for e in h])
    if not runs:
        return None
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


def welch(a, b):
    return (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a)
                                             + b.var(ddof=1) / len(b))


CONDS = ["baseline", "oracle", "progress"]
WINDOWS = [("paper's budget (c300-400)", 300, 400),
           ("converged     (c800-1200)", 800, 1200)]

for key, label in (("probe_acc", "PROBE ACCURACY (fixed external geometric decoder)"),
                   ("probe_ce", "PROBE CROSS-ENTROPY (lower = more legible)")):
    data = {c: load(c, key) for c in CONDS}
    if any(v is None for v in data.values()):
        print("%s: not logged\n" % label)
        continue
    n = min(v.shape[1] for v in data.values())
    print("=" * 72)
    print(label)
    print("=" * 72)
    print("%-28s %10s %10s %10s   %s" % ("window", "baseline", "oracle",
                                         "progress", "progress vs base"))
    for lab, lo, hi in WINDOWS:
        vals = {}
        for c in CONDS:
            w = data[c][:, lo // PER:min(hi, n * PER) // PER]
            vals[c] = w.mean(axis=1)
        t = welch(vals["progress"], vals["baseline"])
        print("%-28s %10.4f %10.4f %10.4f   d=%+.4f  t=%5.2f"
              % (lab, vals["baseline"].mean(), vals["oracle"].mean(),
                 vals["progress"].mean(),
                 vals["progress"].mean() - vals["baseline"].mean(), t))
    # trajectory, coarse
    print("\n  trajectory (every 200 cycles):")
    print("  %-10s %s" % ("cycle", "  ".join("%7d" % c for c in
                                             range(200, n * PER + 1, 200))))
    for c in CONDS:
        pts = [data[c][:, max(0, cyc // PER - 10):cyc // PER].mean()
               for cyc in range(200, n * PER + 1, 200)]
        print("  %-10s %s" % (c, "  ".join("%7.4f" % v for v in pts)))
    print()
