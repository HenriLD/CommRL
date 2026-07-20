"""Is the widening spread in capture real seed noise, or a ratio artifact?

Two hypotheses make different predictions:

  H1 (SAC noise grows)   the ABSOLUTE per-seed spread in task reward should be
                         larger at the long budget.
  H2 (ratio artifact)    absolute spread stays put; the capture ratio's spread
                         inflates purely because the denominator (the premium)
                         shrank. Predicted sd = sd(absolute gain) / premium.

Also separates across-seed spread from within-run temporal wobble, so genuine
late-training drift would show up rather than hiding inside the seed spread.
"""
import glob
import json
import os

import numpy as np

PER = 10
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load(cond, root="results_conv"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, cond + "_s*", "history.json"))):
        runs.append([e["r_ext"] for e in json.load(open(p))["history"]])
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


b, o, p = load("baseline"), load("oracle"), load("progress")
n = min(b.shape[1], o.shape[1], p.shape[1])
b, o, p = b[:, :n], o[:, :n], p[:, :n]

WINDOWS = [("paper's budget (c300-400)", 300, 400),
           ("converged     (c800-1200)", 800, 1200)]


def win(a, lo, hi):
    return a[:, lo // PER:hi // PER]


print("ACROSS-SEED sd of task reward (the quantity SAC noise would inflate)")
print("%-28s %9s %9s %9s" % ("window", "baseline", "oracle", "progress"))
for lab, lo, hi in WINDOWS:
    sds = [win(a, lo, hi).mean(axis=1).std(ddof=1) for a in (b, o, p)]
    print("%-28s %9.4f %9.4f %9.4f" % (lab, *sds))

print("\nWITHIN-RUN temporal sd across evaluations inside the window")
print("(late-training drift or eval noise would show up here)")
print("%-28s %9s %9s %9s" % ("window", "baseline", "oracle", "progress"))
for lab, lo, hi in WINDOWS:
    sds = [win(a, lo, hi).std(axis=1, ddof=1).mean() for a in (b, o, p)]
    print("%-28s %9.4f %9.4f %9.4f" % (lab, *sds))

print("\nCAPTURE RATIO: observed spread vs what the shrinking denominator alone predicts")
print("%-28s %8s %8s %10s %10s" % ("window", "premium", "sd(gain)",
                                   "predicted", "observed"))
for lab, lo, hi in WINDOWS:
    bb, oo, pp = (win(a, lo, hi).mean(axis=1) for a in (b, o, p))
    prem = oo.mean() - bb.mean()
    gain = pp - bb.mean()                 # absolute per-seed gain
    predicted = gain.std(ddof=1) / prem   # H2's prediction
    observed = (gain / prem).std(ddof=1)
    print("%-28s %8.4f %8.4f %10.3f %10.3f"
          % (lab, prem, gain.std(ddof=1), predicted, observed))

bb1, pp1 = (win(a, 300, 400).mean(axis=1) for a in (b, p))
bb2, pp2 = (win(a, 800, 1200).mean(axis=1) for a in (b, p))
g1, g2 = pp1 - bb1.mean(), pp2 - bb2.mean()
print("\nsd of the ABSOLUTE gain: %.4f -> %.4f  (x%.2f)"
      % (g1.std(ddof=1), g2.std(ddof=1), g2.std(ddof=1) / g1.std(ddof=1)))
pr1 = win(o, 300, 400).mean() - bb1.mean()
pr2 = win(o, 800, 1200).mean() - bb2.mean()
print("premium:                %.4f -> %.4f  (x%.2f)" % (pr1, pr2, pr2 / pr1))
print("=> ratio sd should scale by %.2f / %.2f = x%.2f if it is pure artifact"
      % (g2.std(ddof=1) / g1.std(ddof=1), pr2 / pr1,
         (g2.std(ddof=1) / g1.std(ddof=1)) / (pr2 / pr1)))
print("   observed ratio sd scaling:                              x%.2f"
      % ((g2 / pr2).std(ddof=1) / (g1 / pr1).std(ddof=1)))
