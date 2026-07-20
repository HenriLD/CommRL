"""Does the oracle premium survive a converged budget?

Every referee raised the same objection: at the paper's 400-cycle budget the
oracle had plateaued while the baseline was still climbing (t=3.9), and the
premium contracted 0.149 -> 0.100 over cycles 320-400. Extrapolating that slope
exhausts the premium near cycle 565, which would make every closure fraction in
the paper an artifact of when training stopped.

This reads the 1200-cycle runs and reports the premium as a function of budget,
plus terminal slopes, so the asymptote is measured rather than extrapolated.
"""
import glob
import json
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT = "results_conv"
PER = 10          # history is logged every 10 cycles


def curves(cond):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, cond + "_s*", "history.json"))):
        h = json.load(open(p))["history"]
        out.append([e["r_ext"] for e in h])
    n = min(len(c) for c in out)
    return np.array([c[:n] for c in out])


def at(curve, cycle, k=3):
    """Mean over the k logged points ending at `cycle`."""
    i = cycle // PER
    return curve[:, max(0, i - k):i].mean(axis=1)


b, o, p = curves("baseline"), curves("oracle"), curves("progress")
n_cycles = min(b.shape[1], o.shape[1], p.shape[1]) * PER
print("completed budget: %d cycles (%.2fM steps), n=%d/%d/%d seeds\n"
      % (n_cycles, n_cycles * 3200 / 1e6, len(b), len(o), len(p)))

print("%8s %10s %10s %10s %10s" % ("cycle", "baseline", "oracle", "premium",
                                   "progress"))
budgets = [c for c in (400, 600, 800, 1000, 1200) if c <= n_cycles]
prem = {}
for c in budgets:
    bb, oo, pp = at(b, c), at(o, c), at(p, c)
    prem[c] = oo.mean() - bb.mean()
    clo = (pp.mean() - bb.mean()) / prem[c] if abs(prem[c]) > 1e-9 else np.nan
    print("%8d %10.4f %10.4f %10.4f %9.0f%%"
          % (c, bb.mean(), oo.mean(), prem[c], 100 * clo))

print("\npremium trajectory: %s"
      % "  ".join("%d:%.4f" % (c, prem[c]) for c in budgets))


def slope(curve, lo, hi):
    """Per-100-cycle slope over [lo, hi], with a Welch t across seeds."""
    a, z = at(curve, lo), at(curve, hi)
    d = (z - a) / ((hi - lo) / 100.0)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return d.mean(), t


print("\nterminal slopes over the last 400 cycles (per 100 cycles):")
lo, hi = max(budgets) - 400, max(budgets)
for name, c in (("baseline", b), ("oracle", o), ("progress", p)):
    m, t = slope(c, lo, hi)
    print("  %-9s %+.4f  (t=%.2f)%s" % (name, m, t,
                                        "  <- still climbing" if t > 2 else ""))

pm, pt = slope(b, lo, hi), None
print("\npaper reported at 400 cycles: baseline slope +0.054/100 (t=3.9), "
      "premium 0.149->0.100")
