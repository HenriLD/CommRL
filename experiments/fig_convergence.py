"""Diagnostic view of the 1200-cycle convergence runs.

Left: task reward vs training budget, per-seed traces plus the seed mean, so
      instability, divergence, or a single collapsing seed is visible rather
      than averaged away.
Right: the oracle premium and the progress listener's capture of it, as a
      function of where you stop training.

Usage: python fig_convergence.py [--out convergence.png]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER = 10
COL = {"baseline": "#7F7F7F", "oracle": "#111111", "progress": "#0072B2"}
LBL = {"baseline": "Baseline", "oracle": "Oracle", "progress": "Progress $L_0$"}


def load(cond, root="results_conv"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, cond + "_s*", "history.json"))):
        runs.append([e["r_ext"] for e in json.load(open(p))["history"]])
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="convergence.png")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    data = {c: load(c) for c in ("baseline", "oracle", "progress")}
    n = min(v.shape[1] for v in data.values())
    x = (np.arange(n) + 1) * PER

    fig, (ax, ax2, ax4) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # ---- left: learning curves, per-seed + mean --------------------------
    for c, v in data.items():
        v = v[:, :n]
        for s in v:                                  # per-seed traces
            ax.plot(x, s, color=COL[c], lw=0.6, alpha=0.28, zorder=1)
        m = v.mean(axis=0)
        se = v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
        ax.plot(x, m, color=COL[c], lw=2.4, zorder=3, label=LBL[c])
        ax.fill_between(x, m - se, m + se, color=COL[c], alpha=0.18, lw=0, zorder=2)

    ax.axvline(400, color="#C0392B", lw=1.2, ls="--", zorder=4)
    ax.text(410, ax.get_ylim()[0] + 0.03, " paper's budget", color="#C0392B",
            fontsize=8, va="bottom")
    ax.set_xlabel("Training budget (cycles)")
    ax.set_ylabel("Task reward $R_{ext}$ (deterministic eval)")
    ax.set_title("Learning curves, 8 seeds each", fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.15)

    # ---- right: premium and capture vs where you stop --------------------
    b, o, p = (data[c][:, :n] for c in ("baseline", "oracle", "progress"))
    prem = o.mean(axis=0) - b.mean(axis=0)
    cap = (p.mean(axis=0) - b.mean(axis=0)) / np.maximum(prem, 1e-9)

    # separate panels, never a second y-scale: the two measures have
    # different units and a twin axis would let the eye compare them falsely.
    ax2.plot(x, prem, color="#111111", lw=2.2)
    ax2.set_xlabel("Training budget (cycles)")
    ax2.set_ylabel("Oracle premium (reward/step)")
    ax2.axvline(400, color="#C0392B", lw=1.2, ls="--")
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_title("The premium is mostly a transient", fontsize=10)
    ax2.grid(alpha=0.15)
    ax2.annotate("baseline still\nnear its floor", xy=(90, prem[8]),
                 xytext=(300, 1.15), fontsize=8, color="#666666",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9))
    ax2.annotate("asymptote $\\approx$0.07", xy=(1000, prem[99]),
                 xytext=(620, 0.42), fontsize=8, color="#666666",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9))

    # capture is a ratio, so it is only interpretable once the denominator
    # has settled; grey out the region where it has not.
    ax4.axvspan(0, 200, color="#DDDDDD", alpha=0.6, lw=0)
    ax4.text(100, 1.32, "ratio\nmeaningless", fontsize=7.5, color="#777777",
             ha="center", va="top")
    ax4.plot(x, np.clip(cap, -0.5, 1.5), color="#0072B2", lw=2.2)
    ax4.axvline(400, color="#C0392B", lw=1.2, ls="--")
    ax4.axhline(0, color="#888888", lw=0.8)
    ax4.set_ylim(-0.5, 1.5)
    ax4.set_xlabel("Training budget (cycles)")
    ax4.set_ylabel("Fraction of premium captured")
    ax4.set_title("Progress $L_0$: capture falls after the budget", fontsize=10)
    ax4.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("wrote", a.out)

    # ---- text diagnostics ------------------------------------------------
    print("\nper-seed final reward (last 40 cycles), to expose collapses:")
    for c, v in data.items():
        f = v[:, -4:].mean(axis=1)
        print("  %-9s %s" % (c, "  ".join("%.3f" % z for z in np.sort(f))))
    print("\nseed-level spread at 1200: baseline %.3f, oracle %.3f, progress %.3f"
          % tuple(data[c][:, -4:].mean(axis=1).std(ddof=1)
                  for c in ("baseline", "oracle", "progress")))


main()
