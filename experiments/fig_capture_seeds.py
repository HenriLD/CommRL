"""Per-seed capture of the oracle premium -- no aggregate line standing in for
eight runs.

The capture fraction is a ratio, so what its denominator is matters. Primary
definition keeps the denominator a population quantity and lets only the
numerator vary by seed:

    capture_i(t) = (progress_i(t) - mean_baseline(t)) / (mean_oracle(t) - mean_baseline(t))

A fully paired alternative (seed i against baseline_i and oracle_i) is also
computed as a robustness check; it is noisier because each seed's premium is a
small difference of two noisy quantities, but if the two disagree about the
trend that is worth knowing.

Usage: python fig_capture_seeds.py [--out capture_seeds.png]
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
BLUE = "#0072B2"
INK = "#222222"


def load(cond, root="results_conv"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, cond + "_s*", "history.json"))):
        runs.append([e["r_ext"] for e in json.load(open(p))["history"]])
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="capture_seeds.png")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    b, o, p = load("baseline"), load("oracle"), load("progress")
    n = min(b.shape[1], o.shape[1], p.shape[1])
    b, o, p = b[:, :n], o[:, :n], p[:, :n]
    x = (np.arange(n) + 1) * PER

    prem = o.mean(axis=0) - b.mean(axis=0)                 # population premium
    cap = (p - b.mean(axis=0)) / np.maximum(prem, 1e-9)    # per seed

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                  gridspec_kw={"width_ratios": [1.6, 1]})

    # ---------- left: every seed's trajectory --------------------------
    ax.axvspan(0, 200, color="#DDDDDD", alpha=0.6, lw=0)
    ax.text(100, 1.42, "premium still\ncollapsing:\nratio meaningless",
            fontsize=7.5, color="#777777", ha="center", va="top")
    for s in cap:
        ax.plot(x, np.clip(s, -0.6, 1.6), color=BLUE, lw=0.9, alpha=0.45)
    med = np.median(cap, axis=0)
    ax.plot(x, np.clip(med, -0.6, 1.6), color=INK, lw=2.4, label="median seed")
    ax.axvline(400, color="#C0392B", lw=1.2, ls="--")
    ax.text(408, -0.5, " paper's budget", color="#C0392B", fontsize=8)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Training budget (cycles)")
    ax.set_ylabel("Fraction of oracle premium captured")
    ax.set_title("Every seed, not the average of them (n=8)", fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(alpha=0.15)

    # ---------- right: each seed, budget vs converged -------------------
    def win(arr, lo, hi):
        return arr[:, lo // PER:hi // PER].mean(axis=1)

    pr_a = win(o, 300, 400).mean() - win(b, 300, 400).mean()
    pr_c = win(o, 800, 1200).mean() - win(b, 800, 1200).mean()
    ca = (win(p, 300, 400) - win(b, 300, 400).mean()) / pr_a
    cc = (win(p, 800, 1200) - win(b, 800, 1200).mean()) / pr_c

    for i, (u, v) in enumerate(zip(ca, cc)):
        ax2.plot([0, 1], [u, v], color=BLUE, lw=1.1, alpha=0.55,
                 marker="o", ms=5, mec="white", mew=0.7)
    ax2.plot([0, 1], [np.median(ca), np.median(cc)], color=INK, lw=2.6,
             marker="o", ms=8, mec="white", mew=1.0, zorder=5, label="median")
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["paper's budget\n(c300-400)", "converged\n(c800-1200)"],
                        fontsize=9)
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_ylabel("Fraction of premium captured")
    ax2.set_title("Per-seed change", fontsize=10)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("wrote", a.out)

    # ---------- numbers ------------------------------------------------
    print("\nper-seed capture (sorted):")
    print("  paper's budget : %s" % "  ".join("%+.2f" % v for v in np.sort(ca)))
    print("  converged      : %s" % "  ".join("%+.2f" % v for v in np.sort(cc)))
    print("\n  median %.2f -> %.2f | mean %.2f -> %.2f | sd %.2f -> %.2f"
          % (np.median(ca), np.median(cc), ca.mean(), cc.mean(),
             ca.std(ddof=1), cc.std(ddof=1)))
    print("  seeds that fell: %d/%d" % (int((cc < ca).sum()), len(ca)))
    print("  seeds above zero: %d/%d -> %d/%d"
          % (int((ca > 0).sum()), len(ca), int((cc > 0).sum()), len(cc)))

    # paired robustness check
    pa = (win(p, 300, 400) - win(b, 300, 400)) / (win(o, 300, 400) - win(b, 300, 400))
    pc = (win(p, 800, 1200) - win(b, 800, 1200)) / (win(o, 800, 1200) - win(b, 800, 1200))
    print("\nfully paired (seed i vs its own baseline_i, oracle_i) -- noisier:")
    print("  median %.2f -> %.2f" % (np.median(pa), np.median(pc)))


main()
