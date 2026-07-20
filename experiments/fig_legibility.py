"""Legibility vs task reward, against training budget.

The paper's claim is about legibility, not task performance. Plotting the two
side by side separates them: the task-reward advantage decays as the baseline
finds its own way to coordinate, while readability by a fixed external decoder
-- one never co-trained with anybody -- keeps rising.

The oracle is the control that makes the probe interpretable: it has no reason
to signal, since its partner already knows the target, so its behaviour should
become LESS readable as it optimises purely for task efficiency.
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
LBL = {"baseline": "Baseline", "oracle": "Oracle (no need to signal)",
       "progress": "Progress $L_0$"}


def load(cond, key, root="results_conv"):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, cond + "_s*", "history.json"))):
        runs.append([e[key] for e in json.load(open(p))["history"]])
    n = min(len(r) for r in runs)
    return np.array([r[:n] for r in runs])


def panel(ax, key, ylabel, title, conds):
    data = {c: load(c, key) for c in conds}
    n = min(v.shape[1] for v in data.values())
    x = (np.arange(n) + 1) * PER
    for c in conds:
        v = data[c][:, :n]
        for s in v:
            ax.plot(x, s, color=COL[c], lw=0.5, alpha=0.20, zorder=1)
        m = v.mean(axis=0)
        se = v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
        # light smoothing for the mean only, so the trend is readable
        k = 5
        ms = np.convolve(m, np.ones(k) / k, mode="same")
        ms[:k], ms[-k:] = m[:k], m[-k:]
        ax.plot(x, ms, color=COL[c], lw=2.4, zorder=3, label=LBL[c])
        ax.fill_between(x, m - se, m + se, color=COL[c], alpha=0.15, lw=0, zorder=2)
    ax.axvline(400, color="#C0392B", lw=1.2, ls="--", zorder=4)
    ax.set_xlabel("Training budget (cycles)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="legibility.png")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conds = ["baseline", "oracle", "progress"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    panel(ax1, "r_ext", "Task reward $R_{ext}$",
          "Task reward: the advantage decays", conds)
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax1.text(410, ax1.get_ylim()[0] + 0.05, " paper's budget",
             color="#C0392B", fontsize=8)

    panel(ax2, "probe_acc", "Probe accuracy (fixed external decoder)",
          "Legibility: the gap widens", conds)
    ax2.text(410, 0.66, " paper's budget", color="#C0392B", fontsize=8)
    ax2.annotate("no incentive to signal", xy=(900, 0.474), xytext=(520, 0.43),
                 fontsize=8, color="#666666",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9))

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("wrote", a.out)


main()
