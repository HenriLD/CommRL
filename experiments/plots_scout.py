"""Aggregate scout-support results into paper figures and a final table.

Usage: python plots_scout.py --resroot results_scout --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperstyle import use_style, plot_series, LABELS, COLORS

ORDER = ["baseline", "oracle", "simple", "exclusivity", "progress", "filter",
         "learned", "learned_prag", "ear", "learned_ear", "filter_ear"]


def load(resroot):
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(resroot, "*", "history.json"))):
        with open(path) as f:
            d = json.load(f)
        data[d["args"]["condition"]].append(d["history"])
    return data


def series(histories, key):
    n = min(len(h) for h in histories)
    steps = np.array([h["steps"] for h in histories[0][:n]])
    vals = np.array([[e[key] for e in h[:n]] for h in histories])
    return steps, vals.mean(axis=0), vals.std(axis=0) / np.sqrt(max(1, vals.shape[0]))


def curve_plot(data, key, ylabel, fname, figdir, conds=None):
    use_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for cond in (conds or ORDER):
        if cond not in data:
            continue
        s, m, e = series(data[cond], key)
        plot_series(ax, s / 1e3, m, e, cond)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, fname))
    plt.close(fig)
    print("wrote", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results_scout")
    p.add_argument("--figdir", default="figs_scout")
    args = p.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    data = load(args.resroot)

    curve_plot(data, "r_ext", "Mean per-step task reward $R_{ext}$",
               "scout_task_reward.png", args.figdir)
    curve_plot(data, "commit_acc", "Supporter commitment accuracy",
               "scout_commit.png", args.figdir)
    curve_plot(data, "pref_bonus", "Mean per-step rendezvous bonus",
               "scout_bonus.png", args.figdir)
    curve_plot(data, "comm_rw", "Communicative reward $R_{comm}$",
               "scout_comm_reward.png", args.figdir,
               conds=["simple", "exclusivity", "progress", "filter",
                      "learned", "learned_prag"])
    curve_plot(data, "listener_acc", "Listener accuracy $L_\\theta$",
               "scout_listener_acc.png", args.figdir,
               conds=["learned", "learned_prag"])

    keys = ["r_ext", "pref_bonus", "commit_acc", "probe_acc", "comm_rw",
            "listener_acc"]
    print("\n=== final performance (mean +/- sem, last 3 evals) ===")
    rows = {}
    for cond in ORDER:
        if cond not in data:
            continue
        row = {}
        for key in keys:
            vals = np.array([np.mean([e[key] for e in h[-3:]]) for h in data[cond]])
            row[key] = (vals.mean(), vals.std() / np.sqrt(len(vals)))
        rows[cond] = row
        print(cond.ljust(13), "  ".join(f"{k}={m:.3f}+/-{s:.3f}" for k, (m, s) in row.items()))
    with open(os.path.join(args.figdir, "final_table_scout.json"), "w") as f:
        json.dump({c: {k: list(v) for k, v in r.items()} for c, r in rows.items()}, f, indent=1)


if __name__ == "__main__":
    main()
