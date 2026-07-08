"""Aggregate results across seeds and produce paper figures + a summary table.

Usage: python plots.py --resroot results --figdir ../papers/Conference_Paper/img
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

ORDER = ["baseline", "oracle", "heuristic", "simple", "learned", "learned_prag"]


def load(resroot):
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(resroot, "*", "history.json"))):
        with open(path) as f:
            d = json.load(f)
        cond = d["args"]["condition"]
        data[cond].append(d["history"])
    return data


def series(histories, key):
    """Return (steps, mean, sem) across seeds for a metric key."""
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
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, fname))
    plt.close(fig)
    print("wrote", fname)


def final_table(data, keys, last_k=3):
    print("\n=== final performance (mean +/- sem over seeds, avg of last "
          f"{last_k} evals) ===")
    rows = {}
    for cond in ORDER:
        if cond not in data:
            continue
        row = {}
        for key in keys:
            vals = []
            for h in data[cond]:
                vals.append(np.mean([e[key] for e in h[-last_k:]]))
            vals = np.array(vals)
            row[key] = (vals.mean(), vals.std() / np.sqrt(len(vals)))
        rows[cond] = row
        print(cond.ljust(14), "  ".join(
            f"{key}={m:.3f}+/-{s:.3f}" for key, (m, s) in row.items()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results")
    p.add_argument("--figdir", default="figs")
    args = p.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    data = load(args.resroot)

    curve_plot(data, "r_ext", "Mean per-step task reward $R_{ext}$",
               "task_reward.png", args.figdir)
    curve_plot(data, "pref_bonus", "Mean per-step preference bonus",
               "pref_bonus.png", args.figdir)
    curve_plot(data, "spec", "Team specialization",
               "specialization.png", args.figdir)
    curve_plot(data, "probe_acc", "Probe intent accuracy",
               "probe_acc.png", args.figdir)
    curve_plot(data, "comm_rw", "Communicative reward $R_{comm}$",
               "comm_reward.png", args.figdir,
               conds=["heuristic", "simple", "learned", "learned_prag"])
    curve_plot(data, "listener_acc", "Listener accuracy $L_\\theta$",
               "listener_acc.png", args.figdir,
               conds=["learned", "learned_prag"])
    curve_plot(data, "dist_pen", "Mean per-step distance penalty",
               "dist_pen.png", args.figdir)
    curve_plot(data, "coll_pen", "Mean per-step collision penalty",
               "coll_pen.png", args.figdir)

    rows = final_table(data, ["r_ext", "pref_bonus", "spec", "probe_acc",
                              "comm_rw", "listener_acc", "dist_pen", "coll_pen"])
    with open(os.path.join(args.figdir, "final_table.json"), "w") as f:
        json.dump({c: {k: list(v) for k, v in r.items()} for c, r in rows.items()},
                  f, indent=1)


if __name__ == "__main__":
    main()
