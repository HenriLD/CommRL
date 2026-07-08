"""Aggregate the lambda ablation into a single figure.

Reads final performance from the main sweep (lambda=0.1) and the ablation
runs, and plots task reward and communicative reward against lambda for the
learned and exclusivity listeners, with the baseline as reference.

Usage: python ablation_plots.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def final(pattern, key, last_k=3):
    vals = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            h = json.load(f)["history"]
        vals.append(np.mean([e[key] for e in h[-last_k:]]))
    v = np.array(vals)
    return v.mean(), v.std() / np.sqrt(max(1, len(v)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    R = os.path.join(here, "results")
    A = os.path.join(here, "results_ablation")

    learned = {
        0.1: final(f"{R}/learned_s*/history.json", "r_ext"),
        0.3: final(f"{A}/learned_lam0.3/learned_s*/history.json", "r_ext"),
        1.0: final(f"{A}/learned_lam1.0/learned_s*/history.json", "r_ext"),
    }
    heuristic = {
        0.03: final(f"{A}/heuristic_lam0.03/heuristic_s*/history.json", "r_ext"),
        0.1: final(f"{R}/heuristic_s*/history.json", "r_ext"),
    }
    base_m, base_e = final(f"{R}/baseline_s*/history.json", "r_ext")

    from paperstyle import use_style
    use_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for d, label, color, marker in [
        (learned, "Learned listener", "#0072B2", "o"),
        (heuristic, "Exclusivity $L_0$", "#D55E00", "s"),
    ]:
        lams = sorted(d)
        m = np.array([d[l][0] for l in lams])
        e = np.array([d[l][1] for l in lams])
        ax.errorbar(lams, m, yerr=e, label=label, color=color,
                    marker=marker, ms=5, lw=2.0, capsize=3)
    ax.axhline(base_m, color="#666666", lw=1.8, ls="--",
               label="Baseline ($\\lambda=0$)")
    ax.axhspan(base_m - base_e, base_m + base_e, color="#666666", alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Communicative weight $\\lambda$")
    ax.set_ylabel("Final task reward $R_{ext}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(args.figdir, "lambda_ablation.png")
    fig.savefig(out)
    print("wrote", out)
    print("learned:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in learned.items()})
    print("heuristic:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in heuristic.items()})
    print("baseline:", round(base_m, 3), round(base_e, 3))


if __name__ == "__main__":
    main()
