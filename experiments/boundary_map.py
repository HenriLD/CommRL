"""Synthesize the boundary map: pragmatic gain as a function of the oracle
premium, across every environment and setting we ran.

For each (results-root, tag) it computes
    premium = oracle - baseline      (what early information is worth)
    gain    = progress - baseline    (what the recipe recovers)
and fits gain = k * premium through the origin. Points that fall off that
line mark the boundary conditions.

Usage: python boundary_map.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from paperstyle import use_style

# (label, resroot, prefix, recipe-condition, marker-group)
POINTS = [
    ("Scout-support",      "results_scout3", "",      "progress",     "core"),
    ("5 meanings",         "results_suite",  "k5_",   "progress",     "core"),
    ("Minefield",          "results_suite",  "mine_", "progress",     "core"),
    ("Partner 0.45",       "results_boundary", "sp045_", "progress",  "slow"),
    ("Partner 0.75",       "results_boundary", "sp075_", "progress",  "core"),
    ("Partner 0.90",       "results_boundary", "sp090_", "progress",  "core"),
    ("Signal cost 2x",     "results_boundary", "sw2_",  "progress",   "cost"),
    ("Signal cost 4x",     "results_boundary", "sw4_",  "progress",   "cost"),
    ("Pref. spread (Env B)", "results",      "",      "learned_prag", "nogap"),
]
COLORS = {"core": "#0072B2", "slow": "#D55E00", "cost": "#E69F00",
          "nogap": "#666666"}


def agg(resroot, cond, key="r_ext"):
    here = os.path.dirname(os.path.abspath(__file__))
    pat = os.path.join(here, resroot, f"{cond}_s*", "history.json")
    v = []
    for p in sorted(glob.glob(pat)):
        h = json.load(open(p))["history"][-3:]
        v.append(np.mean([e[key] for e in h]))
    return np.array(v)


def sem(v):
    return v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    args = p.parse_args()

    rows = []
    for label, root, pre, recipe, grp in POINTS:
        b = agg(root, f"{pre}baseline")
        o = agg(root, f"{pre}oracle")
        g = agg(root, f"{pre}{recipe}")
        if not len(b) or not len(o) or not len(g):
            print(f"skip {label} (missing runs)")
            continue
        prem, gain = o.mean() - b.mean(), g.mean() - b.mean()
        prem_e = math.sqrt(sem(o) ** 2 + sem(b) ** 2)
        gain_e = math.sqrt(sem(g) ** 2 + sem(b) ** 2)
        t = gain / gain_e if gain_e else 0
        rows.append(dict(label=label, grp=grp, prem=prem, gain=gain,
                         prem_e=prem_e, gain_e=gain_e, t=t,
                         n=min(len(b), len(o), len(g))))

    print(f"{'setting':22s} {'n':>2s} {'premium':>14s} {'gain':>14s} {'ratio':>6s} {'t':>5s}")
    for r in rows:
        ratio = f"{r['gain']/r['prem']:.2f}" if abs(r["prem"]) > 3 * r["prem_e"] else "  n/a"
        print(f"{r['label']:22s} {r['n']:2d} {r['prem']:7.3f}+/-{r['prem_e']:.3f} "
              f"{r['gain']:7.3f}+/-{r['gain_e']:.3f} {ratio:>6s} {r['t']:5.2f}")

    # fit gain = k * premium through the origin, using only settings whose
    # premium is itself resolvable (>3 sem) -- elsewhere the ratio is undefined
    fit = [r for r in rows if r["prem"] > 3 * r["prem_e"]]
    x = np.array([r["prem"] for r in fit])
    y = np.array([r["gain"] for r in fit])
    k = float((x * y).sum() / (x * x).sum())
    print(f"\nproportional fit over {len(fit)} resolvable settings: gain = {k:.2f} x premium")

    LABEL_OFF = {
        "Scout-support": (-14, 10), "Minefield": (8, 3),
        "5 meanings": (7, 6), "Partner 0.75": (9, 6),
        "Partner 0.90": (9, 0), "Partner 0.45": (6, 6),
        "Signal cost 2x": (-16, -15),
    }
    use_style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 3.5), gridspec_kw=dict(width_ratios=[2.6, 1]))

    # --- left: the law, over the range where a premium is resolvable ---
    resolv = [r for r in rows if r["prem"] > 3 * r["prem_e"]]
    unres = [r for r in rows if r["prem"] <= 3 * r["prem_e"]]
    xs = np.linspace(0, 0.175, 50)
    ax.plot(xs, xs, color="#aaaaaa", lw=0.9, ls=":", zorder=1)
    ax.text(0.098, 0.094, "oracle", fontsize=6.8, color="#888888",
            ha="right", va="bottom", rotation=34)
    ax.plot(xs, k * xs, color="#111111", lw=1.6, ls="--", zorder=2,
            label=f"gain = {k:.2f} $\\times$ premium")
    seen = set()
    for r in resolv:
        c = COLORS[r["grp"]]
        lbl = {"core": "recipe captures $\\approx$ half",
               "slow": "partner too slow", "cost": "signal too costly"}[r["grp"]]
        ax.errorbar(r["prem"], r["gain"], xerr=r["prem_e"], yerr=r["gain_e"],
                    fmt="o", ms=6.5, color=c, capsize=2.5, lw=1.4, zorder=3,
                    label=lbl if lbl not in seen else None)
        seen.add(lbl)
        ax.annotate(r["label"], (r["prem"], r["gain"]), fontsize=6.4,
                    xytext=LABEL_OFF.get(r["label"], (5, 6)),
                    textcoords="offset points", color="#333333", zorder=4)
    ax.axhline(0, color="#cccccc", lw=0.7, zorder=0)
    ax.set_xlabel("Oracle premium (reward/step)")
    ax.set_ylabel("Pragmatic gain (reward/step)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    ax.set_xlim(0, 0.19)
    ax.set_ylim(-0.005, 0.122)
    ax.set_title("Where a premium exists", fontsize=9)

    # --- right: the categorical boundary -- nothing to win, or no channel ---
    cats = [(r["label"].replace(" (Env B)", ""), r["gain"], r["gain_e"], "#666666")
            for r in unres]
    blind_b = agg("results_suite", "blind_baseline")
    blind_p = agg("results_suite", "blind_progress")
    if len(blind_b) and len(blind_p):
        cats.append(("Channel severed", blind_p.mean() - blind_b.mean(),
                     math.sqrt(sem(blind_p) ** 2 + sem(blind_b) ** 2), "#D55E00"))
    ys = np.arange(len(cats))[::-1]
    for y, (lab, g, e, c) in zip(ys, cats):
        ax2.errorbar(g, y, xerr=e, fmt="o", ms=6.5, color=c, capsize=2.5, lw=1.4)
    ax2.axvline(0, color="#999999", lw=0.9, ls="--")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([c[0] for c in cats], fontsize=7.2)
    ax2.set_ylim(-0.6, len(cats) - 0.4)
    ax2.set_xlabel("Pragmatic gain")
    ax2.set_title("No premium / no channel", fontsize=9)
    fig.tight_layout()
    out = os.path.join(args.figdir, "boundary_map.png")
    os.makedirs(args.figdir, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
