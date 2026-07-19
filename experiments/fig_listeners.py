"""The listener-space result as one figure, replacing the old forest plot and
the full Environment A table.

Design notes (why it looks like this):
  * per-seed dots, not just mean +- s.e.m. At n=8-10 a summary interval hides
    the distribution; the exclusivity listener is bimodal (seven seeds high,
    one collapsed) and a symmetric whisker misrepresents that as vague
    uncertainty rather than an occasional failure.
  * one ink colour for every condition, with emphasis reserved for the two
    rows the argument turns on. Ten categorical hues carried no identity load.
  * sorted by effect, grouped by listener family, so the reader can rank
    without reading numbers.
  * baseline and oracle are references, not series: hairline rules.

Usage: python fig_listeners.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from paperstyle import use_style

INK = "#4C4C4C"          # every condition, unless emphasised
HI = "#0072B2"           # the two rows the argument turns on
REF_B, REF_O = "#888888", "#111111"
XLO = -1.05          # left clip; seeds beyond it are marked, not dropped

# (label, glob, emphasise)
ROWS = [
    ("Progress $L_0$",                 "results_scout3/progress_s*",        False),
    ("Filter $L_0$",                   "results_scout3/filter_s*",          False),
    ("Simple $L_0$",                   "results_scout3/simple_s*",          False),
    ("Exclusivity $L_0$",              "results_scout3/exclusivity_s*",     False),
    (None, None, None),
    ("Learned $L_\\theta$",            "results_scout3/learned_s*",         False),
    ("  $+$ viewpoint bound, $+$RSA",  "results_levers/lam06_learned_pre_prag_s*", False),
    ("  frozen, fit on baseline",      "results_frozen/frozenref_s*",       True),
    ("  frozen, saturated",            "results_frozen/frozensat_s*",       True),
    (None, None, None),
    ("Info-reg. [Strouse]",            "results_baselines/inforeg_s*",      False),
    ("Partner belief [Tian]",          "results_baselines/partner_belief_s*", False),
    (None, None, None),
    ("IPL $+$ RSA",                    "results_ipl/iplprag06_s*",          True),
]


def seed_vals(pat, key="r_ext", last_k=3):
    out = []
    for p in sorted(glob.glob(os.path.join(pat, "history.json"))):
        if not os.path.exists(os.path.join(os.path.dirname(p), "model.pt")):
            continue
        h = json.load(open(p))["history"]
        out.append(np.mean([e[key] for e in h[-last_k:]]))
    return np.array(out)


def main(figdir):
    base, orac = seed_vals("results_scout3/baseline_s*"), seed_vals("results_scout3/oracle_s*")
    prem = orac.mean() - base.mean()
    closure = lambda v: (v - base.mean()) / prem

    use_style()
    fig, ax = plt.subplots(figsize=(5.3, 3.9))
    ypos, labels = [], []
    y = 0
    for label, pat, emph in ROWS:
        if label is None:            # family separator
            y -= 0.5
            continue
        v = seed_vals(pat)
        if len(v) == 0:
            continue
        c = closure(v)
        col = HI if emph else INK
        vis = c[c >= XLO]
        ax.scatter(vis, np.full_like(vis, y), s=13, color=col, alpha=0.32,
                   linewidths=0, zorder=2)                     # per-seed dots
        off = c[c < XLO]                                       # off-scale seeds
        if len(off):
            ax.plot(XLO + 0.02, y, marker="<", ms=6, color=col, zorder=4,
                    clip_on=False)
            ax.text(XLO + 0.07, y + 0.32,
                    f"{len(off)} seed at ${off.min():.1f}$" if len(off) == 1
                    else f"{len(off)} seeds $\\leq{off.max():.1f}$",
                    fontsize=6.6, color=col, ha="left", va="center")
        m = c.mean()
        se = c.std(ddof=1) / len(c) ** .5
        ax.plot([max(m - se, XLO), m + se], [y, y], color=col, lw=2.4,
                solid_capstyle="butt", zorder=3)
        ax.plot(m, y, "o", ms=6.5, color=col, mec="white", mew=1.0, zorder=4)
        ypos.append(y); labels.append(label)
        y -= 1

    ax.axvline(0, color=REF_B, lw=1.0, zorder=1)
    ax.axvline(1, color=REF_O, lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.text(0, 1.2, "baseline", fontsize=7.5, color=REF_B, ha="center", va="bottom")
    ax.text(1, 1.2, "oracle", fontsize=7.5, color=REF_O, ha="center", va="bottom")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(y + 0.4, 1.6)
    ax.set_xlim(XLO, 1.25)
    ax.set_xlabel("Fraction of the oracle premium closed")
    ax.grid(axis="y", alpha=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "listeners.png"))
    plt.close(fig)
    print("wrote listeners.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(a.figdir)
