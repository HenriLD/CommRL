"""A schematic of scout-support: why this environment can test legibility.

A rendered trajectory shows what one episode did; it does not show why the
task is constructed the way it is. The reader needs three facts before any
result means anything: the scout alone knows the target, the first leg is
forced through a waypoint and is therefore identical whatever the target is,
and the supporter is slow enough that it must commit before the scout's
motion has disambiguated anything. Those are geometric facts, so they are
drawn, not plotted.

Usage: python fig_schematic.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from paperstyle import use_style

SITE = "#4C4C4C"
TARGET = "#CC79A7"
SCOUT = "#0072B2"
SUPP = "#E69F00"
MUTED = "#8A8A8A"


def main(figdir):
    use_style()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # three candidate sites on a circle; one is the (privately known) target
    ang = np.array([90, 210, 330]) * math.pi / 180
    sx, sy = np.cos(ang), np.sin(ang)
    for i, (x, y) in enumerate(zip(sx, sy)):
        is_t = (i == 0)
        ax.add_patch(Circle((x, y), 0.155, facecolor=TARGET if is_t else "none",
                            edgecolor=TARGET if is_t else SITE,
                            lw=1.6, alpha=0.95 if is_t else 1.0, zorder=3))
    ax.text(sx[0], sy[0] + 0.30, "target\n(scout only)", fontsize=7.2,
            color=TARGET, ha="center", va="bottom")
    ax.text(sx[1] - 0.10, sy[1] - 0.28, "candidate sites", fontsize=7.2,
            color=SITE, ha="center", va="top")

    # the pickup waypoint at the centre
    ax.add_patch(Circle((0, 0), 0.13, facecolor="none", edgecolor=MUTED,
                        lw=1.2, ls=(0, (2, 2)), zorder=3))
    ax.plot(0, 0, "x", color=MUTED, ms=6, mew=1.6, zorder=4)
    ax.text(0.16, -0.02, "waypoint", fontsize=7.2, color=MUTED,
            ha="left", va="center")

    # scout: forced first leg (identical for every target), then a free leg
    start = np.array([-1.18, 0.62])
    ax.plot(*start, "o", color=SCOUT, ms=7, zorder=5)
    ax.text(start[0], start[1] + 0.16, "scout", fontsize=7.6, color=SCOUT,
            ha="center", va="bottom")
    ax.add_patch(FancyArrowPatch(start, (-0.14, 0.06), arrowstyle="-|>",
                                 mutation_scale=11, lw=2.0, color=SCOUT,
                                 shrinkA=6, shrinkB=2, zorder=4))
    ax.text(-0.82, -0.08, "leg 1: forced,\nsame for all targets",
            fontsize=6.9, color=SCOUT, ha="center", va="top")
    ax.add_patch(FancyArrowPatch((0.06, 0.12), (sx[0] - 0.02, sy[0] - 0.19),
                                 arrowstyle="-|>", mutation_scale=11, lw=2.0,
                                 color=SCOUT, ls=(0, (3, 2)), shrinkA=2,
                                 shrinkB=4, zorder=4))
    ax.text(0.16, 0.52, "leg 2:\nreveals it", fontsize=6.9, color=SCOUT,
            ha="left", va="center")

    # supporter: slow, and must pick a site before leg 2 resolves anything.
    # short stubs toward each candidate read as "which one?" without dragging
    # long arrows across the whole panel.
    sup = np.array([1.30, -1.02])
    ax.plot(*sup, "o", color=SUPP, ms=7, zorder=5)
    ax.text(sup[0] + 0.10, sup[1] - 0.12, "supporter\n(slower)", fontsize=7.4,
            color=SUPP, ha="right", va="top")
    for tgt in range(3):
        d = np.array([sx[tgt], sy[tgt]]) - sup
        end = sup + 0.22 * d
        ax.add_patch(FancyArrowPatch(sup, end, arrowstyle="-|>",
                                     mutation_scale=8, lw=1.1, color=SUPP,
                                     alpha=0.55, shrinkA=6, shrinkB=0, zorder=2))
    ax.text(0.32, -1.26, "which site?\nmust commit early", fontsize=6.9,
            color=SUPP, ha="center", va="center")

    ax.set_xlim(-1.55, 1.60)
    ax.set_ylim(-1.55, 1.50)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(figdir, "schematic.png"))
    plt.close(fig)
    print("wrote schematic.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(a.figdir)
