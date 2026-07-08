"""Shared figure style for the paper: CVD-validated palette, reference-line
styling for baseline/oracle, recessive axes, and time-gradient trajectories.

Palette validated with the six-check categorical validator (light surface):
lightness band, chroma floor, CVD separation (worst adjacent dE 18.3), with
legends providing the identity relief required by the contrast warning.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ---- categorical palette (fixed assignment, never cycled) ----
COLORS = {
    "simple": "#E69F00",        # hand-crafted family: warm hues
    "exclusivity": "#D55E00",
    "heuristic": "#D55E00",     # Env B name for the exclusivity listener
    "progress": "#CC79A7",
    "filter": "#996F00",
    "learned": "#0072B2",       # learned family: cool hues
    "learned_prag": "#009E73",
    "ear": "#56B4E9",           # receiver-side family
    "learned_ear": "#882255",
    "filter_ear": "#996F00",
}
# baseline and oracle are references, not series
REF_STYLE = {
    "baseline": dict(color="#666666", ls="--", lw=1.8),
    "oracle": dict(color="#111111", ls=":", lw=2.0),
}

LABELS = {
    "baseline": "Baseline",
    "oracle": "Oracle",
    "simple": "Simple $L_0$",
    "exclusivity": "Exclusivity $L_0$",
    "heuristic": "Exclusivity $L_0$",
    "progress": "Progress $L_0$",
    "filter": "Filter $L_0$",
    "learned": "Learned $L_\\theta$",
    "learned_prag": "Learned + RSA",
    "ear": "Ear ($\\lambda{=}0$)",
    "learned_ear": "Ear + $R_{comm}$",
    "filter_ear": "Filter + ear",
}

RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def use_style():
    plt.rcParams.update(RC)


def plot_series(ax, x, mean, sem, cond):
    """One condition's training curve with its fixed identity."""
    if cond in REF_STYLE:
        ax.plot(x, mean, label=LABELS[cond], **REF_STYLE[cond])
        ax.fill_between(x, mean - sem, mean + sem,
                        color=REF_STYLE[cond]["color"], alpha=0.12, lw=0)
    else:
        c = COLORS[cond]
        ax.plot(x, mean, label=LABELS[cond], color=c, lw=2.0)
        ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.16, lw=0)


# ---- time-gradient trajectories ----

def _ramp(hex_color):
    """Sequential ramp from a light tint to the full color (time = darkness)."""
    return LinearSegmentedColormap.from_list("r", ["#ffffff", hex_color])


def draw_timed_path(ax, xy, color, every=5, lw=2.2, t0_frac=0.25):
    """Draw a trajectory with time encoded as light-to-dark color and dots at
    regular timestep intervals; a stationary agent shows as stacked dots.

    xy: (T+1, 2) positions. t0_frac: ramp start (avoid near-white segments).
    """
    T = xy.shape[0] - 1
    cmap = _ramp(color)
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    fr = t0_frac + (1 - t0_frac) * np.linspace(0, 1, T)
    lc = LineCollection(segs, colors=cmap(fr), linewidth=lw,
                        capstyle="round", zorder=3)
    ax.add_collection(lc)
    idx = np.arange(0, T + 1, every)
    ax.scatter(xy[idx, 0], xy[idx, 1], s=11,
               color=cmap(t0_frac + (1 - t0_frac) * idx / T),
               edgecolor="white", linewidth=0.4, zorder=4)
    ax.scatter(*xy[0], s=42, facecolor="white", edgecolor=color,
               linewidth=1.4, zorder=5)                       # start: open circle
    ax.scatter(*xy[-1], s=120, marker="*", color=color,
               edgecolor="black", linewidth=0.5, zorder=5)    # end: star


def time_legend_handles(items):
    """Legend handles: one mid-ramp line per agent + note that dark = late."""
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=c, lw=2.2, label=l) for l, c in items]
    handles.append(mlines.Line2D([], [], color="none",
                                 label="light $\\to$ dark = time"))
    return handles
