"""Two figures added in the visual-polish pass, both with the paper's printed
numbers as literals (sources noted inline) so the plotted quantity is exactly
the claimed quantity.

  meaning_axis   literal vs. recursed closure across the three meaning-space
                 regimes (Sec. 5.4); honest tier encoding (open = suggestive
                 t=2.2-2.7, filled = headline t>=5.8).
  transparency   left: I(Z;behaviour) rises while I(Z;private) stays flat
                 below the K=255 critic ceiling; right: the beta sweep prunes
                 noise leakage while the true bearing survives (Sec. 5.4.2).

Usage: python paper_figs2.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from paperstyle import use_style

# fixed identities, drawn from the CVD-safe house palette
C_LIT = "#0072B2"      # literal learned family (cool)
C_REC = "#009E73"      # + RSA recursion
C_IPL = "#CC79A7"      # the inverse-planning headline point
C_BEH = "#0072B2"
C_PRIV = "#666666"


def fig_meaning_axis(figdir):
    # closure as a fraction of the oracle premium; sems are per-premium.
    # discrete: viewpoint-bounded literal (Table 2) vs. +RSA confirmation cohort
    #   (Sec 5.1.1, 27%, t=2.7); continuous: pooled n=16 (Sec 5.4.1, 17.7%,
    #   t=2.2); self-assigned: Sec 5.4.2 (25.7%, t=2.3).
    regimes = ["Discrete\nmeanings", "Continuous\n$S^1$", "Self-assigned\nlatent"]
    lit = np.array([0.04, 0.049, 0.085])
    lit_se = np.array([0.15, 0.10, 0.11])
    rec = np.array([0.27, 0.177, 0.257])
    rec_se = np.array([0.05, 0.08, 0.11])
    x = np.arange(3)

    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.axhline(0, **{"color": "#666666", "ls": "--", "lw": 1.4})
    ax.axhline(1, **{"color": "#111111", "ls": ":", "lw": 1.6})
    ax.text(0.5, 0.045, "baseline", fontsize=7.5, color="#666666", va="bottom", ha="center")
    ax.text(0.5, 0.955, "oracle", fontsize=7.5, color="#111111", va="top", ha="center")

    dx = 0.13
    # connectors first (the recursion lift), then markers on top
    for xi, l, r in zip(x, lit, rec):
        ax.plot([xi - dx, xi + dx], [l, r], color="#bbbbbb", lw=1.3, zorder=1)
    ax.errorbar(x - dx, lit, yerr=lit_se, fmt="s", ms=6, color=C_LIT,
                mfc="white", mec=C_LIT, mew=1.5, capsize=2.5, lw=1.2,
                zorder=3, label="learned listener, no recursion")
    ax.errorbar(x + dx, rec, yerr=rec_se, fmt="o", ms=7, color=C_REC,
                mfc="white", mec=C_REC, mew=1.7, capsize=2.5, lw=1.2,
                zorder=3, label="$+$ RSA recursion (suggestive)")
    # the one headline-tier point: IPL + RSA in the discrete regime
    ax.errorbar([0], [0.70], yerr=[0.08], fmt="*", ms=15, color=C_IPL,
                mec="black", mew=0.5, capsize=2.5, lw=1.2, zorder=4,
                label="IPL + RSA (headline, $t{=}5.9$)")

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-0.25, 1.08)
    ax.set_ylabel("Oracle premium closed")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.90),
              fontsize=7.6, handletextpad=0.4, borderaxespad=0.2)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "meaning_axis.png"))
    plt.close(fig)
    print("wrote meaning_axis.png")


def fig_transparency(figdir):
    """Left column: two zoomed strips sharing the condition axis, each on its
    own scale so the dissociation is legible (private near-flat below the
    critic ceiling; behaviour rising with the reward). Right: the beta sweep."""
    use_style()
    fig = plt.figure(figsize=(5.4, 2.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.12],
                          height_ratios=[1, 1], hspace=0.28, wspace=0.46)
    ax_priv = fig.add_subplot(gs[0, 0])
    ax_beh = fig.add_subplot(gs[1, 0], sharex=ax_priv)
    axR = fig.add_subplot(gs[:, 1])

    # values from the K=1023 critic (results_bn/probes_k1023.json), the most
    # converged lower bound; ceiling log(1024)=6.93
    conds = ["base", "literal", "$+$RSA"]
    x = np.arange(3)
    i_priv = np.array([5.95, 5.98, 5.98]); i_priv_se = np.array([.022, .012, .018])
    i_beh = np.array([1.762, 1.781, 1.814]); i_beh_se = np.array([.004, .006, .010])

    # private strip: flat, sitting well below the critic ceiling (not pinned)
    ax_priv.axhline(6.93, color="#999999", ls=(0, (4, 3)), lw=1.0)
    ax_priv.text(2.02, 6.93, "ceiling", fontsize=6.8, color="#888888",
                 ha="right", va="bottom")
    ax_priv.errorbar(x, i_priv, yerr=i_priv_se, fmt="o-", color=C_PRIV, ms=5,
                     lw=1.6, capsize=3)
    ax_priv.set_ylim(5.7, 7.05)
    ax_priv.set_yticks([5.8, 6.2, 6.6, 7.0])
    ax_priv.set_ylabel("$I(Z;\\mathrm{priv})$", fontsize=8.5)
    ax_priv.tick_params(labelbottom=False)
    ax_priv.set_title("private flat", fontsize=8, pad=3)

    # behaviour strip: rises with the reward (own zoomed scale)
    ax_beh.errorbar(x, i_beh, yerr=i_beh_se, fmt="o-", color=C_BEH, ms=5,
                    lw=1.6, capsize=3)
    ax_beh.set_ylim(1.72, 1.85)
    ax_beh.set_yticks([1.75, 1.80, 1.85])
    ax_beh.set_ylabel("$I(Z;\\mathrm{beh})$", fontsize=8.5)
    ax_beh.set_xticks(x)
    ax_beh.set_xticklabels(conds)
    ax_beh.set_title("behaviour rises ($t{\\approx}4.6$)", fontsize=8, pad=3)

    # right: beta sweep, R^2 of each private channel decoded from z
    beta = np.array([1e-3, 3e-3, 1e-2, 3e-2])
    r2_true = np.array([0.999, 0.999, 0.996, 0.97])
    r2_decoy = np.array([0.21, 0.19, 0.12, 0.15])
    r2_noise = np.array([0.49, 0.15, 0.11, 0.12])
    axR.plot(beta, r2_true, "o-", color="#009E73", ms=5, lw=1.8, label="true bearing")
    axR.plot(beta, r2_decoy, "s--", color="#E69F00", ms=5, lw=1.5, label="decoys")
    axR.plot(beta, r2_noise, "^:", color="#D55E00", ms=5, lw=1.5, label="noise")
    axR.set_xscale("log")
    axR.set_xlabel("compression $\\beta$")
    axR.set_ylabel("decoded $R^2$ from $z$")
    axR.set_ylim(-0.03, 1.06)
    axR.legend(frameon=False, loc="center right", fontsize=7.4, handletextpad=0.4)
    axR.set_title("leakage pruned, content kept", fontsize=8, pad=3)

    fig.savefig(os.path.join(figdir, "transparency.png"))
    plt.close(fig)
    print("wrote transparency.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig_meaning_axis(a.figdir)
    fig_transparency(a.figdir)
