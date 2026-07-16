"""Synthesis figures for the paper, designed so the plotted quantity IS the
claimed quantity:

  forest   gap closure (with bootstrap 95% CI) per condition and environment
  delta    supporter commitment relative to baseline over the episode
  budget   estimated gap closure as a function of training budget (the
           pre-convergence-reversal figure)

Usage: python paper_figs.py --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from paperstyle import use_style, COLORS, REF_STYLE

RS3 = "results_scout3"
SUITE = "results_suite"


def seed_means(root, cond, key="r_ext", last_k=3):
    v = [np.mean([e[key] for e in json.load(open(p))["history"][-last_k:]])
         for p in sorted(glob.glob(os.path.join(root, f"{cond}_s*", "history.json")))]
    return np.array(v)


def closure_ci(prog, base, orac, n_boot=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    def stat(p, b, o):
        return (p.mean() - b.mean()) / max(1e-9, o.mean() - b.mean())
    boots = [stat(rng.choice(prog, len(prog)), rng.choice(base, len(base)),
                  rng.choice(orac, len(orac))) for _ in range(n_boot)]
    return stat(prog, base, orac), np.percentile(boots, [2.5, 97.5])


def fig_forest(figdir):
    rows = []  # (label, closure, lo, hi, color)
    def add(label, root, cond, bcond, ocond, color):
        p, b, o = seed_means(root, cond), seed_means(root, bcond), seed_means(root, ocond)
        m, (lo, hi) = closure_ci(p, b, o)
        rows.append((label, m, lo, hi, color))

    add("Progress $L_0$ (3 meanings)", RS3, "progress", "baseline", "oracle", COLORS["progress"])
    add("Progress $L_0$ (5 meanings)", SUITE, "k5_progress", "k5_baseline", "k5_oracle", COLORS["progress"])
    add("Progress $L_0$ (minefield)", SUITE, "mine_progress", "mine_baseline", "mine_oracle", COLORS["progress"])
    add("Filter $L_0$", RS3, "filter", "baseline", "oracle", COLORS["filter"])
    add("Simple $L_0$", RS3, "simple", "baseline", "oracle", COLORS["simple"])
    add("Exclusivity $L_0$", RS3, "exclusivity", "baseline", "oracle", COLORS["exclusivity"])
    add("Ear + $R_{comm}$", RS3, "learned_ear", "baseline", "oracle", COLORS["learned_ear"])
    add("Ear ($\\lambda{=}0$)", RS3, "ear", "baseline", "oracle", COLORS["ear"])
    add("Learned $L_\\theta$ (speaker only)", RS3, "learned", "baseline", "oracle", COLORS["learned"])
    # blind control: gain under blinding, scaled by the sighted premium
    bp, bb = seed_means(SUITE, "blind_progress"), seed_means(SUITE, "blind_baseline")
    b0, o0 = seed_means(RS3, "baseline"), seed_means(RS3, "oracle")
    prem = o0.mean() - b0.mean()
    rng = np.random.default_rng(1)
    boots = [(rng.choice(bp, len(bp)).mean() - rng.choice(bb, len(bb)).mean()) / prem
             for _ in range(10000)]
    rows.append(("Progress, blind partner", (bp.mean() - bb.mean()) / prem,
                 *np.percentile(boots, [2.5, 97.5]), COLORS["progress"]))

    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, m, lo, hi, color) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.0, solid_capstyle="butt")
        ax.plot(m, yi, "o", color=color, ms=6, mec="white", mew=0.6)
    ax.axvline(0, color="#666666", lw=1.2, ls="--")
    ax.axvline(1, color="#111111", lw=1.2, ls=":")
    ax.text(0, len(rows) - 0.2, "baseline", fontsize=7.5, color="#666666", ha="center")
    ax.text(1, len(rows) - 0.2, "oracle", fontsize=7.5, color="#111111", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("Fraction of oracle premium closed")
    ax.set_xlim(-1.15, 1.15)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "forest.png"))
    plt.close(fig)
    print("wrote forest.png")


def fig_budget(figdir):
    """Estimated gap closure vs training budget for the headline conditions."""
    def curves(root, cond):
        hs = [json.load(open(p))["history"] for p in
              sorted(glob.glob(os.path.join(root, f"{cond}_s*", "history.json")))]
        n = min(len(h) for h in hs)
        steps = np.array([e["steps"] for e in hs[0][:n]])
        return steps, np.array([[e["r_ext"] for e in h[:n]] for h in hs])

    steps, B = curves(RS3, "baseline")
    _, O = curves(RS3, "oracle")
    rng = np.random.default_rng(0)
    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for cond, color in [("progress", COLORS["progress"]),
                        ("learned_ear", COLORS["learned_ear"]),
                        ("learned", COLORS["learned"])]:
        _, P = curves(RS3, cond)
        n = min(P.shape[1], B.shape[1], O.shape[1])
        m, lo, hi = [], [], []
        for t in range(n):
            c, (l, h) = closure_ci(P[:, t], B[:, t], O[:, t], n_boot=3000, rng=rng)
            m.append(c); lo.append(l); hi.append(h)
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.plot(steps[:n] / 1e6, np.clip(m, -1.5, 1.5), color=color, lw=2.0,
                label={"progress": "Progress $L_0$", "learned_ear": "Ear + $R_{comm}$",
                       "learned": "Learned $L_\\theta$"}[cond])
        ax.fill_between(steps[:n] / 1e6, np.clip(lo, -1.5, 1.5),
                        np.clip(hi, -1.5, 1.5), color=color, alpha=0.13, lw=0)
    ax.axhline(0, color="#666666", lw=1.2, ls="--")
    ax.axhline(1, color="#111111", lw=1.2, ls=":")
    ax.axvline(0.48, color="#888888", lw=1.0, ls="-.", alpha=0.8)
    ax.text(0.48, 1.35, " 150-cycle budget", fontsize=7.5, color="#666666")
    ax.set_xlabel("Training budget (millions of environment steps)")
    ax.set_ylabel("Estimated premium closed")
    ax.set_ylim(-1.5, 1.55)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "budget_dynamics.png"))
    plt.close(fig)
    print("wrote budget_dynamics.png")


def fig_delta_commit(figdir):
    from commit_curve import commit_curve
    conds = ["oracle", "progress", "filter", "learned_ear"]
    base = np.stack([commit_curve(d) for d in
                     sorted(glob.glob(os.path.join(RS3, "baseline_s*")))
                     if os.path.exists(os.path.join(d, "model.pt"))])
    bm, bs = base.mean(0), base.std(0) / np.sqrt(base.shape[0])
    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for cond in conds:
        runs = [d for d in sorted(glob.glob(os.path.join(RS3, f"{cond}_s*")))
                if os.path.exists(os.path.join(d, "model.pt"))]
        c = np.stack([commit_curve(d) for d in runs])
        m, s = c.mean(0) - bm, np.sqrt(c.var(0) / c.shape[0] + bs ** 2)
        t = np.arange(1, len(m) + 1)
        style = REF_STYLE.get(cond, {})
        color = style.get("color", COLORS.get(cond))
        ax.plot(t, m, lw=style.get("lw", 2.0), ls=style.get("ls", "-"),
                color=color, label={"oracle": "Oracle", "progress": "Progress $L_0$",
                                    "filter": "Filter $L_0$",
                                    "learned_ear": "Ear + $R_{comm}$"}[cond])
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.13, lw=0)
    ax.axhline(0, color="#666666", lw=1.2, ls="--")
    ax.text(30, 0.004, "baseline", fontsize=7.5, color="#666666")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Commitment accuracy $-$ baseline")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "commit_delta.png"))
    plt.close(fig)
    print("wrote commit_delta.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    p.add_argument("--only", nargs="+", default=["forest", "budget", "delta"])
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if "forest" in a.only:
        fig_forest(a.figdir)
    if "budget" in a.only:
        fig_budget(a.figdir)
    if "delta" in a.only:
        fig_delta_commit(a.figdir)
