"""Qualitative episode rendering for scout-support checkpoints.

Draws the waypoint (black cross), the three sites (squares; the true target
filled), the scout's path (solid) and the supporter's path (dashed).

Usage: python render_scout.py --runs results_scout/baseline_s0 results_scout/filter_s0 \
                              --labels Baseline "Filter L0" --episode_seed 5 --out traj.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scout_support as S
from train_masac import Actor

SITE_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]


def rollout(run_dir, episode_seed):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    env = S.ScoutSupportEnv(1, oracle=("oracle" in os.path.basename(run_dir)),
                            seed=episode_seed)
    obs = env.reset()
    traj = [env.pos[0].clone().numpy()]
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            obs, _, _ = env.step(a)
            traj.append(env.pos[0].clone().numpy())
    return (np.stack(traj), env.sites[0].numpy(), int(env.target[0]),
            env.waypoint[0].numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--episode_seed", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, run, label in zip(axes, args.runs, args.labels):
        traj, sites, target, wp = rollout(run, args.episode_seed)
        for i in range(sites.shape[0]):
            filled = SITE_COLORS[i] if i == target else "none"
            ax.scatter(*sites[i], marker="s", s=240, facecolor=filled,
                       edgecolor=SITE_COLORS[i], linewidth=2.2, zorder=1,
                       alpha=0.85 if i == target else 1.0)
        ax.scatter(*wp, marker="X", s=120, color="k", zorder=2)
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], color="#e377c2", lw=1.8,
                zorder=3, label="scout")
        ax.plot(traj[:, 1, 0], traj[:, 1, 1], color="#17becf", lw=1.8,
                ls="--", zorder=3, label="supporter")
        for i, c in [(0, "#e377c2"), (1, "#17becf")]:
            ax.scatter(*traj[0, i], color=c, marker="o", s=45, zorder=4,
                       edgecolor="k", linewidth=0.6)
            ax.scatter(*traj[-1, i], color=c, marker="*", s=140, zorder=4,
                       edgecolor="k", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
