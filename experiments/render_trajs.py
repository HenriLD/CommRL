"""Render qualitative episode trajectories from trained checkpoints.

For each requested run, rolls out one deterministic episode from a fixed seed
and draws agent trajectories (colored by private preference) and landmarks
(colored by category). Produces side-by-side comparisons for the paper.

Usage: python render_trajs.py --runs results/baseline_s0 results/learned_s0 \
                              --labels "Baseline" "Learned listener" \
                              --episode_seed 7 --out ../papers/Conference_Paper/img/trajectories.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pragmatic_spread import PragmaticSpreadEnv, EPISODE_LEN, N_AGENTS
from train_masac import Actor

CMAP = ["#d62728", "#2ca02c", "#1f77b4"]  # meaning colors: red, green, blue


def rollout(run_dir, episode_seed, oracle=False):
    actor = Actor()
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=True)
    actor.load_state_dict(ckpt["actor"])
    env = PragmaticSpreadEnv(1, oracle=oracle, seed=episode_seed)
    obs = env.reset()
    traj = [env.pos[0].clone().numpy()]
    with torch.no_grad():
        for t in range(EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            obs, info, done = env.step(a)
            traj.append(env.pos[0].clone().numpy())
    return (np.stack(traj), env.lm_pos[0].numpy(), env.lm_color[0].numpy(),
            env.pref[0].numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--episode_seed", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, run, label in zip(axes, args.runs, args.labels):
        traj, lm_pos, lm_color, pref = rollout(
            run, args.episode_seed, oracle=("oracle" in run))
        for l in range(lm_pos.shape[0]):
            ax.scatter(*lm_pos[l], marker="s", s=220, facecolor="none",
                       edgecolor=CMAP[lm_color[l]], linewidth=2.2, zorder=1)
        for i in range(N_AGENTS):
            c = CMAP[pref[i]]
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=c, lw=1.6, alpha=0.85, zorder=2)
            ax.scatter(*traj[0, i], color=c, marker="o", s=45, zorder=3,
                       edgecolor="k", linewidth=0.6)
            ax.scatter(*traj[-1, i], color=c, marker="*", s=130, zorder=3,
                       edgecolor="k", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
