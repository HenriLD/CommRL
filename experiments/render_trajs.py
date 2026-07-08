"""Render qualitative episode trajectories from preference-spread checkpoints.

Agent paths are colored by the agent's private preference and darken with
time; dots every 5 steps make speed visible (a stationary agent shows as
stacked dots). Landmarks are squares colored by category.

Usage: python render_trajs.py --runs results/baseline_s0 results/learned_s0 \
                              --labels "Baseline" "Learned listener" \
                              --episode_seed 21 --out ../papers/Conference_Paper/img/trajectories.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from pragmatic_spread import PragmaticSpreadEnv, EPISODE_LEN, N_AGENTS
from train_masac import Actor
from paperstyle import use_style, draw_timed_path

CMAP = ["#D55E00", "#009E73", "#0072B2"]  # meaning colors


def rollout(run_dir, episode_seed, oracle=False):
    actor = Actor()
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
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
    p.add_argument("--episode_seed", type=int, default=21)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    use_style()

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.1))
    if n == 1:
        axes = [axes]
    for ax, run, label in zip(axes, args.runs, args.labels):
        traj, lm_pos, lm_color, pref = rollout(
            run, args.episode_seed, oracle=("oracle" in run))
        for l in range(lm_pos.shape[0]):
            ax.scatter(*lm_pos[l], marker="s", s=220, facecolor="none",
                       edgecolor=CMAP[lm_color[l]], linewidth=2.0, zorder=1)
        for i in range(N_AGENTS):
            draw_timed_path(ax, traj[:, i], CMAP[pref[i]])
        ax.set_title(label)
        ax.set_xlim(-1.58, 1.58); ax.set_ylim(-1.58, 1.58)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6)
    axes[0].text(0.03, 0.02, "light $\\to$ dark = time;\ndots every 5 steps",
                 transform=axes[0].transAxes, fontsize=7.5, color="#444444")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
