"""Qualitative episode rendering for scout-support checkpoints.

Time is encoded along each trajectory: paths darken from episode start to end
and carry a dot every 5 steps, so a slow or stationary agent shows up as
tightly stacked dots. Start = open circle, end = star. Sites are squares (the
true target filled), the pickup waypoint is a black cross.

Usage: python render_scout.py --runs results_scout2/baseline_s0 results_scout2/learned_ear_s0 \
                              --labels Baseline "Ear + R_comm" --episode_seed 5 --out traj.png
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear
from paperstyle import use_style, draw_timed_path, time_legend_handles

SITE_COLORS = ["#D55E00", "#009E73", "#0072B2"]
SCOUT_COLOR = "#882255"
SUPPORT_COLOR = "#0072B2"


def rollout(run_dir, episode_seed):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    listener = None
    name = os.path.basename(run_dir)
    ear = ("ear" in name.replace("learned", "")) or name.startswith("ear")
    if ear and ckpt.get("listener") is not None:
        listener = S.ScoutListener()
        listener.load_state_dict(ckpt["listener"])
    env = S.ScoutSupportEnv(1, oracle=name.startswith("oracle"), seed=episode_seed)
    obs = env.reset()
    if ear:
        obs = inject_ear(obs, torch.full((1, S.N_MEANINGS), 1 / 3))
    traj = [env.pos[0].clone().numpy()]
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            next_obs, _, _ = env.step(a)
            if ear and listener is not None:
                post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            elif ear:  # filter_ear
                next_obs = inject_ear(next_obs, env.log_belief.exp())
            obs = next_obs
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
    use_style()

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.1))
    if n == 1:
        axes = [axes]
    for ax, run, label in zip(axes, args.runs, args.labels):
        traj, sites, target, wp = rollout(run, args.episode_seed)
        for i in range(sites.shape[0]):
            filled = SITE_COLORS[i] if i == target else "none"
            ax.scatter(*sites[i], marker="s", s=230, facecolor=filled,
                       edgecolor=SITE_COLORS[i], linewidth=2.0, zorder=1,
                       alpha=0.9 if i == target else 1.0)
        ax.scatter(*wp, marker="X", s=110, color="k", zorder=2)
        ax.add_patch(plt.Circle(wp, S.KEY_RADIUS, fill=False, color="k",
                                lw=0.8, ls=(0, (2, 2)), alpha=0.55, zorder=2))
        draw_timed_path(ax, traj[:, 0], SCOUT_COLOR)
        draw_timed_path(ax, traj[:, 1], SUPPORT_COLOR)
        # mark the supporter's commitment moment: first step from which its
        # nearest site is the true target for the rest of the episode
        sup = traj[:, 1]
        d = np.linalg.norm(sup[:, None, :] - sites[None], axis=-1)
        right = d.argmin(-1) == target
        tc = len(right)
        for i in range(len(right) - 1, -1, -1):
            if right[i]:
                tc = i
            else:
                break
        if tc < len(right):
            ax.plot(*sup[tc], marker="D", ms=7, color=SUPPORT_COLOR,
                    mec="k", mew=0.8, zorder=6)
            ax.annotate(f"commits $t{{=}}{tc}$", sup[tc], fontsize=7.0,
                        xytext=(6, -11), textcoords="offset points",
                        color="#222222", zorder=6)
        ax.set_title(label)
        ax.set_xlim(-1.58, 1.58); ax.set_ylim(-1.58, 1.58)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(0.6)
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        ax.spines["top"].set_linewidth(0.6); ax.spines["right"].set_linewidth(0.6)
    axes[0].legend(handles=time_legend_handles(
        [("scout", SCOUT_COLOR), ("supporter", SUPPORT_COLOR)]),
        loc="lower left", frameon=False, handlelength=1.4, fontsize=7.5)
    axes[0].text(0.03, 0.97, "dots every 5 steps", transform=axes[0].transAxes,
                 fontsize=7.5, color="#444444", va="top")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
