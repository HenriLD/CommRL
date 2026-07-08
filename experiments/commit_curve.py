"""Time-resolved commitment accuracy for scout-support checkpoints.

For each condition, rolls out held-out episodes with the final deterministic
policy and plots the supporter's commitment accuracy as a function of episode
timestep: how early does the supporter know where to go?

Usage: python commit_curve.py --resroot results_scout --out ../papers/Conference_Paper/img/scout_commit_curve.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear
from paperstyle import use_style, LABELS, COLORS, REF_STYLE
from plots_scout import ORDER


def commit_curve(run_dir, n_envs=256, seed=777):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    name = os.path.basename(run_dir)
    listener = None
    ear = ("ear" in name.replace("learned", "")) or name.startswith("ear")
    if ear and ckpt.get("listener") is not None:
        listener = S.ScoutListener()
        listener.load_state_dict(ckpt["listener"])
    env = S.ScoutSupportEnv(n_envs, oracle=name.startswith("oracle"), seed=seed)
    obs = env.reset()
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    acc = []
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            next_obs, info, _ = env.step(a)
            acc.append(info["commit_acc"].mean().item())
            if ear:
                if listener is not None:
                    post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                else:
                    post = env.log_belief.exp()
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
    return np.array(acc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results_scout")
    p.add_argument("--out", default="scout_commit_curve.png")
    p.add_argument("--conds", nargs="+", default=None)
    args = p.parse_args()

    curves = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(args.resroot, "*"))):
        if not os.path.exists(os.path.join(run, "model.pt")):
            continue
        cond = os.path.basename(run).rsplit("_s", 1)[0]
        if args.conds and cond not in args.conds:
            continue
        curves[cond].append(commit_curve(run))
        print("rolled out", os.path.basename(run), flush=True)

    use_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for cond in ORDER:
        if cond not in curves:
            continue
        c = np.stack(curves[cond])
        m, e = c.mean(0), c.std(0) / np.sqrt(c.shape[0])
        t = np.arange(1, len(m) + 1)
        if cond in REF_STYLE:
            ax.plot(t, m, label=LABELS[cond], **REF_STYLE[cond])
            ax.fill_between(t, m - e, m + e, color=REF_STYLE[cond]["color"],
                            alpha=0.10, lw=0)
        else:
            ax.plot(t, m, label=LABELS[cond], color=COLORS[cond], lw=2.0)
            ax.fill_between(t, m - e, m + e, color=COLORS[cond], alpha=0.15, lw=0)
    ax.axhline(1 / 3, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.text(49, 1 / 3 + 0.012, "chance", fontsize=7.5, color="#555555", ha="right")
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Supporter commitment accuracy")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)

    summary = {c: {"auc": float(np.stack(v).mean()),
                   "acc_t10": float(np.stack(v)[:, 9].mean()),
                   "acc_t20": float(np.stack(v)[:, 19].mean())}
               for c, v in curves.items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
