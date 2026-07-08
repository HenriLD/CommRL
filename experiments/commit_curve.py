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
from plots_scout import LABELS, COLORS, ORDER


def commit_curve(run_dir, n_envs=256, seed=777):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    env = S.ScoutSupportEnv(n_envs, oracle=("oracle" in os.path.basename(run_dir)),
                            seed=seed)
    obs = env.reset()
    acc = []
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            obs, info, _ = env.step(a)
            acc.append(info["commit_acc"].mean().item())
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

    plt.figure(figsize=(5.2, 3.4))
    for cond in ORDER:
        if cond not in curves:
            continue
        c = np.stack(curves[cond])
        m, e = c.mean(0), c.std(0) / np.sqrt(c.shape[0])
        t = np.arange(1, len(m) + 1)
        plt.plot(t, m, label=LABELS[cond], color=COLORS[cond], lw=1.8)
        plt.fill_between(t, m - e, m + e, color=COLORS[cond], alpha=0.15, lw=0)
    plt.axhline(1 / 3, color="k", lw=0.8, ls=":", alpha=0.6)
    plt.xlabel("Episode timestep")
    plt.ylabel("Supporter commitment accuracy")
    plt.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    plt.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print("wrote", args.out)

    summary = {c: {"auc": float(np.stack(v).mean()),
                   "acc_t10": float(np.stack(v)[:, 9].mean()),
                   "acc_t20": float(np.stack(v)[:, 19].mean())}
               for c, v in curves.items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
