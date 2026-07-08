"""Zero-shot cross-play: pair the SCOUT policy from one run with the
SUPPORTER policy from another. If the implicit protocol is grounded, cross-
seed pairs should retain most of their self-play performance; if it is a
co-adapted convention, cross-play should collapse toward baseline.

Actors are parameter-shared with a role flag, so a cross-play team uses
actor A for the scout slot and actor B for the supporter slot. Ear
conditions use the SUPPORTER side's listener (its own equipment).

Usage: python cross_play.py --resroot results_scout3 --conds learned_ear learned baseline --seeds 0 1 2 3 4 5
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear


def load(run_dir):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    listener = None
    if ckpt.get("listener") is not None:
        listener = S.ScoutListener()
        listener.load_state_dict(ckpt["listener"])
    return actor, listener


def team_rollout(scout_actor, sup_actor, sup_listener, ear, n_envs=256, seed=555):
    env = S.ScoutSupportEnv(n_envs, seed=seed)
    obs = env.reset()
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    r_tot, commit = 0.0, 0.0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a_s, _ = scout_actor.sample(obs, deterministic=True)
            a_p, _ = sup_actor.sample(obs, deterministic=True)
            a = torch.stack([a_s[:, 0], a_p[:, 1]], dim=1)
            next_obs, info, _ = env.step(a)
            r_tot += info["r_ext"].mean().item()
            commit += info["commit_acc"].mean().item()
            if ear and sup_listener is not None:
                post = torch.softmax(sup_listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
    return r_tot / S.EPISODE_LEN, commit / S.EPISODE_LEN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results_scout3")
    p.add_argument("--conds", nargs="+", default=["baseline", "learned", "learned_ear"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = {}
    for c in args.conds:
        for s in args.seeds:
            d = os.path.join(args.resroot, f"{c}_s{s}")
            if os.path.exists(os.path.join(d, "model.pt")):
                runs[(c, s)] = load(d)

    results = {}
    for cond in args.conds:
        seeds = [s for (c, s) in runs if c == cond]
        self_r, cross_r, self_c, cross_c = [], [], [], []
        ear = "ear" in cond.replace("learned", "") or cond.startswith("ear")
        for si, sj in itertools.product(seeds, seeds):
            scout_actor, _ = runs[(cond, si)]
            sup_actor, sup_lis = runs[(cond, sj)]
            r, cm = team_rollout(scout_actor, sup_actor, sup_lis, ear)
            (self_r if si == sj else cross_r).append(r)
            (self_c if si == sj else cross_c).append(cm)
        results[cond] = {
            "self_r": float(np.mean(self_r)), "cross_r": float(np.mean(cross_r)),
            "self_commit": float(np.mean(self_c)), "cross_commit": float(np.mean(cross_c)),
            "n_seeds": len(seeds),
        }
        print(f"{cond:14s} self {np.mean(self_r):+.3f}  cross {np.mean(cross_r):+.3f} "
              f"(retention {np.mean(cross_r)/np.mean(self_r)*100 if np.mean(self_r) else 0:.0f}%)  "
              f"commit self {np.mean(self_c):.3f} cross {np.mean(cross_c):.3f}", flush=True)

    out = args.out or os.path.join(args.resroot, "cross_play.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
