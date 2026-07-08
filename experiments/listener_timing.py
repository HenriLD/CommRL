"""Diagnose WHEN the listener can decode the scout: per-timestep accuracy.

Rolls out converged policies and reports the co-trained listener's decode
accuracy at each episode step, split by whether the scout has the key.

Usage: python listener_timing.py --run results_scout/learned_ear_s0
"""

import argparse
import os

import numpy as np
import torch

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_envs", type=int, default=512)
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    listener = S.ScoutListener()
    listener.load_state_dict(ckpt["listener"])

    env = S.ScoutSupportEnv(args.n_envs, seed=4242)
    obs = env.reset()
    ear = "ear" in os.path.basename(args.run)
    if ear:
        obs = inject_ear(obs, torch.full((args.n_envs, S.N_MEANINGS), 1 / 3))

    correct_pre, n_pre, correct_post, n_post = 0.0, 0, 0.0, 0
    pre_acc_t = []
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            pred = listener(obs[:, 0], a[:, 0]).argmax(-1)
            hit = (pred == env.target).float()
            no_key = env.has_key < 0.5
            correct_pre += hit[no_key].sum().item(); n_pre += int(no_key.sum())
            correct_post += hit[~no_key].sum().item(); n_post += int((~no_key).sum())
            pre_acc_t.append(hit[no_key].mean().item() if no_key.any() else float("nan"))
            next_obs, info, _ = env.step(a)
            if ear:
                post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            obs = next_obs

    print("t   pre-key-env decode acc")
    for t in range(0, S.EPISODE_LEN, 2):
        print(f"{t:3d} {pre_acc_t[t]:.3f}")
    print(f"\ndecode acc on (env,t) with NO key yet: {correct_pre/max(1,n_pre):.3f} "
          f"(chance .333, n={n_pre})")
    print(f"decode acc on (env,t) with key:        {correct_post/max(1,n_post):.3f} "
          f"(n={n_post})")


if __name__ == "__main__":
    main()
