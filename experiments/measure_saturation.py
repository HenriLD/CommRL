"""Measure the antecedent of the gradient-saturation proposition directly.

For each converged run we replay the trained policy under its own trained
listener and record, over the policy's occupancy, the listener's posterior
mass on the TRUE meaning: the mean posterior, the fraction of occupancy with
posterior >= 0.95 (the proposition's 1 - delta), and the empirical standard
deviation of R_comm (the spread the shaping gradient can exploit).

Full-context listeners are expected to concentrate (saturate); listeners
bounded to the audience's viewpoint should concentrate less.

Usage: python measure_saturation.py [--out results_scout3/saturation.json]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import scout_support as S
from train_masac import Actor

# (label, run glob, listener input mode)
GROUPS = [
    ("full-context", "results_scout3/learned_s*", "full"),
    ("viewpoint-bounded", "results_boundedL/learned_act_s*", "act"),
    ("window-bounded", "results_prekey/learned_pre_s*", "act"),
]


def measure(run_dir, inputs, n_envs=64, n_resets=8, seed=12345):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                    weights_only=True)
    if ck.get("listener") is None:
        return None
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    listener = S.ScoutListener(inputs=inputs)
    listener.load_state_dict(ck["listener"])
    listener.eval()

    post, rcomm = [], []
    with torch.no_grad():
        for r in range(n_resets):
            env = S.ScoutSupportEnv(n_envs, seed=seed + r)
            obs = env.reset()
            for t in range(S.EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=False)   # on-policy occupancy
                p = F.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                pt = p.gather(-1, env.target.unsqueeze(-1)).squeeze(-1)
                post.append(pt)
                rcomm.append(torch.log(pt.clamp(min=1e-8)) + math.log(S.N_MEANINGS))
                obs, _, _ = env.step(a)
    post = torch.cat(post).numpy()
    rcomm = torch.cat(rcomm).numpy()
    return {"mean_posterior": float(post.mean()),
            "p_ge_95": float((post >= 0.95).mean()),
            "rcomm_sd": float(rcomm.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_scout3/saturation.json")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for label, pat, inputs in GROUPS:
        rows = []
        for d in sorted(glob.glob(pat)):
            if not os.path.exists(os.path.join(d, "model.pt")):
                continue
            m = measure(d, inputs)
            if m:
                m["run"] = os.path.basename(d)
                rows.append(m)
        out[label] = rows
        if rows:
            f = lambda k: np.array([r[k] for r in rows])
            print(f"{label:18s} n={len(rows):2d}  mean posterior "
                  f"{f('mean_posterior').mean():.3f}  "
                  f"P(>=.95) {f('p_ge_95').mean():.3f}  "
                  f"R_comm sd {f('rcomm_sd').mean():.3f}", flush=True)
        else:
            print(f"{label:18s} no runs found ({pat})")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
