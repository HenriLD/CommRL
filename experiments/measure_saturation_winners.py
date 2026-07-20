"""Saturation antecedents for the WINNING listeners.

The referee round's sharpest open question: Proposition 3's antecedent
(posterior concentration under the policy's own occupancy) was measured only
for the learned listeners -- the conditions that fail. Clause (a) of the
principle is a claim about the successful conditions, so measure them under
the same replay protocol: stochastic on-policy actions, 64 envs x 8 resets,
posterior mass on the TRUE meaning, P(>= .95), and the empirical R_comm s.d.

Readers: the three hand-crafted listeners on their own converged policies
(posterior recovered from the reward construction itself) and the IPL literal
on the ipl_prag policies (softmax over meaning-conditioned counterfactuals).

Usage: python measure_saturation_winners.py
"""
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import scout_support as S
from train_masac import Actor
from train_scout import IPLWrapper

LOG3 = math.log(3)

GROUPS = [
    ("progress", "results_scout3/progress_s*"),
    ("filter", "results_scout3/filter_s*"),
    ("simple", "results_scout3/simple_s*"),
    ("ipl-literal", "results_ipl/iplprag06_s*"),
]


def measure(run_dir, kind, ipl=None, n_envs=64, n_resets=8, seed=12345):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                    weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    gen = torch.Generator().manual_seed(seed)

    post, rcomm = [], []
    with torch.no_grad():
        for r in range(n_resets):
            env = S.ScoutSupportEnv(n_envs, seed=seed + r)
            obs = env.reset()
            for t in range(S.EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=False)  # on-policy
                if kind == "ipl-literal":
                    mus = ipl.means(obs[:, 0])                 # (E,M,2)
                    d = (a[:, 0].unsqueeze(1) - mus).pow(2).sum(-1)
                    p = F.softmax(-d / (2 * IPLWrapper.KAPPA ** 2), dim=-1)
                    pt = p.gather(-1, env.target.unsqueeze(-1)).squeeze(-1)
                    post.append(pt)
                    rcomm.append(torch.log(pt.clamp(min=1e-8)) + LOG3)
                    obs, _, _ = env.step(a)
                else:
                    # hand-crafted kinds need pre-step geometry and the
                    # post-step env; recover the posterior from the reward,
                    # which is log p_true + log 3 by construction
                    pre_pos = env.pos[:, 0].clone()
                    obs, _, _ = env.step(a)
                    rc = S.scout_comm_reward(env, a, kind, gen=gen,
                                             pre_pos=pre_pos)
                    rcomm.append(rc)
                    post.append(torch.exp(rc - LOG3))
    post = torch.cat(post).numpy()
    rcomm = torch.cat(rcomm).numpy()
    return {"mean_posterior": float(post.mean()),
            "p_ge_95": float((post >= 0.95).mean()),
            "rcomm_sd": float(rcomm.std())}


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ipl = None
    out = {}
    for label, pat in GROUPS:
        if label == "ipl-literal" and ipl is None:
            ck = torch.load(os.path.join("results_scout3", "baseline_s9",
                                         "model.pt"), map_location="cpu",
                            weights_only=True)
            ref = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
            ref.load_state_dict(ck["actor"])
            ref.eval()
            ipl = IPLWrapper(ref)
        rows = []
        for d in sorted(glob.glob(pat)):
            if not os.path.exists(os.path.join(d, "model.pt")):
                continue
            m = measure(d, label, ipl=ipl)
            m["run"] = os.path.basename(d)
            rows.append(m)
        out[label] = rows
        f = lambda k: np.array([r[k] for r in rows])
        print("%-12s n=%2d  mean posterior %.3f  P(>=.95) %.3f  "
              "R_comm sd %.3f"
              % (label, len(rows), f("mean_posterior").mean(),
                 f("p_ge_95").mean(), f("rcomm_sd").mean()), flush=True)

    with open("results_scout3/saturation_winners.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results_scout3/saturation_winners.json")
    print("\nlearned family, same protocol (from the paper): full-context "
          "0.966/0.926/0.45; viewpoint 0.92/0.81/0.49; window 0.88/0.60/0.47")


if __name__ == "__main__":
    main()
