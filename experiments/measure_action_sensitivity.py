"""Decompose each listener's reward variance into action-driven and
context-driven parts.

The totals tie: the IPL-literal's R_comm spread (0.47) equals the learned
family's (0.45-0.49), yet one shapes behavior and the other does not. A
shaping gradient can only exploit variance the ACTION controls, so the
discriminating quantity is the within-state spread: fix the state, resample
K actions from the policy's own distribution, and measure how much R_comm
moves. Both readers are plain (s,a)-functions, so no env stepping is needed.

  within-state sd   sqrt(E_s[ Var_a(R | s) ])  -- action-driven
  total sd          over the joint occupancy    -- action + context

Prediction: the learned listener's variance is almost entirely contextual
(saturated per state); the IPL's is almost entirely action-driven.

Usage: python measure_action_sensitivity.py
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
K = 8              # resampled actions per state
N_ENVS = 64
N_RESETS = 4


def rollout_states(actor, seed):
    """States (scout obs) and targets over the policy's own occupancy."""
    obs_all, tgt_all = [], []
    with torch.no_grad():
        for r in range(N_RESETS):
            env = S.ScoutSupportEnv(N_ENVS, seed=seed + r)
            obs = env.reset()
            for t in range(S.EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=False)
                obs_all.append(obs.clone())
                tgt_all.append(env.target.clone())
                obs, _, _ = env.step(a)
    return torch.cat(obs_all), torch.cat(tgt_all)


def decompose(run_dir, reader, seed=54321):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                    weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    obs, tgt = rollout_states(actor, seed)                  # (N,2,OBS),(N,)

    with torch.no_grad():
        # K policy-sampled actions per state
        rep = obs.repeat_interleave(K, dim=0)
        a, _ = actor.sample(rep, deterministic=False)
        a0 = a[:, 0]                                        # (N*K, 2)
        s0 = rep[:, 0]

        if reader == "learned":
            listener = S.ScoutListener()
            listener.load_state_dict(ck["listener"])
            listener.eval()
            p = F.softmax(listener(s0, a0), dim=-1)
        else:                                               # ipl
            mus = reader.means(s0)
            d = (a0.unsqueeze(1) - mus).pow(2).sum(-1)
            p = F.softmax(-d / (2 * IPLWrapper.KAPPA ** 2), dim=-1)

        pt = p.gather(-1, tgt.repeat_interleave(K).unsqueeze(-1)).squeeze(-1)
        r = (torch.log(pt.clamp(min=1e-8)) + LOG3).reshape(-1, K)  # (N,K)

    within = r.var(dim=1, unbiased=True).mean().sqrt().item()
    total = r.reshape(-1).std().item()
    return {"within_sd": within, "total_sd": total,
            "frac_action": within ** 2 / max(total ** 2, 1e-12)}


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    torch.set_num_threads(2)

    ck = torch.load(os.path.join("results_scout3", "baseline_s9", "model.pt"),
                    map_location="cpu", weights_only=True)
    ref = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    ref.load_state_dict(ck["actor"])
    ref.eval()
    ipl = IPLWrapper(ref)

    out = {}
    for label, pat, reader in [
            ("learned full-context", "results_scout3/learned_s*", "learned"),
            ("ipl-literal", "results_ipl/iplprag06_s*", ipl)]:
        rows = []
        for d in sorted(glob.glob(pat)):
            if not os.path.exists(os.path.join(d, "model.pt")):
                continue
            if reader == "learned":
                probe = torch.load(os.path.join(d, "model.pt"),
                                   map_location="cpu", weights_only=True)
                if probe.get("listener") is None:
                    continue
            rows.append(decompose(d, reader))
            print("  done", os.path.basename(d), flush=True)
        out[label] = rows
        f = lambda k: np.array([r[k] for r in rows])
        print("%-22s n=%d  within-state sd %.3f  total sd %.3f  "
              "action-driven fraction %.2f"
              % (label, len(rows), f("within_sd").mean(), f("total_sd").mean(),
                 f("frac_action").mean()), flush=True)

    with open("results_scout3/action_sensitivity.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results_scout3/action_sensitivity.json")


if __name__ == "__main__":
    main()
