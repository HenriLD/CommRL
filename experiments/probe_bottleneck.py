"""Post-hoc probes for the stage-2 bottleneck suite (the decisive tests).

Content probe: closed-form ridge regression from the per-episode latent
mu_z to the true bearing and to each decoy bearing. If task pressure (not
the legibility reward) selects content, the true bearing should be
decodable from z and the decoys should not, in every condition.

Transparency probes: matched InfoNCE critics estimate lower bounds on
I(Z; behavior) (fresh critic on masked public context + action vs. z --
fair to base2, which trained no listener) and I(Z; private). Their ratio
is the transparency coefficient: how much of what the agent knows-and-uses
is readable from how it acts. The legibility arms should raise the
numerator; beta is fixed so the denominator should hold still.

Usage: python probe_bottleneck.py [--conds base2 vibnce03 vibnceprag03]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scout_support as S
import scout_bottleneck as B

N_PART = 15
ROLLOUT_RESETS = 6
N_ENVS = 128


def collect(actor, seed):
    """Rollouts -> per-episode (priv, mu_z, theta, decoys) and
    per-step (masked public+action features, episode index)."""
    ep_priv, ep_z, ep_th, ep_dec = [], [], [], []
    feats, ep_idx = [], []
    ep0 = 0
    for r in range(ROLLOUT_RESETS):
        env = B.DecoyScoutEnv(N_ENVS, seed=seed * 100 + r)
        obs = env.reset()
        with torch.no_grad():
            mu, _ = actor.encode(obs[:, 0])
        ep_priv.append(obs[:, 0, B.PRIV_SLICE].clone())
        ep_z.append(mu)
        ep_th.append(env.theta.clone())
        ep_dec.append(env.decoys.clone())
        with torch.no_grad():
            for t in range(S.EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=True)
                x = obs[:, 0, :S.OBS_DIM].clone()
                x[:, S.PREF_SLICE] = 0.0
                x[:, S.PARTNER_PREF_SLICE] = 0.0
                feats.append(torch.cat([x, a[:, 0]], dim=1))
                ep_idx.append(torch.arange(ep0, ep0 + N_ENVS))
                obs, _, _ = env.step(a)
        ep0 += N_ENVS
    return (torch.cat(ep_priv), torch.cat(ep_z), torch.cat(ep_th),
            torch.cat(ep_dec), torch.cat(feats), torch.cat(ep_idx))


def ridge_r2(z, target, lam=1e-3):
    """Closed-form ridge z -> target (n, d); returns R^2."""
    zc = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1)
    A = zc.T @ zc + lam * torch.eye(zc.shape[1])
    W = torch.linalg.solve(A, zc.T @ target)
    pred = zc @ W
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean(0)).pow(2).sum()
    return (1 - ss_res / ss_tot).item()


class NCECritic(nn.Module):
    def __init__(self, x_dim, z_dim=B.Z_DIM, hidden=128, emb=32):
        super().__init__()
        self.fx = nn.Sequential(nn.Linear(x_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, emb))
        self.fz = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, emb))
        self.scale = emb ** -0.5

    def bound(self, x, z, steps=600, batch=512, lr=1e-3, seed=0):
        """Train InfoNCE and return the converged bound in nats
        (mean of the last 50 steps), capped at log(K+1)."""
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        tail = []
        for it in range(steps):
            i = torch.randint(0, x.shape[0], (batch,), generator=g)
            perm = torch.randint(0, x.shape[0], (batch, N_PART), generator=g)
            zs = torch.cat([z[i].unsqueeze(1), z[perm]], dim=1)
            e = self.fx(x[i])
            f = self.fz(zs)
            sc = torch.einsum("be,bke->bk", e, f) * self.scale
            loss = F.cross_entropy(sc, torch.zeros(batch, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step()
            if it >= steps - 50:
                tail.append(math.log(N_PART + 1) - loss.item())
        return float(np.mean(tail))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conds", nargs="+",
                   default=["base2", "vibnce03", "vibnceprag03"])
    p.add_argument("--out", default="results_bn/probes.json")
    args = p.parse_args()
    torch.set_num_threads(4)

    results = {}
    for cond in args.conds:
        rows = []
        for d in sorted(glob.glob(f"results_bn/{cond}_s*")):
            mp = os.path.join(d, "model.pt")
            if not os.path.exists(mp):
                continue
            actor = B.VIBActor()
            actor.load_state_dict(torch.load(mp, map_location="cpu")["actor"])
            actor.eval()
            seed = int(d.split("_s")[-1])
            priv, z, th, dec, feats, ep_idx = collect(actor, seed)
            tgt_true = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
            r2_true = ridge_r2(z, tgt_true)
            r2_dec = []
            for k in range(B.N_DECOY):
                t = torch.stack([torch.cos(dec[:, k]), torch.sin(dec[:, k])], dim=1)
                r2_dec.append(ridge_r2(z, t))
            r2_noise = ridge_r2(z, priv[:, -B.N_NOISE:])
            i_beh = NCECritic(feats.shape[1]).bound(feats, z[ep_idx], seed=seed)
            i_priv = NCECritic(B.PRIV_DIM).bound(priv, z, seed=seed)
            rows.append({"seed": seed, "r2_true": r2_true,
                         "r2_decoy": float(np.mean(r2_dec)),
                         "r2_noise": r2_noise,
                         "i_behavior": i_beh, "i_private": i_priv,
                         "transparency": i_beh / max(i_priv, 1e-6)})
            print(f"[{cond} s{seed}] R2 true {r2_true:.3f} decoy "
                  f"{np.mean(r2_dec):.3f} noise {r2_noise:.3f}  "
                  f"I(Z;beh) {i_beh:.2f} I(Z;priv) {i_priv:.2f} "
                  f"T {i_beh/max(i_priv,1e-6):.3f}", flush=True)
        results[cond] = rows

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\nsummary (mean +- sem over seeds):")
    for cond, rows in results.items():
        for k in ["r2_true", "r2_decoy", "r2_noise", "i_behavior",
                  "i_private", "transparency"]:
            v = np.array([r[k] for r in rows])
            print(f"  {cond:14s} {k:12s} {v.mean():.3f} +- "
                  f"{v.std(ddof=1)/len(v)**.5:.3f}")


if __name__ == "__main__":
    main()
