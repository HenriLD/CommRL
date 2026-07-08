"""Is the supporter network expressive enough to decode the signal while
doing the task? Three separable diagnostics on a trained checkpoint:

1. input probes  — linear / MLP-64 / actor-sized MLP trained (supervised) to
   decode the scout's target from the SUPPORTER's observations, split by
   pre-key vs post-key steps. Tests whether the information is present and
   extractable at the actor's capacity.
2. hidden probes — linear decoders on the trained actor's hidden activations
   for supporter inputs. Tests whether RL training already extracted the
   signal internally.
3. intervention  — rollouts with the supporter's ear features forced to the
   true one-hot / uniform / the normal posterior. Tests whether the trained
   policy USES the ear when it is confident.

Usage: python capacity_probe.py --run results_scout2/learned_ear_s0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear


def collect(actor, listener, ear, n_envs, seed, deterministic=True):
    env = S.ScoutSupportEnv(n_envs, seed=seed)
    obs = env.reset()
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    sup_obs, tgt, key, ts = [], [], [], []
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=deterministic)
            sup_obs.append(obs[:, 1].clone())
            tgt.append(env.target.clone())
            key.append(env.has_key.clone())
            ts.append(torch.full((n_envs,), t))
            next_obs, _, _ = env.step(a)
            if ear:
                post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
    return (torch.cat(sup_obs), torch.cat(tgt), torch.cat(key), torch.cat(ts))


def train_probe(net, X, y, epochs=6, batch=512, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            loss = F.cross_entropy(net(X[j]), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def probe_acc(net, X, y, key):
    with torch.no_grad():
        hit = (net(X).argmax(-1) == y).float()
    pre, post = hit[key < 0.5], hit[key >= 0.5]
    return pre.mean().item() if len(pre) else float("nan"), \
           post.mean().item() if len(post) else float("nan")


def intervention(actor, listener, mode, n_envs=512, seed=999):
    """mode: normal | true | uniform"""
    env = S.ScoutSupportEnv(n_envs, seed=seed)
    obs = env.reset()
    m_oh = F.one_hot(env.target, S.N_MEANINGS).float()
    uni = torch.full((n_envs, S.N_MEANINGS), 1 / 3)
    obs = inject_ear(obs, m_oh if mode == "true" else uni)
    commit, r_ext = [], 0.0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            next_obs, info, _ = env.step(a)
            commit.append(info["commit_acc"].mean().item())
            r_ext += info["r_ext"].mean().item()
            if mode == "true":
                post = m_oh
            elif mode == "uniform":
                post = uni
            else:
                post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
            next_obs = inject_ear(next_obs, post)
            obs = next_obs
    commit = np.array(commit)
    return {"r_ext_per_step": r_ext / S.EPISODE_LEN,
            "commit_t10": commit[9], "commit_t20": commit[19],
            "commit_mean": commit.mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_envs", type=int, default=512)
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    listener = None
    ear = ckpt.get("listener") is not None
    if ear:
        listener = S.ScoutListener()
        listener.load_state_dict(ckpt["listener"])

    print(f"run: {args.run}  (ear={ear})")
    Xtr, ytr, ktr, _ = collect(actor, listener, ear, args.n_envs, seed=31,
                               deterministic=False)
    Xte, yte, kte, _ = collect(actor, listener, ear, args.n_envs // 2, seed=32)

    print("\n1) input probes on SUPPORTER observations (decode acc pre/post key, chance .33)")
    D = Xtr.shape[-1]
    for name, net in [
        ("linear", nn.Linear(D, 3)),
        ("mlp-64", nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 3))),
        ("mlp-256x256 (actor-sized)", nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 3))),
    ]:
        train_probe(net, Xtr, ytr)
        pre, post = probe_acc(net, Xte, yte, kte)
        print(f"   {name:28s} pre-key {pre:.3f}   post-key {post:.3f}")

    print("\n2) linear probes on trained actor hidden activations (supporter inputs)")
    with torch.no_grad():
        h1_tr = actor.net[1](actor.net[0](Xtr)); h2_tr = actor.net(Xtr)
        h1_te = actor.net[1](actor.net[0](Xte)); h2_te = actor.net(Xte)
    for name, htr, hte in [("layer1", h1_tr, h1_te), ("layer2", h2_tr, h2_te)]:
        net = train_probe(nn.Linear(htr.shape[-1], 3), htr, ytr)
        pre, post = probe_acc(net, hte, yte, kte)
        print(f"   {name:28s} pre-key {pre:.3f}   post-key {post:.3f}")

    if ear:
        print("\n3) ear intervention (256 envs, deterministic policy)")
        for mode in ["normal", "true", "uniform"]:
            r = intervention(actor, listener, mode)
            print(f"   ear={mode:8s} r_ext/step {r['r_ext_per_step']:+.3f}  "
                  f"commit@t10 {r['commit_t10']:.3f}  commit@t20 {r['commit_t20']:.3f}  "
                  f"mean {r['commit_mean']:.3f}")


if __name__ == "__main__":
    main()
