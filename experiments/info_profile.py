"""Per-timestep information profile of every condition's converged behavior.

For each run: train a fresh probe listener on final-policy rollouts, then
plot decode accuracy as a function of episode timestep, averaged per
condition. Shows WHERE in the episode each condition's behavior carries
target information (chance = 1/3), independently of any training listener.

Usage: python info_profile.py --resroot results_scout3 --figdir ../papers/Conference_Paper/img
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import scout_support as S
from train_masac import Actor
from train_scout import inject_ear
from paperstyle import use_style, LABELS, COLORS, REF_STYLE

ORDER = ["baseline", "oracle", "filter", "learned", "learned_ear"]


def collect(actor, listener, ear, n_envs, seed, deterministic):
    env = S.ScoutSupportEnv(n_envs, seed=seed)
    obs = env.reset()
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    O, A, Y, T = [], [], [], []
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=deterministic)
            O.append(obs[:, 0].clone()); A.append(a[:, 0].clone())
            Y.append(env.target.clone())
            T.append(torch.full((n_envs,), t))
            next_obs, _, _ = env.step(a)
            if ear and listener is not None:
                post = torch.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
    return torch.cat(O), torch.cat(A), torch.cat(Y), torch.cat(T)


def profile_run(run_dir, device):
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

    Otr, Atr, Ytr, _ = collect(actor, listener, ear, 256, 71, deterministic=False)
    Ote, Ate, Yte, Tte = collect(actor, listener, ear, 128, 72, deterministic=True)

    probe = S.ScoutListener().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Otr, Atr, Ytr = Otr.to(device), Atr.to(device), Ytr.to(device)
    n = Otr.shape[0]
    for ep in range(4):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            j = perm[i:i + 1024]
            loss = F.cross_entropy(probe(Otr[j], Atr[j]), Ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        hit = (probe(Ote.to(device), Ate.to(device)).argmax(-1).cpu() == Yte).float()
    acc_t = np.array([hit[Tte == t].mean().item() for t in range(S.EPISODE_LEN)])
    return acc_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results_scout3")
    p.add_argument("--figdir", default="../papers/Conference_Paper/img")
    p.add_argument("--device", default="cpu")
    p.add_argument("--conds", nargs="+", default=ORDER)
    args = p.parse_args()
    device = torch.device(args.device)

    curves = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(args.resroot, "*"))):
        name = os.path.basename(run)
        cond = name.rsplit("_s", 1)[0]
        if cond not in args.conds or not os.path.exists(os.path.join(run, "model.pt")):
            continue
        curves[cond].append(profile_run(run, device))
        print("profiled", name, flush=True)

    use_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for cond in args.conds:
        if cond not in curves:
            continue
        c = np.stack(curves[cond])
        m, e = c.mean(0), c.std(0) / np.sqrt(c.shape[0])
        t = np.arange(1, len(m) + 1)
        if cond in REF_STYLE:
            ax.plot(t, m, label=LABELS[cond], **REF_STYLE[cond])
        else:
            ax.plot(t, m, label=LABELS[cond], color=COLORS[cond], lw=2.0)
        ax.fill_between(t, m - e, m + e,
                        color=REF_STYLE.get(cond, {}).get("color", COLORS.get(cond)),
                        alpha=0.13, lw=0)
    ax.axhline(1 / 3, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.text(49, 1 / 3 + 0.012, "chance", fontsize=7.5, color="#555555", ha="right")
    ax.set_xlabel("Episode timestep")
    ax.set_ylabel("Post-hoc probe decode accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(args.figdir, exist_ok=True)
    out = os.path.join(args.figdir, "info_profile.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
