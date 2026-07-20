"""Convention vs grounding: what is each policy's early behavior readable BY?

Every reader is grounded in something; none is a view from nowhere. The
experiment orders four readers by what grounds them, and asks how much of each
policy's early-window information each grounding recovers:

  geometric   grounded in a HUMAN PRIOR (curving toward what you mean --
              the paper's fixed probe, never trained on anything)
  ipl         grounded in a FROZEN NON-COMMUNICATIVE POLICY (the paper's
              inverse-planning listener; competence as common ground)
  fit-self    grounded in THE SPEAKER'S OWN EQUILIBRIUM: a fresh decoder
              fitted to this seed's converged rollouts, evaluated on held-out
              episodes. Reads any regularity, including an arbitrary code.
  fit-cross   grounded in OTHER EQUILIBRIA: the same decoder fitted on the
              other seeds' rollouts. A convention is equilibrium-specific,
              so it should not transfer; a grounded signal should.

The two derived quantities:

  residue  = fit-self - geometric   information present but invisible to the
                                    human prior (candidate convention)
  transfer = fit-cross - chance     how much of the code is shared across
                                    equilibria (grounded codes transfer)

All readers see only supporter-visible statistics (site bearings from the
scout + the scout's action), the paper's viewpoint-bounded family. Primary
window is the pre-key leg (t <= 15), where the value of information is at
full weight. CPU-only; safe to run alongside GPU training.

Usage: python convention_vs_grounded.py [--root results_conv] [--quick]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import scout_support as S
from train_masac import Actor
from train_scout import IPLWrapper

REF_CKPT = os.path.join("results_scout3", "baseline_s9", "model.pt")
N_ENVS = 256
EARLY = 15
CHANCE = 1.0 / 3.0


# ------------------------------ rollouts ------------------------------

def collect(run_dir, n_resets, seed0, oracle):
    """Deterministic rollouts. Returns dict of stacked per-step tensors."""
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    out = {k: [] for k in ("dirs", "act", "sobs", "target", "t", "key")}
    with torch.no_grad():
        for r in range(n_resets):
            env = S.ScoutSupportEnv(N_ENVS, oracle=oracle, seed=seed0 + r)
            obs = env.reset()
            for t in range(S.EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=True)
                out["dirs"].append(env.goal_dirs()[:, 0])      # (E,M,2)
                out["act"].append(a[:, 0])                     # (E,2)
                out["sobs"].append(obs[:, 0])                  # (E,OBS)
                out["target"].append(env.target)               # (E,)
                out["t"].append(torch.full((N_ENVS,), t))
                out["key"].append(env.has_key.clone())         # state when acting
                obs, _, _ = env.step(a)
    return {k: torch.cat(v) for k, v in out.items()}


def features(d, scout_only=True):
    """Fitted-reader inputs: scout-centric state with the meaning slot zeroed
    (exactly as IPLWrapper builds its counterfactuals) plus the action.

    scout_only additionally zeroes the partner block (supporter relative
    position and velocity). Without this the decoder reads the TARGET off the
    co-trained supporter's drift -- the oracle's supporter walks straight at
    the answer from step 0 -- which measures the pair's joint state, not the
    scout's signaling. The with-partner variant is kept as a diagnostic."""
    s = d["sobs"].clone()
    s[..., S.PREF_SLICE] = 0.0
    s[..., S.PARTNER_PREF_SLICE] = 0.0
    if scout_only:
        pb = slice(S.PARTNER_PREF_SLICE.start - 4, S.PARTNER_PREF_SLICE.start)
        s[..., pb] = 0.0                      # partner_rel + partner_vel
    return torch.cat([s, d["act"],
                      (d["t"].float() / S.EPISODE_LEN).unsqueeze(-1)], dim=-1)


def mask_early(d):
    """The structurally uninformative leg: before the key is picked up. A fixed
    t-cutoff leaks post-pickup steps (the scout then walks straight at the
    target and every reader trivially succeeds), so gate on the flag itself."""
    return d["key"] < 0.5


# ------------------------------ readers -------------------------------

def read_geometric(d, m):
    acc, _ = S.probe_intent_metrics(d["dirs"][m].unsqueeze(1),
                                    d["target"][m], d["act"][m].unsqueeze(1))
    return acc.mean().item()


class _IPL:
    def __init__(self):
        ck = torch.load(REF_CKPT, map_location="cpu", weights_only=True)
        ref = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
        ref.load_state_dict(ck["actor"])
        ref.eval()
        self.w = IPLWrapper(ref)

    def acc(self, d, m):
        with torch.no_grad():
            mus = self.w.means(d["sobs"][m])                   # (N,M,2)
            dist = (d["act"][m].unsqueeze(1) - mus).pow(2).sum(-1)
            pred = (-dist / (2 * IPLWrapper.KAPPA ** 2)).argmax(-1)
        return (pred == d["target"][m]).float().mean().item()


def fit_decoder(x, y, epochs, gen):
    net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, S.N_MEANINGS))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(x)
    for _ in range(epochs):
        idx = torch.randperm(n, generator=gen)
        for i in range(0, n, 4096):
            j = idx[i:i + 4096]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(x[j]), y[j])
            loss.backward()
            opt.step()
    net.eval()
    return net


def decoder_acc(net, x, y):
    with torch.no_grad():
        return (net(x).argmax(-1) == y).float().mean().item()


# ------------------------------ analysis ------------------------------

def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.mean() - b.mean()) / math.sqrt(
        a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_conv")
    ap.add_argument("--conds", nargs="+",
                    default=["baseline", "oracle", "progress"])
    ap.add_argument("--resets", type=int, default=4)   # 3 train + 1 eval
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    torch.set_num_threads(2)          # do not fight the training workers

    ipl = _IPL()
    gen = torch.Generator().manual_seed(0)

    # ---- collect all rollouts first (train resets + one held-out eval) ----
    data = {}
    for cond in args.conds:
        runs = sorted(glob.glob(os.path.join(args.root, cond + "_s*")))
        runs = [r for r in runs if os.path.exists(os.path.join(r, "model.pt"))]
        if args.quick:
            runs = runs[:2]
        for r in runs:
            seed = int(r.rsplit("_s", 1)[1])
            data[(cond, seed)] = collect(r, args.resets, 20_000 + 100 * seed,
                                         oracle=(cond == "oracle"))
            print("collected %s_s%d" % (cond, seed), flush=True)

    n_eval = N_ENVS * S.EPISODE_LEN                      # last reset = eval
    # precompute per-run masks and per-family features once
    trearly, feats = {}, {}
    for k, d in data.items():
        tr = torch.arange(len(d["target"])) < (args.resets - 1) * n_eval
        trearly[k] = tr & mask_early(d)
        for fam, so in (("", True), ("_full", False)):
            feats[k + (fam,)] = features(d, scout_only=so)

    results = {}
    for (cond, seed), d in sorted(data.items()):
        y = d["target"]
        early = mask_early(d)
        tr = torch.arange(len(y)) < (args.resets - 1) * n_eval
        ev = ~tr
        m_ev = ev & early                                # primary window
        m_post = ev & ~early                             # post-key contrast

        row = {"geometric": read_geometric(d, m_ev), "ipl": ipl.acc(d, m_ev),
               "geo_post": read_geometric(d, m_post),
               "ipl_post": ipl.acc(d, m_post)}
        # mean pickup step: pre-key steps per episode
        row["pickup"] = d["key"].reshape(args.resets, S.EPISODE_LEN, N_ENVS) \
                                .lt(0.5).float().sum(1).mean().item()

        peers = [(c, s) for (c, s) in data if c == cond and s != seed]
        for fam in ("", "_full"):
            x = feats[(cond, seed, fam)]
            xt, yt = x[tr & early], y[tr & early]
            net = fit_decoder(xt, yt, args.epochs, gen)
            row["fit_self" + fam] = decoder_acc(net, x[m_ev], y[m_ev])
            if fam == "":
                row["fit_self_train"] = decoder_acc(net, xt, yt)
            if peers:
                xs = torch.cat([feats[p + (fam,)][trearly[p]] for p in peers])
                ys = torch.cat([data[p]["target"][trearly[p]] for p in peers])
                # matched-n: the cross reader gets the SAME training budget as
                # the self reader, else data volume confounds grounding source
                sel = torch.randperm(len(ys), generator=gen)[:len(yt)]
                netc = fit_decoder(xs[sel], ys[sel], args.epochs, gen)
                row["fit_cross" + fam] = decoder_acc(netc, x[m_ev], y[m_ev])
        results[(cond, seed)] = row
        print("  %s_s%d  %s" % (cond, seed, "  ".join(
            "%s=%.3f" % (k, v) for k, v in row.items())), flush=True)

    # ------------------------------ report ------------------------------
    readers = ["geometric", "ipl", "fit_self", "fit_cross"]
    print("\nPRE-KEY WINDOW accuracy, mean +- se over seeds; chance %.3f"
          % CHANCE)
    print("%-10s" % "cond" + "".join("%16s" % r for r in readers)
          + "%16s%16s" % ("residue", "transfer"))
    agg = {}
    for cond in args.conds:
        vals = {r: [results[k][r] for k in results if k[0] == cond
                    and r in results[k]] for r in readers}
        agg[cond] = vals
        res = [s - g for s, g in zip(vals["fit_self"], vals["geometric"])]
        trn = [c - CHANCE for c in vals["fit_cross"]]
        cells = "".join("%9.3f+-%.3f" % (np.mean(v), np.std(v, ddof=1)
                                         / math.sqrt(len(v)))
                        for v in (vals[r] for r in readers))
        print("%-10s%s%9.3f+-%.3f%9.3f+-%.3f"
              % (cond, cells, np.mean(res),
                 np.std(res, ddof=1) / math.sqrt(len(res)),
                 np.mean(trn), np.std(trn, ddof=1) / math.sqrt(len(trn))))

    print("\nCONTEXT: post-key, pickup timing, and the with-partner diagnostic")
    print("%-10s%16s%16s%16s%16s%16s%16s"
          % ("cond", "geo_post", "ipl_post", "pickup_step", "fit_train",
             "self_full", "cross_full"))
    for cond in args.conds:
        cols = []
        for key in ("geo_post", "ipl_post", "pickup", "fit_self_train",
                    "fit_self_full", "fit_cross_full"):
            v = [results[k][key] for k in results
                 if k[0] == cond and key in results[k]]
            cols.append("%9.3f+-%.3f" % (np.mean(v), np.std(v, ddof=1)
                                         / math.sqrt(len(v))))
        print("%-10s%s" % (cond, "".join(cols)))

    print("\nkey contrasts (Welch t):")
    for cond in args.conds:
        v = agg[cond]
        print("  %-9s fit_self vs geometric  t=%5.2f   fit_self vs fit_cross  t=%5.2f"
              % (cond, welch(v["fit_self"], v["geometric"]),
                 welch(v["fit_self"], v["fit_cross"])))
    if "baseline" in agg and "progress" in agg:
        rb = [s - g for s, g in zip(agg["baseline"]["fit_self"],
                                    agg["baseline"]["geometric"])]
        rp = [s - g for s, g in zip(agg["progress"]["fit_self"],
                                    agg["progress"]["geometric"])]
        print("  residue baseline vs progress: t=%5.2f" % welch(rb, rp))

    with open("convention_vs_grounded.json", "w", encoding="utf-8") as f:
        json.dump({"%s_s%d" % k: v for k, v in results.items()}, f, indent=1)
    print("\nwrote convention_vs_grounded.json")


if __name__ == "__main__":
    main()
