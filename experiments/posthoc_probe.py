"""Post-hoc legibility probe: for each trained run, roll out the final
deterministic policy, train a fresh listener from scratch on those rollouts,
and report held-out decoding accuracy of private preferences.

This measures how much intent information is decodable from each condition's
converged behavior under an identical decoding budget, independently of any
listener used during training.

Usage: python posthoc_probe.py --resroot results --out results/posthoc_probe.json
"""

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

from pragmatic_spread import PragmaticSpreadEnv, LearnedListener, EPISODE_LEN
from train_masac import Actor

N_TRAIN_STEPS = 40      # vectorized episodes of data collection (train)
N_TEST_STEPS = 10       # held-out episodes
N_ENVS = 64
EPOCHS = 4
BATCH = 512


def collect(actor, oracle, seed, n_cycles):
    env = PragmaticSpreadEnv(N_ENVS, oracle=oracle, seed=seed)
    X_obs, X_act, Y = [], [], []
    with torch.no_grad():
        for c in range(n_cycles):
            obs = env.reset()
            for t in range(EPISODE_LEN):
                a, _ = actor.sample(obs, deterministic=True)
                X_obs.append(obs.reshape(-1, obs.shape[-1]).clone())
                X_act.append(a.reshape(-1, 2).clone())
                Y.append(env.pref.reshape(-1).clone())
                obs, _, _ = env.step(a)
    return torch.cat(X_obs), torch.cat(X_act), torch.cat(Y)


def probe_run(run_dir, seed=0):
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                      weights_only=True)
    actor = Actor(); actor.load_state_dict(ckpt["actor"])
    oracle = "oracle" in os.path.basename(run_dir)

    obs_tr, act_tr, y_tr = collect(actor, oracle, seed=1000 + seed, n_cycles=N_TRAIN_STEPS)
    obs_te, act_te, y_te = collect(actor, oracle, seed=2000 + seed, n_cycles=N_TEST_STEPS)

    torch.manual_seed(seed)
    probe = LearnedListener()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n = obs_tr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            loss = F.cross_entropy(probe(obs_tr[j], act_tr[j]), y_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = probe(obs_te, act_te)
        acc = (logits.argmax(-1) == y_te).float().mean().item()
        ce = F.cross_entropy(logits, y_te).item()
    return acc, ce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results")
    p.add_argument("--out", default="results/posthoc_probe.json")
    args = p.parse_args()

    results = {}
    for run in sorted(glob.glob(os.path.join(args.resroot, "*"))):
        if not os.path.exists(os.path.join(run, "model.pt")):
            continue
        name = os.path.basename(run)
        acc, ce = probe_run(run)
        results[name] = {"posthoc_acc": acc, "posthoc_ce": ce}
        print(f"{name}: post-hoc decode acc {acc:.3f}  ce {ce:.3f}", flush=True)

    # aggregate by condition
    agg = {}
    for name, r in results.items():
        cond = name.rsplit("_s", 1)[0]
        agg.setdefault(cond, []).append(r["posthoc_acc"])
    print("\n=== post-hoc decodability by condition ===")
    for cond, accs in agg.items():
        t = torch.tensor(accs)
        print(f"{cond.ljust(14)} {t.mean():.3f} +/- {t.std().item()/max(1,(len(accs)-1))**0.5:.3f}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
