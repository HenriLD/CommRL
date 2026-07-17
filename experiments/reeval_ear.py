"""Corrected evaluation for the bounded-listener ear condition.

An earlier snapshot's evaluate() omitted the ear injection this condition was
trained with, producing an artifactual 0.551 in the archived histories. This
script re-evaluates the saved checkpoints with the injection restored and
prints the corrected final-policy numbers reported in the paper (0.841+/-.023).

Usage: python reeval_ear.py --resroot results_boundedL
"""

import argparse
import glob
import os

import numpy as np
import torch

import scout_support as S
from train_masac import Actor
from train_scout import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resroot", default="results_boundedL")
    p.add_argument("--cond", default="learned_act_ear")
    args = p.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.resroot, f"{args.cond}_s*"))):
        ckpt = torch.load(os.path.join(d, "model.pt"), map_location="cpu",
                          weights_only=True)
        actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
        actor.load_state_dict(ckpt["actor"])
        listener = S.ScoutListener(inputs=ckpt.get("listener_inputs") or "act")
        listener.load_state_dict(ckpt["listener"])
        m = evaluate(actor, args.cond, listener, seed=12345)
        rows.append((os.path.basename(d), m["r_ext"], m["commit_acc"]))
        print(f"{rows[-1][0]:24s} r_ext {m['r_ext']:.3f}  commit {m['commit_acc']:.3f}")

    r = np.array([x[1] for x in rows])
    print(f"\ncorrected: r_ext {r.mean():.3f}+/-{r.std(ddof=1)/np.sqrt(len(r)):.3f}"
          f"  (archived histories contain the pre-fix 0.551 artifact)")


if __name__ == "__main__":
    main()
