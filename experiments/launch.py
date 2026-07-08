"""Launch the full experiment sweep: conditions x seeds, a few processes at a time.

Usage: python launch.py --outroot results --seeds 0 1 2 --cycles 150 --workers 4
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

CONDITIONS = ["baseline", "oracle", "heuristic", "simple", "learned", "learned_prag"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cycles", type=int, default=150)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--script", default="train_masac.py")
    p.add_argument("--voi", type=float, default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    jobs = []
    for cond, seed in itertools.product(args.conditions, args.seeds):
        outdir = os.path.join(args.outroot, f"{cond}_s{seed}")
        if os.path.exists(os.path.join(outdir, "model.pt")):
            print(f"skip {cond} s{seed} (done)")
            continue
        jobs.append((cond, seed, outdir))

    running = []
    while jobs or running:
        running = [(proc, tag) for proc, tag in running if proc.poll() is None]
        while jobs and len(running) < args.workers:
            cond, seed, outdir = jobs.pop(0)
            os.makedirs(outdir, exist_ok=True)
            log = open(os.path.join(outdir, "log.txt"), "w")
            cmd = [sys.executable, os.path.join(here, args.script),
                   "--condition", cond, "--seed", str(seed),
                   "--cycles", str(args.cycles), "--lam", str(args.lam),
                   "--outdir", outdir]
            if args.voi is not None:
                cmd += ["--voi", str(args.voi)]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, f"{cond}_s{seed}"))
            print(f"launched {cond} s{seed} (pid {proc.pid}); "
                  f"{len(jobs)} queued, {len(running)} running", flush=True)
        time.sleep(20)
    print("ALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
