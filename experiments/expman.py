"""Experiment manager: launch queues of runs, check status, and report
aggregate statistics -- all with compact, terminal-friendly output.

Jobs are declared in a JSON spec: a list of {name, args} entries where args
are extra CLI flags for train_scout.py. The manager runs `workers` jobs at a
time, skips completed runs (model.pt present), and writes everything under
--outroot/<name>/.

  python expman.py launch --spec suite.json --outroot results_suite --workers 5
  python expman.py status --outroot results_suite
  python expman.py report --outroot results_suite --baseline baseline_a --oracle oracle_a

`status` prints one line per run (cycle, wall min). `report` prints one line
per condition group (mean +/- sem, gap closure vs the named baseline/oracle,
Welch t) -- grouping runs named <cond>_s<seed>.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_PY = r"C:\Users\henri\AppData\Local\Programs\Python\Python313\python.exe"


WORKER_CTRL = os.path.join(HERE, "workers.txt")

# Trainers run below-normal priority with OpenMP spin-waiting off: the worker
# cap controls GPU parallelism, but interactive responsiveness comes from
# priority — idle-loop spinning otherwise pegs every core regardless of count.
CHILD_FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
CHILD_ENV = {**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"}


def _worker_cap(default):
    """Dynamic worker cap: write an integer to experiments/workers.txt at any
    time to throttle (0 = pause launching; in-flight runs always finish);
    delete the file to restore the --workers default. Re-read every poll, so
    changes take effect as running jobs complete."""
    try:
        with open(WORKER_CTRL) as f:
            return max(0, int(f.read().strip()))
    except (OSError, ValueError):
        return default


def launch(args):
    with open(args.spec) as f:
        jobs = json.load(f)
    queue = []
    for job in jobs:
        outdir = os.path.join(args.outroot, job["name"])
        if os.path.exists(os.path.join(outdir, "model.pt")):
            continue
        queue.append((job["name"], outdir, job["args"]))
    print(f"{len(queue)} jobs queued ({len(jobs) - len(queue)} already done)")
    running = []
    cap_seen = args.workers
    while queue or running:
        running = [(p, n) for p, n in running if p.poll() is None]
        cap = _worker_cap(args.workers)
        if cap != cap_seen:
            print(f"worker cap -> {cap} (workers.txt)", flush=True)
            cap_seen = cap
        while queue and len(running) < cap:
            name, outdir, extra = queue.pop(0)
            os.makedirs(outdir, exist_ok=True)
            cmd = [args.python or GPU_PY, os.path.join(HERE, args.script),
                   "--outdir", outdir] + extra
            log = open(os.path.join(outdir, "log.txt"), "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=CHILD_FLAGS, env=CHILD_ENV)
            running.append((p, name))
            print(f"launched {name} ({len(queue)} left)", flush=True)
        time.sleep(15)
    print("ALL JOBS DONE")


def status(args):
    for d in sorted(glob.glob(os.path.join(args.outroot, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        done = os.path.exists(os.path.join(d, "model.pt"))
        line = ""
        log = os.path.join(d, "log.txt")
        if os.path.exists(log):
            with open(log, errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                line = lines[-1][-72:]
        print(f"{'DONE ' if done else 'run  '}{name:28s} {line}")


def _agg(outroot, cond, key):
    vals = []
    for p in sorted(glob.glob(os.path.join(outroot, f"{cond}_s*", "history.json"))):
        # only completed runs: an in-flight run has history.json but no model.pt
        if not os.path.exists(os.path.join(os.path.dirname(p), "model.pt")):
            continue
        h = json.load(open(p))["history"][-3:]
        vals.append(np.mean([e[key] for e in h]))
    return np.array(vals)


def report(args):
    conds = defaultdict(int)
    for d in glob.glob(os.path.join(args.outroot, "*_s*")):
        name = os.path.basename(d)
        if os.path.exists(os.path.join(d, "model.pt")):
            conds[name.rsplit("_s", 1)[0]] += 1
    base = _agg(args.outroot, args.baseline, "r_ext") if args.baseline else None
    orac = _agg(args.outroot, args.oracle, "r_ext") if args.oracle else None
    print(f"{'condition':22s} {'n':>2s} {'r_ext':>14s} {'commit':>7s} {'gap%':>6s} {'t':>6s}")
    for cond in sorted(conds):
        v = _agg(args.outroot, cond, "r_ext")
        c = _agg(args.outroot, cond, "commit_acc")
        m, s = v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0
        gap = t = ""
        if base is not None and orac is not None and cond not in (args.baseline, args.oracle):
            gap = f"{(m - base.mean()) / max(1e-9, orac.mean() - base.mean()) * 100:.0f}"
            va, vb = v.var(ddof=1) / len(v), base.var(ddof=1) / len(base)
            t = f"{(m - base.mean()) / math.sqrt(va + vb):.2f}"
        print(f"{cond:22s} {len(v):2d} {m:7.3f}+/-{s:.3f} {c.mean():7.3f} {gap:>6s} {t:>6s}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("launch")
    pl.add_argument("--spec", required=True)
    pl.add_argument("--outroot", required=True)
    pl.add_argument("--workers", type=int, default=5)
    pl.add_argument("--script", default="train_scout.py",
                    help="trainer module (e.g. train_merge.py, train_masac.py)")
    pl.add_argument("--python", default=None)
    ps = sub.add_parser("status")
    ps.add_argument("--outroot", required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--outroot", required=True)
    pr.add_argument("--baseline", default=None)
    pr.add_argument("--oracle", default=None)
    args = p.parse_args()
    {"launch": launch, "status": status, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
