"""Benchmark the SAC update step (our actual workload shape) on CPU vs GPU.

Measures updates/sec for the twin-critic + actor + listener backward passes
at several batch sizes, matching train_scout.py's network sizes.

Usage: python bench_device.py --device cuda
"""

import argparse
import time

import torch
import torch.nn as nn

OBS_DIM, ACT_DIM, N_AGENTS = 49, 2, 2


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, o))


def bench(device, batch, iters=200, threads=4, hidden=256):
    torch.set_num_threads(threads)
    dev = torch.device(device)
    critic1 = mlp(N_AGENTS * (OBS_DIM + ACT_DIM), hidden, 1).to(dev)
    critic2 = mlp(N_AGENTS * (OBS_DIM + ACT_DIM), hidden, 1).to(dev)
    actor = mlp(OBS_DIM, hidden, 2 * ACT_DIM).to(dev)
    listener = mlp(OBS_DIM + ACT_DIM, hidden // 2, 3).to(dev)
    opt = torch.optim.Adam([*critic1.parameters(), *critic2.parameters(),
                            *actor.parameters(), *listener.parameters()], lr=3e-4)
    x_joint = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM), device=dev)
    x_obs = torch.randn(batch * N_AGENTS, OBS_DIM, device=dev)
    x_lis = torch.randn(batch, OBS_DIM + ACT_DIM, device=dev)
    y = torch.randn(batch, device=dev)
    lab = torch.randint(0, 3, (batch,), device=dev)

    def step():
        loss = ((critic1(x_joint).squeeze(-1) - y) ** 2).mean() + \
               ((critic2(x_joint).squeeze(-1) - y) ** 2).mean() + \
               actor(x_obs).pow(2).mean() + \
               nn.functional.cross_entropy(listener(x_lis), lab)
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(20):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return iters / dt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=None,
                   help="single batch size (for concurrency tests)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()
    if args.batch:
        ups = bench(args.device, args.batch, iters=args.iters,
                    threads=args.threads, hidden=args.hidden)
        print(f"{ups:.1f}")
    else:
        if args.device == "cuda":
            print("device:", torch.cuda.get_device_name(0), torch.__version__)
        for batch in [512, 2048, 8192]:
            ups = bench(args.device, batch)
            print(f"batch {batch:5d}: {ups:7.1f} updates/s "
                  f"({ups * batch:,.0f} samples/s)")
