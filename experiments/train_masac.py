"""MASAC (parameter-shared actors, centralized twin critic) on the
preference Simple Spread task, with pluggable pragmatic reward conditions.

Conditions:
  baseline   lambda = 0
  oracle     lambda = 0, observations include teammate preferences
  heuristic  thesis PRM: exclusivity L0 + RSA recursion (Algorithm 1)
  simple     thesis ablation: plain cosine L0 + RSA recursion
  learned    learned listener L_theta, R_comm = log L_theta(m|s,a) + log M
  learned_prag  learned listener + RSA recursion over sampled alternatives

Usage: python train_masac.py --condition learned --seed 0 --outdir results/learned_s0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pragmatic_spread import (
    PragmaticSpreadEnv, LearnedListener, heuristic_comm_reward,
    probe_intent_metrics, OBS_DIM, ACT_DIM, N_AGENTS, N_MEANINGS,
    EPISODE_LEN, PREF_SLICE,
)

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Linear(hidden, ACT_DIM)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        a = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class CentralCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        in_dim = N_AGENTS * OBS_DIM + N_AGENTS * ACT_DIM
        def q():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1, self.q2 = q(), q()

    def forward(self, obs, act):
        x = torch.cat([obs.flatten(1), act.flatten(1)], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, N_AGENTS, OBS_DIM))
        self.act = torch.zeros((capacity, N_AGENTS, ACT_DIM))
        self.r_ext = torch.zeros(capacity)
        self.r_comm = torch.zeros((capacity, N_AGENTS))
        self.next_obs = torch.zeros((capacity, N_AGENTS, OBS_DIM))
        self.done = torch.zeros(capacity)
        self.pref = torch.zeros((capacity, N_AGENTS), dtype=torch.long)
        self.idx, self.full = 0, False

    def push(self, obs, act, r_ext, r_comm, next_obs, done, pref):
        n = obs.shape[0]
        i = torch.arange(self.idx, self.idx + n) % self.capacity
        self.obs[i] = obs.cpu()
        self.act[i] = act.cpu()
        self.r_ext[i] = r_ext.cpu()
        self.r_comm[i] = r_comm.cpu()
        self.next_obs[i] = next_obs.cpu()
        self.done[i] = float(done)
        self.pref[i] = pref.cpu()
        self.idx = int((self.idx + n) % self.capacity)
        if self.idx < n:
            self.full = True
        self.size = self.capacity if self.full else self.idx

    def sample(self, batch):
        i = torch.randint(0, self.size, (batch,))
        to = lambda x: x[i].to(self.device)
        return (to(self.obs), to(self.act), to(self.r_ext), to(self.r_comm),
                to(self.next_obs), to(self.done), to(self.pref))


def evaluate(actor, cond, listener, device, seed, n_envs=64):
    env = PragmaticSpreadEnv(n_envs, device=device, oracle=(cond == "oracle"), seed=seed)
    obs = env.reset()
    gen = torch.Generator().manual_seed(seed)
    tot = {k: 0.0 for k in ["r_ext", "pref_bonus", "dist_pen", "coll_pen",
                            "probe_acc", "probe_ce", "comm_rw", "spec", "listener_acc"]}
    steps = 0
    with torch.no_grad():
        for t in range(EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            dirs, pref = env.goal_dirs(), env.pref
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "pref_bonus", "dist_pen", "coll_pen"]:
                tot[k] += info[k].mean().item()
            acc, ce = probe_intent_metrics(dirs, pref, a)
            tot["probe_acc"] += acc.mean().item()
            tot["probe_ce"] += ce.mean().item()
            tot["spec"] += env.specialization().mean().item()
            if cond in ("heuristic", "simple"):
                rc = heuristic_comm_reward(dirs, pref, a,
                                           exclusivity=(cond == "heuristic"), gen=gen)
                tot["comm_rw"] += rc.mean().item()
            elif listener is not None:
                rc = listener.comm_reward(obs, a, pref)
                tot["comm_rw"] += rc.mean().item()
                pred = listener(obs, a).argmax(-1)
                tot["listener_acc"] += (pred == pref).float().mean().item()
            obs = next_obs
            steps += 1
    return {k: v / steps for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["baseline", "oracle", "heuristic", "simple", "learned", "learned_prag"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--cycles", type=int, default=150)      # cycle = 1 episode across all envs
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--updates_per_step", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--outdir", required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.outdir, exist_ok=True)

    lam = 0.0 if args.condition in ("baseline", "oracle") else args.lam
    use_listener = args.condition in ("learned", "learned_prag")

    env = PragmaticSpreadEnv(args.n_envs, device=device,
                             oracle=(args.condition == "oracle"), seed=args.seed)
    actor = Actor().to(device)
    critic = CentralCritic().to(device)
    critic_t = CentralCritic().to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(ACT_DIM)

    listener = LearnedListener().to(device) if use_listener else None
    opt_l = torch.optim.Adam(listener.parameters(), lr=1e-3) if use_listener else None

    buf = ReplayBuffer(400_000, device)
    gen = torch.Generator().manual_seed(args.seed + 999)

    history = []
    t0 = time.time()
    total_steps = 0

    for cycle in range(args.cycles):
        obs = env.reset()
        for t in range(EPISODE_LEN):
            with torch.no_grad():
                if total_steps < 2000:
                    a = torch.rand((args.n_envs, N_AGENTS, ACT_DIM), generator=gen) * 2 - 1
                else:
                    a, _ = actor.sample(obs)
            dirs, pref = env.goal_dirs(), env.pref
            next_obs, info, done = env.step(a)

            # pragmatic reward at collection time (static heuristics only;
            # the learned listener recomputes at update time to track co-adaptation)
            if args.condition in ("heuristic", "simple"):
                with torch.no_grad():
                    r_comm = heuristic_comm_reward(
                        dirs, pref, a, exclusivity=(args.condition == "heuristic"), gen=gen)
            else:
                r_comm = torch.zeros((args.n_envs, N_AGENTS))

            buf.push(obs, a, info["r_ext"], r_comm, next_obs,
                     done and t == EPISODE_LEN - 1, pref)
            obs = next_obs
            total_steps += args.n_envs

            if buf.size < 4 * args.batch:
                continue
            for _ in range(args.updates_per_step):
                b_obs, b_act, b_rext, b_rcomm, b_next, b_done, b_pref = buf.sample(args.batch)

                if use_listener:
                    # supervised listener update
                    logits = listener(b_obs, b_act)
                    l_loss = F.cross_entropy(logits.reshape(-1, N_MEANINGS), b_pref.reshape(-1))
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    with torch.no_grad():
                        if args.condition == "learned":
                            b_rcomm = listener.comm_reward(b_obs, b_act, b_pref)
                        else:  # learned_prag
                            alt = torch.rand((*b_act.shape[:-1], 16, 2)) * 2 - 1
                            b_rcomm = listener.comm_reward(
                                b_obs, b_act, b_pref, pragmatic=True, alt_actions=alt)

                r_total = b_rext + lam * b_rcomm.sum(dim=1)
                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    a2, logp2 = actor.sample(b_next.reshape(-1, OBS_DIM))
                    a2 = a2.reshape(args.batch, N_AGENTS, ACT_DIM)
                    logp2 = logp2.reshape(args.batch, N_AGENTS).sum(dim=1)
                    q1t, q2t = critic_t(b_next, a2)
                    qt = torch.min(q1t, q2t) - alpha * logp2
                    y = r_total + args.gamma * (1 - b_done) * qt
                q1, q2 = critic(b_obs, b_act)
                c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_c.zero_grad(); c_loss.backward(); opt_c.step()

                a_new, logp_new = actor.sample(b_obs.reshape(-1, OBS_DIM))
                a_new = a_new.reshape(args.batch, N_AGENTS, ACT_DIM)
                logp_new = logp_new.reshape(args.batch, N_AGENTS).sum(dim=1)
                q1p, q2p = critic(b_obs, a_new)
                a_loss = (log_alpha.exp().detach() * logp_new - torch.min(q1p, q2p)).mean()
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()

                al_loss = -(log_alpha.exp() * (logp_new.detach() / N_AGENTS + target_entropy)).mean()
                opt_alpha.zero_grad(); al_loss.backward(); opt_alpha.step()

                with torch.no_grad():
                    for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * ps)

        if (cycle + 1) % 10 == 0 or cycle == args.cycles - 1:
            m = evaluate(actor, args.condition, listener, device, seed=12345)
            m.update(cycle=cycle + 1, steps=total_steps,
                     wall_min=round((time.time() - t0) / 60, 1))
            history.append(m)
            print(f"[{args.condition} s{args.seed}] cyc {cycle+1}/{args.cycles} "
                  f"R_ext {m['r_ext']:.2f} pref {m['pref_bonus']:.2f} "
                  f"spec {m['spec']:.2f} probe_acc {m['probe_acc']:.2f} "
                  f"comm {m['comm_rw']:.2f} ({m['wall_min']} min)", flush=True)
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=1)

    torch.save({"actor": actor.state_dict(),
                "listener": listener.state_dict() if listener else None},
               os.path.join(args.outdir, "model.pt"))
    print(f"DONE {args.condition} seed {args.seed} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
