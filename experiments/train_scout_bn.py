"""MASAC on decoy-enriched scout-support with the VIB bottleneck actor
(stage 2 of the continuous-meaning program: legibility of the agent's OWN
decision latent, see scout_bottleneck.py).

Conditions (all share the bottleneck architecture and the same beta, so the
anchors price communication, not capacity):
  base2      lambda = 0
  oracle2    lambda = 0, supporter told the true bearing (public channel)
  blind2     lambda = 0, supporter cannot observe the scout
  vib_nce       R_comm = InfoNCE decodability of mu_z from (context, action)
  vib_nce_prag  + one Sinkhorn iteration (RSA recursion over particles)

Usage: python train_scout_bn.py --condition vib_nce_prag --seed 0 --outdir results_bn/vibnceprag_s0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ckpt
import scout_support as S
import scout_bottleneck as B
from train_masac import CentralCritic

N_PART = 15  # meaning particles (contrast set size K); reward cap log(K+1)


class Buffer:
    def __init__(self, capacity, device, obs_dim):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros((capacity, S.N_AGENTS, obs_dim))
        self.act = torch.zeros((capacity, S.N_AGENTS, S.ACT_DIM))
        self.r_ext = torch.zeros(capacity)
        self.next_obs = torch.zeros((capacity, S.N_AGENTS, obs_dim))
        self.done = torch.zeros(capacity)
        self.theta = torch.zeros(capacity)
        self.idx, self.full = 0, False

    def push(self, obs, act, r_ext, next_obs, done, theta):
        n = obs.shape[0]
        i = torch.arange(self.idx, self.idx + n) % self.capacity
        for buf, val in [(self.obs, obs), (self.act, act), (self.r_ext, r_ext),
                         (self.next_obs, next_obs), (self.theta, theta)]:
            buf[i] = val.cpu()
        self.done[i] = float(done)
        self.idx = int((self.idx + n) % self.capacity)
        if self.idx < n:
            self.full = True
        self.size = self.capacity if self.full else self.idx

    def sample(self, batch):
        i = torch.randint(0, self.size, (batch,))
        to = lambda x: x[i].to(self.device)
        return (to(self.obs), to(self.act), to(self.r_ext),
                to(self.next_obs), to(self.done), to(self.theta))


def z_particles(mu_z, k):
    """(B, Z) own meanings -> (B, k+1, Z) with the truth in column 0 and
    other episodes' latents as in-batch contrast particles."""
    perm = torch.randint(0, mu_z.shape[0], (mu_z.shape[0], k),
                         device=mu_z.device)
    return torch.cat([mu_z.unsqueeze(1), mu_z[perm]], dim=1)


def evaluate(actor, cond, listener, seed, n_envs=64, device=torch.device("cpu")):
    env = B.DecoyScoutEnv(n_envs, oracle=(cond == "oracle2"),
                          blind=(cond == "blind2"), seed=seed)
    obs = env.reset()
    tot = {k: 0.0 for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc",
                            "comm_rw", "listener_acc", "kl"]}
    steps = 0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs.to(device), deterministic=True)
            a = a.cpu()
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc"]:
                tot[k] += info[k].mean().item()
            tot["kl"] += actor.kl(obs[:, 0].to(device)).mean().item()
            if listener is not None:
                mu_z, _ = actor.encode(obs[:, 0].to(device))
                zs = z_particles(mu_z, N_PART)
                sc = listener.scores(obs[:, 0].to(device), a[:, 0].to(device), zs)
                tot["comm_rw"] += (F.log_softmax(sc, dim=-1)[:, 0]
                                   + math.log(N_PART + 1)).mean().item()
                tot["listener_acc"] += (sc.argmax(-1) == 0).float().mean().item()
            obs = next_obs
            steps += 1
    return {k: v / steps for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["base2", "oracle2", "blind2", "vib_nce",
                            "vib_nce_prag"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--cycles", type=int, default=400)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--updates_per_step", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--outdir", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--voi", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    lam = 0.0 if args.condition in ("base2", "oracle2", "blind2") else args.lam
    use_nce = args.condition in ("vib_nce", "vib_nce_prag")

    env = B.DecoyScoutEnv(args.n_envs, oracle=(args.condition == "oracle2"),
                          blind=(args.condition == "blind2"), seed=args.seed)
    actor = B.VIBActor(hidden=args.hidden).to(device)
    critic = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS,
                           obs_dim=B.OBS_DIM_BN, act_dim=S.ACT_DIM).to(device)
    critic_t = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS,
                             obs_dim=B.OBS_DIM_BN, act_dim=S.ACT_DIM).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(S.ACT_DIM)

    listener = B.NCEListenerZ().to(device) if use_nce else None
    opt_l = torch.optim.Adam(listener.parameters(), lr=1e-3) if use_nce else None

    buf = Buffer(400_000, device, B.OBS_DIM_BN)
    gen = torch.Generator().manual_seed(args.seed + 999)

    history = []
    t0 = time.time()
    total_steps = 0
    start_cycle = 0
    resumed_at = None
    nets_d = {"actor": actor, "critic": critic, "critic_t": critic_t}
    if listener is not None:
        nets_d["listener"] = listener
    opts_d = {"a": opt_a, "c": opt_c, "alpha": opt_alpha}
    if opt_l is not None:
        opts_d["l"] = opt_l
    res = ckpt.load(args.outdir, nets_d, opts_d)
    if res:
        start_cycle, total_steps, history, ex = res
        with torch.no_grad():
            log_alpha.copy_(ex["log_alpha"].to(device))
        resumed_at = start_cycle
        print(f"resumed at cycle {start_cycle}", flush=True)

    for cycle in range(start_cycle, args.cycles):
        obs = env.reset()
        for t in range(S.EPISODE_LEN):
            with torch.no_grad():
                if total_steps < 2000:
                    a = torch.rand((args.n_envs, S.N_AGENTS, S.ACT_DIM),
                                   generator=gen) * 2 - 1
                else:
                    a, _ = actor.sample(obs.to(device))
                    a = a.cpu()
            next_obs, info, done = env.step(a)
            buf.push(obs, a, info["r_ext"], next_obs,
                     done and t == S.EPISODE_LEN - 1, env.theta)
            obs = next_obs
            total_steps += args.n_envs

            if buf.size < 4 * args.batch:
                continue
            for _ in range(args.updates_per_step):
                b_obs, b_act, b_rext, b_next, b_done, b_th = \
                    buf.sample(args.batch)

                r_total = b_rext
                if use_nce:
                    with torch.no_grad():
                        mu_z, _ = actor.encode(b_obs[:, 0])
                    zs = z_particles(mu_z, N_PART)
                    l_loss = listener.nce_loss(b_obs[:, 0], b_act[:, 0], zs)
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    with torch.no_grad():
                        if args.condition == "vib_nce_prag":
                            alt = torch.rand((args.batch, 16, S.ACT_DIM),
                                             device=device) * 2 - 1
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                      zs, pragmatic=True,
                                                      alt_actions=alt)
                        else:
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0], zs)
                        b_pre = 1.0 - b_obs[:, 0, S.KEY_INDEX]
                        rc = rc * (args.voi + (1 - args.voi) * b_pre)
                        r_total = b_rext + lam * rc

                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    a2, logp2 = actor.sample(b_next.reshape(-1, B.OBS_DIM_BN))
                    a2 = a2.reshape(args.batch, S.N_AGENTS, S.ACT_DIM)
                    logp2 = logp2.reshape(args.batch, S.N_AGENTS).sum(dim=1)
                    q1t, q2t = critic_t(b_next, a2)
                    y = r_total + args.gamma * (1 - b_done) * \
                        (torch.min(q1t, q2t) - alpha * logp2)
                q1, q2 = critic(b_obs, b_act)
                c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_c.zero_grad(); c_loss.backward(); opt_c.step()

                a_new, logp_new = actor.sample(b_obs.reshape(-1, B.OBS_DIM_BN))
                a_new = a_new.reshape(args.batch, S.N_AGENTS, S.ACT_DIM)
                logp_new = logp_new.reshape(args.batch, S.N_AGENTS).sum(dim=1)
                q1p, q2p = critic(b_obs, a_new)
                a_loss = (log_alpha.exp().detach() * logp_new
                          - torch.min(q1p, q2p)).mean()
                # VIB: information budget on the scout's private-branch latent
                a_loss = a_loss + args.beta * actor.kl(b_obs[:, 0]).mean()
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()

                al_loss = -(log_alpha.exp()
                            * (logp_new.detach() / S.N_AGENTS + target_entropy)).mean()
                opt_alpha.zero_grad(); al_loss.backward(); opt_alpha.step()

                with torch.no_grad():
                    for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * ps)

        if (cycle + 1) % 10 == 0 or cycle == args.cycles - 1:
            m = evaluate(actor, args.condition, listener, seed=12345, device=device)
            m.update(cycle=cycle + 1, steps=total_steps,
                     wall_min=round((time.time() - t0) / 60, 1))
            if resumed_at is not None:
                m["resumed_at"] = resumed_at
                resumed_at = None
            history.append(m)
            print(f"[{args.condition} s{args.seed}] cyc {cycle+1}/{args.cycles} "
                  f"R_ext {m['r_ext']:.2f} commit {m['commit_acc']:.2f} "
                  f"comm {m['comm_rw']:.2f} kl {m['kl']:.2f} "
                  f"({m['wall_min']} min)", flush=True)
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=1)
        if (cycle + 1) % ckpt.CKPT_EVERY == 0 and cycle + 1 < args.cycles:
            ckpt.save(args.outdir, cycle + 1, total_steps, history,
                      nets_d, opts_d, {"log_alpha": log_alpha.detach().cpu()})

    torch.save({"actor": actor.state_dict(),
                "listener": listener.state_dict() if listener else None},
               os.path.join(args.outdir, "model.pt"))
    ckpt.clear(args.outdir)
    print(f"DONE {args.condition} seed {args.seed} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
