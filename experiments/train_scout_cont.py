"""MASAC on continuous-bearing scout-support (meaning space S^1).

Conditions: baseline | oracle | blind | progress_part | nce | nce_prag
(see DESIGN_continuous_meanings.md and scout_continuous.py).

Usage: python train_scout_cont.py --condition nce_prag --seed 0 --outdir results_cont/nce_prag_s0
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
import scout_continuous as C
from train_masac import Actor, CentralCritic

N_PART = 15  # meaning particles (contrast set size K); reward cap log(K+1)


class ContBuffer:
    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros((capacity, S.N_AGENTS, S.OBS_DIM))
        self.act = torch.zeros((capacity, S.N_AGENTS, S.ACT_DIM))
        self.r_ext = torch.zeros(capacity)
        self.r_comm = torch.zeros((capacity, S.N_AGENTS))
        self.next_obs = torch.zeros((capacity, S.N_AGENTS, S.OBS_DIM))
        self.done = torch.zeros(capacity)
        self.theta = torch.zeros(capacity)
        self.idx, self.full = 0, False

    def push(self, obs, act, r_ext, r_comm, next_obs, done, theta):
        n = obs.shape[0]
        i = torch.arange(self.idx, self.idx + n) % self.capacity
        for buf, val in [(self.obs, obs), (self.act, act), (self.r_ext, r_ext),
                         (self.r_comm, r_comm), (self.next_obs, next_obs),
                         (self.theta, theta)]:
            buf[i] = val.cpu()
        self.done[i] = float(done)
        self.idx = int((self.idx + n) % self.capacity)
        if self.idx < n:
            self.full = True
        self.size = self.capacity if self.full else self.idx

    def sample(self, batch):
        i = torch.randint(0, self.size, (batch,))
        to = lambda x: x[i].to(self.device)
        return (to(self.obs), to(self.act), to(self.r_ext), to(self.r_comm),
                to(self.next_obs), to(self.done), to(self.theta))


def particles(theta, k, gen=None, device=None):
    """(B,) true bearings -> (B, k+1) with the truth in column 0."""
    alt = torch.rand((theta.shape[0], k), generator=gen) * 2 * math.pi
    return torch.cat([theta.unsqueeze(1), alt.to(theta.device)], dim=1)


def evaluate(actor, cond, listener, seed, n_envs=64, device=torch.device("cpu")):
    env = C.ContinuousScoutEnv(n_envs, oracle=(cond == "oracle"),
                               blind=(cond == "blind"), seed=seed)
    obs = env.reset()
    gen = torch.Generator().manual_seed(seed)
    tot = {k: 0.0 for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc",
                            "comm_rw", "listener_acc"]}
    steps = 0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs.to(device), deterministic=True)
            a = a.cpu()
            pre_pos = env.pos[:, 0].clone()
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc"]:
                tot[k] += info[k].mean().item()
            th = particles(env.theta, N_PART, gen=gen)
            if cond == "progress_part":
                tot["comm_rw"] += C.particle_progress_reward(env, pre_pos, th).mean().item()
            elif listener is not None:
                zs = torch.stack([torch.cos(th), torch.sin(th)], dim=-1).to(device)
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
                   choices=["baseline", "oracle", "blind", "progress_part",
                            "nce", "nce_prag"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.3)
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

    lam = 0.0 if args.condition in ("baseline", "oracle", "blind") else args.lam
    use_nce = args.condition in ("nce", "nce_prag")

    env = C.ContinuousScoutEnv(args.n_envs, oracle=(args.condition == "oracle"),
                               blind=(args.condition == "blind"), seed=args.seed)
    actor = Actor(hidden=args.hidden, obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM).to(device)
    critic = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS,
                           obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM).to(device)
    critic_t = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS,
                             obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(S.ACT_DIM)

    listener = C.NCEListener().to(device) if use_nce else None
    opt_l = torch.optim.Adam(listener.parameters(), lr=1e-3) if use_nce else None

    buf = ContBuffer(400_000, device)
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
            pre_pos = env.pos[:, 0].clone()
            pre_key = 1.0 - env.has_key.clone()
            next_obs, info, done = env.step(a)

            r_comm = torch.zeros((args.n_envs, S.N_AGENTS))
            if args.condition == "progress_part":
                with torch.no_grad():
                    th = particles(env.theta, N_PART, gen=gen)
                    rc = C.particle_progress_reward(env, pre_pos, th)
                    r_comm[:, 0] = rc * (args.voi + (1 - args.voi) * pre_key)

            buf.push(obs, a, info["r_ext"], r_comm, next_obs,
                     done and t == S.EPISODE_LEN - 1, env.theta)
            obs = next_obs
            total_steps += args.n_envs

            if buf.size < 4 * args.batch:
                continue
            for _ in range(args.updates_per_step):
                b_obs, b_act, b_rext, b_rcomm, b_next, b_done, b_th = \
                    buf.sample(args.batch)

                if use_nce:
                    # in-batch meaning particles: truth + K other episodes
                    perm = torch.randint(0, args.batch, (args.batch, N_PART),
                                         device=device)
                    th = torch.cat([b_th.unsqueeze(1), b_th[perm]], dim=1)
                    zs = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
                    l_loss = listener.nce_loss(b_obs[:, 0], b_act[:, 0], zs)
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    with torch.no_grad():
                        if args.condition == "nce_prag":
                            alt = torch.rand((args.batch, 16, S.ACT_DIM),
                                             device=device) * 2 - 1
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                      zs, pragmatic=True,
                                                      alt_actions=alt)
                        else:
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0], zs)
                        b_pre = 1.0 - b_obs[:, 0, S.KEY_INDEX]
                        rc = rc * (args.voi + (1 - args.voi) * b_pre)
                        b_rcomm = torch.zeros_like(b_rcomm)
                        b_rcomm[:, 0] = rc

                r_total = b_rext + lam * b_rcomm.sum(dim=1)
                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    a2, logp2 = actor.sample(b_next.reshape(-1, S.OBS_DIM))
                    a2 = a2.reshape(args.batch, S.N_AGENTS, S.ACT_DIM)
                    logp2 = logp2.reshape(args.batch, S.N_AGENTS).sum(dim=1)
                    q1t, q2t = critic_t(b_next, a2)
                    y = r_total + args.gamma * (1 - b_done) * \
                        (torch.min(q1t, q2t) - alpha * logp2)
                q1, q2 = critic(b_obs, b_act)
                c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_c.zero_grad(); c_loss.backward(); opt_c.step()

                a_new, logp_new = actor.sample(b_obs.reshape(-1, S.OBS_DIM))
                a_new = a_new.reshape(args.batch, S.N_AGENTS, S.ACT_DIM)
                logp_new = logp_new.reshape(args.batch, S.N_AGENTS).sum(dim=1)
                q1p, q2p = critic(b_obs, a_new)
                a_loss = (log_alpha.exp().detach() * logp_new
                          - torch.min(q1p, q2p)).mean()
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
                  f"comm {m['comm_rw']:.2f} ({m['wall_min']} min)", flush=True)
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
