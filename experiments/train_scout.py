"""MASAC on the scout-support task with the expanded listener family.

Conditions:
  baseline     lambda = 0
  oracle       lambda = 0, supporter observes the scout's target
  simple       cosine listener + RSA over sampled alternatives
  exclusivity  cosine-margin listener + RSA (thesis Algorithm 1)
  progress     per-step distance-progress listener (Dragan-style)
  filter       recursive Bayesian evidence accumulation over the episode
  learned      learned listener L_theta on the scout, N_r = 0
  learned_prag learned listener + one RSA step over 16 alternatives

Usage: python train_scout.py --condition filter --seed 0 --outdir results_scout/filter_s0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import scout_support as S
from train_masac import Actor, CentralCritic, ReplayBuffer

HANDCRAFTED = ("simple", "exclusivity", "progress", "filter")


def inject_ear(obs, posterior):
    """Fill the supporter's partner-meaning slot with the listener posterior."""
    obs = obs.clone()
    obs[:, 1, S.PARTNER_PREF_SLICE] = posterior
    return obs


def evaluate(actor, cond, listener, seed, n_envs=64):
    env = S.ScoutSupportEnv(n_envs, oracle=(cond == "oracle"), seed=seed)
    obs = env.reset()
    ear = cond in ("ear", "learned_ear")
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    gen = torch.Generator().manual_seed(seed)
    keys = ["r_ext", "pref_bonus", "dist_pen", "commit_acc",
            "probe_acc", "probe_ce", "comm_rw", "listener_acc"]
    tot = {k: 0.0 for k in keys}
    steps = 0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs, deterministic=True)
            dirs, target = env.goal_dirs(), env.target
            pre_pos = env.pos[:, 0].clone()
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc"]:
                tot[k] += info[k].mean().item()
            acc, ce = S.probe_intent_metrics(dirs, target, a)
            tot["probe_acc"] += acc.mean().item()
            tot["probe_ce"] += ce.mean().item()
            if cond in HANDCRAFTED:
                rc = S.scout_comm_reward(env, a, cond, gen=gen, pre_pos=pre_pos)
                tot["comm_rw"] += rc.mean().item()
            elif listener is not None:
                rc = listener.comm_reward(obs[:, 0], a[:, 0], target)
                tot["comm_rw"] += rc.mean().item()
                pred = listener(obs[:, 0], a[:, 0]).argmax(-1)
                tot["listener_acc"] += (pred == target).float().mean().item()
            if ear:
                post = F.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
            steps += 1
    return {k: v / steps for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["baseline", "oracle", "simple", "exclusivity",
                            "progress", "filter", "learned", "learned_prag",
                            "ear", "learned_ear"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--cycles", type=int, default=150)
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
    os.makedirs(args.outdir, exist_ok=True)

    lam = 0.0 if args.condition in ("baseline", "oracle", "ear") else args.lam
    use_listener = args.condition in ("learned", "learned_prag", "ear", "learned_ear")
    ear = args.condition in ("ear", "learned_ear")

    env = S.ScoutSupportEnv(args.n_envs, oracle=(args.condition == "oracle"),
                            seed=args.seed)
    actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    critic = CentralCritic(n_agents=S.N_AGENTS, obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    critic_t = CentralCritic(n_agents=S.N_AGENTS, obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(S.ACT_DIM)

    listener = S.ScoutListener() if use_listener else None
    opt_l = torch.optim.Adam(listener.parameters(), lr=1e-3) if use_listener else None

    buf = ReplayBuffer(400_000, torch.device("cpu"), n_agents=S.N_AGENTS,
                       obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    gen = torch.Generator().manual_seed(args.seed + 999)

    history = []
    t0 = time.time()
    total_steps = 0

    for cycle in range(args.cycles):
        obs = env.reset()
        if ear:
            obs = inject_ear(obs, torch.full((args.n_envs, S.N_MEANINGS), 1 / 3))
        for t in range(S.EPISODE_LEN):
            with torch.no_grad():
                if total_steps < 2000:
                    a = torch.rand((args.n_envs, S.N_AGENTS, S.ACT_DIM),
                                   generator=gen) * 2 - 1
                else:
                    a, _ = actor.sample(obs)
            pre_pos = env.pos[:, 0].clone()
            target = env.target
            next_obs, info, done = env.step(a)

            r_comm = torch.zeros((args.n_envs, S.N_AGENTS))
            if args.condition in HANDCRAFTED:
                with torch.no_grad():
                    r_comm[:, 0] = S.scout_comm_reward(env, a, args.condition,
                                                       gen=gen, pre_pos=pre_pos)
            if ear:
                with torch.no_grad():
                    post = F.softmax(listener(obs[:, 0], a[:, 0]), dim=-1)
                next_obs = inject_ear(next_obs, post)

            buf.push(obs, a, info["r_ext"], r_comm, next_obs,
                     done and t == S.EPISODE_LEN - 1, env.pref)
            obs = next_obs
            total_steps += args.n_envs

            if buf.size < 4 * args.batch:
                continue
            for _ in range(args.updates_per_step):
                b_obs, b_act, b_rext, b_rcomm, b_next, b_done, b_pref = buf.sample(args.batch)
                b_target = b_pref[:, 0]

                if use_listener:
                    logits = listener(b_obs[:, 0], b_act[:, 0])
                    l_loss = F.cross_entropy(logits, b_target)
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    if lam > 0:
                        with torch.no_grad():
                            if args.condition == "learned_prag":
                                alt = torch.rand((args.batch, 16, 2)) * 2 - 1
                                rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                          b_target, pragmatic=True,
                                                          alt_actions=alt)
                            else:
                                rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                          b_target)
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
                a_loss = (alpha * logp_new - torch.min(q1p, q2p)).mean()
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()

                al_loss = -(log_alpha.exp() *
                            (logp_new.detach() / S.N_AGENTS + target_entropy)).mean()
                opt_alpha.zero_grad(); al_loss.backward(); opt_alpha.step()

                with torch.no_grad():
                    for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * ps)

        if (cycle + 1) % 10 == 0 or cycle == args.cycles - 1:
            m = evaluate(actor, args.condition, listener, seed=12345)
            m.update(cycle=cycle + 1, steps=total_steps,
                     wall_min=round((time.time() - t0) / 60, 1))
            history.append(m)
            print(f"[{args.condition} s{args.seed}] cyc {cycle+1}/{args.cycles} "
                  f"R_ext {m['r_ext']:.2f} bonus {m['pref_bonus']:.2f} "
                  f"commit {m['commit_acc']:.2f} probe {m['probe_acc']:.2f} "
                  f"comm {m['comm_rw']:.2f} ({m['wall_min']} min)", flush=True)
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=1)

    torch.save({"actor": actor.state_dict(),
                "listener": listener.state_dict() if listener else None},
               os.path.join(args.outdir, "model.pt"))
    print(f"DONE {args.condition} seed {args.seed} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
