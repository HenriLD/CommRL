"""MASAC on the highway-fork negotiation task (Environment C, fork design).

Conditions:
  baseline       lambda = 0
  oracle         lambda = 0, cooperators observe the ego's intended branch
  blind          lambda = 0, cooperators cannot observe the ego (control)
  corridorprog   hand-crafted completion-point progress listener
  learned_pre_prag  bounded learned listener (audience viewpoint, pre-commit
                    training window) + one RSA step -- the paper's recipe
  inforeg        Strouse-style variational I(A;M|S) (two decoders)

Usage: python train_corridor.py --condition oracle --seed 0 --outdir results_corridor/oracle_s0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ckpt
import highway_corridor as H
from train_masac import Actor, CentralCritic, ReplayBuffer


def evaluate(actor, cond, listener, seed, n_envs=64, device=torch.device("cpu"),
             listener2=None):
    env = H.HighwayCorridorEnv(n_envs, oracle=(cond == "oracle"),
                               blind=(cond == "blind"), seed=seed)
    obs = env.reset()
    keys = ["r_ext", "merge_rate", "fail_rate", "harsh", "coll", "comm_rw",
            "listener_acc"]
    tot = {k: 0.0 for k in keys}
    steps = 0
    with torch.no_grad():
        for t in range(H.EPISODE_LEN):
            a, _ = actor.sample(obs.to(device), deterministic=True)
            a = a.cpu()
            pre_x = env.x[:, 0].clone()
            pre_y = env.y[:, 0].clone()
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "harsh", "coll"]:
                tot[k] += info[k].mean().item()
            if cond == "corridorprog":
                tot["comm_rw"] += H.corridor_progress_reward(env, pre_x, pre_y).mean().item()
            elif listener is not None:
                so, sa = obs[:, 0].to(device), a[:, 0].to(device)
                if listener2 is not None:
                    g = env.target.to(device).unsqueeze(-1)
                    lp1 = F.log_softmax(listener(so, sa), dim=-1)
                    lp0 = F.log_softmax(listener2(so, sa), dim=-1)
                    rc = (torch.gather(lp1, -1, g) - torch.gather(lp0, -1, g)).squeeze(-1)
                else:
                    rc = listener.comm_reward(so, sa, env.target.to(device))
                tot["comm_rw"] += rc.mean().item()
                pred = listener(so, sa).argmax(-1).cpu()
                tot["listener_acc"] += (pred == env.target).float().mean().item()
            obs = next_obs
            steps += 1
    m = {k: v / steps for k, v in tot.items()}
    m["merge_rate"] = info["merge_rate"].mean().item()
    m["fail_rate"] = info["fail_rate"].mean().item()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["baseline", "oracle", "blind", "corridorprog",
                            "learned_pre_prag", "inforeg"])
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
    p.add_argument("--voi", type=float, default=0.2,
                   help="post-commit weight on R_comm (pre-commit weight is 1)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    lam = 0.0 if args.condition in ("baseline", "oracle", "blind") else args.lam
    use_listener = args.condition in ("learned_pre_prag", "inforeg")
    inforeg = args.condition == "inforeg"
    prekey = args.condition == "learned_pre_prag"

    env = H.HighwayCorridorEnv(args.n_envs, oracle=(args.condition == "oracle"),
                               blind=(args.condition == "blind"), seed=args.seed)
    actor = Actor(hidden=args.hidden, obs_dim=H.OBS_DIM, act_dim=H.ACT_DIM).to(device)
    critic = CentralCritic(hidden=args.hidden, n_agents=H.N_AGENTS,
                           obs_dim=H.OBS_DIM, act_dim=H.ACT_DIM).to(device)
    critic_t = CentralCritic(hidden=args.hidden, n_agents=H.N_AGENTS,
                             obs_dim=H.OBS_DIM, act_dim=H.ACT_DIM).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(H.ACT_DIM)

    listener = (H.CorridorListener(inputs="full" if inforeg else "act").to(device)
                if use_listener else None)
    opt_l = (torch.optim.Adam(listener.parameters(), lr=1e-3)
             if use_listener else None)
    listener2 = H.CorridorListener(inputs="state").to(device) if inforeg else None
    opt_l2 = (torch.optim.Adam(listener2.parameters(), lr=1e-3)
              if inforeg else None)

    buf = ReplayBuffer(400_000, device, n_agents=H.N_AGENTS,
                       obs_dim=H.OBS_DIM, act_dim=H.ACT_DIM)
    gen = torch.Generator().manual_seed(args.seed + 999)

    history = []
    t0 = time.time()
    total_steps = 0
    start_cycle = 0
    resumed_at = None
    nets_d = {"actor": actor, "critic": critic, "critic_t": critic_t}
    if listener is not None:
        nets_d["listener"] = listener
    if listener2 is not None:
        nets_d["listener2"] = listener2
    opts_d = {"a": opt_a, "c": opt_c, "alpha": opt_alpha}
    if opt_l is not None:
        opts_d["l"] = opt_l
    if opt_l2 is not None:
        opts_d["l2"] = opt_l2
    res = ckpt.load(args.outdir, nets_d, opts_d)
    if res:
        start_cycle, total_steps, history, ex = res
        with torch.no_grad():
            log_alpha.copy_(ex["log_alpha"].to(device))
        resumed_at = start_cycle
        print(f"resumed at cycle {start_cycle}", flush=True)

    for cycle in range(start_cycle, args.cycles):
        obs = env.reset()
        for t in range(H.EPISODE_LEN):
            with torch.no_grad():
                if total_steps < 2000:
                    a = torch.rand((args.n_envs, H.N_AGENTS, H.ACT_DIM),
                                   generator=gen) * 2 - 1
                else:
                    a, _ = actor.sample(obs.to(device))
                    a = a.cpu()
            pre_x = env.x[:, 0].clone()
            pre_y = env.y[:, 0].clone()
            pre_commit = env.pre_flag()
            target = env.target
            next_obs, info, done = env.step(a)

            r_comm = torch.zeros((args.n_envs, H.N_AGENTS))
            if args.condition == "corridorprog":
                with torch.no_grad():
                    rc = H.corridor_progress_reward(env, pre_x, pre_y)
                    r_comm[:, 0] = rc * (args.voi + (1 - args.voi) * pre_commit)

            buf.push(obs, a, info["r_ext"], r_comm, next_obs,
                     done and t == H.EPISODE_LEN - 1,
                     target.unsqueeze(1).expand(-1, H.N_AGENTS))
            obs = next_obs
            total_steps += args.n_envs

            if buf.size < 4 * args.batch:
                continue
            for _ in range(args.updates_per_step):
                b_obs, b_act, b_rext, b_rcomm, b_next, b_done, b_pref = \
                    buf.sample(args.batch)
                b_target = b_pref[:, 0]

                if use_listener:
                    logits = listener(b_obs[:, 0], b_act[:, 0])
                    if prekey:
                        # pre-commit window: ego still centered in its lane
                        w = b_obs[:, 0, H.KEY_INDEX]
                        ce = F.cross_entropy(logits, b_target, reduction="none")
                        l_loss = (ce * w).sum() / w.sum().clamp(min=1.0)
                    else:
                        l_loss = F.cross_entropy(logits, b_target)
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    if inforeg:
                        logits2 = listener2(b_obs[:, 0], b_act[:, 0])
                        l2_loss = F.cross_entropy(logits2, b_target)
                        opt_l2.zero_grad(); l2_loss.backward(); opt_l2.step()
                    with torch.no_grad():
                        if prekey:
                            alt = torch.rand((args.batch, 16, H.ACT_DIM),
                                             device=device) * 2 - 1
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                      b_target, pragmatic=True,
                                                      alt_actions=alt)
                        else:
                            lp1 = F.log_softmax(logits, dim=-1)
                            lp0 = F.log_softmax(logits2, dim=-1)
                            gt = b_target.unsqueeze(-1)
                            rc = (torch.gather(lp1, -1, gt)
                                  - torch.gather(lp0, -1, gt)).squeeze(-1)
                        b_pre = b_obs[:, 0, H.KEY_INDEX]
                        rc = rc * (args.voi + (1 - args.voi) * b_pre)
                        b_rcomm = torch.zeros_like(b_rcomm)
                        b_rcomm[:, 0] = rc

                r_total = b_rext + lam * b_rcomm.sum(dim=1)
                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    a2, logp2 = actor.sample(b_next.reshape(-1, H.OBS_DIM))
                    a2 = a2.reshape(args.batch, H.N_AGENTS, H.ACT_DIM)
                    logp2 = logp2.reshape(args.batch, H.N_AGENTS).sum(dim=1)
                    q1t, q2t = critic_t(b_next, a2)
                    y = r_total + args.gamma * (1 - b_done) * \
                        (torch.min(q1t, q2t) - alpha * logp2)
                q1, q2 = critic(b_obs, b_act)
                c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_c.zero_grad(); c_loss.backward(); opt_c.step()

                a_new, logp_new = actor.sample(b_obs.reshape(-1, H.OBS_DIM))
                a_new = a_new.reshape(args.batch, H.N_AGENTS, H.ACT_DIM)
                logp_new = logp_new.reshape(args.batch, H.N_AGENTS).sum(dim=1)
                q1p, q2p = critic(b_obs, a_new)
                a_loss = (log_alpha.exp().detach() * logp_new
                          - torch.min(q1p, q2p)).mean()
                opt_a.zero_grad(); a_loss.backward(); opt_a.step()

                al_loss = -(log_alpha.exp()
                            * (logp_new.detach() / H.N_AGENTS + target_entropy)).mean()
                opt_alpha.zero_grad(); al_loss.backward(); opt_alpha.step()

                with torch.no_grad():
                    for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * ps)

        if (cycle + 1) % 10 == 0 or cycle == args.cycles - 1:
            m = evaluate(actor, args.condition, listener, seed=12345,
                         device=device, listener2=listener2)
            m.update(cycle=cycle + 1, steps=total_steps,
                     wall_min=round((time.time() - t0) / 60, 1))
            if resumed_at is not None:
                m["resumed_at"] = resumed_at
                resumed_at = None
            history.append(m)
            print(f"[{args.condition} s{args.seed}] cyc {cycle+1}/{args.cycles} "
                  f"R_ext {m['r_ext']:.3f} succ {m['merge_rate']:.2f} "
                  f"fail {m['fail_rate']:.2f} coll {m['coll']:.2f} "
                  f"({m['wall_min']} min)", flush=True)
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=1)
        if (cycle + 1) % ckpt.CKPT_EVERY == 0 and cycle + 1 < args.cycles:
            ckpt.save(args.outdir, cycle + 1, total_steps, history,
                      nets_d, opts_d, {"log_alpha": log_alpha.detach().cpu()})

    torch.save({"actor": actor.state_dict(),
                "listener": listener.state_dict() if listener else None,
                "listener2": listener2.state_dict() if listener2 else None},
               os.path.join(args.outdir, "model.pt"))
    ckpt.clear(args.outdir)
    print(f"DONE {args.condition} seed {args.seed} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
