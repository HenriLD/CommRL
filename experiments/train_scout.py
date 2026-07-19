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
  learned_act* listener bounded to the audience viewpoint (site bearings + action)
  learned_pre* act-bounded listener trained on pre-key steps only, so decode
               accuracy can only come from early signaling, never from the
               post-commitment leg

Usage: python train_scout.py --condition filter --seed 0 --outdir results_scout/filter_s0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import ckpt
import scout_support as S
from train_masac import Actor, CentralCritic, ReplayBuffer

HANDCRAFTED = ("simple", "exclusivity", "progress", "filter")


class IPLWrapper:
    """Inverse-planning listener: the RSA-textbook L0 instantiated by a
    FROZEN, independently trained, non-communicative goal-conditioned policy.
    L0(m|s,a) = pi_ref(a|s,m) / sum_m' pi_ref(a|s,m') -- Bayes over the
    reference policy's action densities with the meaning slot substituted.
    No hand-crafted geometry, no free parameters, no co-adaptation: the
    common ground is competence itself."""

    KAPPA = 0.5  # RSA soft-rationality tolerance (action units), the one dial

    def __init__(self, ref_actor):
        self.ref = ref_actor

    def means(self, obs):
        """Competent action for each candidate meaning: tanh(mu_ref(s, m))."""
        mus = []
        for m in range(S.N_MEANINGS):
            o = obs.clone()
            o[..., S.PREF_SLICE] = 0.0
            o[..., S.PREF_SLICE.start + m] = 1.0
            mu, _ = self.ref(o)
            mus.append(torch.tanh(mu))
        return torch.stack(mus, dim=-2)              # (..., M, ACT_DIM)

    def logits(self, obs, actions):
        mu = self.means(obs)
        d2 = ((actions.unsqueeze(-2) - mu) ** 2).sum(-1)
        return -d2 / (2 * self.KAPPA ** 2)           # (..., M)

    def __call__(self, obs, actions):
        return self.logits(obs, actions)

    def comm_reward(self, obs, actions, target, pragmatic=False,
                    alt_actions=None, n_iter=1):
        import math
        if not pragmatic:
            logp = F.log_softmax(self.logits(obs, actions), dim=-1)
            lp = torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
            return lp + math.log(S.N_MEANINGS)
        acts = torch.cat([actions.unsqueeze(-2), alt_actions], dim=-2)
        mu = self.means(obs)                          # (batch, M, ACT)
        d2 = ((acts.unsqueeze(-2) - mu.unsqueeze(-3)) ** 2).sum(-1)  # (b, A, M)
        L = F.softmax(-d2 / (2 * self.KAPPA ** 2), dim=-1)
        for _ in range(n_iter):
            Sp = L / L.sum(dim=-2, keepdim=True).clamp(min=1e-12)
            L = Sp / Sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = torch.gather(L[..., 0, :], -1, target.unsqueeze(-1)).squeeze(-1)
        return torch.log(p.clamp(min=1e-8)) + math.log(S.N_MEANINGS)


def inject_ear(obs, posterior):
    """Fill the supporter's partner-meaning slot with the listener posterior."""
    obs = obs.clone()
    obs[:, 1, S.PARTNER_PREF_SLICE] = posterior
    return obs


def evaluate(actor, cond, listener, seed, n_envs=64, device=torch.device("cpu"),
             blind=False, listener2=None):
    env = S.ScoutSupportEnv(n_envs, oracle=(cond == "oracle"), seed=seed, blind=blind)
    obs = env.reset()
    ear = cond in ("ear", "learned_ear", "filter_ear", "learned_act_ear")
    if ear:
        obs = inject_ear(obs, torch.full((n_envs, S.N_MEANINGS), 1 / 3))
    gen = torch.Generator().manual_seed(seed)
    keys = ["r_ext", "pref_bonus", "dist_pen", "commit_acc",
            "probe_acc", "probe_ce", "comm_rw", "listener_acc"]
    tot = {k: 0.0 for k in keys}
    steps = 0
    with torch.no_grad():
        for t in range(S.EPISODE_LEN):
            a, _ = actor.sample(obs.to(device), deterministic=True)
            a = a.cpu()
            dirs, target = env.goal_dirs(), env.target
            pre_pos = env.pos[:, 0].clone()
            next_obs, info, done = env.step(a)
            for k in ["r_ext", "pref_bonus", "dist_pen", "commit_acc"]:
                tot[k] += info[k].mean().item()
            acc, ce = S.probe_intent_metrics(dirs, target, a)
            tot["probe_acc"] += acc.mean().item()
            tot["probe_ce"] += ce.mean().item()
            if cond in HANDCRAFTED or cond == "filter_ear":
                kind = "filter" if cond == "filter_ear" else cond
                rc = S.scout_comm_reward(env, a, kind, gen=gen, pre_pos=pre_pos)
                tot["comm_rw"] += rc.mean().item()
            elif listener is not None:
                li = 1 if cond == "partner_belief" else 0
                so, sa = obs[:, li].to(device), a[:, 0].to(device)
                if listener2 is not None:  # inforeg: action info beyond state
                    g = target.to(device).unsqueeze(-1)
                    lp1 = F.log_softmax(listener(so, sa), dim=-1)
                    lp0 = F.log_softmax(listener2(so, sa), dim=-1)
                    rc = (torch.gather(lp1, -1, g)
                          - torch.gather(lp0, -1, g)).squeeze(-1)
                else:
                    rc = listener.comm_reward(so, sa, target.to(device))
                tot["comm_rw"] += rc.mean().item()
                pred = listener(so, sa).argmax(-1).cpu()
                tot["listener_acc"] += (pred == target).float().mean().item()
            if ear:
                if cond == "filter_ear":
                    post = env.log_belief.exp()
                else:
                    post = F.softmax(listener(obs[:, 0].to(device),
                                              a[:, 0].to(device)), dim=-1).cpu()
                next_obs = inject_ear(next_obs, post)
            obs = next_obs
            steps += 1
    return {k: v / steps for k, v in tot.items()}


def pretrain_listener_on_ref(listener, args, device, n_resets=20, steps=2000):
    """Fit the bounded listener by ML on rollouts of the frozen, seed-disjoint,
    non-communicative baseline (the same checkpoint the IPL inverts), using the
    pre-commitment window so the fit cannot be earned from post-key motion.
    Returns nothing; the listener is trained in place and then frozen by the
    caller."""
    ref_ckpt = torch.load(args.ref_ckpt, map_location="cpu", weights_only=True)
    ref_actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    ref_actor.load_state_dict(ref_ckpt["actor"])
    ref_actor.to(device).eval()
    obs_l, act_l, tgt_l = [], [], []
    with torch.no_grad():
        for r in range(n_resets):
            env = S.ScoutSupportEnv(args.n_envs, seed=args.seed * 977 + r)
            o = env.reset()
            for t in range(S.EPISODE_LEN):
                a, _ = ref_actor.sample(o.to(device), deterministic=False)
                a = a.cpu()
                pre = o[:, 0, S.KEY_INDEX] < 0.5          # pre-commitment only
                if pre.any():
                    obs_l.append(o[pre, 0].clone())
                    act_l.append(a[pre, 0].clone())
                    tgt_l.append(env.target[pre].clone())
                o, _, _ = env.step(a)
    X = torch.cat(obs_l).to(device)
    A = torch.cat(act_l).to(device)
    Y = torch.cat(tgt_l).to(device)
    n_ho = max(1, X.shape[0] // 5)                     # held-out split, so the
    Xh, Ah, Yh = X[:n_ho], A[:n_ho], Y[:n_ho]          # reported fit is not
    X, A, Y = X[n_ho:], A[n_ho:], Y[n_ho:]             # memorization
    opt = torch.optim.Adam(listener.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(args.seed + 4242)
    for it in range(steps):
        i = torch.randint(0, X.shape[0], (512,), generator=g).to(device)
        loss = F.cross_entropy(listener(X[i], A[i]), Y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (listener(X, A).argmax(-1) == Y).float().mean().item()
        acc_h = (listener(Xh, Ah).argmax(-1) == Yh).float().mean().item()
    print(f"[frozen listener] pretrained on {X.shape[0]} pre-key transitions "
          f"from {args.ref_ckpt}; train acc {acc:.3f}, held-out {acc_h:.3f} "
          f"(chance {1/S.N_MEANINGS:.3f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["baseline", "oracle", "simple", "exclusivity",
                            "progress", "filter", "learned", "learned_prag",
                            "ear", "learned_ear", "filter_ear",
                            "learned_act", "learned_act_prag", "learned_act_ear",
                            "learned_pre", "learned_pre_prag",
                            "learned_frozen_prag", "learned_frozensat_prag",
                            "inforeg", "partner_belief",
                            "ipl", "ipl_prag"])
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
    p.add_argument("--voi", type=float, default=1.0,
                   help="post-key weight on R_comm (pre-key weight is 1); "
                        "<1 concentrates the subsidy where information matters")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_sites", type=int, default=3)
    p.add_argument("--minefield", action="store_true")
    p.add_argument("--blind", action="store_true",
                   help="control: supporter cannot observe the scout")
    p.add_argument("--sup_speed", type=float, default=None,
                   help="supporter max speed (controls the oracle premium)")
    p.add_argument("--shape_w", type=float, default=None,
                   help="scout shaping weight (controls signal cost)")
    p.add_argument("--listener_lr", type=float, default=1e-3,
                   help="reward-listener learning rate; below the actor lr it "
                        "bounds how fast the audience adapts to the speaker")
    p.add_argument("--alt_policy", action="store_true",
                   help="draw RSA alternative actions from the current policy "
                        "instead of uniform random (proper RSA alternative set)")
    p.add_argument("--frozen_ckpt", default=None,
                   help="converged learned listener to freeze (frozensat)")
    p.add_argument("--ref_ckpt",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_scout3", "baseline_s9", "model.pt"),
                   help="frozen non-communicative reference policy for the "
                        "inverse-planning listener (seed-disjoint from runs)")
    args = p.parse_args()
    S.configure(args.n_sites, args.minefield, args.sup_speed, args.shape_w)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    lam = 0.0 if args.condition in ("baseline", "oracle", "ear") else args.lam
    use_listener = args.condition in ("learned", "learned_prag", "ear", "learned_ear",
                                      "learned_act", "learned_act_prag",
                                      "learned_act_ear",
                                      "learned_pre", "learned_pre_prag",
                                      "learned_frozen_prag", "learned_frozensat_prag",
                                      "inforeg", "partner_belief")
    listener_inputs = ("act" if (args.condition.startswith(("learned_act", "learned_pre"))
                         or args.condition == "learned_frozen_prag")
                       else "partner" if args.condition == "partner_belief"
                       else "full")
    listener_prekey = args.condition.startswith("learned_pre")
    frozen_listener = args.condition == "learned_frozen_prag"
    frozen_sat = args.condition == "learned_frozensat_prag"
    # external baselines: Strouse-style variational I(A;M|S) needs a second,
    # state-only decoder; Tian-style partner belief reads the supporter's obs
    inforeg = args.condition == "inforeg"
    pbelief = args.condition == "partner_belief"
    ipl = args.condition in ("ipl", "ipl_prag")
    ear = args.condition in ("ear", "learned_ear", "filter_ear", "learned_act_ear")

    env = S.ScoutSupportEnv(args.n_envs, oracle=(args.condition == "oracle"),
                            seed=args.seed, blind=args.blind)
    actor = Actor(hidden=args.hidden, obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM).to(device)
    critic = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS, obs_dim=S.OBS_DIM,
                           act_dim=S.ACT_DIM).to(device)
    critic_t = CentralCritic(hidden=args.hidden, n_agents=S.N_AGENTS, obs_dim=S.OBS_DIM,
                             act_dim=S.ACT_DIM).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(S.ACT_DIM)

    listener = S.ScoutListener(inputs=listener_inputs).to(device) if use_listener else None
    opt_l = (torch.optim.Adam(listener.parameters(), lr=args.listener_lr)
             if use_listener else None)
    if frozen_sat:
        # Reviewer-requested cell: freeze the CONVERGED, saturated, full-context
        # learned listener and use it as a static reward. Isolates "frozen"
        # from "restricted/competence-grounded".
        sat_ck = args.frozen_ckpt or f"results_scout3/learned_s{args.seed}/model.pt"
        sd = torch.load(sat_ck, map_location="cpu", weights_only=True)["listener"]
        listener.load_state_dict(sd)
        for prm in listener.parameters():
            prm.requires_grad_(False)
        opt_l = None
        print(f"[frozen-saturated listener] loaded {sat_ck}", flush=True)
    if frozen_listener:
        # Ablation separating the IPL's two changes. The IPL is both frozen
        # (cannot co-adapt) and grounded in a task-competent reference policy.
        # Here we keep the freeze but drop the grounding: fit the same bounded
        # listener by maximum likelihood on rollouts of the SAME frozen
        # seed-disjoint baseline, then freeze it for the whole run.
        pretrain_listener_on_ref(listener, args, device)
        for prm in listener.parameters():
            prm.requires_grad_(False)
        opt_l = None
    listener2 = S.ScoutListener(inputs="state").to(device) if inforeg else None
    opt_l2 = (torch.optim.Adam(listener2.parameters(), lr=args.listener_lr)
              if inforeg else None)
    if ipl:
        ref_ckpt = torch.load(args.ref_ckpt, map_location="cpu",
                              weights_only=True)
        ref_actor = Actor(obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
        ref_actor.load_state_dict(ref_ckpt["actor"])
        ref_actor.to(device).eval()
        for prm in ref_actor.parameters():
            prm.requires_grad_(False)
        listener = IPLWrapper(ref_actor)

    buf = ReplayBuffer(400_000, device, n_agents=S.N_AGENTS,
                       obs_dim=S.OBS_DIM, act_dim=S.ACT_DIM)
    gen = torch.Generator().manual_seed(args.seed + 999)

    history = []
    t0 = time.time()
    total_steps = 0
    start_cycle = 0
    resumed_at = None
    nets_d = {"actor": actor, "critic": critic, "critic_t": critic_t}
    if hasattr(listener, "state_dict"):
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
        if ear:
            obs = inject_ear(obs, torch.full((args.n_envs, S.N_MEANINGS), 1 / 3))
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
            target = env.target
            next_obs, info, done = env.step(a)

            r_comm = torch.zeros((args.n_envs, S.N_AGENTS))
            if args.condition in HANDCRAFTED or args.condition == "filter_ear":
                kind = "filter" if args.condition == "filter_ear" else args.condition
                with torch.no_grad():
                    rc = S.scout_comm_reward(env, a, kind, gen=gen, pre_pos=pre_pos)
                    r_comm[:, 0] = rc * (args.voi + (1 - args.voi) * pre_key)
            if ear:
                with torch.no_grad():
                    if args.condition == "filter_ear":
                        post = env.log_belief.exp()
                    else:
                        post = F.softmax(listener(obs[:, 0].to(device),
                                                  a[:, 0].to(device)), dim=-1).cpu()
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

                if use_listener and not (frozen_listener or frozen_sat):
                    l_in = b_obs[:, 1] if pbelief else b_obs[:, 0]
                    logits = listener(l_in, b_act[:, 0])
                    if listener_prekey:
                        w = 1.0 - b_obs[:, 0, S.KEY_INDEX]
                        ce = F.cross_entropy(logits, b_target, reduction="none")
                        l_loss = (ce * w).sum() / w.sum().clamp(min=1.0)
                    else:
                        l_loss = F.cross_entropy(logits, b_target)
                    opt_l.zero_grad(); l_loss.backward(); opt_l.step()
                    if inforeg:
                        logits2 = listener2(b_obs[:, 0], b_act[:, 0])
                        l2_loss = F.cross_entropy(logits2, b_target)
                        opt_l2.zero_grad(); l2_loss.backward(); opt_l2.step()
                    if lam > 0:
                        with torch.no_grad():
                            if args.condition in ("learned_prag", "learned_act_prag",
                                                  "learned_pre_prag",
                                                  "learned_frozen_prag",
                                                  "learned_frozensat_prag"):
                                if args.alt_policy:
                                    o_rep = b_obs[:, 0].repeat_interleave(16, dim=0)
                                    alt, _ = actor.sample(o_rep)
                                    alt = alt.reshape(args.batch, 16, S.ACT_DIM)
                                else:
                                    alt = torch.rand((args.batch, 16, 2),
                                                     device=device) * 2 - 1
                                rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                          b_target, pragmatic=True,
                                                          alt_actions=alt)
                            elif inforeg:
                                # variational I(A; M | S): what the action adds
                                # beyond the (masked) state
                                lp1 = F.log_softmax(logits, dim=-1)
                                lp0 = F.log_softmax(logits2, dim=-1)
                                g = b_target.unsqueeze(-1)
                                rc = (torch.gather(lp1, -1, g)
                                      - torch.gather(lp0, -1, g)).squeeze(-1)
                            else:
                                rc = listener.comm_reward(l_in, b_act[:, 0],
                                                          b_target)
                            b_pre_key = 1.0 - b_obs[:, 0, S.KEY_INDEX]
                            rc = rc * (args.voi + (1 - args.voi) * b_pre_key)
                            b_rcomm = torch.zeros_like(b_rcomm)
                            b_rcomm[:, 0] = rc

                if ipl and lam > 0:
                    with torch.no_grad():
                        if args.condition == "ipl_prag":
                            alt = torch.rand((args.batch, 16, 2),
                                             device=device) * 2 - 1
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                      b_target, pragmatic=True,
                                                      alt_actions=alt)
                        else:
                            rc = listener.comm_reward(b_obs[:, 0], b_act[:, 0],
                                                      b_target)
                        b_pre_key = 1.0 - b_obs[:, 0, S.KEY_INDEX]
                        rc = rc * (args.voi + (1 - args.voi) * b_pre_key)
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
            m = evaluate(actor, args.condition, listener, seed=12345, device=device,
                         blind=args.blind, listener2=listener2)
            m.update(cycle=cycle + 1, steps=total_steps,
                     wall_min=round((time.time() - t0) / 60, 1))
            if resumed_at is not None:
                m["resumed_at"] = resumed_at
                resumed_at = None
            history.append(m)
            print(f"[{args.condition} s{args.seed}] cyc {cycle+1}/{args.cycles} "
                  f"R_ext {m['r_ext']:.2f} bonus {m['pref_bonus']:.2f} "
                  f"commit {m['commit_acc']:.2f} probe {m['probe_acc']:.2f} "
                  f"comm {m['comm_rw']:.2f} ({m['wall_min']} min)", flush=True)
            with open(os.path.join(args.outdir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=1)
        if (cycle + 1) % ckpt.CKPT_EVERY == 0 and cycle + 1 < args.cycles:
            ckpt.save(args.outdir, cycle + 1, total_steps, history,
                      nets_d, opts_d, {"log_alpha": log_alpha.detach().cpu()})

    save_l = listener if hasattr(listener, "state_dict") else None
    torch.save({"actor": actor.state_dict(),
                "listener": save_l.state_dict() if save_l else None,
                "listener_inputs": listener_inputs if save_l else None,
                "listener2": listener2.state_dict() if listener2 else None,
                "ref_ckpt": args.ref_ckpt if ipl else None},
               os.path.join(args.outdir, "model.pt"))
    ckpt.clear(args.outdir)
    print(f"DONE {args.condition} seed {args.seed} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
