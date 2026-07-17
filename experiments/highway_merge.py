"""Highway merge negotiation (Environment C), vectorized in torch.

A torch-vectorized implementation of the standard on-ramp merge scenario
(cf. highway-env, Leurent 2018), recast as an implicit-communication task.
A *merger* on a finite ramp privately intends one of three outcomes: continue
to its ramp EXIT, or merge into gap G1 or G2 of a main-lane platoon ordered
(front to back) C_lead, G1, L1, G2, L2 -- where L1 and L2 are RL-controlled
*listener* vehicles and C_lead is a scripted cruiser. Each gap opens only if
the car behind it brakes early (sustained, comfortable deceleration), which
costs progress -- so a rational listener brakes only once it believes the
merger wants its gap. The ramp's approach phase is structurally uninformative
(closing distance to the platoon zone is optimal for every intent), so early
information must be actively signaled through the merger's observable speed
profile; a finite ramp plus limited braking authority make late inference
costly. The oracle-gap criterion certifies the resulting premium empirically.

Meanings: 0 = EXIT (no merge), 1 = G1 (ahead of L1), 2 = G2 (ahead of L2).
Vocabulary: one-dimensional speed modulation -- deliberately disjoint from
scout-support's radial site geometry.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- constants
N_AGENTS = 3          # 0 = merger (speaker), 1 = L1, 2 = L2 (listeners)
N_MEANINGS = 3
ACT_DIM = 2           # [accel -1..1 -> A_MIN..A_MAX, merger-only: commit]
EPISODE_LEN = 70
DT = 0.2

V_NOM = 10.0
V_MAX = 16.0
A_MAX = 2.0
A_MIN = -3.0
HARSH = 2.0

CAR_LEN = 5.0
TIGHT_GAP = 6.5       # initial bumper gaps: too tight to merge
SAFE_GAP = 10.0       # gap needed to merge safely
RAMP_END = 160.0      # merge (or exit) must happen before this x

HIST_LEN = 6

W_PROGRESS = 0.1
W_HARSH = 0.3
W_FUEL = 0.08         # per-step cost of listener speed deviation: makes
                      # unconditional gap-opening dominated by targeted,
                      # inference-driven opening (gate iteration 4)
MERGE_BONUS = 12.0
EXIT_BONUS = 4.0
FAIL_PENALTY = 4.0
W_COLL = 4.0
W_TAILGATE = 0.05

# per-agent observation layout:
#   role one-hot (3), own x/50, own v/VMAX, own lane,
#   meaning one-hot (3)  [merger only; oracle: listeners too],
#   slot-center rel x (3)/50,
#   merger rel x/50, merger v/VMAX, merger lane,
#   L1 rel x/50, L1 v/VMAX, L2 rel x/50, L2 v/VMAX,
#   merger accel history (HIST_LEN)/A_MAX,
#   lead rel x/50, lead v/VMAX
OBS_DIM = 3 + 3 + N_MEANINGS + N_MEANINGS + 3 + 4 + HIST_LEN + 2
PREF_SLICE = slice(6, 6 + N_MEANINGS)
SLOT_SLICE = slice(6 + N_MEANINGS, 6 + 2 * N_MEANINGS)
KEY_INDEX = 5         # own lane: merger 1 on the ramp (pre-commit), 0 after


class HighwayMergeEnv:
    def __init__(self, n_envs, oracle=False, blind=False, seed=0):
        self.n = n_envs
        self.oracle = oracle
        self.blind = blind
        self.gen = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        n, g = self.n, self.gen
        self.t = 0
        self.target = torch.randint(0, N_MEANINGS, (n,), generator=g)
        base = 26.0 + torch.rand((n,), generator=g) * 6.0    # L2 start
        step = CAR_LEN + TIGHT_GAP
        self.x = torch.zeros((n, N_AGENTS))
        self.x[:, 2] = base                                   # L2
        self.x[:, 1] = base + step                            # L1
        self.lead = base + 2 * step                           # scripted C_lead
        self.lead_v = torch.full((n,), V_NOM)
        self.x[:, 0] = base - 16.0 - torch.rand((n,), generator=g) * 8.0
        self.v = torch.full((n, N_AGENTS), V_NOM)
        self.v[:, 0] += torch.rand((n,), generator=g) * 2.0 - 1.0
        self.lane = torch.zeros((n, N_AGENTS))
        self.lane[:, 0] = 1.0
        self.merged = torch.zeros(n, dtype=torch.bool)
        self.failed = torch.zeros(n, dtype=torch.bool)
        self.exited = torch.zeros(n, dtype=torch.bool)
        self.hist = torch.zeros((n, HIST_LEN))
        return self.obs()

    def slot_centers(self):
        ext = torch.full((self.n,), RAMP_END)
        g1 = (self.lead + self.x[:, 1]) * 0.5
        g2 = (self.x[:, 1] + self.x[:, 2]) * 0.5
        return torch.stack([ext, g1, g2], dim=1)

    def obs(self):
        n = self.n
        slots = self.slot_centers()
        obs = []
        for i in range(N_AGENTS):
            role = F.one_hot(torch.tensor(i), N_AGENTS).float().expand(n, -1)
            own = torch.stack([self.x[:, i] / 50.0, self.v[:, i] / V_MAX,
                               self.lane[:, i]], dim=1)
            pref = torch.zeros((n, N_MEANINGS))
            if i == 0 or self.oracle:
                pref = F.one_hot(self.target, N_MEANINGS).float()
            slot_rel = (slots - self.x[:, i:i + 1]) / 50.0
            merger = torch.stack([(self.x[:, 0] - self.x[:, i]) / 50.0,
                                  self.v[:, 0] / V_MAX, self.lane[:, 0]], dim=1)
            hist = self.hist / A_MAX
            if i > 0 and self.blind:
                merger = torch.zeros_like(merger)
                hist = torch.zeros_like(hist)
            peers = torch.stack([(self.x[:, 1] - self.x[:, i]) / 50.0,
                                 self.v[:, 1] / V_MAX,
                                 (self.x[:, 2] - self.x[:, i]) / 50.0,
                                 self.v[:, 2] / V_MAX], dim=1)
            lead = torch.stack([(self.lead - self.x[:, i]) / 50.0,
                                self.lead_v / V_MAX], dim=1)
            obs.append(torch.cat([role, own, pref, slot_rel, merger, peers,
                                  hist, lead], dim=1))
        return torch.stack(obs, dim=1)

    def step(self, actions):
        n = self.n
        a = actions[:, :, 0].clamp(-1, 1)
        accel = torch.where(a >= 0, a * A_MAX, a * (-A_MIN))
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = accel[:, 0]

        self.v = (self.v + accel * DT).clamp(0.0, V_MAX)
        self.x = self.x + self.v * DT
        self.lead_v = self.lead_v + 0.2 * (V_NOM - self.lead_v) * DT
        self.lead = self.lead + self.lead_v * DT

        slots = self.slot_centers()
        tgt_slot = torch.gather(slots, 1, self.target.unsqueeze(1)).squeeze(1)

        gap1 = self.lead - self.x[:, 1] - CAR_LEN
        gap2 = self.x[:, 1] - self.x[:, 2] - CAR_LEN
        open_ = torch.stack([torch.zeros(n, dtype=torch.bool),
                             gap1 > SAFE_GAP, gap2 > SAFE_GAP], dim=1)
        tgt_open = torch.gather(open_, 1, self.target.unsqueeze(1)).squeeze(1)
        can_zone = (self.x[:, 0] > self.x[:, 2] - 10.0)
        aligned = (self.x[:, 0] - tgt_slot).abs() < 4.0
        commit = ((actions[:, 0, 1] > 0) & can_zone & aligned & tgt_open
                  & (~self.merged) & (~self.failed) & (~self.exited))
        newly = commit
        self.merged = self.merged | commit
        self.lane[:, 0] = torch.where(self.merged, 0.0, self.lane[:, 0])

        crossing = ((~self.merged) & (~self.failed) & (~self.exited)
                    & (self.x[:, 0] > RAMP_END))
        is_exit = self.target == 0
        newly_exited = crossing & is_exit
        newly_failed = crossing & (~is_exit)
        self.failed = self.failed | newly_failed
        self.exited = self.exited | newly_exited

        prog = W_PROGRESS * (self.v.sum(dim=1) / (N_AGENTS * V_NOM) * 2.0)
        harsh = W_HARSH * F.relu(accel.abs() - HARSH).sum(dim=1)
        ramp_frac = ((RAMP_END - self.x[:, 0]) / RAMP_END).clamp(0, 1)
        bonus = MERGE_BONUS * (0.5 + ramp_frac) * newly.float()
        exit_b = EXIT_BONUS * (self.v[:, 0] / V_MAX) * newly_exited.float()
        fail = FAIL_PENALTY * newly_failed.float()

        overlap = torch.zeros(n)
        for other in (self.lead, self.x[:, 1], self.x[:, 2]):
            overlap = overlap + ((self.x[:, 0] - other).abs() < CAR_LEN).float() \
                * newly.float()
        coll = W_COLL * overlap
        crowd = (F.relu(1.0 - (self.lead - self.x[:, 1] - CAR_LEN) / 2.0)
                 + F.relu(1.0 - (self.x[:, 1] - self.x[:, 2] - CAR_LEN) / 2.0))
        tailgate = W_TAILGATE * crowd
        fuel = W_FUEL * accel[:, 1:].abs().sum(dim=1)

        r = prog + bonus + exit_b - harsh - fail - coll - tailgate - fuel
        self.t += 1
        done = self.t >= EPISODE_LEN
        success = torch.where(is_exit, self.exited, self.merged)
        info = {"r_ext": r, "merge_rate": success.float(),
                "fail_rate": self.failed.float(), "harsh": harsh,
                "bonus": bonus + exit_b, "open_correct": tgt_open.float()}
        return self.obs(), info, done


# ------------------------------------------------------- listener machinery
def slot_progress_reward(env, pre_x, pre_slots):
    """Gap-progress listener: per-step reduction of the merger's misalignment
    to each candidate slot (1D progress analog; vocabulary = speed profile)."""
    slots = env.slot_centers()
    err_now = (slots - env.x[:, 0:1]).abs()
    err_pre = (pre_slots - pre_x.unsqueeze(1)).abs()
    prog = (err_pre - err_now) / (V_MAX * DT)
    L = F.softmax(5.0 * prog, dim=1)
    p = torch.gather(L, 1, env.target.unsqueeze(1)).squeeze(1)
    return torch.log(p.clamp(min=1e-8)) + math.log(N_MEANINGS)


def simple_speed_reward(env):
    """Simple listener: speed-deviation direction vs. each slot."""
    slots = env.slot_centers()
    rel = slots - env.x[:, 0:1]
    dv = (env.v[:, 0:1] - V_NOM) / V_MAX
    score = 5.0 * dv * torch.sign(rel) / (rel.abs() / 20.0 + 1.0)
    L = F.softmax(score, dim=1)
    p = torch.gather(L, 1, env.target.unsqueeze(1)).squeeze(1)
    return torch.log(p.clamp(min=1e-8)) + math.log(N_MEANINGS)


class MergeListener(nn.Module):
    """Learned listener; inputs='act' = audience viewpoint (slot bearings +
    action), 'full' = masked observation + action, 'state' = masked obs only."""

    def __init__(self, hidden=128, inputs="act"):
        super().__init__()
        self.inputs = inputs
        in_dim = {"full": OBS_DIM + ACT_DIM, "act": N_MEANINGS + ACT_DIM,
                  "state": OBS_DIM}[inputs]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MEANINGS),
        )

    def features(self, obs, actions):
        if self.inputs == "act":
            return torch.cat([obs[..., SLOT_SLICE], actions], dim=-1)
        x = obs.clone()
        x[..., PREF_SLICE] = 0.0
        if self.inputs == "state":
            return x
        return torch.cat([x, actions], dim=-1)

    def forward(self, obs, actions):
        return self.net(self.features(obs, actions))

    def comm_reward(self, obs, actions, target, pragmatic=False,
                    alt_actions=None, n_iter=1):
        if not pragmatic:
            logp = F.log_softmax(self(obs, actions), dim=-1)
            lp = torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
            return lp + math.log(N_MEANINGS)
        acts = torch.cat([actions.unsqueeze(-2), alt_actions], dim=-2)
        A = acts.shape[-2]
        obs_e = obs.unsqueeze(-2).expand(*obs.shape[:-1], A, obs.shape[-1])
        logits = self(obs_e, acts)
        L = F.softmax(logits, dim=-1)
        for _ in range(n_iter):
            S = L / L.sum(dim=-2, keepdim=True).clamp(min=1e-12)
            L = S / S.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = torch.gather(L[..., 0, :], -1, target.unsqueeze(-1)).squeeze(-1)
        return torch.log(p.clamp(min=1e-8)) + math.log(N_MEANINGS)
