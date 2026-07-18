"""Highway fork negotiation (Environment C, fork design), torch-vectorized.

A three-lane highway approaching a fork. An *ego* vehicle in the middle lane
privately intends one of three routes: leave on the LEFT branch, leave on the
RIGHT branch, or continue THRU on the middle carriageway. The left and right
lanes each carry a dense co-moving chain [flow, C_a, C_b, flow]; C_a and C_b
are RL *cooperators* (lane-locked), flow cars are scripted. Initial bumper
gaps (6 m) are far below the safe entry gap (10 m), so an unassisted lane
change is kinematically infeasible: the ego reaches its branch only where a
cooperator brakes early and holds a gap open -- and braking propagates down
the chain (followers keep their distance), so every opening costs several
cars' speed for as long as it is held.

Why the premium is structural rather than priced in:
  - Accommodations are disjoint by identity: a left-chain gap is useless for
    a RIGHT ego and vice versa, and a THRU ego needs neither. Hedging means
    opening BOTH chains -- twice the cascade cost, all episode.
  - Only canonical driving rewards appear: speed tracking (all nine cars,
    i.e. traffic flow), comfort, collision, branch/completion bonuses.
    Density, chain spacing, and the fork deadline are scenario geometry.
  - The certification protocol (certify_corridor.py) scores the informed
    schedule against the BEST uninformed script -- passive, hedge-both,
    guess-left, guess-right -- the policy families that defeated the on-ramp
    design, not merely a passive strawman.

Meanings: 0 = LEFT, 1 = RIGHT, 2 = THRU.
Vocabulary: the ego's observable speed profile and lateral positioning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- constants
N_AGENTS = 5          # 0 = ego (speaker); 1,2 = C_a,C_b left; 3,4 = right
N_COOP = 4
N_FLOW = 4            # scripted: lead + tail per chain
N_MEANINGS = 3
ACT_DIM = 2           # [longitudinal accel, lateral velocity (ego only)]
EPISODE_LEN = 75
DT = 0.2

V_NOM = 10.0
V_MAX = 14.0
A_MAX = 2.0
A_MIN = -3.0
HARSH = 2.0
V_LAT = 0.7           # lanes per second; one lane change ~1.4 s

CAR_LEN = 5.0
TIGHT_GAP = 6.0       # initial bumper gaps: unthreadable
SAFE_GAP = 10.0       # bumper gap needed to enter a lane

X_FORK = 95.0         # exit deadline: be on your branch lane when crossing
ROAD_END = 150.0      # THRU completion (middle lane)

W_SPEED = 0.1
W_HARSH = 0.3
W_COLL = 4.0
EXIT_BONUS = 10.0
THRU_BONUS = 4.0
FAIL_PENALTY = 4.0

HIST_LEN = 6

# lane index of each meaning's completion: LEFT -> y=0, RIGHT -> y=2, THRU -> y=1
MEAN_Y = torch.tensor([0.0, 2.0, 1.0])
MEAN_X = torch.tensor([X_FORK, X_FORK, ROAD_END])

# hoisted constants (allocation-free obs/step hot paths)
_ROLE = torch.eye(5)
_MEANX_ROW = MEAN_X.view(1, 3)
_MEANY_ROW = MEAN_Y.view(1, 3)
_NOT_EYE9 = (~torch.eye(9, dtype=torch.bool)).unsqueeze(0)

# per-agent observation layout:
#   role one-hot (5), own x/50 v/VMAX y/2,
#   ego-pre flag (1: ego still centered in the middle lane),
#   meaning one-hot (3) [ego only; oracle: cooperators too],
#   completion-point rel x (3)/50   [the audience-viewpoint "bearings"],
#   ego rel x/50, ego v/VMAX, ego y/2          [masked if blind],
#   ego action history (HIST_LEN x [accel, lat])  [masked if blind],
#   others block: 4 cooperators + 4 flow cars, each [rel x/50, v/VMAX, y/2]
OBS_DIM = 5 + 3 + 1 + N_MEANINGS + 3 + 3 + 2 * HIST_LEN + 3 * (N_COOP + N_FLOW)
PREF_SLICE = slice(9, 9 + N_MEANINGS)
SLOT_SLICE = slice(9 + N_MEANINGS, 9 + 2 * N_MEANINGS)
KEY_INDEX = 8         # ego-pre flag


def _chain_x(base):
    """Front-to-back x positions [flow_lead, C_a, C_b, flow_tail]."""
    step = CAR_LEN + TIGHT_GAP
    return torch.stack([base + 3 * step, base + 2 * step, base + step, base],
                       dim=1)


class HighwayCorridorEnv:
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
        baseL = 6.0 + torch.rand((n,), generator=g) * 6.0
        baseR = 6.0 + torch.rand((n,), generator=g) * 6.0
        cL = _chain_x(baseL)                     # lane 0: [F, C_a, C_b, F]
        cR = _chain_x(baseR)                     # lane 2
        self.x = torch.zeros((n, N_AGENTS))
        # ego roughly abeam C_a's front gap midpoints
        self.x[:, 0] = 0.5 * (baseL + baseR) + 2.0 * (CAR_LEN + TIGHT_GAP) \
            + torch.rand((n,), generator=g) * 6.0 - 3.0
        self.x[:, 1], self.x[:, 2] = cL[:, 1], cL[:, 2]
        self.x[:, 3], self.x[:, 4] = cR[:, 1], cR[:, 2]
        self.y = torch.zeros((n, N_AGENTS))
        self.y[:, 0] = 1.0
        self.y[:, 3:5] = 2.0
        self.v = torch.full((n, N_AGENTS), V_NOM)
        self.v[:, 0] += torch.rand((n,), generator=g) * 2.0 - 1.0
        # scripted flow cars: [leadL, tailL, leadR, tailR]
        self.fx = torch.stack([cL[:, 0], cL[:, 3], cR[:, 0], cR[:, 3]], dim=1)
        self.fv = torch.full((n, N_FLOW), V_NOM)
        self.fy = torch.tensor([0.0, 0.0, 2.0, 2.0]).expand(n, -1).clone()
        self.exited = torch.zeros(n, dtype=torch.bool)
        self.done_thru = torch.zeros(n, dtype=torch.bool)
        self.failed = torch.zeros(n, dtype=torch.bool)
        self.hist = torch.zeros((n, HIST_LEN, 2))
        return self.obs()

    # ------------------------------------------------------------- helpers
    def pre_flag(self):
        return ((self.y[:, 0] - 1.0).abs() < 0.4).float()

    def gaps(self):
        """Bumper gaps ahead of C_a and C_b per side: (n, 4) as
        [L: F-C_a, C_a-C_b, R: F-C_a, C_a-C_b]."""
        gL1 = self.fx[:, 0] - self.x[:, 1] - CAR_LEN
        gL2 = self.x[:, 1] - self.x[:, 2] - CAR_LEN
        gR1 = self.fx[:, 2] - self.x[:, 3] - CAR_LEN
        gR2 = self.x[:, 3] - self.x[:, 4] - CAR_LEN
        return torch.stack([gL1, gL2, gR1, gR2], dim=1)

    def all_x(self):
        return torch.cat([self.x, self.fx], dim=1)      # (n, 9)

    def all_y(self):
        return torch.cat([self.y, self.fy], dim=1)

    def all_v(self):
        return torch.cat([self.v, self.fv], dim=1)

    def obs(self):
        n = self.n
        wc = _MEANX_ROW.expand(n, -1)
        obs = []
        pre = self.pre_flag().unsqueeze(1)
        for i in range(N_AGENTS):
            role = _ROLE[i].expand(n, -1)
            own = torch.stack([self.x[:, i] / 50.0, self.v[:, i] / V_MAX,
                               self.y[:, i] / 2.0], dim=1)
            pref = torch.zeros((n, N_MEANINGS))
            if i == 0 or self.oracle:
                pref = F.one_hot(self.target, N_MEANINGS).float()
            slot_rel = (wc - self.x[:, i:i + 1]) / 50.0
            ego = torch.stack([(self.x[:, 0] - self.x[:, i]) / 50.0,
                               self.v[:, 0] / V_MAX, self.y[:, 0] / 2.0], dim=1)
            hist = self.hist.reshape(n, -1)
            if i > 0 and self.blind:
                ego = torch.zeros_like(ego)
                hist = torch.zeros_like(hist)
            oth_x, oth_y, oth_v = self.all_x(), self.all_y(), self.all_v()
            others = torch.stack([(oth_x[:, 1:] - self.x[:, i:i + 1]) / 50.0,
                                  oth_v[:, 1:] / V_MAX,
                                  oth_y[:, 1:] / 2.0], dim=2).reshape(n, -1)
            obs.append(torch.cat([role, own, pre, pref, slot_rel, ego, hist,
                                  others], dim=1))
        return torch.stack(obs, dim=1)

    # ---------------------------------------------------------------- step
    def _flow_accel(self):
        """Scripted flow: hold V_NOM; tails keep distance to C_b of their
        chain (the natural cascade that prices every opening)."""
        acc = 0.6 * (V_NOM - self.fv)
        for j, leader in [(1, 2), (3, 4)]:
            gap = self.x[:, leader] - self.fx[:, j] - CAR_LEN
            acc[:, j] = acc[:, j] - 2.0 * F.relu(TIGHT_GAP * 0.7 - gap)
        return acc.clamp(A_MIN, A_MAX)

    def step(self, actions):
        n = self.n
        a_long = actions[:, :, 0].clamp(-1, 1)
        accel = torch.where(a_long >= 0, a_long * A_MAX, a_long * (-A_MIN))
        frozen0 = self.exited | self.done_thru | self.failed
        accel[:, 0] = torch.where(frozen0, torch.zeros(n), accel[:, 0])
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1, 0] = accel[:, 0] / A_MAX
        self.hist[:, -1, 1] = actions[:, 0, 1].clamp(-1, 1)

        self.v = (self.v + accel * DT).clamp(0.0, V_MAX)
        self.x = self.x + self.v * DT
        self.fv = (self.fv + self._flow_accel() * DT).clamp(0.0, V_MAX)
        self.fx = self.fx + self.fv * DT

        # ego lateral motion (cooperators are lane-locked)
        dy = actions[:, 0, 1].clamp(-1, 1) * V_LAT * DT
        dy = torch.where(frozen0, torch.zeros(n), dy)
        self.y[:, 0] = (self.y[:, 0] + dy).clamp(0.0, 2.0)

        # collisions: any pair sharing a lane band closer than CAR_LEN;
        # settled (exited/failed/thru) egos are off the carriageway
        ax, ay = self.all_x(), self.all_y()
        dx = (ax.unsqueeze(2) - ax.unsqueeze(1)).abs()
        dyy = (ay.unsqueeze(2) - ay.unsqueeze(1)).abs()
        hit = (dx < CAR_LEN) & (dyy < 0.6)
        hit = hit & _NOT_EYE9
        hit[:, 0] = hit[:, 0] & ~frozen0.unsqueeze(1)
        hit[:, :, 0] = hit[:, :, 0] & ~frozen0.unsqueeze(1)
        n_hit = hit[:, :N_AGENTS].any(dim=2).float().sum(dim=1)

        # fork / completion logic
        xe, ye = self.x[:, 0], self.y[:, 0]
        live = ~frozen0
        at_fork = live & (xe > X_FORK)
        side_y = MEAN_Y[self.target]
        on_branch = (ye - side_y).abs() < 0.3
        wants_exit = self.target < 2
        newly_exit = at_fork & wants_exit & on_branch
        newly_fail = at_fork & wants_exit & (~on_branch)
        newly_thru = live & (self.target == 2) & (xe > ROAD_END) \
            & ((ye - 1.0).abs() < 0.5)
        self.exited = self.exited | newly_exit
        self.failed = self.failed | newly_fail
        self.done_thru = self.done_thru | newly_thru

        # canonical rewards: traffic speed, comfort, collision, bonuses
        ego_v = torch.where(frozen0 | newly_exit | newly_thru,
                            torch.full((n,), V_NOM), self.v[:, 0])
        vs = torch.cat([ego_v.unsqueeze(1), self.v[:, 1:], self.fv], dim=1)
        speed = W_SPEED * (1.0 - (vs - V_NOM).abs() / V_NOM).sum(dim=1) \
            * (N_AGENTS / vs.shape[1])
        harsh = W_HARSH * F.relu(accel.abs() - HARSH).sum(dim=1)
        coll = W_COLL * n_hit
        bonus = EXIT_BONUS * newly_exit.float() \
            + THRU_BONUS * newly_thru.float() \
            - FAIL_PENALTY * newly_fail.float()

        r = speed + bonus - harsh - coll
        self.t += 1
        done = self.t >= EPISODE_LEN
        success = torch.where(wants_exit, self.exited, self.done_thru)
        info = {"r_ext": r, "merge_rate": success.float(),
                "fail_rate": self.failed.float(),
                "harsh": harsh, "bonus": bonus,
                "coll": (n_hit > 0).float()}
        return self.obs(), info, done


# ------------------------------------------------------- listener machinery
def corridor_progress_reward(env, pre_x, pre_y):
    """Progress listener: per-step reduction of the ego's distance to each
    meaning's completion point (fork on the branch lane, road end on the
    middle); one lane of lateral offset weighted like 10 m of road."""
    tx = _MEANX_ROW.expand(env.n, -1)
    ty = _MEANY_ROW.expand(env.n, -1)
    d_now = ((tx - env.x[:, 0:1]).abs() + 10.0 * (ty - env.y[:, 0:1]).abs())
    d_pre = ((tx - pre_x.unsqueeze(1)).abs() + 10.0 * (ty - pre_y.unsqueeze(1)).abs())
    prog = (d_pre - d_now) / (V_MAX * DT)
    L = F.softmax(5.0 * prog, dim=1)
    p = torch.gather(L, 1, env.target.unsqueeze(1)).squeeze(1)
    return torch.log(p.clamp(min=1e-8)) + math.log(N_MEANINGS)


class CorridorListener(nn.Module):
    """Learned listener; inputs='act' = audience viewpoint (completion-point
    bearings + action), 'full' = masked observation + action, 'state' =
    masked observation only."""

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
