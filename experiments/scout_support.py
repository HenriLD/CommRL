"""Scout-support environment and its listener family.

Two agents: a SCOUT that privately knows which of three sites is the true
target, and a slower SUPPORTER that must be at the target together with the
scout to earn the team bonus. The supporter never observes the target; it can
only infer it from the scout's motion, so the task value of legible behavior
is structural: an oracle supporter (told the target) commits immediately,
while a naive one loses reward to late or wrong commitment.

Sites are placed on a circle around the origin and the scout starts near the
center, so early scout motion is maximally ambiguous and disambiguation is a
choice, not an accident.

Listener family for R_comm on the scout (the supporter has no private state):
  - simple       cosine(action, direction-to-site), softmax
  - exclusivity  cosine margin against best competing site (thesis Alg. 1)
  - progress     per-step reduction of distance-to-site (Dragan-style)
  - filter       recursive Bayesian evidence accumulation over the episode
  - learned      L_theta(m | s, a) trained on replay (+ optional RSA)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_AGENTS = 2            # 0 = scout, 1 = supporter
N_SITES = 3
N_MEANINGS = 3
DT = 0.1
DAMPING = 0.25
ACCEL = 5.0
SCOUT_SPEED = 1.0
SUPPORT_SPEED = 0.6
SITE_RADIUS = 1.0
SITE_JITTER = 0.1
COVER_RADIUS = 0.3
KEY_RADIUS = 0.3
BONUS = 3.0
EPISODE_LEN = 50
HIST_LEN = 6              # steps of scout motion history in observations

# The scout must first visit a pickup waypoint near the center before its
# presence at the target counts. Its first-leg motion is therefore
# task-uninformative about the target: any information the supporter gets
# early must be actively signaled. The scout starts on an outer annulus so
# this ambiguous first leg is long enough for signaling to matter.

# observation layout per agent (K = N_SITES):
#   own pos (2) | own vel (2) | role one-hot (2)
#   own meaning one-hot (K; zeros for supporter)
#   site rel pos (2K) | waypoint rel pos (2) | has-key flag (1)
#   partner rel pos (2) | partner vel (2)
#   partner meaning one-hot (K; zeros unless oracle supporter)
#   scout motion history (HIST_LEN * (vel 2 + action 2); public, same for both)
ACT_DIM = 2
MINEFIELD = False
MINE_PENALTY = 1.5
SHAPE_W = 1.0    # multiplier on the scout's distance shaping (signal cost)


def configure(n_sites=3, minefield=False, support_speed=None, shape_w=None):
    """Set the environment family: meaning-space size, reward variant,
    supporter speed (controls the oracle premium), and scout shaping weight
    (controls the economic cost of signaling detours).
    Must be called before constructing envs, listeners, or policies."""
    global N_SITES, N_MEANINGS, MINEFIELD, OBS_DIM, PREF_SLICE, KEY_INDEX, \
        PARTNER_PREF_SLICE, SUPPORT_SPEED, SHAPE_W
    N_SITES = N_MEANINGS = n_sites
    MINEFIELD = minefield
    if support_speed is not None:
        SUPPORT_SPEED = support_speed
    if shape_w is not None:
        SHAPE_W = shape_w
    K = n_sites
    OBS_DIM = 37 + 4 * K
    PREF_SLICE = slice(6, 6 + K)
    KEY_INDEX = 8 + 3 * K
    PARTNER_PREF_SLICE = slice(13 + 3 * K, 13 + 4 * K)


configure(3, False)


class ScoutSupportEnv:
    def __init__(self, n_envs, device="cpu", oracle=False, seed=0, blind=False):
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.oracle = oracle
        self.blind = blind    # supporter cannot observe the scout (control)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.t = 0
        self.reset()

    def reset(self):
        E = self.n_envs
        self.t = 0
        # sites on a circle with random rotation and radial jitter
        theta = torch.rand((E, 1), generator=self.gen) * 2 * math.pi
        angles = theta + (2 * math.pi / N_SITES) * torch.arange(N_SITES).view(1, N_SITES)
        radius = SITE_RADIUS + (torch.rand((E, N_SITES), generator=self.gen) * 2 - 1) * SITE_JITTER
        self.sites = torch.stack([radius * torch.cos(angles),
                                  radius * torch.sin(angles)], dim=-1).to(self.device)
        # pickup waypoint near the center; scout starts on an outer annulus so
        # the ambiguous first leg is long
        self.waypoint = ((torch.rand((E, 2), generator=self.gen) * 2 - 1) * 0.25).to(self.device)
        ang = torch.rand((E,), generator=self.gen) * 2 * math.pi
        rad = 0.9 + torch.rand((E,), generator=self.gen) * 0.4
        scout_pos = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)
        sup_pos = (torch.rand((E, 2), generator=self.gen) * 2 - 1)
        self.pos = torch.stack([scout_pos, sup_pos], dim=1).to(self.device)
        self.vel = torch.zeros((E, N_AGENTS, 2), device=self.device)
        self.has_key = torch.zeros(E, device=self.device)
        self.target = torch.randint(0, N_SITES, (E,), generator=self.gen).to(self.device)
        self.hist = torch.zeros((E, HIST_LEN, 4), device=self.device)
        # belief state for the filter listener
        self.log_belief = torch.full((E, N_MEANINGS), -math.log(N_MEANINGS),
                                     device=self.device)
        return self.obs()

    @property
    def pref(self):
        """Meaning per agent; the supporter's slot mirrors the target but is
        never observed or rewarded (kept for interface compatibility)."""
        return self.target.unsqueeze(1).expand(-1, N_AGENTS)

    def obs(self):
        E = self.n_envs
        m_oh = F.one_hot(self.target, N_MEANINGS).float()          # (E,3)
        zeros3 = torch.zeros_like(m_oh)
        obs = []
        for i in range(N_AGENTS):
            own_pos = self.pos[:, i]
            own_vel = self.vel[:, i]
            role = torch.zeros((E, 2), device=self.device)
            role[:, i] = 1.0
            own_m = m_oh if i == 0 else zeros3
            site_rel = (self.sites - own_pos.unsqueeze(1)).reshape(E, -1)
            wp_rel = self.waypoint - own_pos
            key = self.has_key.unsqueeze(1)
            j = 1 - i
            partner_rel = self.pos[:, j] - own_pos
            partner_vel = self.vel[:, j]
            partner_m = m_oh if (i == 1 and self.oracle) else zeros3
            hist = self.hist.reshape(E, -1)
            if i == 1 and self.blind:
                # control condition: sever the visual channel to the scout
                partner_rel = torch.zeros_like(partner_rel)
                partner_vel = torch.zeros_like(partner_vel)
                hist = torch.zeros_like(hist)
            obs.append(torch.cat([own_pos, own_vel, role, own_m, site_rel,
                                  wp_rel, key, partner_rel, partner_vel,
                                  partner_m, hist], dim=1))
        return torch.stack(obs, dim=1)  # (E, N, OBS_DIM)

    def step(self, actions):
        actions = actions.clamp(-1, 1)
        # roll the scout motion-history window (velocity before this step,
        # action taken this step)
        self.hist = torch.cat([
            self.hist[:, 1:],
            torch.cat([self.vel[:, 0], actions[:, 0]], dim=-1).unsqueeze(1),
        ], dim=1)
        self.vel = self.vel * (1 - DAMPING) + actions * ACCEL * DT
        speed = self.vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_speed = torch.tensor([SCOUT_SPEED, SUPPORT_SPEED],
                                 device=self.device).view(1, N_AGENTS, 1)
        self.vel = torch.where(speed > max_speed, self.vel / speed * max_speed, self.vel)
        self.pos = (self.pos + self.vel * DT).clamp(-1.5, 1.5)
        self.t += 1

        # pick up the key when the scout reaches the waypoint
        d_wp = (self.pos[:, 0] - self.waypoint).norm(dim=-1)
        self.has_key = torch.maximum(self.has_key, (d_wp < KEY_RADIUS).float())

        tgt = torch.gather(self.sites, 1,
                           self.target.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)  # (E,2)
        d_scout = (self.pos[:, 0] - tgt).norm(dim=-1)
        d_sup = (self.pos[:, 1] - tgt).norm(dim=-1)
        both_in = (d_scout < COVER_RADIUS) & (d_sup < COVER_RADIUS) & (self.has_key > 0)
        # scout shaping: reach the waypoint first, then the target
        scout_shape = torch.where(self.has_key > 0, d_scout, d_wp + 1.0)
        r_ext = BONUS * both_in.float() - SHAPE_W * scout_shape - d_sup
        if MINEFIELD:
            # entering any non-target site is penalized: wrong commitment is
            # costly, not merely slow
            d_all = (self.pos[:, 1].unsqueeze(1) - self.sites).norm(dim=-1)  # (E,K)
            inside = (d_all < COVER_RADIUS).float()
            inside.scatter_(1, self.target.unsqueeze(1), 0.0)
            r_ext = r_ext - MINE_PENALTY * inside.sum(dim=1)

        done = self.t >= EPISODE_LEN
        info = {
            "r_ext": r_ext,
            "pref_bonus": BONUS * both_in.float(),
            "dist_pen": scout_shape + d_sup,
            "coll_pen": torch.zeros_like(r_ext),
            "commit_acc": self.commitment_accuracy(),
        }
        return self.obs(), info, done

    # ---- helpers ----

    def goal_dirs(self):
        """Unit vectors from the scout to each site: (E, 1, M, 2), matching the
        (E, N, M, 2) interface with N=1 (only the scout communicates)."""
        vec = self.sites - self.pos[:, 0].unsqueeze(1)               # (E,M,2)
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return vec.unsqueeze(1)

    def commitment_accuracy(self):
        """Whether the supporter's nearest site is the true target."""
        d = (self.pos[:, 1].unsqueeze(1) - self.sites).norm(dim=-1)  # (E,M)
        return (d.argmin(dim=1) == self.target).float()

    def specialization(self):
        return self.commitment_accuracy()  # interface compatibility


# --------------------------- listeners ---------------------------

def _cos(a, v):
    an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (an.unsqueeze(-2) * v).sum(dim=-1)


def scout_comm_reward(env, actions_true, kind, temp=5.0, n_samples=64, n_iter=1,
                      gen=None, pre_pos=None, pre_log_belief=None):
    """R_comm for the scout under a hand-crafted listener. Must be called
    AFTER env.step() with `pre_pos` = scout position before the step (needed
    by `progress`); direction-based kinds use pre-step geometry implicitly via
    pre_pos. Returns (E,) rewards for the scout.

    kinds: simple | exclusivity | progress | filter
    """
    E = env.n_envs
    dev = actions_true.device
    a = actions_true[:, 0]                                           # (E,2) scout action
    log3 = math.log(N_MEANINGS)

    if kind == "progress":
        # per-site progress made this step, normalized by max step length
        d_pre = (pre_pos.unsqueeze(1) - env.sites).norm(dim=-1)      # (E,M)
        d_post = (env.pos[:, 0].unsqueeze(1) - env.sites).norm(dim=-1)
        prog = (d_pre - d_post) / (SCOUT_SPEED * DT)
        post = F.softmax(temp * prog, dim=-1)
        p_true = torch.gather(post, 1, env.target.unsqueeze(1)).squeeze(1)
        return torch.log(p_true.clamp(min=1e-8)) + log3

    dirs = (env.sites - pre_pos.unsqueeze(1))
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)    # (E,M,2)

    if kind == "filter":
        # recursive Bayesian update of the episode belief from this action,
        # with exponential forgetting so old evidence decays and the reward
        # scale stays bounded
        loglik = 2.0 * _cos(a.unsqueeze(1), dirs.unsqueeze(1)).squeeze(1)   # (E,M)
        env.log_belief = 0.9 * env.log_belief + loglik
        env.log_belief = env.log_belief - env.log_belief.logsumexp(dim=1, keepdim=True)
        lp_true = torch.gather(env.log_belief, 1, env.target.unsqueeze(1)).squeeze(1)
        return lp_true.clamp(min=-4.0) + log3

    # sampled-alternative RSA listeners (simple / exclusivity)
    alt = torch.rand((E, n_samples, 2), generator=gen).to(dev) * 2 - 1
    acts = torch.cat([a.unsqueeze(1), alt], dim=1)                   # (E,A,2)
    cs = _cos(acts, dirs.unsqueeze(1))                               # (E,A,M)
    s = temp * cs
    if kind == "exclusivity":
        M = s.shape[-1]
        mask = ~torch.eye(M, dtype=torch.bool, device=dev)
        others = s.unsqueeze(-2).expand(*s.shape[:-1], M, M)
        s = s - others.masked_fill(~mask, -1e9).amax(dim=-1)
    L = F.softmax(s, dim=-1)
    for _ in range(n_iter):
        S = L / L.sum(dim=-2, keepdim=True).clamp(min=1e-12)
        L = S / S.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p_true = torch.gather(L[:, 0], 1, env.target.unsqueeze(1)).squeeze(1)
    return torch.log(p_true.clamp(min=1e-8)) + log3


class ScoutListener(nn.Module):
    """L_theta(m | ., a) for the scout.

    inputs='full': the whole observation (private blocks masked) + action.
    Competent enough to decode any behavior from context -- saturates.
    inputs='act':  only the action and the site bearings from the scout's
    position -- the at-a-glance viewpoint of an audience. Task-driven motion
    toward the waypoint is uninformative under this listener, so the reward
    retains gradient: this is the bounded literal listener RSA prescribes,
    learned rather than hand-crafted.
    """

    def __init__(self, hidden=128, inputs="full"):
        super().__init__()
        self.inputs = inputs
        in_dim = {"full": OBS_DIM + ACT_DIM,       # masked context + action
                  "act": 2 * N_SITES + ACT_DIM,    # audience viewpoint
                  "state": OBS_DIM,                # masked context, no action
                  "partner": OBS_DIM}[inputs]      # the partner's own obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MEANINGS),
        )

    def features(self, obs, actions):
        if self.inputs == "act":
            site_rel = obs[..., 6 + N_MEANINGS: 6 + 3 * N_MEANINGS]
            return torch.cat([site_rel, actions], dim=-1)
        x = obs.clone()
        x[..., PREF_SLICE] = 0.0
        x[..., PARTNER_PREF_SLICE] = 0.0
        if self.inputs in ("state", "partner"):
            return x
        return torch.cat([x, actions], dim=-1)

    def forward(self, obs, actions):
        return self.net(self.features(obs, actions))

    def comm_reward(self, obs, actions, target, pragmatic=False,
                    alt_actions=None, n_iter=1):
        """obs/actions: scout slots only (..., OBS_DIM) / (..., 2)."""
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
        p_true = torch.gather(L[..., 0, :], -1, target.unsqueeze(-1)).squeeze(-1)
        return torch.log(p_true.clamp(min=1e-8)) + math.log(N_MEANINGS)


def probe_intent_metrics(dirs, target, actions):
    """Fixed geometric probe on the scout's action. dirs: (E,1,M,2)."""
    cs = _cos(actions[:, 0].unsqueeze(1), dirs[:, 0].unsqueeze(1)).squeeze(1)  # (E,M)
    pred = cs.argmax(dim=-1)
    acc = (pred == target).float()
    logp = F.log_softmax(5.0 * cs, dim=-1)
    ce = -torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
    return acc, ce
