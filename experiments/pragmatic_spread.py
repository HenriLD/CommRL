"""Vectorized 'preference Simple Spread' environment and listener models.

Self-contained PyTorch re-implementation of the modified Simple Spread task
from the thesis (agents with private color preferences, colored landmarks,
preference bonus / distance penalty / collision penalty), plus the family of
listener models used to compute the pragmatic reward R_comm:

  - heuristic exclusivity L0 (thesis Algorithm 1)
  - simple cosine L0 (thesis ablation)
  - learned listener L_theta (amortized / variational listener)

All computation is batched over (n_envs, n_agents).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_AGENTS = 3
N_LANDMARKS = 3
N_MEANINGS = 3          # color categories
DT = 0.1
DAMPING = 0.25
ACCEL = 5.0
MAX_SPEED = 1.0
AGENT_SIZE = 0.1
COVER_RADIUS = 0.4
PREF_BONUS = 2.0
COLL_PENALTY = 1.0
EPISODE_LEN = 50

# observation layout per agent:
#   own pos (2) | own vel (2) | own pref one-hot (3)
#   landmark rel pos (3*2) | landmark colors one-hot (3*3)
#   other agents rel pos (2*2) | other agents vel (2*2)
#   teammate pref one-hots (2*3)  [zeroed unless oracle condition]
OBS_DIM = 2 + 2 + 3 + 6 + 9 + 4 + 4 + 6  # = 36
ACT_DIM = 2
PREF_SLICE = slice(4, 7)  # own preference one-hot within the observation


class PragmaticSpreadEnv:
    def __init__(self, n_envs, device="cpu", oracle=False, seed=0):
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.oracle = oracle
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.t = 0
        self.reset()

    def reset(self):
        E = self.n_envs
        self.t = 0
        self.pos = (torch.rand((E, N_AGENTS, 2), generator=self.gen) * 2 - 1).to(self.device)
        self.vel = torch.zeros((E, N_AGENTS, 2), device=self.device)
        self.lm_pos = (torch.rand((E, N_LANDMARKS, 2), generator=self.gen) * 2 - 1).to(self.device)
        # landmark colors: a random permutation of the 3 categories per env
        self.lm_color = torch.argsort(torch.rand((E, N_LANDMARKS), generator=self.gen), dim=1).to(self.device)
        # private preferences: uniform with replacement (conflicts possible)
        self.pref = torch.randint(0, N_MEANINGS, (E, N_AGENTS), generator=self.gen).to(self.device)
        return self.obs()

    def obs(self):
        E = self.n_envs
        dev = self.device
        pref_oh = F.one_hot(self.pref, N_MEANINGS).float()               # (E,N,3)
        lm_color_oh = F.one_hot(self.lm_color, N_MEANINGS).float()       # (E,L,3)
        obs = []
        for i in range(N_AGENTS):
            own_pos = self.pos[:, i]
            own_vel = self.vel[:, i]
            own_pref = pref_oh[:, i]
            lm_rel = (self.lm_pos - own_pos.unsqueeze(1)).reshape(E, -1)
            lm_col = lm_color_oh.reshape(E, -1)
            others = [j for j in range(N_AGENTS) if j != i]
            oth_rel = (self.pos[:, others] - own_pos.unsqueeze(1)).reshape(E, -1)
            oth_vel = self.vel[:, others].reshape(E, -1)
            if self.oracle:
                oth_pref = pref_oh[:, others].reshape(E, -1)
            else:
                oth_pref = torch.zeros((E, 2 * N_MEANINGS), device=dev)
            obs.append(torch.cat([own_pos, own_vel, own_pref, lm_rel, lm_col, oth_rel, oth_vel, oth_pref], dim=1))
        return torch.stack(obs, dim=1)  # (E, N, OBS_DIM)

    def step(self, actions):
        """actions: (E, N, 2) in [-1, 1]. Returns obs, reward dict, done flag."""
        actions = actions.clamp(-1, 1)
        self.vel = self.vel * (1 - DAMPING) + actions * ACCEL * DT
        speed = self.vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.vel = torch.where(speed > MAX_SPEED, self.vel / speed * MAX_SPEED, self.vel)
        self.pos = (self.pos + self.vel * DT).clamp(-1.5, 1.5)
        self.t += 1

        # distances (E, L, N): landmark x agent
        d = (self.lm_pos.unsqueeze(2) - self.pos.unsqueeze(1)).norm(dim=-1)
        min_d, closest = d.min(dim=2)                                    # (E,L)
        dist_pen = min_d.sum(dim=1)                                      # (E,)
        closest_pref = torch.gather(self.pref, 1, closest)               # (E,L)
        matched = (closest_pref == self.lm_color) & (min_d < COVER_RADIUS)
        pref_bonus = PREF_BONUS * matched.float().sum(dim=1)             # (E,)

        pd = (self.pos.unsqueeze(2) - self.pos.unsqueeze(1)).norm(dim=-1)  # (E,N,N)
        iu = torch.triu_indices(N_AGENTS, N_AGENTS, offset=1)
        coll = (pd[:, iu[0], iu[1]] < 2 * AGENT_SIZE).float().sum(dim=1)
        coll_pen = COLL_PENALTY * coll                                   # (E,)

        r_ext = pref_bonus - dist_pen - coll_pen
        done = self.t >= EPISODE_LEN
        info = {
            "r_ext": r_ext, "pref_bonus": pref_bonus,
            "dist_pen": dist_pen, "coll_pen": coll_pen,
        }
        return self.obs(), info, done

    # ---- helpers used by listeners and metrics ----

    def goal_dirs(self):
        """Unit vectors from each agent to the landmark of each color.
        Returns (E, N, M, 2)."""
        # landmark of color m per env: since colors are a permutation,
        # invert the permutation: slot[m] = index of landmark with color m
        slot = torch.argsort(self.lm_color, dim=1)                       # (E,L)
        lm_by_color = torch.gather(self.lm_pos, 1, slot.unsqueeze(-1).expand(-1, -1, 2))  # (E,M,2)
        vec = lm_by_color.unsqueeze(1) - self.pos.unsqueeze(2)           # (E,N,M,2)
        return vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def specialization(self):
        """Fraction of unique 'claimed' landmarks (closest landmark per agent)."""
        d = (self.pos.unsqueeze(2) - self.lm_pos.unsqueeze(1)).norm(dim=-1)  # (E,N,L)
        claim = d.argmin(dim=2)                                          # (E,N)
        uniq = F.one_hot(claim, N_LANDMARKS).amax(dim=1).sum(dim=1).float()
        return uniq / N_AGENTS                                           # (E,)


# --------------------------- listener models ---------------------------

def _cos(a, v):
    """a: (..., 2) actions, v: (..., M, 2) directions -> (..., M)."""
    an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (an.unsqueeze(-2) * v).sum(dim=-1)


def heuristic_l0_scores(actions, dirs, temp, exclusivity=True):
    """actions: (E,N,A,2) candidate action sets, dirs: (E,N,M,2).
    Returns scores (E,N,A,M)."""
    cs = _cos(actions, dirs.unsqueeze(2))          # (E,N,A,M)
    s = temp * cs
    if exclusivity:
        M = s.shape[-1]
        mask = ~torch.eye(M, dtype=torch.bool, device=s.device)
        others = s.unsqueeze(-2).expand(*s.shape[:-1], M, M)
        max_other = others.masked_fill(~mask, -1e9).amax(dim=-1)
        s = s - max_other
    return s


def rsa_posterior(l0_logits, n_iter=1):
    """l0_logits: (E,N,A,M) unnormalized. Returns L*(m|a): (E,N,A,M) probs."""
    L = F.softmax(l0_logits, dim=-1)
    for _ in range(n_iter):
        S = L / L.sum(dim=-2, keepdim=True).clamp(min=1e-12)   # speaker: normalize over actions
        L = S / S.sum(dim=-1, keepdim=True).clamp(min=1e-12)   # listener: normalize over meanings
    return L


def heuristic_comm_reward(dirs, pref, actions_true, temp=5.0, n_samples=64, n_iter=1,
                          exclusivity=True, gen=None):
    """Thesis PRM (Algorithm 1), vectorized. `dirs` and `pref` must come from
    the state s_t in which `actions_true` were taken. Returns R_comm (E,N)."""
    E = dirs.shape[0]
    alt = torch.rand((E, N_AGENTS, n_samples, 2), generator=gen).to(actions_true.device) * 2 - 1
    acts = torch.cat([actions_true.unsqueeze(2), alt], dim=2)      # (E,N,A,2)
    logits = heuristic_l0_scores(acts, dirs, temp, exclusivity)
    post = rsa_posterior(logits, n_iter)                           # (E,N,A,M)
    p_true = torch.gather(post[:, :, 0], 2, pref.unsqueeze(-1)).squeeze(-1)
    return torch.log(p_true.clamp(min=1e-8)) + torch.log(torch.tensor(float(N_MEANINGS)))


class LearnedListener(nn.Module):
    """L_theta(m | s, a): predicts an agent's private preference from its
    publicly observable context and action. The private preference block of
    the observation is zeroed before it is fed to the network."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MEANINGS),
        )

    @staticmethod
    def features(obs, actions):
        x = obs.clone()
        x[..., PREF_SLICE] = 0.0          # hide the private state
        x[..., -6:] = 0.0                 # hide oracle teammate preferences too
        return torch.cat([x, actions], dim=-1)

    def forward(self, obs, actions):
        return self.net(self.features(obs, actions))  # logits (…, M)

    def comm_reward(self, obs, actions, pref, pragmatic=False, alt_actions=None, n_iter=1):
        """R_comm = log L(m_true | s, a) + log M  (info gain over uniform prior).
        If pragmatic=True, applies RSA recursion over an alternative action set."""
        if not pragmatic:
            logp = F.log_softmax(self(obs, actions), dim=-1)
            lp = torch.gather(logp, -1, pref.unsqueeze(-1)).squeeze(-1)
            return lp + torch.log(torch.tensor(float(N_MEANINGS)))
        acts = torch.cat([actions.unsqueeze(-2), alt_actions], dim=-2)      # (E,N,A,2)
        A = acts.shape[-2]
        obs_e = obs.unsqueeze(-2).expand(*obs.shape[:-1], A, obs.shape[-1])
        logits = self(obs_e, acts)                                          # (E,N,A,M)
        post = rsa_posterior(logits, n_iter)
        p_true = torch.gather(post[..., 0, :], -1, pref.unsqueeze(-1)).squeeze(-1)
        return torch.log(p_true.clamp(min=1e-8)) + torch.log(torch.tensor(float(N_MEANINGS)))


def probe_intent_metrics(dirs, pref, actions):
    """Fixed geometric probe used identically across all conditions.
    `dirs`/`pref` must come from the state s_t in which `actions` were taken.
    Returns (accuracy (E,N) bool as float, cross-entropy (E,N))."""
    cs = _cos(actions, dirs)                       # (E,N,M)
    pred = cs.argmax(dim=-1)
    acc = (pred == pref).float()
    logp = F.log_softmax(5.0 * cs, dim=-1)
    ce = -torch.gather(logp, -1, pref.unsqueeze(-1)).squeeze(-1)
    return acc, ce
