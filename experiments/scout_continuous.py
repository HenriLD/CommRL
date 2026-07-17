"""Continuous-bearing scout-support: the meaning space is S^1.

The scout's private target is an arbitrary point on the site circle (bearing
theta ~ U[0, 2pi)) rather than one of K discrete sites; everything else --
waypoint, annulus start, key mechanics, premium structure -- is inherited
from scout_support, so the oracle-gap machinery transfers. Meanings are
embedded as z = (cos theta, sin theta), packed into the discrete pref slot.

Listeners (see DESIGN_continuous_meanings.md):
  particle progress   softmax over sampled bearing particles of per-step
                      distance progress -- the hand-crafted listener
                      generalized for free
  NCE                 contrastive InfoNCE critic over meaning particles;
                      reward bounded by log(K+1), the log|M| analog
  NCE + Sinkhorn      RSA recursion = alternating row/column normalization
                      of the (alternatives x particles) score matrix
The Gaussian-density listener is deliberately absent: its reward is unbounded
under decoder variance collapse (the density analog of saturation).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import scout_support as S


class ContinuousScoutEnv(S.ScoutSupportEnv):
    def reset(self):
        E = self.n_envs
        th = torch.rand((E,), generator=self.gen) * 2 * math.pi
        self.theta = th.to(self.device)
        self.tpoint = S.SITE_RADIUS * torch.stack(
            [torch.cos(self.theta), torch.sin(self.theta)], dim=-1)
        super().reset()   # parent obs() call now finds theta
        # nearest discrete site kept for interface compatibility only
        d = (self.tpoint.unsqueeze(1) - self.sites).norm(dim=-1)
        self.target = d.argmin(dim=1)
        return self.obs()

    @property
    def z(self):
        return torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=-1)

    def obs(self):
        o = super().obs()
        E = self.n_envs
        emb = torch.cat([self.z, torch.zeros((E, S.N_MEANINGS - 2),
                                             device=self.device)], dim=1)
        o[:, 0, S.PREF_SLICE] = emb                       # scout knows theta
        o[:, 1, S.PREF_SLICE] = 0.0
        o[:, :, S.PARTNER_PREF_SLICE] = 0.0
        if self.oracle:
            o[:, 1, S.PARTNER_PREF_SLICE] = emb           # oracle channel
        return o

    def step(self, actions):
        # replicate the parent step with the continuous target point
        _, _, done = None, None, None
        actions = actions.clamp(-1, 1)
        self.hist = torch.cat([
            self.hist[:, 1:],
            torch.cat([self.vel[:, 0], actions[:, 0]], dim=-1).unsqueeze(1),
        ], dim=1)
        self.vel = self.vel * (1 - S.DAMPING) + actions * S.ACCEL * S.DT
        speed = self.vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_speed = torch.tensor([S.SCOUT_SPEED, S.SUPPORT_SPEED],
                                 device=self.device).view(1, S.N_AGENTS, 1)
        self.vel = torch.where(speed > max_speed,
                               self.vel / speed * max_speed, self.vel)
        self.pos = (self.pos + self.vel * S.DT).clamp(-1.5, 1.5)
        self.t += 1
        d_wp = (self.pos[:, 0] - self.waypoint).norm(dim=-1)
        self.has_key = torch.maximum(self.has_key,
                                     (d_wp < S.KEY_RADIUS).float())
        d_scout = (self.pos[:, 0] - self.tpoint).norm(dim=-1)
        d_sup = (self.pos[:, 1] - self.tpoint).norm(dim=-1)
        both_in = ((d_scout < S.COVER_RADIUS) & (d_sup < S.COVER_RADIUS)
                   & (self.has_key > 0))
        scout_shape = torch.where(self.has_key > 0, d_scout, d_wp + 1.0)
        r_ext = S.BONUS * both_in.float() - S.SHAPE_W * scout_shape - d_sup
        done = self.t >= S.EPISODE_LEN
        info = {"r_ext": r_ext, "pref_bonus": S.BONUS * both_in.float(),
                "dist_pen": scout_shape + d_sup,
                "coll_pen": torch.zeros_like(r_ext),
                "commit_acc": self.commitment_accuracy()}
        return self.obs(), info, done

    def commitment_accuracy(self):
        """Supporter within 60 degrees of the true bearing (chance = 1/3,
        matching the discrete metric's chance level)."""
        p = self.pos[:, 1]
        ang = torch.atan2(p[:, 1], p[:, 0])
        diff = torch.remainder(ang - self.theta + math.pi, 2 * math.pi) - math.pi
        return (diff.abs() < math.pi / 3).float()


def particle_progress_reward(env, pre_pos, thetas):
    """Particle progress listener: softmax over K+1 bearing particles (true
    bearing first) of per-step distance progress toward each particle point.
    thetas: (E, K+1) with thetas[:, 0] = env.theta."""
    pts = S.SITE_RADIUS * torch.stack(
        [torch.cos(thetas), torch.sin(thetas)], dim=-1)      # (E, K+1, 2)
    d_now = (pts - env.pos[:, 0].unsqueeze(1)).norm(dim=-1)
    d_pre = (pts - pre_pos.unsqueeze(1)).norm(dim=-1)
    prog = (d_pre - d_now) / (S.SCOUT_SPEED * S.DT)
    L = F.softmax(5.0 * prog, dim=1)
    return torch.log(L[:, 0].clamp(min=1e-8)) + math.log(thetas.shape[1])


class NCEListener(nn.Module):
    """Contrastive listener over meaning particles. Embeds the (masked
    context, action) pair and the meaning vector; score = scaled dot product.
    Literal reward = InfoNCE at the true meaning among K particles (bounded
    by log(K+1)); pragmatic reward = Sinkhorn iterations on the
    (alternatives x particles) score matrix."""

    def __init__(self, hidden=128, emb=32):
        super().__init__()
        self.enc_sa = nn.Sequential(
            nn.Linear(S.OBS_DIM + S.ACT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, emb))
        self.enc_z = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, emb))
        self.scale = emb ** -0.5

    def sa_embed(self, obs, actions):
        x = obs.clone()
        x[..., S.PREF_SLICE] = 0.0
        x[..., S.PARTNER_PREF_SLICE] = 0.0
        return self.enc_sa(torch.cat([x, actions], dim=-1))

    def scores(self, obs, actions, zs):
        """obs (B, OBS), actions (B, ACT) or (B, A, ACT); zs (B, K+1, 2).
        Returns (B, [A,] K+1)."""
        f = self.enc_z(zs)                                   # (B, K+1, E)
        if actions.dim() == obs.dim():                       # single action
            e = self.sa_embed(obs, actions)                  # (B, E)
            return torch.einsum("be,bke->bk", e, f) * self.scale
        A = actions.shape[-2]
        obs_e = obs.unsqueeze(-2).expand(*obs.shape[:-1], A, obs.shape[-1])
        e = self.sa_embed(obs_e, actions)                    # (B, A, E)
        return torch.einsum("bae,bke->bak", e, f) * self.scale

    def nce_loss(self, obs, actions, zs):
        """InfoNCE with the true meaning in column 0."""
        sc = self.scores(obs, actions, zs)
        return F.cross_entropy(sc, torch.zeros(sc.shape[0], dtype=torch.long,
                                               device=sc.device))

    def comm_reward(self, obs, actions, zs, pragmatic=False,
                    alt_actions=None, n_iter=1):
        K1 = zs.shape[1]
        if not pragmatic:
            sc = self.scores(obs, actions, zs)
            logp = F.log_softmax(sc, dim=-1)
            return logp[:, 0] + math.log(K1)
        acts = torch.cat([actions.unsqueeze(-2), alt_actions], dim=-2)
        L = F.softmax(self.scores(obs, acts, zs), dim=-1)    # (B, A, K+1)
        for _ in range(n_iter):
            Sp = L / L.sum(dim=-2, keepdim=True).clamp(min=1e-12)
            L = Sp / Sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.log(L[:, 0, 0].clamp(min=1e-8)) + math.log(K1)
