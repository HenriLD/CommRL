"""Stage 2 of the continuous-meaning program: the bottleneck inversion.

The meaning is no longer a hand-specified embedding of a task variable. The
scout's PRIVATE observation block passes through a variational encoder to a
latent z that is the ONLY path from private observation to the action head
(concat [z, public] after the bottleneck, per DESIGN_continuous_meanings.md).
Task pressure decides WHAT enters z; the legibility reward makes behavior
expose it; VIB (KL to N(0,I)) prices the information budget.

The decisive test enriches the private block with decoy bearings and noise
channels, only the true bearing being decision-relevant: z should come to
carry the true bearing only, and R_comm should make exactly that readable
(agent-selected content, derived rather than specified).

This module is not imported by any stage-1 file; the running stage-1 suite
launches fresh processes that only read scout_continuous.py, so nothing here
can perturb it.
"""

import math

import torch
import torch.nn as nn

import scout_support as S
import scout_continuous as C

N_DECOY = 2
N_NOISE = 2
PRIV_DIM = 2 + 2 * N_DECOY + N_NOISE       # true (cos,sin) + decoys + noise
Z_DIM = 4                                   # < PRIV_DIM: selection is forced
OBS_DIM_BN = S.OBS_DIM + PRIV_DIM
PRIV_SLICE = slice(S.OBS_DIM, OBS_DIM_BN)
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class DecoyScoutEnv(C.ContinuousScoutEnv):
    """Continuous-bearing scout-support with a decoy-enriched private block.

    The scout's clean pref slot is zeroed: the true bearing reaches the policy
    only through the appended private block [cos/sin true, cos/sin decoy x2,
    noise x2], all constant within an episode. The supporter's block is zeros.
    The oracle channel (supporter told the true bearing, public side) is
    inherited unchanged -- the oracle is told the target, not the decoys.
    """

    def reset(self):
        E = self.n_envs
        d = torch.rand((E, N_DECOY), generator=self.gen) * 2 * math.pi
        self.decoys = d.to(self.device)
        self.noise = torch.randn((E, N_NOISE), generator=self.gen).to(self.device)
        return super().reset()

    def obs(self):
        o = super().obs()
        E = self.n_envs
        o[:, 0, S.PREF_SLICE] = 0.0
        ang = torch.cat([self.theta.unsqueeze(1), self.decoys], dim=1)
        trig = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).reshape(E, -1)
        block = torch.zeros((E, S.N_AGENTS, PRIV_DIM), device=self.device)
        block[:, 0] = torch.cat([trig, self.noise], dim=1)
        return torch.cat([o, block], dim=-1)


class VIBActor(nn.Module):
    """private block -> q(z|priv) -> concat [public, z] -> tanh-Gaussian head.

    z is resampled per forward pass (reparameterized); the reported log-prob
    is conditional on the drawn z, the standard latent-variable-policy bound.
    encode() exposes the per-episode meaning mu_z (the private block is
    constant within an episode, so mu_z is too).
    """

    def __init__(self, hidden=256, z_dim=Z_DIM, priv_dim=PRIV_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(priv_dim, 64), nn.ReLU(), nn.Linear(64, 2 * z_dim))
        self.net = nn.Sequential(
            nn.Linear(S.OBS_DIM + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, S.ACT_DIM)
        self.log_std = nn.Linear(hidden, S.ACT_DIM)

    def encode(self, obs):
        h = self.enc(obs[..., PRIV_SLICE])
        mu, log_std = h.chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, 2.0)

    def kl(self, obs):
        """KL(q(z|priv) || N(0,I)) per sample."""
        mu, log_std = self.encode(obs)
        return 0.5 * (mu.pow(2) + (2 * log_std).exp() - 1 - 2 * log_std).sum(-1)

    def sample(self, obs, deterministic=False):
        mu_z, ls_z = self.encode(obs)
        z = mu_z if deterministic else mu_z + ls_z.exp() * torch.randn_like(mu_z)
        h = self.net(torch.cat([obs[..., :S.OBS_DIM], z], dim=-1))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        pre = mu + std * torch.randn_like(mu)
        a = torch.tanh(pre)
        logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std
                - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class NCEListenerZ(C.NCEListener):
    """Contrastive listener whose meaning space is the speaker's own latent:
    decodes mu_z from (masked public context, action) against in-batch
    particles. Scores/loss/Sinkhorn recursion inherit from the stage-1
    listener; only the input dims and the masking differ (the private block
    is excluded by construction -- the listener is the audience)."""

    def __init__(self, hidden=128, emb=32, z_dim=Z_DIM):
        nn.Module.__init__(self)
        self.enc_sa = nn.Sequential(
            nn.Linear(S.OBS_DIM + S.ACT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, emb))
        self.enc_z = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb))
        self.scale = emb ** -0.5

    def sa_embed(self, obs, actions):
        x = obs[..., :S.OBS_DIM].clone()
        x[..., S.PREF_SLICE] = 0.0
        x[..., S.PARTNER_PREF_SLICE] = 0.0
        return self.enc_sa(torch.cat([x, actions], dim=-1))
