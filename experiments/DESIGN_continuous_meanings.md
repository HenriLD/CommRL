# Continuous meaning spaces: design note

Goal (user directive 2026-07-18): drop the discrete hand-specified meaning
assumption. Meanings become "some vector the policy is materially conditioned
on"; the objective inverts to *make the latent that drives your decisions
legible to other agents*.

## Tractability: particles + Sinkhorn

What breaks at large/continuous |M| is not the literal decoder but the RSA
recursion's listener normalization over meanings. Fix: contrast PARTICLES.
Sample K alternative meanings from the batch (other episodes' latents) and K'
alternative actions; both RSA normalizations become finite softmaxes. The
recursion is alternating row/column normalization of a (K'+1)x(K+1) score
matrix -- a Sinkhorn iteration -- and at N_r=0 it IS InfoNCE. The reward is
bounded by log(K+1), the exact analog of log|M|; Props 1-2 carry over with
the InfoNCE MI bound; the discrete machinery is the particles-enumerate-M
special case. Cite: Cohn-Gordon et al. (distractor-based RSA in pragmatic
captioning) for sampled-meaning RSA in NLP; van den Oord et al. for InfoNCE.

## Rejected design: density listener

A Gaussian decoder q(z|s,a) with reward log q(z|s,a) - log p(z) admits
decoder variance collapse: the reward inflates unboundedly with zero new
information (the density analog of saturation, unbounded). The contrastive
ratio form is the principled choice, not a convenience. This deserves its own
theory paragraph (Prop-2-style: bounded rewards or bust).

## The inversion needs a bottleneck

z is meaningful only if it is the ONLY path from private observation to the
policy: private obs -> encoder -> z -> pi(a|s,z). Then task performance
forces z to carry decision-relevant content and decodability-of-z equals
decodability-of-what-drives-behavior; the agent learns WHAT to communicate
(self-assigned meanings). Without the bottleneck the policy routes around z.
Stage 1 (this repo): z = fixed embedding (cos t, sin t) of a continuous
target bearing -- tests the machinery. Stage 2: learned encoder bottleneck.

## Testbed: continuous-bearing scout-support

Target = point on the site circle at angle t ~ U[0, 2pi); meaning space S^1;
waypoint/annulus/key mechanics unchanged so the oracle-gap machinery
transfers. Obs packing keeps OBS_DIM: PREF one-hot slot (3) carries
(cos t, sin t, 0); oracle fills the supporter's PARTNER_PREF slot likewise.
Success metric: both agents within COVER_RADIUS of the target point;
commitment = supporter within angular tolerance of t.

Conditions: baseline / oracle / blind / progress_part (particle progress
listener: softmax over sampled bearing-particles of per-step distance
progress -- the hand-crafted listener generalizes for free) / nce (literal
contrastive) / nce_prag (+1 Sinkhorn iteration) / ipl_cont (reference policy
conditioned on particle bearings -- the IPL generalizes for free).

Files: scout_continuous.py (env + listeners), train_scout_cont.py (trainer,
cloned from train_scout.py). Gate first: baseline/oracle x 4 seeds must
certify a premium at 400 cycles before the listener suite runs.
