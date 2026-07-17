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

## The inversion needs a bottleneck (architecture settled with user, 2026-07-18)

Split the policy input: PRIVATE perspective -> encoder -> z (the bottleneck),
then concat [z, PUBLIC perspective] -> action head. z is the only path from
private observation to the policy, so task performance forces z to carry the
decision-relevant private content; decodability-of-z = decodability of what
drives behavior.

Resolution of the context-dependence concern: RSA places context-dependence
in the SPEAKER's utterance choice, not the meaning -- the action head (and
the listener) both condition on public context freely, so context-dependent
compression happens at the channel. Cost: the bottleneck width must fit the
union over contexts of decision-relevant private content (no per-context
pruning); acceptable. Keeping z private-only is REQUIRED by our own
saturation analysis: public features in the decoding target reintroduce the
context shortcut on the meaning side. "Public meanings" are excluded by the
oracle-gap logic itself: anything commonly observed has zero communication
premium by definition. Hence z = (private INTERSECT decision-relevant) --
"what is worth communicating," derived rather than specified.

Stage-2 additions:
- VIB bottleneck (KL to N(0,I) <= C) gives a tunable information budget; the
  TRANSPARENCY COEFFICIENT I(Z; behavior) / I(Z; private) -- both estimable
  with the contrastive machinery -- measures how much of what the agent
  knows-and-uses is readable. Headline metric.
- The decisive test needs DECOYS: private obs = true bearing + decoy bearings
  + noise channels, only part decision-relevant. Demonstration = the
  bottleneck encodes the true bearing only and the legibility reward makes
  exactly that readable (agent-selected content).

Stage 1 (implemented): z = fixed embedding (cos t, sin t) of a continuous
target bearing -- tests the particle/Sinkhorn machinery first.

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
