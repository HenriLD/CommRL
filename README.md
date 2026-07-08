# CommRL — Pragmatic Rewards for Legible Multi-Agent RL

Code and papers for a framework that derives an intrinsic "legibility" reward
for cooperative RL agents from the Rational Speech Act model of pragmatic
inference. An agent is rewarded for actions that let an observer infer its
private intention, so communication happens through movement rather than a
dedicated channel.

## Layout

- `experiments/` — environment, training, and evaluation code for the
  conference paper (see its README for details and reproduction commands).
- `papers/Conference_Paper/` — current paper source (`main.tex`, figures).
- `papers/Master_Thesis/` — thesis this work grew out of.
- `papers/ICASSP_Paper/` — earlier 4-page draft, superseded by the conference paper.

The original 2025 prototype (PettingZoo simple_tag + SAC) was removed from the
working tree; it remains in git history.
