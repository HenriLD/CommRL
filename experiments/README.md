# Pragmatic Reward Experiments

Self-contained PyTorch implementations of the two environments in
`papers/Conference_Paper/` (`main.tex` = full-length version, `aamas.tex` =
anonymized ACM submission): *scout-support* (Environment A, structural oracle
gap — communication has task value) and *preference Simple Spread*
(Environment B, near-zero oracle gap — control). Everything is vectorized over
environments in torch; no external MARL framework.

## Environment A (primary)

- `scout_support.py` — env (scout knows the target site, must visit a pickup
  waypoint first; slower supporter must co-locate at the target; `configure()`
  for the five-meaning / minefield / partner-speed / signal-cost variants) and
  the listener family: simple / exclusivity / progress / Bayesian filter /
  learned (`ScoutListener` input modes: `full`, `act` = audience viewpoint,
  `state` = no action, `partner` = supporter's own observation).
- `train_scout.py` — MASAC with conditions
  `baseline | oracle | simple | exclusivity | progress | filter |
  learned(-_prag) | learned_act(_prag/_ear) | learned_pre(_prag) |
  ear | learned_ear | filter_ear | inforeg | partner_belief`.
  `learned_pre*` bound the listener to the audience viewpoint and the
  pre-commitment window; `inforeg` / `partner_belief` are the external
  baselines (Strouse-style variational I(A;M|S), Tian-style partner belief).
  Flags: `--lam --voi --listener_lr --alt_policy --device --n_sites
  --minefield --blind --sup_speed --shape_w`.
- `expman.py` — token-cheap experiment manager: `launch --spec <suite>.json
  --outroot <dir> --workers 6`, `status`, `report` (gates on completed runs).
  The `suite_*.json` specs determine every run in the paper exactly.
- Analysis / figures: `paper_figs.py` (forest, budget dynamics, commit delta),
  `boundary_map.py` (premium-vs-gain fits with bootstrap CIs),
  `info_profile.py` (per-timestep post-hoc decodability), `commit_curve.py`,
  `render_scout.py` (trajectory panels with commitment markers),
  `cross_play.py`, `posthoc_probe.py`, `capacity_probe.py`.

## Environment B (control)

- `pragmatic_spread.py` — env + thesis-era listener family.
- `train_masac.py` — MASAC with conditions
  `baseline | oracle | heuristic | simple | learned | learned_prag`.
- `launch.py` — sweep driver (`--script` selects the trainer).
- `plots.py`, `ablation_plots.py` (λ sweep from `results_ablation/`),
  `render_trajs.py`.

## Reproduce (GPU: system Python 3.13 with torch ROCm; CPU also works)

```
# Environment A headline suite (400 cycles ≈ converged; ~20 min/run on GPU)
python expman.py launch --spec suite_stage1.json --outroot results_scout3 --workers 6
python expman.py report --outroot results_scout3

# Environment B control at the converged budget
python launch.py --outroot results_envB400 --seeds 0 1 2 3 4 --cycles 400 \
    --lam 0.1 --device cuda --threads 2 --workers 6
```

Hard-won conventions: never conclude below ~400 cycles (the paper documents
three spurious pre-convergence reversals); final performance is the mean of the
last three evaluations on the fixed 64-episode set; headline claims need
Welch t ≥ 4 (Bonferroni across the paper's ~30 tests), everything else is
labeled suggestive. Results roots kept for paper provenance: `results_scout3*`
(headline + grid/wide/novoi checks), `results_suite` (variants + blind),
`results_boundary`, `results_boundedL` / `results_prekey` / `results_levers`
(bounded-listener program incl. the pre-registered λ=0.6 confirmation),
`results_baselines` (external baselines), `results_recipe` (λ×VOI surface),
`results` / `results_envB400` / `results_ablation` (Environment B),
`results_scout2` (150-cycle archive behind the pre-convergence-hazard section),
`results_scout_lam` (hyperparameter provenance).
