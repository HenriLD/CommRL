# Pragmatic Reward Experiments (Conference Paper)

Self-contained PyTorch implementation of the *preference Simple Spread* task and
the pragmatic-reward conditions compared in `papers/Conference_Paper/main.tex`.
No VMAS/BenchMARL dependency — everything is vectorized over environments in torch.

## Files

- `pragmatic_spread.py` — environment (3 agents, 3 colored landmarks, private
  color preferences; preference bonus / distance penalty / collision penalty)
  and the listener family: heuristic exclusivity L0 (thesis Algorithm 1),
  simple cosine L0, learned listener L_theta, RSA recursion, fixed geometric probe.
- `train_masac.py` — parameter-shared MASAC (centralized twin critic, auto
  entropy) with pluggable reward conditions:
  `baseline | oracle | heuristic | simple | learned | learned_prag`.
- `launch.py` — full sweep (conditions x seeds), a few worker processes at a time.
- `plots.py` — aggregates `results/*/history.json` into paper figures + final table.
- `ablation_plots.py` — lambda-sweep figure from `results_ablation/`.
- `posthoc_probe.py` — trains a fresh listener on each run's final-policy rollouts
  to measure converged-behavior decodability (fair across conditions).
- `render_trajs.py` — qualitative trajectory figures from saved checkpoints.

## Reproduce

```
conda activate commrl-working
python launch.py --outroot results --seeds 0 1 2 --cycles 150 --workers 4
python plots.py --resroot results --figdir ../papers/Conference_Paper/img
```

One run is 480k env steps (~1h on CPU). The learned-listener conditions
recompute R_comm at replay time so the reward tracks the co-adapting listener;
heuristic conditions compute it at collection time (the listener is static).
