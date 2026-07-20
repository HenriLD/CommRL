"""Build the long-budget convergence suite.

Every referee in the last round made the same objection: at the 400-cycle
terminal budget the oracle has plateaued (slope +0.006/100 cycles, t=0.4) but
the baseline is still improving (+0.054/100, t=3.9), and the premium contracts
from 0.149 to 0.100 over cycles 320-400. Linear extrapolation of that slope
exhausts the premium near cycle 563, which would turn every closure fraction in
the paper into a stopping-time artifact and reduce the contribution to a
sample-efficiency claim.

This suite runs the four conditions the argument actually needs, three times
past the current budget, so the premium's asymptote can be read directly
instead of extrapolated:

    baseline    the term that is still moving
    oracle      the term that has plateaued
    progress    best hand-crafted listener
    ipl_prag    the constructive recipe (lambda=0.6)

Writes suite_convergence.json. It does NOT launch anything. To run:

    python expman.py launch --spec suite_convergence.json \
        --outroot results_conv --workers 6 --script train_scout.py

Throttle at any time by editing workers.txt; expman re-reads it every poll.
"""
import json

CYCLES = 1200          # 3x the current 400-cycle budget
SEEDS = range(8)       # matches the n=8 conditions in Table 1

# (condition, lambda) -- baseline and oracle take no legibility reward
CONDS = [
    ("baseline", 0.0),
    ("oracle", 0.0),
    ("progress", 0.3),
    ("ipl_prag", 0.6),
]

spec = []
for cond, lam in CONDS:
    for s in SEEDS:
        spec.append({
            "name": "%s_s%d" % (cond, s),
            "args": [
                "--condition", cond,
                "--seed", str(s),
                "--cycles", str(CYCLES),
                "--lam", str(lam),
                "--voi", "0.2",
                "--device", "cuda",
                "--threads", "2",
            ],
        })

with open("suite_convergence.json", "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=1)

steps = CYCLES * 3200
print("wrote suite_convergence.json: %d runs (%d conditions x %d seeds)"
      % (len(spec), len(CONDS), len(SEEDS)))
print("each run %d cycles = %.2fM env steps (current budget: 1.28M)"
      % (CYCLES, steps / 1e6))
print("\nnot launched. to start:")
print("  python expman.py launch --spec suite_convergence.json "
      "--outroot results_conv --workers 6 --script train_scout.py")
