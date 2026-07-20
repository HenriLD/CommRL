"""Second convergence suite: the listeners needed to test the ORDERING.

The first suite settled whether the premium survives a converged budget (it
does, at roughly half its 400-cycle size). But the paper's actual claim is
about which listener pays, and that cannot be checked with progress alone.
These two conditions complete the contrast at 1200 cycles:

    learned   the co-adapting ML listener the paper reports as inert
    filter    the second motion-grounded listener (67% at 400 cycles)

Baseline and oracle anchors come from results_conv, so they are not re-run.

Writes suite_convergence2.json. Does NOT launch.
"""
import json

CYCLES = 1200
SEEDS = range(8)
CONDS = [("learned", 0.3), ("filter", 0.3)]

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

with open("suite_convergence2.json", "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=1)

print("wrote suite_convergence2.json: %d runs (%d conditions x %d seeds)"
      % (len(spec), len(CONDS), len(SEEDS)))
print("launch with --outroot results_conv2")
