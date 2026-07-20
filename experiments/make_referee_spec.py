"""Referee-item suite: the permuted-meaning placebo and the IPL confirmation
cohort, both at the standard 400-cycle evaluation budget.

placebo    progress_placebo, seeds 0-7, lam=0.3 (progress's weight): identical
           reward form, density, and VOI schedule with a per-episode permuted
           target. Three referees independently requested exactly this. If
           densification explains the gains, it should gain; if content does,
           it should not (the blind control predicts a cost).

ipl conf   ipl_prag at the selected lam=0.6, seeds 20-27 -- disjoint from the
           selection cohort (0-7) AND from the frozen reference's seed (9) --
           matching the selection/confirmation protocol the bounded listener
           got. The paper currently flags this cohort as missing.

Writes suite_referee.json; does not launch.
"""
import json

spec = []
for s in range(8):
    spec.append({"name": "progress_placebo_s%d" % s,
                 "args": ["--condition", "progress_placebo", "--seed", str(s),
                          "--cycles", "400", "--lam", "0.3", "--voi", "0.2",
                          "--device", "cuda", "--threads", "2"]})
for s in range(20, 28):
    spec.append({"name": "iplconf_s%d" % s,
                 "args": ["--condition", "ipl_prag", "--seed", str(s),
                          "--cycles", "400", "--lam", "0.6", "--voi", "0.2",
                          "--device", "cuda", "--threads", "2"]})

with open("suite_referee.json", "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=1)
print("wrote suite_referee.json: %d runs" % len(spec))
