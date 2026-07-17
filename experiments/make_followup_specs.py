"""Generate the follow-up suite specs: lever controls, recipe sensitivity,
and n-extension of the two n=3 headline-table rows."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = ["--cycles", "400", "--voi", "0.2", "--device", "cuda", "--threads", "2"]

# 1) controls into results_levers: literal @ lam06 (is it the recursion or
#    just the weight?) + lam10 dose point for the pragmatic reward
jobs = []
for s in range(8):
    jobs.append({"name": f"lam06_learned_pre_s{s}",
                 "args": ["--condition", "learned_pre", "--seed", str(s),
                          "--lam", "0.6"] + BASE})
    jobs.append({"name": f"lam10_learned_pre_prag_s{s}",
                 "args": ["--condition", "learned_pre_prag", "--seed", str(s),
                          "--lam", "1.0"] + BASE})
json.dump(jobs, open(os.path.join(HERE, "suite_levers2.json"), "w"), indent=1)

# 2) progress-recipe lambda x VOI sensitivity at the converged budget
jobs = []
for lam in ("0.1", "0.3", "0.6"):
    for voi in ("0.2", "1.0"):
        if lam == "0.3" and voi == "0.2":
            continue  # exists at n=10 in results_scout3
        for s in range(3):
            tag = f"p_l{lam.replace('.', '')}_v{voi.replace('.', '')}"
            jobs.append({"name": f"{tag}_s{s}",
                         "args": ["--condition", "progress", "--seed", str(s),
                                  "--cycles", "400", "--lam", lam, "--voi", voi,
                                  "--device", "cuda", "--threads", "2"]})
json.dump(jobs, open(os.path.join(HERE, "suite_recipe.json"), "w"), indent=1)

# 3) extend the two n=3 rows of Table 2 to n=8 (same recipe as originals)
jobs = []
for cond in ("exclusivity", "learned_prag"):
    for s in range(3, 8):
        jobs.append({"name": f"{cond}_s{s}",
                     "args": ["--condition", cond, "--seed", str(s),
                              "--lam", "0.3"] + BASE})
json.dump(jobs, open(os.path.join(HERE, "suite_extend.json"), "w"), indent=1)
print("specs written: levers2=16, recipe=15, extend=10")
