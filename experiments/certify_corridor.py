"""Scripted certification of the highway-fork premium (protocol v2).

Lesson of the on-ramp gates: certifying against a PASSIVE strawman is not
enough -- RL found uninformed policies (always-accommodate; solo threading)
that beat passive and erased the premium. Here the informed schedule is
scored against the best of an explicit uninformed-adversary set:

  passive     cooperators hold V_NOM (also: solo-infeasibility check)
  hedge       both chains open from t=0, hold until the fork is decided
  hedge_ad    adaptive hedge: both open, RECOVER once the ego commits a side
  guessL/R    run the full informed schedule for a fixed guessed side

plus a diagnostic (not an adversary -- it IS communication through action):
  reactive    open when the ego noses toward my lane; ego noses from t=0

Premium := E_m r(informed) - max over adversaries of E_m r(adversary),
with every script's ego playing its best scripted response given m.

Usage: python certify_corridor.py [--episodes 512]
"""

import argparse

import torch

import highway_corridor as H

V = H.V_NOM


def p_speed(v, v_cmd):
    a = 1.5 * (v_cmd - v)
    return torch.where(a >= 0, (a / H.A_MAX).clamp(max=1.0),
                       (a / -H.A_MIN).clamp(min=-1.0))


def gap_centers(env):
    """Physical centers of C_a's front gap per side: (n, 2) [L, R]."""
    gcL = (env.fx[:, 0] - H.CAR_LEN + env.x[:, 1]) / 2
    gcR = (env.fx[:, 2] - H.CAR_LEN + env.x[:, 3]) / 2
    return torch.stack([gcL, gcR], dim=1)


def ego_action(env, m, nose=False):
    """Scripted ego, meaning-aware. Exits: align with the intended side's
    C_a front gap, cross when it is open; THRU: hold V_NOM, stay centered."""
    n = env.n
    a = torch.zeros((n, 2))
    y = env.y[:, 0]
    side = torch.where(torch.tensor(m == 0).expand(n), 0.0, 2.0)
    if m == 2:
        a[:, 0] = p_speed(env.v[:, 0], torch.full((n,), V))
        a[:, 1] = (1.0 - y).clamp(-1, 1)          # recenter if drifted
        return a
    gc = gap_centers(env)[:, 0 if m == 0 else 1]
    gap = env.gaps()[:, 0 if m == 0 else 2]
    v_cmd = (V + 0.4 * (gc - env.x[:, 0])).clamp(6.0, 13.0)
    a[:, 0] = p_speed(env.v[:, 0], v_cmd)
    aligned = (env.x[:, 0] - gc).abs() < 3.5
    go = (gap > H.SAFE_GAP) & aligned
    committed = (y - 1.0).abs() > 0.25            # once moving, keep moving
    toward = -1.0 if m == 0 else 1.0
    lat = torch.zeros(n)
    lat = torch.where(go | (committed & ((y - side).abs() > 0.05)),
                      torch.full((n,), toward), lat)
    if nose and m != 2:
        # advertise the side early with a small in-lane bias
        want = 1.0 + 0.3 * toward
        lat = torch.where(~go & ~committed, (want - y).clamp(-1, 1) * 0.6, lat)
    a[:, 1] = lat
    return a


def coop_action(env, mode, m, t):
    """Scripted cooperators. Returns (n, 4) longitudinal actions for
    [C_a L, C_b L, C_a R, C_b R]. C_b always follows safely (the cascade)."""
    n = env.n
    v = env.v[:, 1:]
    tgt = torch.full((n, 4), V)
    gaps = env.gaps()
    ego_y = env.y[:, 0]
    decided = (ego_y - 1.0).abs() > 0.7           # ego has entered a side lane
    past = env.x[:, 0] > H.X_FORK

    def open_side(side_mask, col):
        tgt[:, col] = torch.where(side_mask, torch.full((n,), V - 3.0),
                                  tgt[:, col])

    on = ~(decided | past)                        # hold only while undecided
    if mode == "informed":
        if m == 0:
            open_side(on, 0)
        elif m == 1:
            open_side(on, 2)
    elif mode == "hedge":
        open_side(~past, 0), open_side(~past, 2)
    elif mode == "hedge_ad":
        open_side(on, 0), open_side(on, 2)
    elif mode == "guessL":
        open_side(on, 0)
    elif mode == "guessR":
        open_side(on, 2)
    elif mode == "reactive":
        noseL = (ego_y < 0.85) & ~past
        noseR = (ego_y > 1.15) & ~past
        open_side(noseL, 0), open_side(noseR, 2)
    # C_b safety: keep distance to the braking C_a ahead (cascade cost)
    a = 1.5 * (tgt - v)
    a[:, 1] = a[:, 1] - 8.0 * torch.relu(H.TIGHT_GAP * 0.7 - gaps[:, 1])
    a[:, 3] = a[:, 3] - 8.0 * torch.relu(H.TIGHT_GAP * 0.7 - gaps[:, 3])
    return torch.where(a >= 0, (a / H.A_MAX).clamp(max=1.0),
                       (a / -H.A_MIN).clamp(min=-1.0))


def rollout(mode, m, episodes, seed=0, nose=None):
    env = H.HighwayCorridorEnv(episodes, seed=seed)
    env.reset()
    env.target[:] = m
    tot_r = torch.zeros(episodes)
    min_gap_open = torch.full((episodes,), 1e9)
    if nose is None:
        nose = (mode == "reactive")
    for t in range(H.EPISODE_LEN):
        acts = torch.zeros((episodes, H.N_AGENTS, H.ACT_DIM))
        acts[:, 0] = ego_action(env, m, nose=nose)
        acts[:, 1:, 0] = coop_action(env, mode, m, t)
        _, info, _ = env.step(acts)
        tot_r += info["r_ext"]
        g = env.gaps()
        side_gap = g[:, 0] if m == 0 else g[:, 2]
        if m != 2:
            min_gap_open = torch.minimum(min_gap_open, side_gap.neg())
    succ = info["merge_rate"].mean().item()
    coll = info["fail_rate"].mean().item()
    return (tot_r.mean().item() / H.EPISODE_LEN, succ,
            -min_gap_open.max().item() if m != 2 else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=512)
    args = p.parse_args()
    modes = ["informed", "passive", "hedge", "hedge_ad", "guessL", "guessR",
             "reactive"]
    names = {0: "LEFT", 1: "RIGHT", 2: "THRU"}
    table = {}
    for mode in modes:
        rs, ss = [], []
        for m in range(3):
            r, s, maxg = rollout(mode, m, args.episodes, seed=7 + m)
            rs.append(r); ss.append(s)
            if mode == "passive" and m == 0:
                print(f"solo check: max LEFT gap ever under passive = "
                      f"{maxg:.1f} m (SAFE_GAP {H.SAFE_GAP})")
        table[mode] = (sum(rs) / 3, rs, ss)
        print(f"{mode:9s} r/step {sum(rs)/3:+.3f}  "
              f"per-m r [{rs[0]:+.3f} {rs[1]:+.3f} {rs[2]:+.3f}]  "
              f"success [{ss[0]:.2f} {ss[1]:.2f} {ss[2]:.2f}]")
    adversaries = ["passive", "hedge", "hedge_ad", "guessL", "guessR"]
    best = max(adversaries, key=lambda k: table[k][0])
    prem = table["informed"][0] - table[best][0]
    print(f"\nbest uninformed adversary: {best} ({table[best][0]:+.3f})")
    print(f"PREMIUM vs best adversary: {prem:+.3f} r/step "
          f"({prem * H.EPISODE_LEN:+.2f}/episode)")
    print(f"reactive (communication-equilibrium diagnostic): "
          f"{table['reactive'][0]:+.3f}")


if __name__ == "__main__":
    main()
