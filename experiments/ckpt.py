"""Mid-run checkpointing so interrupted runs resume instead of restarting.

Granularity: every CKPT_EVERY cycles a small checkpoint (networks, optimizer
states, RNG, progress, history -- NOT the replay buffer) is written to
<outdir>/ckpt.pt. On start, a trainer with a ckpt and no model.pt resumes
from it: the replay buffer is refilled by collecting REFILL_CYCLES episodes
with the restored policy before updates continue, and the history entry is
flagged resumed=True for disclosure (a resumed run's trajectory is not
bit-identical to an uninterrupted one). The checkpoint is deleted when the
final model.pt is written.
"""

import os

import torch

CKPT_EVERY = 50
REFILL_CYCLES = 3


def save(outdir, cycle, total_steps, history, nets, opts, extras=None):
    torch.save({
        "cycle": cycle,
        "total_steps": total_steps,
        "history": history,
        "nets": {k: v.state_dict() for k, v in nets.items()},
        "opts": {k: v.state_dict() for k, v in opts.items()},
        "extras": extras or {},
        "torch_rng": torch.get_rng_state(),
    }, os.path.join(outdir, "ckpt.pt"))


def load(outdir, nets, opts):
    """Returns (start_cycle, total_steps, history, extras) or None."""
    path = os.path.join(outdir, "ckpt.pt")
    if not os.path.exists(path) or os.path.exists(os.path.join(outdir, "model.pt")):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in ck["nets"].items():
        if k in nets and nets[k] is not None:
            nets[k].load_state_dict(v)
    for k, v in ck["opts"].items():
        if k in opts and opts[k] is not None:
            opts[k].load_state_dict(v)
    torch.set_rng_state(ck["torch_rng"])
    return ck["cycle"], ck["total_steps"], ck["history"], ck.get("extras", {})


def clear(outdir):
    path = os.path.join(outdir, "ckpt.pt")
    if os.path.exists(path):
        os.remove(path)
