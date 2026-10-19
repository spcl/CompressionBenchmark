"""Slurm worker placement."""

import json
import math
import os
import socket


def layout(ranks, per_node):
    """Return node and task requests for one worker per Slurm task."""
    if type(ranks) is not int or ranks < 1 or type(per_node) is not int or per_node < 1:
        raise ValueError("Worker counts and ranks_per_node must be positive integers")
    return math.ceil(ranks / per_node), ranks, min(ranks, per_node)


def worker():
    """Return the global Slurm rank and worker count."""
    count = int(os.environ["COMPBENCH_ALPS_RANKS"])
    tasks = int(os.environ.get("SLURM_STEP_NUM_TASKS") or os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    if count < 1 or tasks != count or not 0 <= rank < count:
        raise ValueError("Slurm tasks do not match the worker count")
    return rank, count


if __name__ == "__main__":
    rank, count = worker()
    print(
        json.dumps(
            dict(
                rank=rank,
                workers=count,
                host=socket.gethostname(),
                affinity=sorted(os.sched_getaffinity(0)),
            )
        )
    )
