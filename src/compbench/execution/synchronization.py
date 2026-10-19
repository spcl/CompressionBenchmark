"""Shared-filesystem rendezvous outside the timed workload."""

import json
from pathlib import Path
from time import monotonic, sleep, time

from compbench.results import save

MAX_START_LAG = 0.1


def wait_for_files(paths, *, timeout=3600.0):
    deadline = monotonic() + timeout
    pending = set(map(Path, paths))
    while pending:
        pending = {p for p in pending if not p.exists()}
        if not pending:
            return
        if monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {len(pending)} worker files")
        sleep(0.1)


def synchronized_start(directory, rank, count, *, timeout=300.0, lead=3.0):
    if not 0 <= rank < count or lead <= 0 or timeout <= 0:
        raise ValueError("Invalid start barrier parameters")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation rejects stale attempts and duplicate rank processes.
    (directory / f"ready_{rank}").touch(exist_ok=False)
    release = directory / "release.json"
    if rank == 0:
        wait_for_files([directory / f"ready_{r}" for r in range(count)], timeout=timeout)
        save(release, dict(workers=count, scheduled_start=time() + lead))
    wait_for_files([release], timeout=timeout)
    payload = json.loads(release.read_text())
    if payload["workers"] != count:
        raise ValueError("Mismatched worker count at start barrier")
    scheduled = payload["scheduled_start"]
    if time() >= scheduled:
        raise RuntimeError("Received start barrier release after its deadline")
    while (remaining := scheduled - time()) > 0:
        sleep(remaining)
    actual = time()
    lag = actual - scheduled
    if not 0 <= lag <= MAX_START_LAG:
        raise RuntimeError(f"Missed synchronized start by {lag:.6f} s")
    return actual, dict(barrier_id=str(directory), scheduled_start=scheduled, start_lag_s=lag)
