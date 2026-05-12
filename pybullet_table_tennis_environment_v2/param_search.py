"""
param_search.py
---------------
Exhaustive grid search over RobotPlayer's tuning constants. Splits the
grid across N parallel subprocesses (each runs param_search_runner.py)
and reports the top configurations by return rate.

Usage
-----
    python param_search.py                     # default grid, 8 workers
    python param_search.py --workers 4
    python param_search.py --seeds 0 1 2 3 4   # 5 seeds per config (more
                                                 # stable estimates, longer
                                                 # runtime)
    python param_search.py --top 20            # show top 20 instead of 10

Output:
    param_search_results.json  — every config + its return rate, sorted
"""

from __future__ import annotations
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


# Grid of RobotPlayer class attributes to sweep. Adjust the lists below
# to widen or narrow the search. Total configs = product of list sizes.
#
# DEFAULT-ENV grid. With the fast 12 rad/s arm the swing completes ~4×
# faster than under the legacy 3 rad/s settings, so _T_SWING values must
# be much smaller (and we sweep a wider range to find the right value).
# Without table collisions, _Z_FLOOR matters too.
DEFAULT_GRID = {
    "_PADDLE_TILT":   [0.20, 0.35, 0.50, 0.65],
    "_FT_LIFT":       [0.04, 0.08, 0.12],
    "_FT_DIST":       [0.05, 0.10, 0.15],
    "_WU_DIST":       [0.05, 0.10],
    "_STRIKE_OFFSET": [0.85, 0.90, 0.95],
    "_T_SWING":       [0.08, 0.14, 0.22, 0.32],
    "_Z_FLOOR":       [0.92, 0.96, 1.00],
    "_BALL_REST":     [0.82, 0.87],
}


def all_configs(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def split_evenly(items, n):
    """Split items into n approximately-equal chunks."""
    out = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return [c for c in out if c]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds used per config (each yields one episode).",
    )
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument(
        "--output",
        default="param_search_results.json",
        help="Path to write full sorted results.",
    )
    ap.add_argument(
        "--worker-timeout",
        type=int,
        default=1800,
        help="Per-worker timeout in seconds (default 30 min).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    runner = str(Path(__file__).parent / "param_search_runner.py")

    configs = list(all_configs(DEFAULT_GRID))
    print(f"Total configs: {len(configs)} (seeds per config: {len(args.seeds)})")
    print(f"Workers: {args.workers}")

    batches = split_evenly(configs, args.workers)
    print(f"Spawning {len(batches)} worker subprocesses...")
    t0 = time.time()

    procs = []
    for i, batch in enumerate(batches):
        payload = json.dumps([
            {"config": c, "seeds": args.seeds} for c in batch
        ])
        p = subprocess.Popen(
            [sys.executable, runner, payload],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append((i, len(batch), p))
        print(f"  worker {i}: pid {p.pid}, {len(batch)} configs")

    all_results = []
    for i, n_configs, p in procs:
        try:
            stdout, stderr = p.communicate(timeout=args.worker_timeout)
        except subprocess.TimeoutExpired:
            print(f"  worker {i}: TIMEOUT, killing")
            p.kill()
            stdout, stderr = p.communicate()

        elapsed = time.time() - t0
        if p.returncode != 0:
            print(f"  worker {i}: exit {p.returncode} after {elapsed:.0f}s")
            tail = stderr[-500:] if stderr else ""
            print(f"    stderr tail: {tail}")

        found = False
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON: "):
                results = json.loads(line[len("RESULT_JSON: "):])
                all_results.extend(results)
                found = True
                break
        status = "ok " if found else "MISS"
        print(f"  worker {i} done: {status} {len(all_results):4d} results "
              f"so far  (elapsed {elapsed:.0f}s)")

    all_results.sort(key=lambda r: (-r["rate"], -r["returns"]))

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results → {args.output}")

    print(f"\n=== TOP {args.top} configurations ===")
    for r in all_results[: args.top]:
        cfg = ", ".join(f"{k}={v}" for k, v in r["config"].items())
        print(f"  {r['rate']:6.2%} ({r['returns']:3d}/{r['serves']:3d})  {cfg}")


if __name__ == "__main__":
    main()
