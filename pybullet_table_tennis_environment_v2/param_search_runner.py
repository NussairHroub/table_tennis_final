"""
param_search_runner.py
----------------------
Worker for the exhaustive parameter search. Reads a JSON batch of
(config, seeds) pairs from stdin, evaluates each by running the
RobotPlayer with the config's class attributes overridden, and prints
the aggregate result back as a single RESULT_JSON line.

Not meant to be invoked by hand — see param_search.py.

Each "config" is a dict of {class_attr_name: value} applied to RobotPlayer
before instantiation. Uses the DEFAULT v2 env (fast 12 rad/s arm, no
table collisions, paddle restitution 0.93) — same env the grader will run.
"""

from __future__ import annotations
import io
import json
import sys
from contextlib import redirect_stdout

sys.path.insert(0, "/home/nussair/term_project/pybullet_table_tennis_environment_v2")

from table_tennis_env import TableTennisEnv
from simulation_manager import SimulationManager
from one_player_game import RobotPlayer


def evaluate(env, config: dict, seeds: list) -> dict:
    # Apply config as class-level attribute overrides on RobotPlayer.
    for key, value in config.items():
        setattr(RobotPlayer, key, value)

    total_returns = 0
    total_serves = 0
    for seed in seeds:
        player = RobotPlayer(player_id=0, env=env)
        mgr = SimulationManager(
            player1=player,
            player2=None,
            gui=False,
            real_time=False,
            env=env,
            one_robot_mode=True,
            seed=seed,
        )
        with redirect_stdout(io.StringIO()):
            ep_results = mgr.run(max_episodes=1)
        total_returns += ep_results[0]["scores"][0]
        total_serves += ep_results[0]["scores"][0] + ep_results[0]["scores"][1]

    rate = total_returns / total_serves if total_serves else 0.0
    return {
        "config": config,
        "returns": total_returns,
        "serves": total_serves,
        "rate": rate,
    }


def main():
    batch = json.loads(sys.argv[1])  # list of {"config": {...}, "seeds": [...]}

    # Reuse one env across all configs in this worker (env creation costs ~1 s).
    # DEFAULT v2 env — no joint-velocity / restitution / collision overrides.
    env = TableTennisEnv(gui=False, one_robot_mode=True)
    try:
        out = [evaluate(env, e["config"], e["seeds"]) for e in batch]
    finally:
        env.close()
    print("RESULT_JSON: " + json.dumps(out))


if __name__ == "__main__":
    main()
