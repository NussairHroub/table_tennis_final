"""
eval_rl.py
----------
Evaluate a trained RL policy on the standard SimulationManager scoring rules.

Usage
-----
    python eval_rl.py --model runs/<run>/best/best_model.zip
    python eval_rl.py --model runs/<run>/model.zip --episodes 5 --gui
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from table_tennis_env import TableTennisEnv
from simulation_manager import SimulationManager
from player import Player
from one_player_game import RobotPlayer
from rl_env import TableTennisRLEnv


class _RLPlayer(Player):
    """Wraps a trained SB3 model as a Player. With residual=True the model's
    [-1, 1]^7 output is added to the RobotPlayer expert's action."""

    def __init__(self, model_path: str, vecnorm_path: str | None = None,
                 algo: str = "ppo", residual: bool = True,
                 env=None):
        cls = PPO if algo == "ppo" else SAC
        self.model = cls.load(model_path, device="cpu")
        self.norm = None
        if vecnorm_path and Path(vecnorm_path).exists():
            dummy = DummyVecEnv([lambda: TableTennisRLEnv(gui=False, residual=residual)])
            self.norm = VecNormalize.load(vecnorm_path, dummy)
            self.norm.training = False
            self.norm.norm_reward = False

        self.residual = residual
        if residual:
            assert env is not None, "residual=True needs the env for the expert"
            self._expert = RobotPlayer(player_id=0, env=env)
            self._scale = TableTennisRLEnv.RESIDUAL_SCALE
            self._lo = np.array(
                [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
                dtype=np.float32,
            )
            self._hi = -self._lo
            self._hi[6] = 3.0543

    def get_action(self, observation: dict) -> list:
        obs = np.concatenate([
            observation["ball_position"],
            observation["ball_velocity"],
            observation["joint_positions"],
            observation["joint_velocities"],
            [
                observation["table_info"]["length"],
                observation["table_info"]["width"],
                observation["table_info"]["height"],
            ],
        ]).astype(np.float32)

        if self.norm is not None:
            obs = self.norm.normalize_obs(obs)

        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        if self.residual:
            expert_a = np.asarray(self._expert.get_action(observation), dtype=np.float32)
            effective = expert_a + self._scale * np.clip(action, -1.0, 1.0)
            effective = np.clip(effective, self._lo, self._hi)
            return [float(a) for a in effective]
        return [float(a) for a in action]

    def reset(self) -> None:
        if self.residual:
            self._expert.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.zip")
    parser.add_argument("--vecnorm", default=None,
                        help="Path to vecnormalize.pkl (auto-detected if omitted)")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-residual", action="store_true",
                        help="Treat model output as raw joint angles "
                             "(for non-residual policies).")
    args = parser.parse_args()

    model_path = Path(args.model)
    vecnorm = args.vecnorm
    if vecnorm is None:
        for candidate in [
            model_path.parent / "vecnormalize.pkl",
            model_path.parent.parent / "vecnormalize.pkl",
        ]:
            if candidate.exists():
                vecnorm = str(candidate)
                break

    print(f"Model:        {model_path}")
    print(f"VecNormalize: {vecnorm}")

    env = TableTennisEnv(gui=args.gui, one_robot_mode=True)
    player = _RLPlayer(
        str(model_path), vecnorm,
        algo=args.algo,
        env=env,
        residual=not args.no_residual,
    )
    mgr = SimulationManager(
        player1=player,
        player2=None,
        gui=args.gui,
        real_time=args.gui,
        seed=args.seed,
        env=env,
        one_robot_mode=True,
    )

    results = mgr.run(max_episodes=args.episodes)
    total_returns = sum(r["scores"][0] for r in results)
    total_serves = sum(r["scores"][0] + r["scores"][1] for r in results)
    rate = (100.0 * total_returns / total_serves) if total_serves else 0.0
    print(f"\n=== {total_returns} / {total_serves} returns "
          f"= {rate:.0f}% over {args.episodes} episode(s) ===")
    env.close()


if __name__ == "__main__":
    main()
