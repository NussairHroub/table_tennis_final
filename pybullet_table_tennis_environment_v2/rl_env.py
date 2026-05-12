"""
rl_env.py
---------
Single-rally Gymnasium environment for training a residual PPO/SAC policy
on top of the classical RobotPlayer expert.

The agent emits a delta in [-1, 1]^7 that is scaled by RESIDUAL_SCALE rad and
added to the expert's joint targets. The expert is the RobotPlayer defined in
one_player_game.py — using it as the anchor means the policy only learns the
small deviations that improve on a working baseline.
"""

from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("rl_env requires gymnasium. Install with: pip install gymnasium")

from table_tennis_env import TableTennisEnv
from simulation_manager import SimulationManager, GameState
from player import Player
from one_player_game import RobotPlayer
import pybullet as p


_JOINT_LOW = np.array(
    [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543],
    dtype=np.float32,
)
_JOINT_HIGH = np.array(
    [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543],
    dtype=np.float32,
)


class _AgentPlayer(Player):
    def __init__(self):
        self._action = [0.0] * 7

    def set_action(self, action):
        self._action = list(action)

    def get_action(self, _observation):
        return self._action


class TableTennisRLEnv(gym.Env):
    """
    Phase-1 single-rally training environment.

    Observation (Box, 23-d float32):
        [0:3]   ball position
        [3:6]   ball velocity
        [6:13]  own joint positions
        [13:20] own joint velocities
        [20:23] table dims (length, width, height)

    Action:
        residual=True  → Box([-1, 1]^7), added to expert action × RESIDUAL_SCALE
        residual=False → Box(KUKA joint limits), raw joint targets
    """

    metadata = {"render_modes": []}

    RESIDUAL_SCALE = 0.5  # rad — max deviation per joint from expert action

    DIST_COEF = 0.01
    SUCCESS_BONUS = 60.0
    CONTACT_BONUS = 5.0
    SWING_BONUS_COEF = 2.0
    SWING_BONUS_CAP = 6.0
    FAULT_DOUBLE_BOUNCE = -2.0
    FAULT_NET = -5.0
    FAULT_OUT = -3.0
    FAULT_FALLEN = -3.0
    FAULT_TIMEOUT = -3.0

    def __init__(
        self,
        gui: bool = False,
        seed: int | None = None,
        max_steps: int = 1500,
        residual: bool = True,
    ):
        super().__init__()

        self._agent = _AgentPlayer()
        self._env = TableTennisEnv(gui=gui, one_robot_mode=True)
        self._manager = SimulationManager(
            player1=self._agent,
            player2=None,
            gui=gui,
            real_time=False,
            seed=seed,
            env=self._env,
            one_robot_mode=True,
        )
        self._residual = residual
        self._expert = RobotPlayer(player_id=0, env=self._env) if residual else None

        if residual:
            self.action_space = spaces.Box(
                low=-np.ones(7, dtype=np.float32),
                high=np.ones(7, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(_JOINT_LOW, _JOINT_HIGH, dtype=np.float32)

        obs_low = np.concatenate([
            np.full(3, -5.0),
            np.full(3, -20.0),
            _JOINT_LOW,
            np.full(7, -10.0),
            np.array([0.0, 0.0, 0.0]),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            np.full(3, 5.0),
            np.full(3, 20.0),
            _JOINT_HIGH,
            np.full(7, 10.0),
            np.array([3.0, 2.0, 1.0]),
        ]).astype(np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._max_steps = max_steps
        self._step_count = 0
        self._paddle_id = self._env.paddle1_id
        self._physics = self._env.physics_client

        self._made_contact = False
        self._post_contact_vx_seen = False
        self._prev_paddle_contact = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._manager._rng.seed(seed)

        self._manager.reset()
        if self._expert is not None:
            self._expert.reset()
        self._manager._do_serve()
        self._step_count = 0
        self._made_contact = False
        self._post_contact_vx_seen = False
        self._prev_paddle_contact = False
        return self._build_obs(), self._build_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._residual:
            obs_dict = self._env.get_observation(0)
            expert_a = np.asarray(
                self._expert.get_action(obs_dict), dtype=np.float32,
            )
            effective = expert_a + self.RESIDUAL_SCALE * np.clip(action, -1.0, 1.0)
            effective = np.clip(effective, _JOINT_LOW, _JOINT_HIGH)
            self._agent.set_action(effective)
        else:
            self._agent.set_action(action)

        info = self._manager.step_once()
        self._step_count += 1

        terminated = False
        truncated = False
        reward = self._shaped_step_reward()

        if self._manager.game_state == GameState.POINT_SCORED:
            reward += self._terminal_reward()
            terminated = True

        if self._step_count >= self._max_steps and not terminated:
            truncated = True

        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._env.close()

    def _shaped_step_reward(self) -> float:
        ball_state = self._env.get_ball_state()
        ball_pos = np.asarray(ball_state["position"], dtype=np.float32)
        ball_vx = float(ball_state["velocity"][0])
        paddle_pos, _ = p.getBasePositionAndOrientation(
            self._paddle_id, physicsClientId=self._physics,
        )
        paddle_pos = np.asarray(paddle_pos, dtype=np.float32)

        if not self._made_contact:
            d = float(np.linalg.norm(ball_pos - paddle_pos))
            r_dist = -self.DIST_COEF * min(d, 1.5)
        else:
            r_dist = 0.0

        in_contact = bool(self._env.get_contact_with_paddle(0))
        r_contact = 0.0
        if in_contact and not self._prev_paddle_contact:
            if not self._made_contact:
                r_contact += self.CONTACT_BONUS
            self._made_contact = True
        self._prev_paddle_contact = in_contact

        r_swing = 0.0
        if self._made_contact and not self._post_contact_vx_seen:
            if abs(ball_vx) > 0.5:
                outward_vx = max(ball_vx, 0.0)
                r_swing = min(self.SWING_BONUS_COEF * outward_vx, self.SWING_BONUS_CAP)
                self._post_contact_vx_seen = True

        return r_dist + r_contact + r_swing

    def _terminal_reward(self) -> float:
        fault = self._manager.last_fault
        if fault == "successful_return":
            return self.SUCCESS_BONUS
        if "double_bounce_on_player0_side" in fault:
            return self.FAULT_DOUBLE_BOUNCE * (0.4 if self._made_contact else 1.0)
        if fault == "ball_hit_net":
            return self.FAULT_NET
        if fault == "ball_out_of_bounds":
            return self.FAULT_OUT
        if fault == "ball_fell_below_table":
            return self.FAULT_FALLEN
        if fault == "rally_timeout":
            return self.FAULT_TIMEOUT
        return 0.0

    def _build_obs(self) -> np.ndarray:
        obs_dict = self._env.get_observation(0)
        return np.concatenate([
            obs_dict["ball_position"],
            obs_dict["ball_velocity"],
            obs_dict["joint_positions"],
            obs_dict["joint_velocities"],
            [
                obs_dict["table_info"]["length"],
                obs_dict["table_info"]["width"],
                obs_dict["table_info"]["height"],
            ],
        ]).astype(np.float32)

    def _build_info(self) -> dict:
        return {"scores": list(self._manager.scores)}
