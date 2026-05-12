"""
gym_wrapper.py
--------------
Optional Gym / Gymnasium wrapper around SimulationManager.

Provides a standard single-agent RL interface:
  - observation_space: Box(23,)   flattened observation vector
  - action_space:      Box(7,)    7 joint target angles (radians)
  - reward:            +1 point scored, -1 point lost, 0 otherwise

The "agent" controls player 0 (robot at x=-2, defends x < 0).
An opponent Player instance is supplied at construction.

This module is importable even if gym / gymnasium is not installed;
importing without gym will raise ImportError only when the class is
instantiated.

Usage
-----
    from gym_wrapper import TableTennisGymEnv
    from my_player import MyOpponent

    env = TableTennisGymEnv(opponent=MyOpponent(), gui=False)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_MODULE = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_MODULE = "gym"
    except ImportError:
        gym = None
        spaces = None
        _GYM_MODULE = None

from table_tennis_env import KUKA_EE_LINK_INDEX
from simulation_manager import SimulationManager, GameState
from player import Player

# Joint limits (low / high) for the 7 KUKA iiwa joints
_JOINT_LOW  = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543], dtype=np.float32)
_JOINT_HIGH = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543], dtype=np.float32)


def _check_gym():
    if _GYM_MODULE is None:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' is installed.\n"
            "Install with:  pip install gymnasium\n"
            "or:            pip install gym"
        )


class _AgentAsPlayer(Player):
    """Internal adapter: converts numpy action arrays to the Player interface."""

    def __init__(self):
        self._action = [0.0] * 7

    def set_action(self, action):
        self._action = list(action)

    def get_action(self, observation):
        return self._action


class TableTennisGymEnv:
    """
    Single-agent Gym/Gymnasium environment for table tennis.

    The agent controls player 0 (robot at x=-2, defends x < 0).
    An opponent Player instance acts as player 1.

    Observation space (Box, shape=(23,), float32)
    ---------------------------------------------
      [0:3]   ball position (x, y, z)
      [3:6]   ball velocity (vx, vy, vz)
      [6:13]  own joint positions (7 values, radians)
      [13:20] own joint velocities (7 values, rad/s)
      [20]    table length (constant 2.74)
      [21]    table width  (constant 1.525)
      [22]    table height (constant 0.76)

    Action space (Box, shape=(7,), float32)
    ----------------------------------------
      7 target joint angles in radians, clipped to KUKA iiwa limits.

    Reward
    ------
      +1  when the agent scores a point
      -1  when the opponent scores a point
       0  otherwise (ongoing rally)
    """

    def __init__(
        self,
        opponent: Player = None,
        gui: bool = False,
        game_point: int = 11,
        real_time: bool = False,
        seed: int = None,
        one_robot_mode: bool = False,
    ):
        _check_gym()

        self._agent  = _AgentAsPlayer()
        self._opponent = opponent

        self._manager = SimulationManager(
            player1=self._agent,
            player2=self._opponent,
            gui=gui,
            game_point=game_point,
            real_time=real_time,
            seed=seed,
            one_robot_mode=one_robot_mode,
        )

        # Observation space: 23-dimensional
        obs_low = np.concatenate([
            np.full(3,  -5.0),   # ball position
            np.full(3, -20.0),   # ball velocity
            _JOINT_LOW,          # own joint positions
            np.full(7, -10.0),   # own joint velocities
            np.array([0.0, 0.0, 0.0]),  # table info (positive constants)
        ]).astype(np.float32)

        obs_high = np.concatenate([
            np.full(3,   5.0),
            np.full(3,  20.0),
            _JOINT_HIGH,
            np.full(7,  10.0),
            np.array([3.0, 2.0, 1.0]),  # generous upper bounds for table dims
        ]).astype(np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(_JOINT_LOW, _JOINT_HIGH, dtype=np.float32)

        self._prev_scores = [0, 0]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            self._manager._rng.seed(seed)

        self._manager.reset()
        self._prev_scores = [0, 0]

        obs = self._build_obs()
        info = self._manager._build_info()
        return obs, info

    def step(self, action):
        """
        Take one physics step with the given joint action.

        Parameters
        ----------
        action : array-like of shape (7,)

        Returns
        -------
        obs          : np.ndarray, shape (23,)
        reward       : float
        terminated   : bool   — True if game is over
        truncated    : bool   — always False (no step limit here)
        info         : dict
        """
        self._agent.set_action(action)

        info = self._manager.step_once()

        # Advance through POINT_SCORED automatically (one reset per point)
        reward = 0.0
        if self._manager.game_state == GameState.POINT_SCORED:
            new_scores = self._manager.scores
            if new_scores[0] > self._prev_scores[0]:
                reward = 1.0    # agent scored
            elif new_scores[1] > self._prev_scores[1]:
                reward = -1.0   # opponent scored
            self._prev_scores = list(new_scores)

            # Prepare for next serve unless game is over
            if self._manager.game_state != GameState.GAME_OVER:
                self._manager.game_state = GameState.WAITING_SERVE
                self._manager.player1.reset()
                if self._manager.player2 is not None:
                    self._manager.player2.reset()
                self._manager.env.reset()
                self._manager._reset_rally_state()

        obs        = self._build_obs()
        terminated = self._manager.game_state == GameState.GAME_OVER
        truncated  = False
        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering is handled by PyBullet's own GUI — no-op here."""
        pass

    def close(self):
        self._manager.env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Flatten agent observation dict to a numpy array."""
        obs_dict = self._manager.env.get_observation(0)  # agent is player 0
        arr = np.concatenate([
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
        return arr
