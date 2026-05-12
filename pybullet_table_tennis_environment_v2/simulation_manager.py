"""
simulation_manager.py
---------------------
Top-level controller that integrates the physics environment, players, and
full game rules for a table tennis match.

Usage
-----
    from table_tennis_env import TableTennisEnv
    from simulation_manager import SimulationManager
    from my_player import MyPlayer

    env = TableTennisEnv(gui=True)
    manager = SimulationManager(MyPlayer(), MyPlayer(), gui=True, real_time=True)
    results = manager.run(max_episodes=1)
"""

import time
import random
from enum import Enum, auto

from table_tennis_env import (
    TableTennisEnv,
    TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT,
    SIM_TIMESTEP,
)
from player import Player


# ---------------------------------------------------------------------------
# Game constants
GAME_POINT = 11 # First to 11 (win by 2)
MAX_STEPS_PER_RALLY = 2400 # ~10 seconds at 240 Hz before timeout
OUT_OF_BOUNDS_MARGIN = 0.1   # metres beyond table edge before calling out
BALL_FALLEN_Z = 0.5          # Ball below this Z → missed → fault

# Serve launch parameters
# Player 0 defends x < 0 → serves from own side (x<0) toward opponent (x>0)
# Player 1 defends x > 0 → serves from own side (x>0) toward opponent (x<0)
# By default, the serving is at a moderate speed toward the opponent, with some noise for variability.
SERVE_POSITIONS = {
    0: [-0.6, 0.0, 1.1],
    1: [ 0.6, 0.0, 1.1],
}
SERVE_VELOCITIES = {
    0: [ 2.8, 0.0, 1.0],   # toward x > 0 (opponent's side)
    1: [-2.8, 0.0, 1.0],   # toward x < 0 (opponent's side)
}
SERVE_NOISE_XY = 0.3  # std dev for x/y velocity noise 0.15
SERVE_NOISE_Z  = 0.15  # std dev for z velocity noise 0.05

# Z tolerance for determining which side the ball is on when it bounces
NET_ZONE_HALF = 0.05   # ±5 cm around x=0 treated as "net zone"


class GameState(Enum):
    WAITING_SERVE = auto() # Waiting for the server to launch the ball
    RALLY         = auto() # Ball is in play, players can hit it
    POINT_SCORED  = auto() # A point was just scored, waiting for reset
    GAME_OVER     = auto() # One player has won the game, no more play until reset


class SimulationManager:
    """
    Manages a full table tennis game between two players.

    Parameters
    ----------
    player1 : Player
        Controls robot 1 (base at x=-2, defends x < 0).
    player2 : Player
        Controls robot 2 (base at x=+2, defends x > 0).
    gui : bool
        Whether to open the PyBullet GUI window.
    game_point : int
        Points needed to win a game (default 11).
    real_time : bool
        If True, insert time.sleep between steps to match real time.
        Recommended when gui=True for a watchable simulation.
    seed : int or None
        Random seed for serve trajectory noise.
    """

    def __init__(self, player1: Player, player2: Player = None, gui: bool = False, game_point: int = GAME_POINT,
                 real_time: bool = False, seed: int = None, env: TableTennisEnv = None,
                 one_robot_mode: bool = False):
        self.player1        = player1
        self.player2        = player2
        self.game_point     = game_point # points needed to win (configurable)
        self.real_time      = real_time
        self.gui            = gui
        self.one_robot_mode = one_robot_mode
        self._rng           = random.Random(seed) # Reproducible randomness for serve noise

        self.env = env if env is not None else TableTennisEnv(gui=gui, one_robot_mode=one_robot_mode)

        # Initialize Game state (initialised by reset())
        self.game_state: GameState = GameState.WAITING_SERVE
        self.scores:    list = [0, 0]
        self.server:    int  = 0          # player_id of current server
        self.last_fault: str = ""
        self.total_steps: int = 0

        # Rally tracking
        self.rally_step_count: int  = 0
        self._last_hitter:     int  = None
        self._bounce_counts:   dict = {0: 0, 1: 0}
        self._prev_contact_table:   bool = False
        self._prev_contact_paddle:  dict = {0: False, 1: False}

    # ------------------------------------------------------------------
    # Public API

    def reset(self) -> dict:
        """
        Reset game state, environment, and both players.

        Returns
        -------
        dict
            Initial info dict (same format as step_once).
        """
        self.scores          = [0, 0]
        self.server          = 0
        self.last_fault      = ""
        self.total_steps     = 0
        self.game_state      = GameState.WAITING_SERVE

        self.player1.reset()
        if self.player2 is not None:
            self.player2.reset()
        self.env.reset()
        self._reset_rally_state()

        return self._build_info()

    def step_once(self) -> dict:
        """
        Advance the simulation by one physics step.

        Handles the full game loop: querying players, stepping physics,
        checking contacts and faults, updating game state.

        Returns
        -------
        dict
            {
              "state":  GameState  — current game state,
              "scores": [int, int] — [player1_score, player2_score],
              "server": int        — player_id who serves next,
              "fault":  str        — description of last fault (or ""),
              "done":   bool       — True if game is over
            }
        """
        
        # GAME dynamics handler:
        # 1. If we're waiting to serve, launch the ball first
        if self.game_state == GameState.WAITING_SERVE:
            self._do_serve()

        # 2. Don't advance if game is over or point was just scored
        if self.game_state in (GameState.GAME_OVER, GameState.POINT_SCORED):
            return self._build_info()

        # 3. Query players for actions (Student has to implement in player.py get_action method)
        obs1 = self.env.get_observation(0) # Get the observation for player to pass to the get_action method
        action1 = self.player1.get_action(obs1)
        action1 = self.player1.clip_action(action1) # Maintain feasibility of actions (joint states)

        if self.one_robot_mode:
            action2 = [0.0] * 7  # no robot2; env.step ignores this
        else:
            obs2 = self.env.get_observation(1)
            action2 = self.player2.get_action(obs2)
            action2 = self.player2.clip_action(action2)

        # 4. Advance physics step
        self.env.step(action1, action2) # using the env step to advance the simulation and apply the actions
        self.rally_step_count += 1
        self.total_steps      += 1

        # 5. Check rules (contacts first, then continuous faults) 
        self._check_ball_contacts()
        if self.game_state == GameState.RALLY:
            self._check_faults()

        # 6. Real-time sync
        if self.real_time:
            time.sleep(SIM_TIMESTEP) # If in GUI mode with real_time=True, this makes the simulation run at real time speed (instead of as fast as possible)

        return self._build_info()

    def run(self, max_episodes: int = 1) -> list:
        """
        Run one or more complete games to GAME_OVER, then return results.

        Parameters
        ----------
        max_episodes : int
            Number of full games to play.

        Returns
        -------
        list of dict, one per episode:
            {
              "scores":      [int, int],
              "winner":      int,   # player_id
              "total_steps": int,
            }
        """
        results = []
        for ep in range(max_episodes):
            # 1. Reset everything to start a new episode (one game = Episode with 11 points)
            self.reset()
            while self.game_state != GameState.GAME_OVER: # Internally controlled game-over
                # 2. Step though spisode getting palyer actions and applying to the env
                self.step_once()
                # 2. If point is scored, change state and reset players and env for next point
                if self.game_state == GameState.POINT_SCORED:
                    if self.one_robot_mode:
                        scored = self.last_fault == "successful_return"
                        label = "Return" if scored else "Miss  "
                        print(f"  {label} ({self.last_fault}) | Returns: {self.scores[0]}")
                    else:
                        print(
                            f"  Point! Fault: {self.last_fault} | "
                            f"Score: {self.scores[0]}-{self.scores[1]}"
                        )
                    self.game_state = GameState.WAITING_SERVE
                    self.player1.reset()
                    if self.player2 is not None:
                        self.player2.reset()
                    self.env.reset()
                    self._reset_rally_state()

            winner = 0 if self.scores[0] > self.scores[1] else 1
            result = {
                "scores":      list(self.scores),
                "winner":      winner,
                "total_steps": self.total_steps,
            }
            results.append(result)
            if self.one_robot_mode:
                total_serves = self.scores[0] + self.scores[1]
                print(
                    f"Episode {ep + 1}: {self.scores[0]} / {total_serves} returns "
                    f"in {self.total_steps} steps"
                )
            else:
                print(
                    f"Episode {ep + 1}: Player {winner + 1} wins "
                    f"{self.scores[0]}-{self.scores[1]} "
                    f"in {self.total_steps} steps"
                )

        return results

    # ------------------------------------------------------------------
    # Internal game logic Definitions
    # Automatic initial serving
    def _do_serve(self) -> None:
        """Launch the ball from the server's side toward the opponent."""
        if self.one_robot_mode:
            self.server = 1  # ball always comes from opponent's side
        base_vel = list(SERVE_VELOCITIES[self.server])
        noisy_vel = [
            base_vel[0] + self._rng.gauss(0, SERVE_NOISE_XY),
            base_vel[1] + self._rng.gauss(0, SERVE_NOISE_XY * 0.3),
            base_vel[2] + self._rng.gauss(0, SERVE_NOISE_Z),
        ]
        self.env.reset(
            ball_position=list(SERVE_POSITIONS[self.server]),
            ball_velocity=noisy_vel,
        )
        self._reset_rally_state()
        # Before any paddle contact, attribute the ball to the server
        # (so if it goes out immediately, the server loses the point)
        self._last_hitter = self.server
        self.game_state = GameState.RALLY

    def _reset_rally_state(self) -> None:
        self.rally_step_count              = 0
        self._last_hitter                  = None
        self._bounce_counts                = {0: 0, 1: 0}
        self._prev_contact_table           = False
        self._prev_contact_paddle          = {0: False, 1: False}
        self.last_fault                    = ""
        self._robot_hit_after_bounce_side0 = False  # True only when robot hit after ball bounced on side 0
    # Count the table contact to determine bounces
    def _check_ball_contacts(self) -> None:
        """
        Detect ball bounces (edge-triggered) and paddle hits.
        Called every physics step during RALLY.
        """
        if self.game_state != GameState.RALLY:
            return

        ball_state = self.env.get_ball_state()
        ball_x = ball_state["position"][0]

        # --- Table bounce detection (edge trigger) ---
        in_contact_table = bool(self.env.get_contact_with_table())
        if in_contact_table and not self._prev_contact_table:
            side = self._get_ball_side(ball_x)
            if side is not None:
                self._bounce_counts[side] += 1
                self._on_new_bounce(side)
        self._prev_contact_table = in_contact_table

        # --- Paddle hit detection (edge trigger per player) ---
        player_ids = [0] if self.one_robot_mode else [0, 1]
        for pid in player_ids:
            in_contact_paddle = bool(self.env.get_contact_with_paddle(pid))
            if in_contact_paddle and not self._prev_contact_paddle[pid]:
                if pid == 0 and self.one_robot_mode:
                    # Valid return only if the ball had already bounced on the robot's side
                    self._robot_hit_after_bounce_side0 = self._bounce_counts[0] > 0
                self._last_hitter = pid
                # Reset bounce counts after a valid hit
                self._bounce_counts = {0: 0, 1: 0}
            self._prev_contact_paddle[pid] = in_contact_paddle

    def _on_new_bounce(self, side: int) -> None:
        """
        Called immediately when a new bounce on 'side' is registered.

        Handles:
        - One-robot mode: first bounce on opponent's half after robot return → robot scores
        - Double bounce: ball bounced twice on the same side → opponent wins
        """
        if self.game_state != GameState.RALLY:
            return

        # One-robot mode: robot scores only if it hit the ball AFTER it bounced on side 0
        if self.one_robot_mode and side == 1 and self._last_hitter == 0 and self._robot_hit_after_bounce_side0:
            self._award_point(0, "successful_return")
            return

        # Double bounce: the player who owns 'side' failed to return in time
        if self._bounce_counts[side] >= 2:
            opponent = 1 - side
            self._award_point(opponent, f"double_bounce_on_player{side}_side")

    def _check_faults(self) -> None:
        """
        TODO: this method detect if the ball is outside table, hit the net or rally timeout
        Continuous per-step checks for ball out-of-bounds, missed, net hit, timeout.
        
        """
        if self.game_state != GameState.RALLY:
            return

        ball_state = self.env.get_ball_state()
        ball_pos   = ball_state["position"]

        # Out of bounds (x or y)
        if self._is_out_of_bounds(ball_pos):
            hitter = self._last_hitter if self._last_hitter is not None else self.server
            self._award_point(1 - hitter, "ball_out_of_bounds")
            return

        # Ball fell below table (missed entirely)
        if ball_pos[2] < BALL_FALLEN_Z:
            hitter = self._last_hitter if self._last_hitter is not None else self.server
            self._award_point(1 - hitter, "ball_fell_below_table")
            return

        # Net contact → last hitter loses the point
        if bool(self.env.get_contact_with_net()):
            hitter = self._last_hitter if self._last_hitter is not None else self.server
            self._award_point(1 - hitter, "ball_hit_net")
            return

        # Rally timeout → server loses
        if self.rally_step_count >= MAX_STEPS_PER_RALLY:
            self._award_point(1 - self.server, "rally_timeout")

    def _award_point(self, winner: int, fault: str) -> None:
        """Update score, set last_fault, transition state."""
        # In one-robot mode, server faults (robot gets point for a reason other than a
        # genuine return) are treated as lets: silently re-serve without counting.
        if self.one_robot_mode and winner == 0 and fault != "successful_return":
            self.game_state = GameState.WAITING_SERVE
            return

        self.scores[winner] += 1
        self.last_fault      = fault
        # In one-robot mode the server is always player 1; otherwise loser serves next
        if self.one_robot_mode:
            self.server = 1
        else:
            self.server = 1 - winner
        self.game_state      = GameState.POINT_SCORED

        # Check win condition: first to game_point with at least 2 point lead
        if (max(self.scores) >= self.game_point and
                abs(self.scores[0] - self.scores[1]) >= 2):
            self.game_state = GameState.GAME_OVER

    def _get_ball_side(self, ball_x: float):
        """
        Return which player's side (0 or 1) the ball is on based on x position.
        Returns None if ball is within the net zone (±NET_ZONE_HALF).
        """
        if ball_x < -NET_ZONE_HALF:
            return 0   # player 0 defends x < 0
        elif ball_x > NET_ZONE_HALF:
            return 1   # player 1 defends x > 0
        return None    # over the net — can't assign to a side

    def _is_out_of_bounds(self, ball_pos: list) -> bool:
        """
        Return True if ball has left the table play area.

        Only triggers when the ball is at or near table height (z <= 0.95 m)
        to avoid calling a ball out while it is still in flight above the table.
        """
        x, y, z = ball_pos
        # Ignore ball while it is in flight well above the table
        if z > TABLE_HEIGHT + 0.19:   # ~0.95 m — allow some arc overhead
            return False
        half_l = TABLE_LENGTH / 2 + OUT_OF_BOUNDS_MARGIN   # 1.47 m
        half_w = TABLE_WIDTH  / 2 + OUT_OF_BOUNDS_MARGIN   # 0.8625 m
        return abs(x) > half_l or abs(y) > half_w

    def _build_info(self) -> dict:
        return {
            "state":  self.game_state,
            "scores": list(self.scores),
            "server": self.server,
            "fault":  self.last_fault,
            "done":   self.game_state == GameState.GAME_OVER,
        }
