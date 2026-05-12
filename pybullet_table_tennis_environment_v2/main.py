"""
main.py
-------
Demo entry point for the single-robot table tennis simulation.

Usage
-----
    python main.py                   # headless, 3 episodes
    python main.py --gui             # PyBullet GUI (needs local/virtual display)
    python main.py --player advanced # run the advanced return controller

Two example players are provided:
  - DoNothingPlayer:      stays at the shared safe/home configuration
  - SimpleTrackingPlayer: uses IK to chase the ball — shows how to use
                          pybullet.calculateInverseKinematics inside a player
  - AdvancedSamplingPlayer: sampling-based controller that plans post-bounce
                            returns and tracks a smooth swing path
"""

import argparse
import pybullet as p

from player import Player
from advanced_player import AdvancedSamplingPlayer
from table_tennis_env import (
    BALL_FRICTION,
    BALL_RESTITUTION,
    KUKA_EE_LINK_INDEX,
    LEGACY_ROBOT1_BASE_POS,
    LEGACY_ROBOT2_BASE_POS,
    READY_JOINT_ANGLES_RAD,
    TableTennisEnv,
)
from simulation_manager import SimulationManager
from utils import run_simulation_with_matplotlib


PLAYER_CHOICES = ("advanced", "simple", "idle")
READY_JOINT_POSE = list(READY_JOINT_ANGLES_RAD)
SIMPLE_HOME_POSE = [0.0] * 7
LEGACY_SIMPLE_MAX_JOINT_FORCE = 250.0
LEGACY_SIMPLE_MAX_JOINT_VELOCITY = 3.0


# ---------------------------------------------------------------------------
# Example players
# ---------------------------------------------------------------------------

class DoNothingPlayer(Player):
    """
    Stays at the shared safe/home configuration.
    The ball will always pass through or fall to the floor.
    Useful as a baseline / sanity-check opponent.
    """

    def get_action(self, _observation: dict) -> list:
        return list(READY_JOINT_POSE)


class SimpleTrackingPlayer(Player):
    """
    Attempts to position the paddle above the ball using inverse kinematics.

    Parameters
    ----------
    player_id : int
        0 for the robot at x=-2 (defends x < 0),
        1 for the robot at x=+2 (defends x > 0).
    env : TableTennisEnv
        The shared environment instance.
    """

    def __init__(self, player_id: int, env: TableTennisEnv):
        self.player_id = player_id
        self.env       = env
        self.robot_id  = env.robot1_id if player_id == 0 else env.robot2_id
        self.ready_pose = list(SIMPLE_HOME_POSE)
        
        """
        TableTennisEnv Features Available to Players: 
        - physics_client: the PyBullet client ID, needed for IK calls
        - robot1_id, robot2_id: the PyBullet body IDs of the two robots (NOT ALLOWED TO CHECK OPPONENT ROBOT'S JOINT STATES)
        - get_observation(player_id): returns the current observation dict, which includes:
            - ball_position: (x, y, z) position of the ball
            - ball_velocity: (vx, vy, vz) velocity of the ball
            - robot_joint_states: player's own robot joint states (positions, velocities, etc.) — can be used for closed-loop control
            - robot_joint_velocities: player's own robot joint velocities
            - table_info: dict with info about the table and player side
        - get_ball_state(): returns the current position and velocity (linear, angular) of the ball, for convenience.
        - get_contact_with_paddle(): returns info about the most recent contact between the ball and the player's paddle, if any.
        - get_contact_with_table(): returns info about the most recent contact between the ball and the table, if any.
        - get_contact_with_net(): returns info about the most recent contact between the ball and the net, if any.
        """

    def get_action(self, observation: dict) -> list:
        ball_pos = observation["ball_position"]
        ball_vel = observation["ball_velocity"]

        # Simple linear prediction: where will the ball be in 0.1 s?
        dt = 0.1
        target = [
            ball_pos[0] + ball_vel[0] * dt,
            ball_pos[1] + ball_vel[1] * dt,
            max(ball_pos[2] + ball_vel[2] * dt, 0.85),  # stay above table
        ]

        # Only move toward the ball when it is on our side
        # BASIC CHECK: Do not move if ball is in other side.
        side = observation["table_info"]["player_side"]
        if side == "negative_x" and ball_pos[0] > 0.1:
            # Ball is on opponent's side — stay at home
            return list(self.ready_pose)
        if side == "positive_x" and ball_pos[0] < -0.1:
            return list(self.ready_pose)

        # Solve IK for the predicted target position
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            KUKA_EE_LINK_INDEX,
            target,
            physicsClientId=self.env.physics_client,
        )
        return list(joint_angles[:7])

    def reset(self) -> None:
        pass


def build_player(name: str, env: TableTennisEnv) -> Player:
    controller = name.lower()
    if controller == "advanced":
        return AdvancedSamplingPlayer(player_id=0, env=env)
    if controller == "simple":
        return SimpleTrackingPlayer(player_id=0, env=env)
    if controller == "idle":
        return DoNothingPlayer()
    raise ValueError(
        f"Unsupported player '{name}'. Available controllers: {', '.join(PLAYER_CHOICES)}"
    )


if __name__ == "__main__":
    ###########################################################################
    parser = argparse.ArgumentParser(
        description="Single-robot table tennis return simulation."
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open the PyBullet GUI window (needs a local or virtual display).",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of complete games to simulate (default: 1).",
    )
    parser.add_argument(
        "--player", default="advanced", choices=PLAYER_CHOICES,
        help="Controller to run on the single robot (default: advanced).",
    )
    args = parser.parse_args()
    
    if args.gui:
        mode_label = "GUI"
    else:
        mode_label = "headless"
    print(
        f"Starting {args.episodes} episode(s) ({mode_label}) "
        f"with player={args.player} ..."
    )

    ################################################################################
    # 1. Define the environment and shared PyBullet client.
    env_kwargs = {"gui": args.gui, "one_robot_mode": True}
    if args.player == "simple":
        env_kwargs["home_joint_angles"] = SIMPLE_HOME_POSE
        env_kwargs["max_joint_force"] = LEGACY_SIMPLE_MAX_JOINT_FORCE
        env_kwargs["max_joint_velocity"] = LEGACY_SIMPLE_MAX_JOINT_VELOCITY
        env_kwargs["robot1_base_pos"] = LEGACY_ROBOT1_BASE_POS
        env_kwargs["robot2_base_pos"] = LEGACY_ROBOT2_BASE_POS
        env_kwargs["table_robot_collisions_enabled"] = True
        env_kwargs["paddle_restitution"] = BALL_RESTITUTION
        env_kwargs["paddle_friction"] = BALL_FRICTION
    env = TableTennisEnv(**env_kwargs)

    # 2. Select the single-robot controller to benchmark.
    player1 = build_player(args.player, env=env)
    
    # 3. Simulation manager handles the game logic and scoring, and runs episodes.
    # - Provide observation to the players
    # - Get actions from player's policy 
    # - Apply actions to the environment
    # - Check for points scored and update scores and game state (serve, rally, point scored, game over)
    manager = SimulationManager(
        player1=player1,
        player2=None, # ONLY ONE ROBOT MODE, so no opponent player
        gui=args.gui, 
        real_time=args.gui,   # slow down to real-time only in GUI mode
        env=env,
        one_robot_mode=True,
    )
    
    # results = manager.run(max_episodes=args.episodes)
    
    results = run_simulation_with_matplotlib(manager=manager, max_episodes=args.episodes, fps=30.0)

    # Print summary of results
    print("\n--- Results ---")
    for i, r in enumerate(results):
        if manager.one_robot_mode:
            total_serves = r['scores'][0] + r['scores'][1]
            print(
                f"  Episode {i + 1}: "
                f"{r['scores'][0]} / {total_serves} returns  |  "
                f"Steps: {r['total_steps']}"
            )
        else:
            print(
                f"  Episode {i + 1}: "
                f"Player 1 {r['scores'][0]} — {r['scores'][1]} Player 2  |  "
                f"Winner: Player {r['winner'] + 1}  |  "
                f"Steps: {r['total_steps']}"
            )

    env.close()
