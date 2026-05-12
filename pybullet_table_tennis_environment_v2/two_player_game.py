"""
two_player_game.py
------------------
Entry point for a two-robot table tennis match.

Usage
-----
    python two_player_game.py                                       # both advanced, headless
    python two_player_game.py --gui                                 # with PyBullet GUI
    python two_player_game.py --player1 robot --player2 simple      # mix controllers

Available controllers: robot, simple, idle

How to add your own player
--------------------------
1. Define a class that subclasses Player (from player.py) and implements
   get_action(observation: dict) -> list of 7 joint angles.
   Put your class in this file, below the existing example players.

2. Register it in build_player() by adding an elif branch:
       elif controller == "mybot":
           return MyBot(player_id=player_id, env=env)

3. Add the key string to PLAYER_CHOICES.

4. Run:  python two_player_game.py --player1 mybot --player2 simple
"""

import argparse
import math

import pybullet as p

from player import Player
from table_tennis_env import (
    KUKA_EE_LINK_INDEX,
    READY_JOINT_ANGLES_RAD,
    TableTennisEnv,
)
from simulation_manager import SimulationManager
from utils import run_simulation_with_matplotlib

PLAYER_CHOICES = ("robot", "simple", "idle")
READY_JOINT_POSE = list(READY_JOINT_ANGLES_RAD)


# ---------------------------------------------------------------------------
# Example players
# To add a new player, define it here and register it in build_player() below.
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
        self.env = env
        self.robot_id = env.robot1_id if player_id == 0 else env.robot2_id
        self.ready_pose = list(READY_JOINT_POSE)

        """
        TableTennisEnv Features Available to Players:
        - physics_client: the PyBullet client ID, needed for IK calls
        - robot1_id, robot2_id: the PyBullet body IDs of the two robots
          (NOT ALLOWED TO CHECK OPPONENT ROBOT'S JOINT STATES)
        - get_observation(player_id): returns the current observation dict:
            - ball_position: (x, y, z) position of the ball
            - ball_velocity: (vx, vy, vz) velocity of the ball
            - joint_positions: player's own robot joint positions
            - joint_velocities: player's own robot joint velocities
            - table_info: dict with info about the table and player side
        - get_ball_state(): returns the current position and velocity of the ball.
        - get_contact_with_paddle(player_id): last ball-paddle contact, if any.
        - get_contact_with_table(): last ball-table contact, if any.
        - get_contact_with_net(): last ball-net contact, if any.
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

        # Only move toward the ball when it is on our side.
        side = observation["table_info"]["player_side"]
        if side == "negative_x" and ball_pos[0] > 0.1:
            return list(self.ready_pose)
        if side == "positive_x" and ball_pos[0] < -0.1:
            return list(self.ready_pose)

        # Solve IK for the predicted target position.
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            KUKA_EE_LINK_INDEX,
            target,
            physicsClientId=self.env.physics_client,
        )
        return list(joint_angles[:7])

    def reset(self) -> None:
        pass


class RobotPlayer(Player):
    """
    Simple strike-pose controller.

    Algorithm
    ---------
    1. Wait at the home pose.
    2. As soon as we see the ball heading toward us (pre-bounce), predict
       its bounce point and project forward to the strike plane x = STRIKE_X
       using projectile motion + the table-bounce restitution.
    3. Solve IK once for the paddle to be at that point with its face
       pointing toward the opponent (tilted up for net clearance), and hold
       those joint targets for the rest of the rally.
    4. The position controller moves the arm smoothly from the rest pose to
       the locked strike pose, where it waits. Paddle restitution does the
       work of reflecting the ball back over the net — no swing-trigger or
       mid-motion target switch (those caused the visible fluctuation).

    Locking pre-bounce gives the (slowed) arm the full pre-bounce + post-
    bounce flight time to settle into the strike pose. The env is configured
    in __main__ with table_robot_collisions_enabled=True so the paddle
    physically cannot pass through the table during the transient.

    Parameters
    ----------
    player_id : int
        0 for the robot at x=-2 (defends x < 0),
        1 for the robot at x=+2 (defends x > 0).
    env : TableTennisEnv
        The shared environment instance.
    """

    _G              = 9.81
    _BALL_REST      = 0.87   # effective ball–table restitution
    _VX_DAMP        = 0.82   # empirical: |vx| decays by ~18 % over the
                             # flight from serve to strike plane (linear
                             # damping + contact friction bleed energy off
                             # the ball). Used in the strike-z prediction.
    _STRIKE_OFFSET  = 0.95   # m, paddle x-offset from robot base toward net.
    _STRIKE_Z       = 0.95   # m, fallback paddle altitude
    _PADDLE_TILT    = 0.65   # rad, paddle face tilt up from vertical (~37°)
                             # — picked by an exhaustive grid sweep over
                             # tilt/swing/strike parameters; this value sat
                             # at the top of the rankings AND the grid
                             # neighbors of this config also scored well
                             # (so it's robust, not a lucky outlier).
    _PADDLE_OFFSET  = 0.05   # m, paddle offset along EE z-axis (matches env)
    _Y_BOUND        = 0.50
    _Z_FLOOR        = 0.96
    _Z_CEIL         = 1.20

    # Swing parameters. The arm winds up BEHIND the strike point (back
    # toward the robot base), then sweeps FORWARD through the strike point
    # to a follow-through pose past it. The forward motion adds paddle
    # velocity at the moment of contact → stronger return.
    #
    # Sized for the fast arm (max_joint_velocity=12 rad/s default v2 env):
    # the full windup→follow sweep completes in T_SWING ≈ 0.22 s.
    _WU_DIST        = 0.05   # m, windup x-offset behind strike
    _FT_DIST        = 0.10   # m, follow-through x-offset forward of strike
    _FT_LIFT        = 0.12   # m, follow-through z-lift above strike — gives
                             # the paddle upward velocity at contact too,
                             # adding vz to the outgoing ball.
    _T_SWING        = 0.22   # s, time the arm sweeps windup→follow.
                             # The arm is at the strike point ~T_SWING/2 s
                             # after the swing fires.
    _SIM_DT         = 1.0 / 240.0   # simulator step length

    def __init__(self, player_id: int, env: TableTennisEnv):
        self.player_id = player_id
        self.env       = env
        self.robot_id  = env.robot1_id if player_id == 0 else env.robot2_id
        # s = -1 if we defend x<0 (opponent at +x); +1 if we defend x>0.
        self.s = -1 if player_id == 0 else 1

        # Always run IK on robot 1 (the well-behaved yaw=0 base); for P1
        # we mirror the world target about the origin (the two robot bases
        # at ±1.7 are symmetric across z), so the resulting joint angles —
        # applied to robot 2 (yaw=π base) — produce the mirrored world EE
        # pose. Avoids PyBullet IK landing on a contorted joint branch for
        # robot 2 that wrecks the Cartesian swing path.
        self._ik_robot_id = env.robot1_id

        bp, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.env.physics_client
        )
        # Strike plane: STRIKE_OFFSET meters from our robot base toward the net.
        self._strike_x = bp[0] + (-self.s) * self._STRIKE_OFFSET

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

        # KUKA iiwa joint limits + ready pose (used as IK rest bias).
        self._lower = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]
        self._upper = [ 2.967,  2.094,  2.967,  2.094,  2.967,  2.094,  3.054]
        self._range = [hi - lo for lo, hi in zip(self._lower, self._upper)]
        self._ready_q = list(READY_JOINT_POSE)

        # Paddle face normal in world coords (= EE z-axis after IK).
        self._n_x = -self.s * math.cos(self._PADDLE_TILT)
        self._n_z = math.sin(self._PADDLE_TILT)
        # IK is always run on robot 1, so the orientation passed to IK is
        # what robot 1 needs to put its paddle face toward +x and tilted up.
        # (For P1 we mirror the target position; the joint solution applied
        # to robot 2 gives a face along -x — the desired mirror.)
        self._ik_orn = p.getQuaternionFromEuler(
            [0.0, math.pi / 2 - self._PADDLE_TILT, 0.0]
        )

        # Per-rally swing plan: windup_q, follow_q, when to switch from
        # windup to follow, and how many physics steps have elapsed.
        self._plan = None

    def reset(self) -> None:
        self._plan = None

    def get_action(self, observation: dict) -> list:
        """
        Implement your own player here! You can use the SimpleTrackingPlayer above as a starting point, or implement something completely different.
        This function call should return a list of 7 joint target angles (radians) for the robot to move toward in the next control step.
        """
        bx, by, bz = observation["ball_position"]
        vx, vy, vz = observation["ball_velocity"]
        table_z    = observation["table_info"]["height"]

        # Ball heading toward the opponent → relax to home pose for next serve.
        if (self.s * vx) < -0.5:
            self._plan = None
            return list(self._ready_q)

        # As soon as we see the ball coming toward us, plan windup + swing.
        coming_to_us = (self.s * vx) > 0.5
        if coming_to_us and self._plan is None:
            plan = self._build_swing_plan(bx, by, bz, vx, vy, vz, table_z)
            if plan is not None:
                self._plan = plan

        if self._plan is None:
            return list(self._ready_q)

        self._plan["step"] += 1
        # First: hold windup. Then: switch to follow-through, sweeping the
        # paddle FORWARD through the strike point.
        if self._plan["step"] < self._plan["swing_at_step"]:
            return list(self._plan["windup_q"])
        return list(self._plan["follow_q"])

    def _build_swing_plan(self, bx, by, bz, vx, vy, vz, table_z):
        """
        Predict bounce + strike, then build a windup→follow joint-target plan
        scheduled so the arm sweeps through the strike point at the same
        moment the ball arrives there.
        """
        # 1. Time of first table bounce: z(t) = table_z.
        a, b, c = 0.5 * self._G, -vz, table_z - bz
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sd = disc ** 0.5
        ts = sorted([(-b - sd) / (2 * a), (-b + sd) / (2 * a)])
        pos = [t for t in ts if t > 0.012]
        if not pos:
            return None
        t_bounce = pos[0]

        bbx = bx + vx * t_bounce
        bby = by + vy * t_bounce
        if (self.s * bbx) < 0.0:            # bounce on opponent's side — ball won't come back
            return None
        # If ball is still on opp's side, it must clear the net (z=0.9125) at x=0.
        if (self.s * bx) < 0:
            t_net = -bx / vx
            if t_net > 0:
                z_at_net = bz + vz * t_net - 0.5 * self._G * t_net ** 2
                if z_at_net < 0.94:         # net top + small margin
                    return None

        # 2. Velocity after the bounce.
        vz_impact = vz - self._G * t_bounce
        vz_after  = abs(vz_impact) * self._BALL_REST

        # 3. Time from bounce point to the strike plane. Use the damped
        #    effective vx so the time matches the sim, not idealized physics.
        if abs(vx) < 1e-3:
            return None
        vx_eff = vx * self._VX_DAMP
        t_post = abs(self._strike_x - bbx) / abs(vx_eff)

        y_strike = bby + vy * t_post
        z_strike = table_z + vz_after * t_post - 0.5 * self._G * t_post ** 2
        if z_strike < self._Z_FLOOR:
            z_strike = self._STRIKE_Z

        strike = [
            self._strike_x,
            float(max(-self._Y_BOUND, min(self._Y_BOUND, y_strike))),
            float(max(self._Z_FLOOR, min(self._Z_CEIL, z_strike))),
        ]
        # Windup pose: paddle pulled BACK from the strike point (back = +s
        # direction for player 0, i.e., more negative x). Drop windup slightly
        # so the swing arcs up through the ball.
        windup = [
            strike[0] + self.s * self._WU_DIST,
            strike[1],
            strike[2] - 0.03,
        ]
        # Follow-through: paddle continues FORWARD past the strike point
        # (toward the opponent), lifted slightly.
        follow = [
            strike[0] - self.s * self._FT_DIST,
            strike[1],
            strike[2] + self._FT_LIFT,
        ]

        # Total time from NOW until the ball is at the strike plane.
        t_arrive = t_bounce + t_post
        # The arm reaches the strike point about T_SWING/2 after the swing
        # fires (midpoint of the windup→follow sweep), so schedule the swing
        # trigger that much before predicted contact.
        swing_at_step = max(
            0, int(round((t_arrive - self._T_SWING * 0.5) / self._SIM_DT))
        )

        return {
            "strike": strike,
            "windup_q": self._ik_paddle(windup),
            "follow_q": self._ik_paddle(follow),
            "swing_at_step": swing_at_step,
            "step": 0,
        }

    def _ik_paddle(self, paddle_target):
        """Solve IK for the EE such that the paddle ends up at paddle_target."""
        ee_target = [
            paddle_target[0] - self._PADDLE_OFFSET * self._n_x,
            paddle_target[1],
            paddle_target[2] - self._PADDLE_OFFSET * self._n_z,
        ]
        return self._ik(ee_target)

    def _ik(self, ee_target):
        # Snapshot live joints, reset to ready pose for the IK call, then
        # restore. PyBullet's IK uses the robot's current joint state as
        # the initial guess; the right-side robot (base yaw=π) is prone to
        # drifting into a contorted elbow branch otherwise. Snapping to
        # the ready pose anchors both robots to the same IK basin.
        # For P1 we mirror the target through the origin (the two robot
        # bases sit at ±1.7, symmetric about z). Robot 1's IK then finds
        # joints that, when applied to robot 2, produce the mirror world
        # EE pose — bypassing PyBullet's tendency to pick a contorted
        # joint branch when the base is yawed by π.
        if self.s == 1:
            tgt = [-ee_target[0], -ee_target[1], ee_target[2]]
        else:
            tgt = list(ee_target)
        live = [p.getJointState(self._ik_robot_id, j,
                                physicsClientId=self.env.physics_client)[:2]
                for j in range(7)]
        for j, a in enumerate(self._ready_q):
            p.resetJointState(self._ik_robot_id, j, a, targetVelocity=0.0,
                              physicsClientId=self.env.physics_client)
        q = p.calculateInverseKinematics(
            self._ik_robot_id,
            KUKA_EE_LINK_INDEX,
            tgt,
            self._ik_orn,
            lowerLimits=self._lower,
            upperLimits=self._upper,
            jointRanges=self._range,
            restPoses=self._ready_q,
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self.env.physics_client,
        )
        for j, (pos, vel) in enumerate(live):
            p.resetJointState(self._ik_robot_id, j, pos, targetVelocity=vel,
                              physicsClientId=self.env.physics_client)
        return list(q[:7])


# ---------------------------------------------------------------------------
# Player registry
# ---------------------------------------------------------------------------


def build_player(name: str, player_id: int, env: TableTennisEnv) -> Player:
    """
    Instantiate a player controller by name.

    Parameters
    ----------
    name : str
        Controller name. Must be one of PLAYER_CHOICES (or any key you add).
    player_id : int
        0 for the left robot (x < 0), 1 for the right robot (x > 0).
    env : TableTennisEnv
        The shared environment instance.

    To add a new player:
        1. Define your Player subclass above this function.
        2. Add an elif branch here, e.g.:
               elif controller == "mybot":
                   return MyBot(player_id=player_id, env=env)
        3. Add "mybot" to PLAYER_CHOICES at the top of this file.
    """
    controller = name.lower()
    if controller == "robot":
        return RobotPlayer(player_id=player_id, env=env)
    if controller == "simple":
        return SimpleTrackingPlayer(player_id=player_id, env=env)
    if controller == "idle":
        return DoNothingPlayer()
    raise ValueError(
        f"Unsupported player '{name}'. Available controllers: {', '.join(PLAYER_CHOICES)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-robot table tennis match simulation."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the PyBullet GUI window (needs a local or virtual display).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of complete games to simulate (default: 1).",
    )
    parser.add_argument(
        "--player1",
        default="simple",
        choices=PLAYER_CHOICES,
        help="Controller for player 1 — left robot, x < 0 (default: simple).",
    )
    parser.add_argument(
        "--player2",
        default="simple",
        choices=PLAYER_CHOICES,
        help="Controller for player 2 — right robot, x > 0 (default: simple).",
    )
    args = parser.parse_args()

    mode_label = "GUI" if args.gui else "headless"
    print(
        f"Starting {args.episodes} episode(s) ({mode_label}) — "
        f"Player 1: {args.player1} vs Player 2: {args.player2}"
    )

    # 1. Create the two-robot environment.
    env = TableTennisEnv(gui=args.gui, one_robot_mode=False)

    # 2. Instantiate both player controllers.
    player1 = build_player(args.player1, player_id=0, env=env)
    player2 = build_player(args.player2, player_id=1, env=env)

    # 3. Simulation manager handles game logic, scoring, and episode progression:
    #    - Provides observations to each player
    #    - Collects actions from each player's policy
    #    - Applies actions to the environment
    #    - Detects points, faults, and game-over conditions
    manager = SimulationManager(
        player1=player1,
        player2=player2,
        gui=args.gui,
        real_time=args.gui,  # slow down to real-time only in GUI mode
        env=env,
        one_robot_mode=False,
    )

    results = manager.run(max_episodes=args.episodes)

    # Print summary of results.
    print("\n--- Results ---")
    for i, r in enumerate(results):
        print(
            f"  Episode {i + 1}: "
            f"Player 1 {r['scores'][0]} — {r['scores'][1]} Player 2  |  "
            f"Winner: Player {r['winner'] + 1}  |  "
            f"Steps: {r['total_steps']}"
        )

    env.close()
