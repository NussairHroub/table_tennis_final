"""
table_tennis_env.py
-------------------

Owns the physics client. Builds the scene (floor, table, net, two KUKA iiwa
robots with paddles, ball) and exposes a clean API used by SimulationManager.

All PyBullet calls use physicsClientId to support multiple simultaneous
simulation instances (e.g. for training).
"""

import math
import pybullet as p
import pybullet_data

from player import JOINT_LIMITS

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
SIM_TIMESTEP = 1.0 / 240.0

# ---------------------------------------------------------------------------
# Table (standard ITTF dimensions)
# ---------------------------------------------------------------------------
TABLE_LENGTH = 2.74     # m, along X axis
TABLE_WIDTH = 1.525     # m, along Y axis
TABLE_HEIGHT = 0.76     # m, top surface Z
TABLE_THICKNESS = 0.05  # m, slab thickness

# When False, robot links and paddles ignore the table so the advanced
# controller can operate without hard robot-table collisions.
TABLE_ROBOT_COLLISIONS_ENABLED = False

# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------
NET_WIDTH = 1.83       # m, along Y (slightly wider than table)
NET_HEIGHT = 0.1525     # m
NET_THICKNESS = 0.01       # m

# ---------------------------------------------------------------------------
# Ball (standard table tennis ball: 40 mm diameter, 2.7 g)
# ---------------------------------------------------------------------------
BALL_RADIUS = 0.02    # m
BALL_MASS = 0.0027    # kg
BALL_RESTITUTION = 0.87
BALL_FRICTION = 0.4
BALL_LINEAR_DAMPING  = 0.01
BALL_ANGULAR_DAMPING = 0.01
PADDLE_RESTITUTION = 0.93
PADDLE_FRICTION = 0.28

# ---------------------------------------------------------------------------
# Robots
# ---------------------------------------------------------------------------
ROBOT1_BASE_POS = [-1.9, 0.0, TABLE_HEIGHT - 0.1]
ROBOT2_BASE_POS = [1.9, 0.0, TABLE_HEIGHT - 0.1]
KUKA_EE_LINK_INDEX = 6     # lbr_iiwa_link_7 (child of joint_7)
NUM_JOINTS = 7

# ---------------------------------------------------------------------------
# Paddle
# ---------------------------------------------------------------------------
PADDLE_RADIUS = 0.08      # m
PADDLE_THICKNESS = 0.005  # m
PADDLE_MASS = 0.1         # kg
PADDLE_EE_OFFSET = [0.0, 0.0, 0.05]   # offset from EE frame origin (m)

# ---------------------------------------------------------------------------
# Joint control
# ---------------------------------------------------------------------------
MAX_JOINT_FORCE = 320.0
MAX_JOINT_VELOCITY = 12.0
LEGACY_ROBOT1_BASE_POS = [-1.7, 0.0, 0.5]
LEGACY_ROBOT2_BASE_POS = [1.7, 0.0, 0.5]

READY_JOINT_ANGLES_DEG = [0, 47, 0, -111, 0, -83, 47]
READY_JOINT_ANGLES_RAD = [
    max(lo, min(hi, math.radians(deg)))
    for deg, (lo, hi) in zip(READY_JOINT_ANGLES_DEG, JOINT_LIMITS)
]


class TableTennisEnv:
    """
    PyBullet-based table tennis physics environment.

    Parameters
    ----------
    gui : bool
        If True, open the PyBullet GUI. If False, run headless (DIRECT mode).
    """

    def __init__(
        self,
        gui: bool = False,
        one_robot_mode: bool = False,
        home_joint_angles: list | None = None,
        max_joint_force: float | None = None,
        max_joint_velocity: float | None = None,
        robot1_base_pos: list | None = None,
        robot2_base_pos: list | None = None,
        table_robot_collisions_enabled: bool | None = None,
        paddle_restitution: float | None = None,
        paddle_friction: float | None = None,
    ):
        self.gui = gui
        self.physics_client = None
        self.one_robot_mode = one_robot_mode
        self.home_joint_angles = (
            list(READY_JOINT_ANGLES_RAD)
            if home_joint_angles is None
            else [float(angle) for angle in home_joint_angles]
        )
        self.max_joint_force = (
            float(MAX_JOINT_FORCE)
            if max_joint_force is None
            else float(max_joint_force)
        )
        self.max_joint_velocity = (
            float(MAX_JOINT_VELOCITY)
            if max_joint_velocity is None
            else float(max_joint_velocity)
        )
        self.robot1_base_pos = (
            list(ROBOT1_BASE_POS)
            if robot1_base_pos is None
            else [float(v) for v in robot1_base_pos]
        )
        self.robot2_base_pos = (
            list(ROBOT2_BASE_POS)
            if robot2_base_pos is None
            else [float(v) for v in robot2_base_pos]
        )
        self.table_robot_collisions_enabled = (
            TABLE_ROBOT_COLLISIONS_ENABLED
            if table_robot_collisions_enabled is None
            else bool(table_robot_collisions_enabled)
        )
        self.paddle_restitution = (
            float(PADDLE_RESTITUTION)
            if paddle_restitution is None
            else float(paddle_restitution)
        )
        self.paddle_friction = (
            float(PADDLE_FRICTION)
            if paddle_friction is None
            else float(paddle_friction)
        )

        # Body IDs (set during _build_scene)
        self.floor_id   = None
        self.table_id   = None
        self.net_id     = None
        self.robot1_id  = None
        self.robot2_id  = None
        self.paddle1_id = None
        self.paddle2_id = None
        self._paddle1_constraint = None
        self._paddle2_constraint = None
        self.ball_id    = None

        self._connect()
        self._build_scene()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=4.5,
                cameraYaw=30,
                cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.0, 0.8],
                physicsClientId=self.physics_client,
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.physics_client,
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(SIM_TIMESTEP, physicsClientId=self.physics_client)

    def _build_scene(self):
        self.floor_id   = self._create_floor()
        self.table_id   = self._create_table()
        self.net_id     = self._create_net()
        self.robot1_id, self.robot2_id = self._create_robots()
        self.paddle1_id, self._paddle1_constraint = self._attach_paddle(self.robot1_id)
        self.paddle2_id, self._paddle2_constraint = self._attach_paddle(self.robot2_id)
        self._configure_table_robot_collisions()
        self.ball_id    = self._create_ball()

    def _create_floor(self) -> int:
        floor = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.physics_client,
        )
        return floor

    def _create_table(self) -> int:
        """Build a table, set collision anv visual"""
        half_l = TABLE_LENGTH    / 2  
        half_w = TABLE_WIDTH     / 2  
        half_t = TABLE_THICKNESS / 2  
        # Top surface center Z = TABLE_HEIGHT - half_t 
        center_z = TABLE_HEIGHT - half_t

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_l, half_w, half_t],
            physicsClientId=self.physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_l, half_w, half_t],
            rgbaColor=[0.05, 0.45, 0.15, 1.0],
            physicsClientId=self.physics_client,
        )
        table = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, center_z],
            physicsClientId=self.physics_client,
        )
        # Change dynamics to make the table more bouncy and less slippery
        p.changeDynamics(
            table, -1,
            restitution=BALL_RESTITUTION,
            lateralFriction=0.6, # higher friction helps prevent excessive sliding after ball-table contact
            physicsClientId=self.physics_client,
        )

        # White center line (visual only, zero collision)
        line_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.005, half_w, 0.001],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=self.physics_client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=line_vis,
            basePosition=[0.0, 0.0, TABLE_HEIGHT + 0.001],
            physicsClientId=self.physics_client,
        )

        # White edge lines along the length (visual only)
        for y_sign in [-1, 1]:
            edge_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half_l, 0.003, 0.001],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=self.physics_client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=edge_vis,
                basePosition=[0.0, y_sign * half_w, TABLE_HEIGHT + 0.001],
                physicsClientId=self.physics_client,
            )

        return table

    def _create_net(self) -> int:
        half_n = NET_HEIGHT    / 2   # 0.07625
        half_w = NET_WIDTH     / 2   # 0.915
        half_t = NET_THICKNESS / 2   # 0.005
        # Net bottom flush with table top → center_z = TABLE_HEIGHT + half_n
        net_z = TABLE_HEIGHT + half_n   # 0.83625

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_t, half_w, half_n],
            physicsClientId=self.physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_t, half_w, half_n],
            rgbaColor=[0.9, 0.9, 0.9, 0.85],
            physicsClientId=self.physics_client,
        )
        net = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, net_z],
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            net, -1,
            restitution=0.5,
            lateralFriction=0.5,
            physicsClientId=self.physics_client,
        )
        return net

    def _create_robots(self):
        q1 = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        q2 = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
        flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT # allow collisions between the two robots, but not between a robot and itself

        robots = []
        r1 = p.loadURDF(
            "kuka_iiwa/model.urdf",
            self.robot1_base_pos, q1,
            flags=flags,
            physicsClientId=self.physics_client,
        )
        robots.append(r1)
        if not self.one_robot_mode:
            r2 = p.loadURDF(
                "kuka_iiwa/model.urdf",
                self.robot2_base_pos, q2,
                flags=flags,
                physicsClientId=self.physics_client,
            )
            robots.append(r2)

        # TODO: Allow or disable the velocity damping in the joints.
        # Disable built-in velocity damping so POSITION_CONTROL works cleanly
        for rid in robots:
            for j in range(NUM_JOINTS):
                p.setJointMotorControl2(
                    rid, j,
                    p.VELOCITY_CONTROL,
                    force=0,
                    physicsClientId=self.physics_client,
                )
        return (r1, r2) if not self.one_robot_mode else (r1, None)

    def _attach_paddle(self, robot_id: int):
        """Attach a flat disk (paddle) rigidly to the KUKA end-effector."""
        # TODO: Instead of a simple cylinder create a paddle with two cylinders, one for the handle and one for the paddle face.
        
        if robot_id is None:
            return None, None
        
        # NOTE: GEOM_CYLINDER uses 'height' in createCollisionShape
        #       but 'length' in createVisualShape — different keyword names.
        paddle_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=PADDLE_RADIUS,
            height=PADDLE_THICKNESS,
            physicsClientId=self.physics_client,
        )
        paddle_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=PADDLE_RADIUS,
            length=PADDLE_THICKNESS,
            rgbaColor=[0.75, 0.1, 0.1, 1.0],
            physicsClientId=self.physics_client,
        )

        # Place the paddle at the world pose implied by the rigid EE offset.
        ee_state = p.getLinkState(
            robot_id, KUKA_EE_LINK_INDEX,
            physicsClientId=self.physics_client,
        )
        paddle_pos, paddle_orn = self._paddle_pose_from_link_state(ee_state)

        paddle = p.createMultiBody(
            baseMass=PADDLE_MASS,
            baseCollisionShapeIndex=paddle_col,
            baseVisualShapeIndex=paddle_vis,
            basePosition=paddle_pos,
            baseOrientation=paddle_orn,
            physicsClientId=self.physics_client,
        )
        p.changeDynamics(
            paddle, -1,
            restitution=self.paddle_restitution,
            lateralFriction=self.paddle_friction,
            # CCD prevents ball from tunneling through paddle
            ccdSweptSphereRadius=0.04, # This is the radius of the sphere ball
            physicsClientId=self.physics_client,
        )

        # Fixed constraint: paddle follows the EE rigidly
        constraint = p.createConstraint(
            robot_id,          KUKA_EE_LINK_INDEX,   # parent
            paddle,            -1,                    # child (base)
            p.JOINT_FIXED,
            [0.0, 0.0, 1.0],                          # joint axis (unused for FIXED)
            PADDLE_EE_OFFSET,                          # parent frame position
            [0.0, 0.0, 0.0],                           # child frame position
            physicsClientId=self.physics_client,
        )
        
        # Set max force to prevent excesive forces when ball hits the paddle while the robot is trying to hold it in place
        p.changeConstraint(
            constraint,
            maxForce=500,
            physicsClientId=self.physics_client,
        )
        return paddle, constraint

    def _paddle_pose_from_link_state(self, link_state):
        """Return the paddle pose implied by the EE link pose plus offset."""
        link_pos = list(link_state[0])
        link_orn = list(link_state[1])
        rot = p.getMatrixFromQuaternion(link_orn)
        offset = [
            rot[0] * PADDLE_EE_OFFSET[0] + rot[1] * PADDLE_EE_OFFSET[1] + rot[2] * PADDLE_EE_OFFSET[2],
            rot[3] * PADDLE_EE_OFFSET[0] + rot[4] * PADDLE_EE_OFFSET[1] + rot[5] * PADDLE_EE_OFFSET[2],
            rot[6] * PADDLE_EE_OFFSET[0] + rot[7] * PADDLE_EE_OFFSET[1] + rot[8] * PADDLE_EE_OFFSET[2],
        ]
        paddle_pos = [
            link_pos[0] + offset[0],
            link_pos[1] + offset[1],
            link_pos[2] + offset[2],
        ]
        return paddle_pos, link_orn

    def _create_ball(self) -> int:
        ball_col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=BALL_RADIUS,
            physicsClientId=self.physics_client,
        )
        ball_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=BALL_RADIUS,
            rgbaColor=[1.0, 0.95, 0.8, 1.0],
            physicsClientId=self.physics_client,
        )
        ball = p.createMultiBody(
            baseMass=BALL_MASS,
            baseCollisionShapeIndex=ball_col,
            baseVisualShapeIndex=ball_vis,
            basePosition=[0.0, 0.0, 1.5],
            physicsClientId=self.physics_client,
        )
        
        # Here we configure the ball dynamics to bounce and not be slippery.
        # Also, configure CCD to prevent tunneling through the table, net, or paddles at high speed.
        # Default restitution is small, not bouncing
        
        p.changeDynamics(
            ball, -1,
            restitution=BALL_RESTITUTION, # make the ball bouncy
            lateralFriction=BALL_FRICTION,  # make the ball less slippery
            linearDamping=BALL_LINEAR_DAMPING, # small linear damping to prevent perpetual motion
            angularDamping=BALL_ANGULAR_DAMPING, # small angular damping to prevent perpetual spinning
            # CCD prevents tunneling through net/table at high speed
            ccdSweptSphereRadius=0.015, # slightly smaller than the ball radius
            physicsClientId=self.physics_client,
        )
        return ball

    def _configure_table_robot_collisions(self) -> None:
        """Enable or disable collisions between the table and robot hardware."""
        enable_collision = 1 if self.table_robot_collisions_enabled else 0
        robot_bodies = [self.robot1_id, self.robot2_id]
        paddle_bodies = [self.paddle1_id, self.paddle2_id]

        for robot_id in robot_bodies:
            if robot_id is None:
                continue
            for link_index in range(-1, NUM_JOINTS):
                p.setCollisionFilterPair(
                    robot_id,
                    self.table_id,
                    linkIndexA=link_index,
                    linkIndexB=-1,
                    enableCollision=enable_collision,
                    physicsClientId=self.physics_client,
                )

        for paddle_id in paddle_bodies:
            if paddle_id is None:
                continue
            p.setCollisionFilterPair(
                paddle_id,
                self.table_id,
                linkIndexA=-1,
                linkIndexB=-1,
                enableCollision=enable_collision,
                physicsClientId=self.physics_client,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, ball_position=None, ball_velocity=None) -> None:
        """
        Reset robots to home position and place the ball.

        Parameters
        ----------
        ball_position : list of 3 floats, optional
            World position [x, y, z] to place the ball. Default: [0, 0, 1.5].
        ball_velocity : list of 3 floats, optional
            Initial linear velocity [vx, vy, vz] of the ball. Default: [0, 0, 0].
        """
        if ball_position is None:
            ball_position = [0.0, 0.0, 1.5]
        if ball_velocity is None:
            ball_velocity = [0.0, 0.0, 0.0]

        home = list(self.home_joint_angles)

        # Reset joint states for both robots
        for rid in [self.robot1_id, self.robot2_id]:
            if rid is None:
                continue
            for j in range(NUM_JOINTS):
                p.resetJointState(
                    rid, j, home[j],
                    physicsClientId=self.physics_client,
                )

        # Reposition paddles so constraint doesn't produce a large impulse
        for rid, pad_id in [(self.robot1_id, self.paddle1_id),
                            (self.robot2_id, self.paddle2_id)]:
            if rid is None or pad_id is None:
                continue
            ee_state = p.getLinkState(
                rid, KUKA_EE_LINK_INDEX,
                physicsClientId=self.physics_client,
            )
            paddle_pos, paddle_orn = self._paddle_pose_from_link_state(ee_state)
            p.resetBasePositionAndOrientation(
                pad_id, paddle_pos, paddle_orn,
                physicsClientId=self.physics_client,
            )

        # Settle constraints (5 steps with zero forces)
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.physics_client)

        # Place ball
        p.resetBasePositionAndOrientation(
            self.ball_id,
            ball_position,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.ball_id,
            linearVelocity=ball_velocity,
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )

    def step(self, action1: list, action2: list) -> None:
        """
        Apply joint position targets to both robots and advance physics by one step.

        Parameters
        ----------
        action1 : list of 7 floats — target joint angles for robot 1 (radians)
        action2 : list of 7 floats — target joint angles for robot 2 (radians)
        """
        self._apply_joint_control(self.robot1_id, action1)
        if self.robot2_id is not None:
            self._apply_joint_control(self.robot2_id, action2)
        p.stepSimulation(physicsClientId=self.physics_client)

    def get_observation(self, player_id: int) -> dict:
        """
        Build the observation dict for a player.

        Parameters
        ----------
        player_id : int — 0 for the robot at x=-2 (defends x<0),
                          1 for the robot at x=+2 (defends x>0)

        Returns
        -------
        dict matching the schema in player.py
        """
        robot_id = self.robot1_id if player_id == 0 else self.robot2_id

        ball_pos, _ = p.getBasePositionAndOrientation(
            self.ball_id, physicsClientId=self.physics_client,
        )
        ball_vel, _ = p.getBaseVelocity(
            self.ball_id, physicsClientId=self.physics_client,
        )

        if robot_id is None:
            return {
                "ball_position":    list(ball_pos),
                "ball_velocity":    list(ball_vel),
                "joint_positions":  [0.0] * NUM_JOINTS,
                "joint_velocities": [0.0] * NUM_JOINTS,
                "table_info": {
                    "length":      TABLE_LENGTH,
                    "width":       TABLE_WIDTH,
                    "height":      TABLE_HEIGHT,
                    "net_height":  TABLE_HEIGHT + NET_HEIGHT,
                    "player_side": "positive_x",
                },
            }

        joint_pos, joint_vel = self._get_joint_states(robot_id)

        return {
            "ball_position":    list(ball_pos),
            "ball_velocity":    list(ball_vel),
            "joint_positions":  joint_pos,
            "joint_velocities": joint_vel,
            "table_info": {
                "length":      TABLE_LENGTH,
                "width":       TABLE_WIDTH,
                "height":      TABLE_HEIGHT,
                "net_height":  TABLE_HEIGHT + NET_HEIGHT,   # 0.9125 m
                # "negative_x": player defends x < 0 (robot at x=-2)
                # "positive_x": player defends x > 0 (robot at x=+2)
                "player_side": "negative_x" if player_id == 0 else "positive_x",
            },
        }

    def get_ball_state(self) -> dict:
        """Return current ball position, velocity, and orientation."""
        pos, orn = p.getBasePositionAndOrientation(
            self.ball_id, physicsClientId=self.physics_client,
        )
        vel, ang_vel = p.getBaseVelocity(
            self.ball_id, physicsClientId=self.physics_client,
        )
        return {
            "position":         list(pos),
            "velocity":         list(vel),
            "angular_velocity": list(ang_vel),
            "orientation":      list(orn),
        }

    def get_contact_with_table(self) -> list:
        """Return contact points between ball and table."""
        return p.getContactPoints(
            self.ball_id, self.table_id,
            physicsClientId=self.physics_client,
        )

    def get_contact_with_net(self) -> list:
        """Return contact points between ball and net."""
        return p.getContactPoints(
            self.ball_id, self.net_id,
            physicsClientId=self.physics_client,
        )

    def get_contact_with_paddle(self, player_id: int) -> list:
        """Return contact points between ball and the specified player's paddle."""
        paddle_id = self.paddle1_id if player_id == 0 else self.paddle2_id
        if paddle_id is None:
            return []
        return p.getContactPoints(
            self.ball_id, paddle_id,
            physicsClientId=self.physics_client,
        )

    def set_ball_state(self, position: list, velocity: list) -> None:
        """Teleport the ball to a given position with a given velocity."""
        p.resetBasePositionAndOrientation(
            self.ball_id, position, [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.ball_id,
            linearVelocity=velocity,
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )

    def close(self) -> None:
        """Disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_joint_control(self, robot_id: int, target_angles: list) -> None:
        if robot_id is None:
            return
        for j in range(NUM_JOINTS):
            p.setJointMotorControl2(
                robot_id, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_angles[j]),
                force=self.max_joint_force,
                maxVelocity=self.max_joint_velocity,
                positionGain=0.52,
                velocityGain=1.0,
                physicsClientId=self.physics_client,
            )

    def _get_joint_states(self, robot_id: int):
        """Return (positions, velocities) each as a list of 7 floats."""
        states = p.getJointStates(
            robot_id,
            list(range(NUM_JOINTS)),
            physicsClientId=self.physics_client,
        )
        positions  = [s[0] for s in states]
        velocities = [s[1] for s in states]
        return positions, velocities
