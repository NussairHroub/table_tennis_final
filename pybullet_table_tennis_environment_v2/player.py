"""
player.py
---------
Abstract base class for table tennis players.

Students should:
  1. Create a new file (e.g. my_player.py)
  2. Import and subclass Player
  3. Implement get_action(observation)

Example
-------
    from player import Player

    class MyPlayer(Player):
        def get_action(self, observation):
            # Stay at home configuration
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
"""

from abc import ABC, abstractmethod


# KUKA iiwa 7 joint limits (radians)
JOINT_LIMITS = [
    (-2.9671, 2.9671),   # joint 1
    (-2.0944, 2.0944),   # joint 2
    (-2.9671, 2.9671),   # joint 3
    (-2.0944, 2.0944),   # joint 4
    (-2.9671, 2.9671),   # joint 5
    (-2.0944, 2.0944),   # joint 6
    (-3.0543, 3.0543),   # joint 7
]


class Player(ABC):
    """
    Abstract base class for table tennis players.

    Subclass this and implement get_action() to create your own player.

    The robot is a KUKA LBR iiwa 7 (7-DOF arm) with a paddle attached
    to its end-effector. You control it by specifying target joint angles.

    Coordinate system
    -----------------
    - X axis runs along the table length (net at x=0)
    - Y axis runs along the table width
    - Z axis points up (table surface at z=0.76 m)

    Robot placement
    ---------------
    - Player defending x < 0: robot base at [-2.0, 0, 0.5], facing +X
    - Player defending x > 0: robot base at [+2.0, 0, 0.5], facing -X
    The "player_side" key in table_info tells you which side you defend.
    """

    @abstractmethod
    def get_action(self, observation: dict) -> list:
        """
        Compute joint target angles given the current observation.

        Parameters
        ----------
        observation : dict
            "ball_position"    : list of 3 floats  [x, y, z] in meters
            "ball_velocity"    : list of 3 floats  [vx, vy, vz] in m/s
            "joint_positions"  : list of 7 floats  current joint angles (rad)
            "joint_velocities" : list of 7 floats  current joint velocities (rad/s)
            "table_info"       : dict
                "length"       : float  table length in meters (2.74)
                "width"        : float  table width in meters (1.525)
                "height"       : float  table surface height in meters (0.76)
                "net_height"   : float  top of net in meters (0.9125)
                "player_side"  : str    "negative_x" or "positive_x"
                                        The side of the table you defend.

        Returns
        -------
        list of 7 floats
            Target joint angles in radians.
            Approximate limits per joint:
              joints 0, 2, 4 : [-2.967,  +2.967] rad
              joints 1, 3, 5 : [-2.094,  +2.094] rad
              joint  6       : [-3.054,  +3.054] rad
            Values outside limits are automatically clipped by the simulation.
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode / point.

        Override this to reset any internal player state such as:
        - History buffers
        - Policy state
        - Planners or controllers

        Default implementation does nothing.
        """
        pass

    def clip_action(self, action: list) -> list:
        """
        Clip joint angles to KUKA iiwa hardware limits.

        Parameters
        ----------
        action : list of 7 floats

        Returns
        -------
        list of 7 floats, each clamped to its joint's [min, max] range.
        """
        return [
            max(lo, min(hi, float(a)))
            for a, (lo, hi) in zip(action, JOINT_LIMITS)
        ]
