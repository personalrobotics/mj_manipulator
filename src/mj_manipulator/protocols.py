"""Protocols defining the contracts between mj_manipulator and robot-specific packages.

Robot-specific packages (geodude, franka_control, xarm_control) implement
these protocols to plug into the generic manipulation framework.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Gripper(Protocol):
    """Protocol for gripper implementations.

    Each robot provides its own concrete gripper:
    - geodude: Robotiq2F140 (parallel jaw, 4-bar linkage)
    - franka_control: FrankaHand (parallel jaw, prismatic)
    - etc.
    """

    @property
    def arm_name(self) -> str:
        """Which arm this gripper belongs to."""
        ...

    @property
    def gripper_body_names(self) -> list[str]:
        """MuJoCo body names for contact detection and collision filtering."""
        ...

    @property
    def attachment_body(self) -> str:
        """MuJoCo body name that objects weld to during kinematic grasping.

        This is the body that makes contact with the object. For a parallel
        jaw gripper, typically one of the finger pads or follower links.
        """
        ...

    @property
    def actuator_id(self) -> int | None:
        """MuJoCo actuator ID for gripper control, or None if no actuator."""
        ...

    @property
    def ctrl_open(self) -> float:
        """Actuator control value for fully open position."""
        ...

    @property
    def ctrl_closed(self) -> float:
        """Actuator control value for fully closed position."""
        ...

    @property
    def is_holding(self) -> bool:
        """Whether the gripper is currently holding an object."""
        ...

    @property
    def held_object(self) -> str | None:
        """Name of the held object, or None."""
        ...

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        """Set which objects the gripper should try to grasp.

        Args:
            objects: List of object names to consider, or None for all.
        """
        ...

    def kinematic_close(self, steps: int = 50) -> str | None:
        """Close gripper in kinematic mode (no physics).

        Detects contact geometrically and returns the grasped object name.

        Args:
            steps: Number of interpolation steps for closing motion.

        Returns:
            Name of grasped object, or None if no object detected.
        """
        ...

    def kinematic_open(self) -> None:
        """Open gripper in kinematic mode."""
        ...

    def get_actual_position(self) -> float:
        """Get current gripper position (0.0 = fully open, 1.0 = fully closed)."""
        ...


@runtime_checkable
class IKSolver(Protocol):
    """Protocol for inverse kinematics solvers.

    Mirrors pycbirrt's IKSolver protocol. Each robot provides its own
    solver (analytical for UR5e/Franka, numerical for others).
    """

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK for a target pose (raw, may include invalid solutions).

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose.
            q_init: Optional initial configuration hint.

        Returns:
            List of joint configurations (may include out-of-limits or colliding).
        """
        ...

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Solve IK and return only collision-free, in-limits solutions.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose.
            q_init: Optional initial configuration hint.

        Returns:
            List of valid joint configurations (may be empty).
        """
        ...


@runtime_checkable
class GraspSource(Protocol):
    """Protocol for providing grasps and placements for objects.

    Robot-specific packages implement this to tell the manipulation
    primitives how to grasp and place objects. For example:
    - geodude: AffordanceRegistry (loads TSR templates from YAML)
    - A learning-based system could implement this with a neural grasp predictor
    """

    def get_grasps(self, object_name: str, hand_type: str) -> list:
        """Get grasp TSRs for an object.

        Args:
            object_name: Name of the object to grasp.
            hand_type: Gripper type string for affordance matching.

        Returns:
            List of TSR objects representing valid grasps.
        """
        ...

    def get_placements(self, destination: str, object_name: str) -> list:
        """Get placement TSRs for an object at a destination.

        Args:
            destination: Where to place the object.
            object_name: What object is being placed.

        Returns:
            List of TSR objects representing valid placements.
        """
        ...

    def get_graspable_objects(self) -> list[str]:
        """Get list of objects that can be grasped."""
        ...

    def get_place_destinations(self, object_name: str) -> list[str]:
        """Get valid placement destinations for an object."""
        ...
