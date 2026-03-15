"""Ready-to-use arm definitions for supported robots.

Provides factory functions that create fully configured Arm instances
with IK solvers, joint limits, and kinematic constants.

Usage:
    from mj_environment import Environment
    from mj_manipulator.arms.ur5e import create_ur5e_arm

    env = Environment("path/to/ur5e/scene.xml")
    arm = create_ur5e_arm(env)
    path = arm.plan_to_pose(target_pose)
"""

from mj_manipulator.arms.franka import add_franka_ee_site, create_franka_arm
from mj_manipulator.arms.ur5e import create_ur5e_arm

__all__ = [
    "create_ur5e_arm",
    "create_franka_arm",
    "add_franka_ee_site",
]
