"""Arm planning demo with UR5e and Franka Panda.

Demonstrates the Arm class with IK and motion planning using arm factories:
  1. Create arms with one-line factory calls
  2. IK: solve for target EE poses
  3. FK: verify IK solutions
  4. Motion planning: plan_to_configuration, plan_to_pose
  5. Trajectory retiming

Usage:
    cd mj_manipulator
    uv run python demos/arm_planning.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from mj_environment import Environment
from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import UR5E_HOME, create_ur5e_arm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
MENAGERIE = WORKSPACE / "mujoco_menagerie"
UR5E_SCENE = MENAGERIE / "universal_robots_ur5e" / "scene.xml"
FRANKA_SCENE = MENAGERIE / "franka_emika_panda" / "scene.xml"


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Factory + state queries
# ---------------------------------------------------------------------------
def demo_factory(arm, home_q, label):
    """Show that arm factories produce working Arm instances."""
    print_header(f"{label} - Factory ({arm.dof}-DOF)")

    print(f"\n  Joint names: {arm.config.joint_names}")
    print(f"  DOF:         {arm.dof}")
    print(f"  Has IK:      {arm.ik_solver is not None}")

    q = arm.get_joint_positions()
    print(f"  Current q:   {np.array2string(q, precision=4)}")

    pose = arm.get_ee_pose()
    print(f"  EE position: {np.array2string(pose[:3, 3], precision=4)}")

    lower, upper = arm.get_joint_limits()
    print(f"  Joint range: [{lower.min():.2f}, {upper.max():.2f}] rad")


# ---------------------------------------------------------------------------
# Demo 2: IK round-trip
# ---------------------------------------------------------------------------
def demo_ik(arm, label):
    """Solve IK for the current EE pose and verify via FK."""
    print_header(f"{label} - IK Round-Trip")

    pose = arm.get_ee_pose()
    pos = pose[:3, 3]
    print(f"\n  Target EE pos: {np.array2string(pos, precision=4)}")

    solutions = arm.ik_solver.solve_valid(pose)
    print(f"  IK solutions:  {len(solutions)}")

    if not solutions:
        print("  WARNING: No IK solutions found!")
        return

    # Verify best solution via FK
    best_err = float("inf")
    best_q = None
    for q in solutions:
        fk_pose = arm.forward_kinematics(q)
        err = np.linalg.norm(fk_pose[:3, 3] - pos)
        if err < best_err:
            best_err = err
            best_q = q

    print(f"  Best FK error: {best_err*1000:.3f} mm")
    print(f"  Best q:        {np.array2string(best_q, precision=4)}")


# ---------------------------------------------------------------------------
# Demo 3: Motion planning
# ---------------------------------------------------------------------------
def demo_planning(arm, q_goal, label):
    """Plan from current config to a goal configuration."""
    print_header(f"{label} - Motion Planning")

    q_start = arm.get_joint_positions()
    print(f"\n  Start: {np.array2string(q_start, precision=4)}")
    print(f"  Goal:  {np.array2string(q_goal, precision=4)}")

    path = arm.plan_to_configuration(q_goal, timeout=10.0, seed=42)

    if path is None:
        print("  Planning failed (no collision-free path found)")
        return

    print(f"  Path:  {len(path)} waypoints")

    # Show trajectory retiming
    traj = arm.plan_trajectory(q_goal, timeout=10.0, seed=42)
    if traj is not None:
        print(f"\n  Trajectory duration: {traj.duration:.3f} s")
        print(f"  Trajectory samples:  {len(traj.positions)}")


# ---------------------------------------------------------------------------
# Demo 4: Plan to pose (IK + planning)
# ---------------------------------------------------------------------------
def demo_plan_to_pose(arm, label):
    """Plan to a target EE pose via IK + motion planning."""
    print_header(f"{label} - Plan to Pose (IK + Planning)")

    # Create a target pose: current pose shifted 5cm in +X
    current_pose = arm.get_ee_pose()
    target_pose = current_pose.copy()
    target_pose[0, 3] += 0.05

    print(f"\n  Current EE: {np.array2string(current_pose[:3, 3], precision=4)}")
    print(f"  Target EE:  {np.array2string(target_pose[:3, 3], precision=4)}")

    path = arm.plan_to_pose(target_pose, timeout=10.0, seed=42)

    if path is None:
        print("  Plan to pose failed (IK or planning failed)")
        return

    print(f"  Path: {len(path)} waypoints")

    # Verify final waypoint FK
    fk_final = arm.forward_kinematics(path[-1])
    pos_err = np.linalg.norm(fk_final[:3, 3] - target_pose[:3, 3])
    print(f"  Final FK error: {pos_err*1000:.3f} mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MENAGERIE.exists():
        print(f"ERROR: mujoco_menagerie not found at {MENAGERIE}")
        sys.exit(1)

    # --- UR5e ---
    ur5e_env = Environment(str(UR5E_SCENE))
    ur5e = create_ur5e_arm(ur5e_env)
    for i, idx in enumerate(ur5e.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

    demo_factory(ur5e, UR5E_HOME, "UR5e")
    demo_ik(ur5e, "UR5e")

    ur5e_goal = UR5E_HOME + np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.0])
    demo_planning(ur5e, ur5e_goal, "UR5e")
    demo_plan_to_pose(ur5e, "UR5e")

    # --- Franka ---
    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)
    franka_dir = FRANKA_SCENE.parent
    tmp_path = franka_dir / "_demo_franka_ee.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        franka_env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    franka = create_franka_arm(franka_env)
    for i, idx in enumerate(franka.joint_qpos_indices):
        franka_env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(franka_env.model, franka_env.data)

    demo_factory(franka, FRANKA_HOME, "Franka Panda")
    demo_ik(franka, "Franka Panda")

    franka_goal = FRANKA_HOME + np.array([0.2, -0.1, 0.0, 0.3, 0.0, -0.2, 0.1])
    demo_planning(franka, franka_goal, "Franka Panda")
    demo_plan_to_pose(franka, "Franka Panda")

    print_header(
        "DONE - Same Arm API with IK + planning\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
