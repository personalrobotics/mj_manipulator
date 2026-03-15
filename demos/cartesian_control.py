"""Cartesian velocity control demo with UR5e and Franka Panda.

Demonstrates that the same QP-based twist controller works with both
6-DOF and 7-DOF arms using real robot models from mujoco_menagerie.

Shows three capabilities:
  1. Jacobian computation and rank analysis
  2. Cartesian twist to joint velocities (QP solver)
  3. Multi-step trajectory following via step_twist

Usage:
    cd mj_manipulator
    uv run python demos/cartesian_control.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from mj_manipulator.cartesian import (
    CartesianControlConfig,
    get_ee_jacobian,
    step_twist,
    twist_to_joint_velocity,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
MENAGERIE = WORKSPACE / "mujoco_menagerie"

UR5E_SCENE = MENAGERIE / "universal_robots_ur5e" / "scene.xml"
FRANKA_SCENE = MENAGERIE / "franka_emika_panda" / "scene.xml"

# ---------------------------------------------------------------------------
# Robot definitions
# ---------------------------------------------------------------------------
UR5E_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
UR5E_HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
UR5E_VEL_LIMITS = np.array([2.094, 2.094, 3.14, 3.14, 3.14, 3.14])

FRANKA_JOINTS = [f"joint{i}" for i in range(1, 8)]
FRANKA_HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
FRANKA_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_arm(scene_path, joint_names, home_config, ee_site_name):
    """Load model, set to home, return (model, data, qpos_idx, qvel_idx, ee_id)."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    qpos_idx, qvel_idx = [], []
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])
        data.qpos[model.jnt_qposadr[jid]] = home_config[i]

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    mujoco.mj_forward(model, data)
    return model, data, qpos_idx, qvel_idx, ee_id


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Jacobian analysis
# ---------------------------------------------------------------------------
def demo_jacobian(model, data, qvel_idx, ee_id, label, dof):
    """Analyze the Jacobian at current configuration."""
    print_header(f"{label} - Jacobian Analysis ({dof}-DOF)")

    J = get_ee_jacobian(model, data, ee_id, qvel_idx)
    print(f"\n  Jacobian shape: {J.shape}")
    print(f"  Jacobian rank:  {np.linalg.matrix_rank(J, tol=1e-6)}")
    print(f"  Frobenius norm: {np.linalg.norm(J):.4f}")

    # Singular values reveal manipulability
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  Singular values: {np.array2string(sv, precision=4)}")
    manipulability = float(np.prod(sv[:min(6, dof)]))
    print(f"  Manipulability:  {manipulability:.6f}")

    if dof == 7:
        print(f"  Null space dim:  1 (redundant arm)")
    elif dof == 6:
        print(f"  Null space dim:  0 (fully actuated at this config)")


# ---------------------------------------------------------------------------
# Demo 2: QP solver comparison
# ---------------------------------------------------------------------------
def demo_qp_solver(model, data, qpos_idx, qvel_idx, ee_id, vel_limits, label, dof):
    """Compare QP solver results for different twists."""
    print_header(f"{label} - QP Solver ({dof}-DOF)")

    J = get_ee_jacobian(model, data, ee_id, qvel_idx)
    q_current = np.array([data.qpos[i] for i in qpos_idx])
    q_min = -np.ones(dof) * 6.28
    q_max = np.ones(dof) * 6.28

    twists = {
        "X linear (5cm/s)": np.array([0.05, 0, 0, 0, 0, 0]),
        "Z linear (5cm/s)": np.array([0, 0, -0.05, 0, 0, 0]),
        "Y rotation (0.2r/s)": np.array([0, 0, 0, 0, 0.2, 0]),
        "Combined": np.array([0.03, 0, -0.02, 0, 0.1, 0]),
    }

    print(f"\n  {'Twist':<22s}  {'Achieved%':>9s}  {'Error':>8s}  {'Limiter':>12s}  "
          f"{'||qd||':>8s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*8}  {'-'*12}  {'-'*8}")

    for name, twist in twists.items():
        result = twist_to_joint_velocity(
            J=J, twist=twist, q_current=q_current,
            q_min=q_min, q_max=q_max, qd_max=vel_limits,
            dt=0.004,
        )
        limiter = result.limiting_factor or "none"
        qd_norm = np.linalg.norm(result.joint_velocities)
        print(
            f"  {name:<22s}  {result.achieved_fraction:>8.1%}  "
            f"{result.twist_error:>8.5f}  {limiter:>12s}  "
            f"{qd_norm:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Demo 3: Multi-step trajectory via step_twist
# ---------------------------------------------------------------------------
def demo_step_twist(model, data, qpos_idx, qvel_idx, ee_id, vel_limits, label, dof):
    """Execute multiple twist steps and show EE motion."""
    print_header(f"{label} - Step Twist Trajectory ({dof}-DOF)")

    q_min = -np.ones(dof) * 6.28
    q_max = np.ones(dof) * 6.28
    dt = 0.004

    # Straight-line motion: 5cm/s in -Z for 20 steps (= 4mm total)
    twist = np.array([0, 0, -0.05, 0, 0, 0])
    n_steps = 20

    ee_start = data.site_xpos[ee_id].copy()
    q_dot_prev = None

    fractions = []
    for i in range(n_steps):
        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=q_min, q_max=q_max, qd_max=vel_limits,
            twist=twist, dt=dt, q_dot_prev=q_dot_prev,
        )
        q_dot_prev = result.joint_velocities

        # Apply to model
        for j, idx in enumerate(qpos_idx):
            data.qpos[idx] = q_new[j]
        mujoco.mj_forward(model, data)

        fractions.append(result.achieved_fraction)

    ee_end = data.site_xpos[ee_id].copy()
    displacement = ee_end - ee_start
    distance = np.linalg.norm(displacement)

    print(f"\n  Direction: -Z (downward) at 5 cm/s")
    print(f"  Steps:     {n_steps} x {dt*1000:.0f}ms = {n_steps*dt*1000:.0f}ms")
    print(f"\n  EE start:  {np.array2string(ee_start, precision=4)}")
    print(f"  EE end:    {np.array2string(ee_end, precision=4)}")
    print(f"  Delta:     {np.array2string(displacement, precision=5)}")
    print(f"  Distance:  {distance*1000:.2f} mm")
    expected = abs(twist[2]) * n_steps * dt
    print(f"  Expected:  {expected*1000:.2f} mm")
    print(f"  Tracking:  {distance/expected*100:.1f}%")
    print(f"  Avg frac:  {np.mean(fractions):.3f}")

    # Hand-frame demo
    print(f"\n  --- Hand frame test ---")
    # Reset to get consistent starting point
    ee_before_hand = data.site_xpos[ee_id].copy()
    twist_hand = np.array([0.05, 0, 0, 0, 0, 0])  # X in hand frame

    R = data.site_xmat[ee_id].reshape(3, 3)
    world_dir = R @ twist_hand[:3]
    print(f"  Hand-frame X twist: {twist_hand[:3]}")
    print(f"  Maps to world dir:  {np.array2string(world_dir, precision=4)}")

    q_world, r_world = step_twist(
        model, data, ee_id, qpos_idx, qvel_idx,
        q_min=q_min, q_max=q_max, qd_max=vel_limits,
        twist=twist_hand, frame="world", dt=dt,
    )
    q_hand, r_hand = step_twist(
        model, data, ee_id, qpos_idx, qvel_idx,
        q_min=q_min, q_max=q_max, qd_max=vel_limits,
        twist=twist_hand, frame="hand", dt=dt,
    )
    q_diff = np.linalg.norm(q_world - q_hand)
    print(f"  World vs hand frame joint diff: {q_diff:.6f}"
          f" ({'different' if q_diff > 1e-6 else 'same'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MENAGERIE.exists():
        print(f"ERROR: mujoco_menagerie not found at {MENAGERIE}")
        print(
            "Clone it:\n"
            "  cd robot-code\n"
            "  git clone https://github.com/google-deepmind/mujoco_menagerie"
        )
        sys.exit(1)

    # --- UR5e (6-DOF) ---
    ur5e = setup_arm(UR5E_SCENE, UR5E_JOINTS, UR5E_HOME, "attachment_site")
    model_u, data_u, qpos_u, qvel_u, ee_u = ur5e

    demo_jacobian(model_u, data_u, qvel_u, ee_u, "UR5e", 6)
    demo_qp_solver(model_u, data_u, qpos_u, qvel_u, ee_u, UR5E_VEL_LIMITS, "UR5e", 6)
    demo_step_twist(model_u, data_u, qpos_u, qvel_u, ee_u, UR5E_VEL_LIMITS, "UR5e", 6)

    # --- Franka Panda (7-DOF) ---
    # Franka doesn't have "attachment_site" — add one via MjSpec
    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    # Find hand body and add EE site
    hand = spec.worldbody.find_child("hand")
    site = hand.add_site()
    site.name = "ee_site"
    site.pos = [0, 0, 0.1034]  # Roughly at fingertip
    model_f = spec.compile()
    data_f = mujoco.MjData(model_f)

    qpos_f, qvel_f = [], []
    for i, name in enumerate(FRANKA_JOINTS):
        jid = mujoco.mj_name2id(model_f, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_f.append(model_f.jnt_qposadr[jid])
        qvel_f.append(model_f.jnt_dofadr[jid])
        data_f.qpos[model_f.jnt_qposadr[jid]] = FRANKA_HOME[i]
    ee_f = mujoco.mj_name2id(model_f, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    mujoco.mj_forward(model_f, data_f)

    demo_jacobian(model_f, data_f, qvel_f, ee_f, "Franka Panda", 7)
    demo_qp_solver(model_f, data_f, qpos_f, qvel_f, ee_f, FRANKA_VEL_LIMITS, "Franka Panda", 7)
    demo_step_twist(model_f, data_f, qpos_f, qvel_f, ee_f, FRANKA_VEL_LIMITS, "Franka Panda", 7)

    print_header(
        "DONE - Same Cartesian controller code\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
