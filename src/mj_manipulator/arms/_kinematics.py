# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Kinematics read from a live MuJoCo model, shared by the IK solvers.

``_extract_hp`` gives the product-of-exponentials parameters of an arm chain (joint axes ``H`` and offsets ``P``, in
the base frame at q = 0), the convention of both ``ssik.Manipulator.from_axes`` and EAIK's ``HPRobot``.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _extract_hp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    joint_qpos_indices: list[int],
    ee_site_id: int,
    base_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract H (joint axes) and P (position offsets) from a MuJoCo model.

    Sets joints to zeros, runs FK, and reads axes/positions in the base
    body's frame. The resulting H/P can be passed directly to EAIK's HPRobot.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (not modified — uses a temporary copy).
        joint_ids: MuJoCo joint IDs for the arm.
        joint_qpos_indices: qpos indices for the arm joints.
        ee_site_id: MuJoCo site ID for the end-effector.
        base_body_id: Body ID of the arm base.

    Returns:
        (H, P, ee_rot) where H is (n_joints, 3) joint axes, P is (n_joints+1, 3)
        position offsets, and ee_rot is (3, 3) orientation of the EE site in the
        base frame at q=zeros. All values are in the base body frame.
    """
    n_joints = len(joint_ids)

    # Use a temporary MjData to avoid disturbing live state
    tmp_data = mujoco.MjData(model)
    np.copyto(tmp_data.qpos, data.qpos)
    for idx in joint_qpos_indices:
        tmp_data.qpos[idx] = 0.0
    mujoco.mj_forward(model, tmp_data)

    # Base frame
    base_pos = tmp_data.xpos[base_body_id].copy()
    base_rot_inv = tmp_data.xmat[base_body_id].reshape(3, 3).T

    # Joint axes and positions in world frame
    H_world = np.zeros((n_joints, 3))
    positions_world = np.zeros((n_joints + 1, 3))

    for i, jid in enumerate(joint_ids):
        body_id = model.jnt_bodyid[jid]
        body_rot = tmp_data.xmat[body_id].reshape(3, 3)
        H_world[i] = body_rot @ model.jnt_axis[jid]
        positions_world[i] = tmp_data.xpos[body_id] + body_rot @ model.jnt_pos[jid]

    positions_world[n_joints] = tmp_data.site_xpos[ee_site_id]

    # EE site orientation in base frame at q=zeros.
    # EAIK's FK at q=zeros gives identity rotation (all joint angles are zero),
    # so the actual site orientation at q=zeros IS the fixed offset that EAIK
    # does not model. We store it so solve() can correct IK targets accordingly.
    ee_rot_world = tmp_data.site_xmat[ee_site_id].reshape(3, 3)
    ee_rot_base = base_rot_inv @ ee_rot_world

    # Convert to base frame
    H_base = (base_rot_inv @ H_world.T).T
    P_offsets = np.zeros((n_joints + 1, 3))
    P_offsets[0] = base_rot_inv @ (positions_world[0] - base_pos)
    for i in range(1, n_joints + 1):
        P_offsets[i] = base_rot_inv @ (positions_world[i] - positions_world[i - 1])

    return H_base, P_offsets, ee_rot_base
