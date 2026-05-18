# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ssik-based analytical IK solver for MuJoCo manipulators.

Wraps a per-arm ``ssik`` artifact module (e.g. ``ssik.prebuilt.jaco2_ik``)
and presents the :class:`~mj_manipulator.protocols.IKSolver` surface so the
factory can dispatch ``ssik`` between EAIK (preferred when it has a known
decomposition for the arm) and mink (numerical fallback). See ssik for
algorithmic background and the per-arm coverage table.

Usage::

    from ssik.prebuilt import jaco2_ik
    solver = MuJoCoSSIKSolver(
        ssik_module=jaco2_ik,
        model=model,
        data=data,
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=...,
    )
    sols = solver.solve(T_world_target)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _read_body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """4x4 world-frame pose of a body."""
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def _read_site_pose(data: mujoco.MjData, site_id: int) -> np.ndarray:
    """4x4 world-frame pose of a site."""
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T


class MuJoCoSSIKSolver:
    """ssik IK solver bound to a MuJoCo arm.

    The ssik artifact's solver returns joint configurations for a target
    pose expressed in the artifact's ``BASE_LINK → EE_LINK`` frame. The
    MuJoCo caller provides world-frame poses at an EE *site* that may not
    coincide with the artifact's EE_LINK origin. This wrapper:

    1. Captures the current arm base pose in the world (assumed static).
    2. Computes the static SE(3) offset between ssik's EE_LINK frame and
       MuJoCo's EE site at the current joint configuration (constant under
       the rigid-body assumption — the offset is purely a frame convention).
    3. On each ``solve`` call, transforms the world-frame target into the
       base frame and applies the offset so the artifact sees an EE_LINK
       target it can solve.

    Args:
        ssik_module: An ssik artifact module (e.g. ``ssik.prebuilt.jaco2_ik``)
            with at least ``solve(T, ...)``, ``fk(q)``, ``DOF``, and
            ``T_HOME`` attributes.
        model: MuJoCo model.
        data: MuJoCo data.
        joint_qpos_indices: qpos indices of the arm joints (length DOF).
        ee_site_id: MuJoCo site ID for the EE the caller targets.
        base_body_id: Body ID of the arm base (root of the kinematic chain).
        joint_limits: Optional ``(lower, upper)`` arrays. Currently the
            ssik artifact respects its own URDF-derived limits via
            ``respect_limits=True``; this argument is accepted for protocol
            consistency.
        respect_limits: Forwarded to ``ssik.solve``. Default ``True``.
        allow_refinement: Forwarded to ``ssik.solve``. Default ``False``.
    """

    def __init__(
        self,
        ssik_module,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: Sequence[int] | np.ndarray,
        ee_site_id: int,
        base_body_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        respect_limits: bool = True,
        allow_refinement: bool = False,
    ):
        if not hasattr(ssik_module, "solve") or not hasattr(ssik_module, "fk"):
            raise TypeError(
                f"ssik_module {ssik_module!r} must expose .solve and .fk (see ssik artifact modules in ssik.prebuilt)."
            )

        self._mod = ssik_module
        self._model = model
        self._data = data
        self._joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=int)
        self._ee_site_id = ee_site_id
        self._base_body_id = base_body_id
        self._joint_limits = joint_limits
        self._respect_limits = respect_limits
        self._allow_refinement = allow_refinement

        dof = int(getattr(ssik_module, "DOF"))
        if len(self._joint_qpos_indices) != dof:
            raise ValueError(
                f"ssik artifact has DOF={dof} but joint_qpos_indices has length {len(self._joint_qpos_indices)}."
            )

        # Compute the static frame offset between ssik's EE_LINK and the
        # MuJoCo EE site. We use the current joint configuration as the
        # reference — the offset is the same at every q for a rigid chain.
        mujoco.mj_forward(model, data)
        q_now = self._read_q()
        T_world_base = _read_body_pose(data, base_body_id)
        T_world_muj_ee = _read_site_pose(data, ee_site_id)
        T_base_muj_ee = np.linalg.inv(T_world_base) @ T_world_muj_ee
        T_base_ssik_ee = np.asarray(ssik_module.fk(q_now), dtype=float)
        # T_base_muj_ee = T_base_ssik_ee @ T_ssik_to_muj_ee  →
        self._T_ssik_to_muj_ee = np.linalg.inv(T_base_ssik_ee) @ T_base_muj_ee

    # ------------------------------------------------------------------
    # Public API (IKSolver protocol)
    # ------------------------------------------------------------------

    def solve(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Solve IK for a world-frame target pose at the MuJoCo EE site.

        Args:
            pose: 4x4 target pose in world frame.
            q_init: Optional joint seed. When provided, ssik returns
                solutions in nearest-to-seed order, which is what
                trajectory-tracking / Cartesian-path callers want.

        Returns:
            List of joint configurations. Empty if ssik finds none.
        """
        T_target_ssik = self._world_target_to_ssik_target(pose)
        q_seed = np.asarray(q_init, dtype=float) if q_init is not None else None
        sols = self._mod.solve(
            T_target_ssik,
            q_seed=q_seed,
            respect_limits=self._respect_limits,
            allow_refinement=self._allow_refinement,
        )
        return [np.asarray(s.q, dtype=float) for s in sols]

    def solve_valid(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Solve IK and return only in-limits solutions.

        ssik's ``respect_limits=True`` already filters out-of-URDF-limit
        branches, so for the default construction this is identical to
        :meth:`solve`. When the caller has built the solver with
        ``respect_limits=False`` (rare — typically used in tests for the
        raw geometric set) and ``joint_limits`` was supplied, we apply
        the limits here.
        """
        sols = self.solve(pose, q_init)
        if self._joint_limits is None:
            return sols
        lo, hi = self._joint_limits
        return [q for q in sols if np.all(q >= lo) and np.all(q <= hi)]

    # ------------------------------------------------------------------
    # Properties / introspection
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        """Number of joints in the artifact."""
        return int(getattr(self._mod, "DOF"))

    @property
    def solver_name(self) -> str:
        """The ssik dispatcher's solver name (e.g. ``ikgeo.three_parallel``)."""
        return getattr(self._mod, "SOLVER_NAME", "ssik")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_q(self) -> np.ndarray:
        return self._data.qpos[self._joint_qpos_indices].copy()

    def _world_target_to_ssik_target(self, pose_world: np.ndarray) -> np.ndarray:
        T_world_base = _read_body_pose(self._data, self._base_body_id)
        T_base_target_muj_ee = np.linalg.inv(T_world_base) @ pose_world
        # T_base_muj_ee = T_base_ssik_ee @ T_ssik_to_muj_ee  →
        return T_base_target_muj_ee @ np.linalg.inv(self._T_ssik_to_muj_ee)
