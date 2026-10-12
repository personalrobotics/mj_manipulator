# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ssik-based analytical IK solver for MuJoCo manipulators: mj_manipulator's IK.

Two ways to build one:

- :meth:`MuJoCoSSIKSolver.from_model` reads the arm's kinematics from the live MuJoCo model (joint axes and offsets at
  q = 0) and builds an ``ssik.Manipulator`` from them, so the solver's FK is the model's. 6R arms dispatch to ikgeo;
  7R arms (Franka, iiwa) to ssik's exact 7R solvers, with no joint locked or discretized. This is what
  ``with_ik="auto"`` and ``"ssik"`` use.
- The constructor wraps any object with ``solve``, ``fk`` and ``DOF`` (or ``dof``): a per-arm ``ssik`` artifact
  module such as ``ssik.prebuilt.jaco2_ik``, or an ``ssik.Manipulator``.

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


def _dof(solver) -> int:
    """Joint count of an ssik artifact module (``DOF``) or ``Manipulator`` (``dof``)."""
    return int(getattr(solver, "DOF", None) or getattr(solver, "dof"))


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
        if not hasattr(ssik_module, "solve") or not hasattr(ssik_module, "fk"):  # a module or a Manipulator
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

        dof = _dof(ssik_module)
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
        out = [np.asarray(s.q, dtype=float) for s in sols]
        if q_seed is not None:
            out = self._with_seed_continuations(T_target_ssik, q_seed, out)
        return out

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
        """Number of joints."""
        return _dof(self._mod)

    @property
    def solver_name(self) -> str:
        """The ssik dispatcher's solver name (e.g. ``ikgeo.three_parallel``)."""
        return getattr(self._mod, "SOLVER_NAME", None) or getattr(self._mod, "solver_name", "ssik")

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_ids: Sequence[int],
        joint_qpos_indices: Sequence[int] | np.ndarray,
        ee_site_id: int,
        base_body_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> MuJoCoSSIKSolver:
        """An SSIK solver whose kinematics are read from the live model: no artifact, no geometry mismatch.

        The joint axes and offsets at q = 0 (``_extract_hp``) build an ``ssik.Manipulator`` via ``from_axes``, which
        dispatches to the arm's analytical solver. The end-effector site's fixed offset is then fitted against
        MuJoCo's FK by the constructor; since both FKs come from the same model, the fit is exact.
        """
        import ssik

        from mj_manipulator.arms._kinematics import _extract_hp

        H, P, _ = _extract_hp(model, data, list(joint_ids), list(joint_qpos_indices), ee_site_id, base_body_id)
        limits = None if joint_limits is None else np.column_stack([joint_limits[0], joint_limits[1]])
        manipulator = ssik.Manipulator.from_axes(H, P, limits=limits)
        return cls(
            manipulator,
            model=model,
            data=data,
            joint_qpos_indices=joint_qpos_indices,
            ee_site_id=ee_site_id,
            base_body_id=base_body_id,
            joint_limits=joint_limits,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _with_seed_continuations(self, T_target: np.ndarray, q_seed: np.ndarray, sols: list[np.ndarray]):
        """On a redundant (7R) arm, add the solutions at the seed's own redundancy coordinate, nearest first.

        ssik samples a 7R pose's one-parameter family of solutions; the sample need not include a point near the seed,
        so a seeded caller (a Cartesian path, teleop) would jump. The pose's self-motion manifold, evaluated at the
        seed's coordinate (``param_of``: joint 6 for the Panda's family, the elbow swivel for SRS arms like the iiwa),
        gives the branches that continue from the seed. 6R arms and modules without charts are returned unchanged.
        """
        self_motion = getattr(self._mod, "self_motion", None)
        if self_motion is None or _dof(self._mod) < 7:
            return sols
        try:
            manifold = self_motion(T_target)
        except NotImplementedError:
            return sols
        t = manifold.param_of(q_seed)
        extra = []
        for chart in manifold.charts:
            if chart.contains(t):
                q = np.asarray(chart.q(t), dtype=float)
                q = q_seed + (q - q_seed + np.pi) % (2 * np.pi) - np.pi  # charts wrap to +-pi: take the seed's winding
                if np.all(np.isfinite(q)) and self._within_limits(q):
                    extra.append(q)

        def distance(q: np.ndarray) -> float:
            return float(np.max(np.abs((q - q_seed + np.pi) % (2 * np.pi) - np.pi)))

        merged: list[np.ndarray] = []
        for q in sorted(extra + sols, key=distance):
            if not any(np.allclose(q, m, atol=1e-9) for m in merged):
                merged.append(q)
        return merged

    def _within_limits(self, q: np.ndarray) -> bool:
        if self._joint_limits is None or not self._respect_limits:
            return True
        lo, hi = self._joint_limits
        return bool(np.all(q >= lo) and np.all(q <= hi))

    def _read_q(self) -> np.ndarray:
        return self._data.qpos[self._joint_qpos_indices].copy()

    def _world_target_to_ssik_target(self, pose_world: np.ndarray) -> np.ndarray:
        T_world_base = _read_body_pose(self._data, self._base_body_id)
        T_base_target_muj_ee = np.linalg.inv(T_world_base) @ pose_world
        # T_base_muj_ee = T_base_ssik_ee @ T_ssik_to_muj_ee  →
        return T_base_target_muj_ee @ np.linalg.inv(self._T_ssik_to_muj_ee)
