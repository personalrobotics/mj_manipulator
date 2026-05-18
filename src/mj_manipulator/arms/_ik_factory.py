# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared logic for resolving IK solvers in arm factories.

Handles the ``with_ik`` parameter (``"auto"`` / ``"eaik"`` / ``"ssik"`` /
``"mink"`` / ``"none"`` / bool) and the EAIK → ssik → mink fallback
chain: EAIK is preferred where it has an analytical decomposition,
ssik picks up arms whose geometry EAIK refuses (non-Pieper 6R, every
7R class), and mink is the last-resort numerical fallback for arms
neither analytical solver covers (or when no ssik artifact is wired).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import IKSolver

logger = logging.getLogger(__name__)

# The union type accepted by all arm factories.
IKMode = Literal["auto", "eaik", "ssik", "mink", "none"] | bool


def resolve_ik_solver(
    arm: "Arm",
    with_ik: IKMode,
    *,
    fixed_joint_index: int | None = None,
    n_discretizations: int = 16,
    ssik_module=None,
) -> "IKSolver | None":
    """Build the right IK solver for an arm based on ``with_ik``.

    The EE site name (for mink's FrameTask) is resolved automatically
    from ``arm.ee_site_id`` — no configuration needed.

    Args:
        arm: A bare Arm (no IK yet) — used to read joint indices/limits.
        with_ik: ``"auto"`` (default), ``"eaik"``, ``"ssik"``, ``"mink"``,
            ``"none"``, or bool for backward compat (``True`` → ``"auto"``).
        fixed_joint_index: For EAIK on 7-DOF arms, which joint to lock.
        n_discretizations: Discretization count for EAIK on 7-DOF arms.
        ssik_module: Per-arm ``ssik`` artifact module (e.g.
            ``ssik.prebuilt.jaco2_ik``). Required when ``with_ik="ssik"``
            or whenever the ``"auto"`` chain reaches the ssik step;
            ``None`` skips ssik in ``"auto"`` and falls through to mink.

    Returns:
        An IKSolver, or None if ``with_ik="none"`` or all attempted
        solvers fail.
    """
    # Normalize bool → string.
    if with_ik is True:
        with_ik = "auto"
    elif with_ik is False:
        with_ik = "none"

    if with_ik == "none":
        return None

    if with_ik == "eaik":
        return _make_eaik(arm, fixed_joint_index, n_discretizations)

    if with_ik == "ssik":
        if ssik_module is None:
            raise ValueError("with_ik='ssik' requires an ssik_module (e.g. ssik.prebuilt.<arm>_ik). Got None.")
        return _make_ssik(arm, ssik_module)

    if with_ik == "mink":
        return _make_mink(arm)

    # "auto": try EAIK → ssik → mink in order.
    eaik = _try_eaik(arm, fixed_joint_index, n_discretizations)
    if eaik is not None:
        logger.debug("Using EAIK analytical IK for '%s'.", arm.config.name)
        return eaik

    if ssik_module is not None:
        ssik = _try_ssik(arm, ssik_module)
        if ssik is not None:
            logger.info(
                "EAIK has no decomposition for '%s'; using ssik analytical IK.",
                arm.config.name,
            )
            return ssik

    logger.info(
        "EAIK has no decomposition for '%s' and no ssik artifact wired; falling back to mink numerical IK.",
        arm.config.name,
    )
    return _make_mink(arm)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _make_eaik(arm, fixed_joint_index, n_discretizations):
    from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver

    env = arm.env
    joint_limits = arm.get_joint_limits()
    first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = int(env.model.body_parentid[first_joint_body])

    kwargs = dict(
        model=env.model,
        data=env.data,
        joint_ids=list(arm.joint_ids),
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=joint_limits,
    )
    if fixed_joint_index is not None:
        kwargs["fixed_joint_index"] = fixed_joint_index
        kwargs["n_discretizations"] = n_discretizations

    return MuJoCoEAIKSolver(**kwargs)


def _try_eaik(arm, fixed_joint_index, n_discretizations):
    """Try EAIK — return solver if it has a known decomposition, else None."""
    try:
        solver = _make_eaik(arm, fixed_joint_index, n_discretizations)
    except Exception as e:
        logger.debug("EAIK construction failed for '%s': %s", arm.config.name, e)
        return None

    # For 6-DOF (no fixed joint): check if EAIK recognizes the kinematics.
    if fixed_joint_index is None and solver.robot is not None:
        if not solver.robot.hasKnownDecomposition():
            logger.debug(
                "EAIK has no known decomposition for '%s' (family: %s).",
                arm.config.name,
                solver.robot.getKinematicFamily(),
            )
            return None

    return solver


def _make_ssik(arm, ssik_module):
    from mj_manipulator.arms.ssik_solver import MuJoCoSSIKSolver

    env = arm.env
    first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = int(env.model.body_parentid[first_joint_body])

    return MuJoCoSSIKSolver(
        ssik_module=ssik_module,
        model=env.model,
        data=env.data,
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=arm.get_joint_limits(),
    )


def _try_ssik(arm, ssik_module):
    """Try ssik — return solver if construction succeeds, else None."""
    try:
        return _make_ssik(arm, ssik_module)
    except Exception as e:
        logger.debug("ssik construction failed for '%s': %s", arm.config.name, e)
        return None


def _make_mink(arm):
    from mj_manipulator.arms.mink_solver import make_mink_solver

    return make_mink_solver(arm)
