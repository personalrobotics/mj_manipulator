# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared logic for resolving IK solvers in arm factories.

Handles the ``with_ik`` parameter (``"auto"`` / ``"ssik"`` / ``"mink"`` /
``"none"`` / bool, and the deprecated ``"eaik"``) and the ssik → mink
fallback chain. ssik is the analytical IK: built from the live model's
kinematics (or from a per-arm artifact when one is given), it covers 6R
arms and 7R arms (Franka, iiwa) in closed form. mink is the numerical
fallback for arms ssik does not cover.

EAIK is deprecated (it handled 7R arms by locking one joint and sweeping
it): ``with_ik="eaik"`` still works when ``eaik`` is installed
(``mj-manipulator[eaik]``) and warns; it will be removed in the next major
release.
"""

from __future__ import annotations

import logging
import warnings
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
    n_discretizations: int | None = None,
    ssik_module=None,
) -> "IKSolver | None":
    """Build the right IK solver for an arm based on ``with_ik``.

    The EE site name (for mink's FrameTask) is resolved automatically
    from ``arm.ee_site_id`` — no configuration needed.

    Args:
        arm: A bare Arm (no IK yet) — used to read joint indices/limits.
        with_ik: ``"auto"`` (default: ssik, then mink), ``"ssik"``,
            ``"mink"``, ``"none"``, or bool for backward compat
            (``True`` → ``"auto"``). ``"eaik"`` is deprecated.
        fixed_joint_index: Deprecated; only ``"eaik"`` uses it (which joint
            of a 7-DOF arm to lock). Ignored, with a warning, otherwise.
        n_discretizations: Deprecated; only ``"eaik"`` uses it (default 16).
            Ignored, with a warning, otherwise.
        ssik_module: Optional per-arm ``ssik`` artifact module (e.g.
            ``ssik.prebuilt.jaco2_ik``). ``None`` builds the ssik solver
            from the live model's kinematics.

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
        warnings.warn(
            "with_ik='eaik' is deprecated and will be removed in the next major release; ssik is the analytical IK "
            "(with_ik='auto' or 'ssik'), and solves 7-DOF arms in closed form.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _make_eaik(arm, fixed_joint_index, 16 if n_discretizations is None else n_discretizations)

    if fixed_joint_index is not None or n_discretizations is not None:
        warnings.warn(
            "fixed_joint_index and n_discretizations are deprecated: only with_ik='eaik' used them (ssik solves "
            "7-DOF arms without locking a joint). They are ignored.",
            DeprecationWarning,
            stacklevel=3,
        )

    if with_ik == "ssik":
        return _make_ssik(arm, ssik_module)

    if with_ik == "mink":
        return _make_mink(arm)

    # "auto": ssik, then mink.
    ssik = _try_ssik(arm, ssik_module)
    if ssik is not None:
        logger.debug("Using ssik analytical IK (%s) for '%s'.", ssik.solver_name, arm.config.name)
        return ssik

    logger.info("ssik does not cover '%s'; falling back to mink numerical IK.", arm.config.name)
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


def _make_ssik(arm, ssik_module=None):
    from mj_manipulator.arms.ssik_solver import MuJoCoSSIKSolver

    env = arm.env
    first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = int(env.model.body_parentid[first_joint_body])

    if ssik_module is None:  # kinematics from the live model
        return MuJoCoSSIKSolver.from_model(
            env.model,
            env.data,
            joint_ids=list(arm.joint_ids),
            joint_qpos_indices=arm.joint_qpos_indices,
            ee_site_id=arm.ee_site_id,
            base_body_id=base_body_id,
            joint_limits=arm.get_joint_limits(),
        )
    return MuJoCoSSIKSolver(
        ssik_module=ssik_module,
        model=env.model,
        data=env.data,
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=arm.get_joint_limits(),
    )


def _try_ssik(arm, ssik_module=None):
    """Try ssik — return solver if construction succeeds, else None."""
    try:
        return _make_ssik(arm, ssik_module)
    except Exception as e:
        logger.debug("ssik construction failed for '%s': %s", arm.config.name, e)
        return None


def _make_mink(arm):
    from mj_manipulator.arms.mink_solver import make_mink_solver

    return make_mink_solver(arm)
