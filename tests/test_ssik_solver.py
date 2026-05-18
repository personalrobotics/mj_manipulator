# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for MuJoCoSSIKSolver — analytical IK via ssik.

Skipped entirely if ssik is not installed (optional in some workspaces).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

ssik = pytest.importorskip("ssik")

from mj_environment import Environment  # noqa: E402

from mj_manipulator.arms._ik_factory import resolve_ik_solver  # noqa: E402
from mj_manipulator.arms.ssik_solver import MuJoCoSSIKSolver  # noqa: E402
from mj_manipulator.protocols import IKSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — UR5e (EAIK-supported, ssik also available)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5e_setup():
    """UR5e arm + ssik solver against the ur5_ik prebuilt."""
    from ssik.prebuilt import ur5_ik

    from mj_manipulator.arms.ur5e import UR5E_HOME, add_ur5e_gravcomp, create_ur5e_arm

    try:
        from mj_manipulator.menagerie import menagerie_scene

        spec = mujoco.MjSpec.from_file(str(menagerie_scene("universal_robots_ur5e")))
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")
    add_ur5e_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm = create_ur5e_arm(env, with_ik=False)

    base_body_id = int(env.model.body_parentid[env.model.jnt_bodyid[arm.joint_ids[0]]])
    solver = MuJoCoSSIKSolver(
        ssik_module=ur5_ik,
        model=env.model,
        data=env.data,
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=arm.get_joint_limits(),
    )
    return arm, solver, np.array(UR5E_HOME)


@pytest.fixture(scope="module")
def jaco2_setup():
    """JACO 2 (via ada_assets) + ssik solver against the jaco2_ik prebuilt.

    JACO 2 is the prime motivation for ssik: EAIK refuses its non-Pieper 6R
    geometry, so ssik is what makes analytical IK available at all.
    """
    ada_assets = pytest.importorskip("ada_assets.assembly")
    from ssik.prebuilt import jaco2_ik

    model, data = ada_assets.assemble_ada(with_human=False, with_camera=False, tool=None)
    env = Environment.from_model(model, data)

    JNT = [
        "j2n6s200_joint_1", "j2n6s200_joint_2", "j2n6s200_joint_3",
        "j2n6s200_joint_4", "j2n6s200_joint_5", "j2n6s200_joint_6",
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JNT]
    qpos_indices = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    base_body_id = int(model.body_parentid[model.jnt_bodyid[joint_ids[0]]])

    solver = MuJoCoSSIKSolver(
        ssik_module=jaco2_ik,
        model=model,
        data=data,
        joint_qpos_indices=qpos_indices,
        ee_site_id=ee_site_id,
        base_body_id=base_body_id,
    )

    # Pull whatever the model's q is now as a "home" for round-trips.
    q_home = data.qpos[qpos_indices].copy()
    return env, solver, qpos_indices, ee_site_id, q_home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_arm_qpos(env, qpos_indices, q: np.ndarray) -> None:
    env.data.qpos[qpos_indices] = q
    mujoco.mj_forward(env.model, env.data)


def _read_ee_pose(env, ee_site_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = env.data.site_xmat[ee_site_id].reshape(3, 3)
    T[:3, 3] = env.data.site_xpos[ee_site_id]
    return T


def _fk_error_arm(arm, q: np.ndarray, target_pose: np.ndarray) -> float:
    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = q[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)
    ee = arm.get_ee_pose()
    return float(np.linalg.norm(ee[:3, 3] - target_pose[:3, 3]))


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestSSIKSolverProtocol:
    def test_implements_ik_solver(self, ur5e_setup):
        _, solver, _ = ur5e_setup
        assert isinstance(solver, IKSolver)

    def test_solve_returns_list(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_arm_qpos(arm.env, arm.joint_qpos_indices, home)
        pose = arm.get_ee_pose()
        result = solver.solve(pose)
        assert isinstance(result, list)

    def test_solve_valid_returns_list(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_arm_qpos(arm.env, arm.joint_qpos_indices, home)
        pose = arm.get_ee_pose()
        result = solver.solve_valid(pose)
        assert isinstance(result, list)

    def test_dof_matches_artifact(self, ur5e_setup):
        _, solver, _ = ur5e_setup
        assert solver.dof == 6

    def test_rejects_non_artifact_module(self):
        with pytest.raises(TypeError, match="must expose .solve and .fk"):
            MuJoCoSSIKSolver(
                ssik_module=object(),  # missing solve/fk
                model=mujoco.MjModel.from_xml_string("<mujoco/>"),
                data=mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>")),
                joint_qpos_indices=[],
                ee_site_id=0,
                base_body_id=0,
            )


# ---------------------------------------------------------------------------
# FK-IK round-trips: JACO 2 (the case that motivated this backend)
# ---------------------------------------------------------------------------


class TestJACO2RoundTrip:
    """ssik IK round-trips for JACO 2 — the case EAIK can't handle."""

    def test_finds_solutions_from_current_pose(self, jaco2_setup):
        env, solver, qpos, ee_id, q_home = jaco2_setup
        _set_arm_qpos(env, qpos, q_home)
        T_target = _read_ee_pose(env, ee_id)
        sols = solver.solve(T_target)
        assert len(sols) >= 1, "ssik must return at least one solution for current pose"

    def test_returns_multiple_branches(self, jaco2_setup):
        """ssik's selling point vs mink: every analytical branch per pose."""
        env, solver, qpos, ee_id, _ = jaco2_setup
        # Use a non-singular config
        q = np.array([0.5, 1.0, -1.0, 0.3, 0.7, 0.2])
        _set_arm_qpos(env, qpos, q)
        T_target = _read_ee_pose(env, ee_id)
        sols = solver.solve(T_target)
        # Non-Pieper 6R typically yields 2-8 branches per reachable pose.
        assert len(sols) >= 2

    def test_solutions_fk_close_to_target(self, jaco2_setup):
        env, solver, qpos, ee_id, _ = jaco2_setup
        q = np.array([0.5, 1.0, -1.0, 0.3, 0.7, 0.2])
        _set_arm_qpos(env, qpos, q)
        T_target = _read_ee_pose(env, ee_id)
        sols = solver.solve(T_target)
        for q_sol in sols:
            _set_arm_qpos(env, qpos, q_sol)
            T_check = _read_ee_pose(env, ee_id)
            pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
            assert pos_err < 1e-3  # 1 mm

    def test_random_poses_above_60_percent(self, jaco2_setup):
        """20 random reachable poses — most return at least one branch with sub-mm FK error.

        Some random configs land near kinematic singularities where ssik refuses;
        a >60% success rate confirms the wrapper is sound (cf. mink, which gets
        ~0% on the same poses with the wrong articutool tilt — that's the gap
        this backend exists to close).
        """
        env, solver, qpos, ee_id, _ = jaco2_setup
        rng = np.random.default_rng(0)
        n_pass = 0
        n_total = 20
        for _ in range(n_total):
            q_truth = rng.uniform(-2.0, 2.0, size=6)
            _set_arm_qpos(env, qpos, q_truth)
            T_target = _read_ee_pose(env, ee_id)
            sols = solver.solve(T_target)
            if not sols:
                continue
            _set_arm_qpos(env, qpos, sols[0])
            T_check = _read_ee_pose(env, ee_id)
            if np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) < 1e-3:
                n_pass += 1
        assert n_pass / n_total >= 0.6, f"Round-trip success {n_pass}/{n_total} below 60%"


# ---------------------------------------------------------------------------
# Factory dispatch — EAIK → ssik → mink
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    """The factory should hand back the right solver for each case."""

    def test_auto_falls_to_ssik_when_eaik_refuses_and_module_given(self, jaco2_setup):
        """JACO 2: EAIK refuses, ssik module supplied → ssik."""
        env, _, qpos, ee_id, q_home = jaco2_setup
        from ssik.prebuilt import jaco2_ik

        from mj_manipulator.arm import Arm
        from mj_manipulator.config import ArmConfig, KinematicLimits, PlanningDefaults

        JNT = [
            "j2n6s200_joint_1", "j2n6s200_joint_2", "j2n6s200_joint_3",
            "j2n6s200_joint_4", "j2n6s200_joint_5", "j2n6s200_joint_6",
        ]
        cfg = ArmConfig(
            name="jaco2", entity_type="arm", joint_names=JNT,
            kinematic_limits=KinematicLimits(
                velocity=np.full(6, 1.5), acceleration=np.full(6, 3.0),
            ),
            ee_site="ee_site",
            planning_defaults=PlanningDefaults(),
        )
        arm = Arm(env, cfg)

        solver = resolve_ik_solver(arm, with_ik="auto", ssik_module=jaco2_ik)
        assert type(solver).__name__ == "MuJoCoSSIKSolver"

    def test_auto_falls_to_mink_when_eaik_refuses_and_no_ssik(self, jaco2_setup):
        """JACO 2: EAIK refuses, no ssik module → legacy mink fallback."""
        env, _, qpos, ee_id, _ = jaco2_setup
        from mj_manipulator.arm import Arm
        from mj_manipulator.config import ArmConfig, KinematicLimits, PlanningDefaults

        JNT = [
            "j2n6s200_joint_1", "j2n6s200_joint_2", "j2n6s200_joint_3",
            "j2n6s200_joint_4", "j2n6s200_joint_5", "j2n6s200_joint_6",
        ]
        cfg = ArmConfig(
            name="jaco2", entity_type="arm", joint_names=JNT,
            kinematic_limits=KinematicLimits(
                velocity=np.full(6, 1.5), acceleration=np.full(6, 3.0),
            ),
            ee_site="ee_site",
            planning_defaults=PlanningDefaults(),
        )
        arm = Arm(env, cfg)

        solver = resolve_ik_solver(arm, with_ik="auto")
        assert type(solver).__name__ == "MinkIKSolver"

    def test_ssik_mode_requires_module(self, jaco2_setup):
        env, _, _, _, _ = jaco2_setup
        from mj_manipulator.arm import Arm
        from mj_manipulator.config import ArmConfig, KinematicLimits, PlanningDefaults

        JNT = [
            "j2n6s200_joint_1", "j2n6s200_joint_2", "j2n6s200_joint_3",
            "j2n6s200_joint_4", "j2n6s200_joint_5", "j2n6s200_joint_6",
        ]
        cfg = ArmConfig(
            name="jaco2", entity_type="arm", joint_names=JNT,
            kinematic_limits=KinematicLimits(
                velocity=np.full(6, 1.5), acceleration=np.full(6, 3.0),
            ),
            ee_site="ee_site",
            planning_defaults=PlanningDefaults(),
        )
        arm = Arm(env, cfg)
        with pytest.raises(ValueError, match="requires an ssik_module"):
            resolve_ik_solver(arm, with_ik="ssik")
