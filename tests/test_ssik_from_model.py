# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ssik built from the live model is the default IK; EAIK is deprecated."""

import importlib.util
import sys
import warnings

import mujoco
import numpy as np
import pytest
from test_arms import franka_arm, franka_env, iiwa14_arm, iiwa14_env, ur5e_arm, ur5e_env  # noqa: F401 (fixtures)

from mj_manipulator.arms._ik_factory import resolve_ik_solver
from mj_manipulator.arms.ssik_solver import MuJoCoSSIKSolver

ARMS = {
    "ur5e": ("ur5e_arm", "ikgeo.three_parallel"),
    "franka": ("franka_arm", "seven_r.spherical_shoulder"),
    "iiwa14": ("iiwa14_arm", "seven_r.srs"),
}


def _site_pose(arm, q):
    env = arm.env
    env.data.qpos[arm.joint_qpos_indices] = q
    mujoco.mj_forward(env.model, env.data)
    T = np.eye(4)
    T[:3, :3] = env.data.site_xmat[arm.ee_site_id].reshape(3, 3)
    T[:3, 3] = env.data.site_xpos[arm.ee_site_id]
    return T


def _samples(arm, n, seed):
    lo, hi = arm.get_joint_limits()
    lo, hi = np.maximum(lo, -2 * np.pi) + 0.05, np.minimum(hi, 2 * np.pi) - 0.05
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi) for _ in range(n)]


@pytest.mark.parametrize("name", list(ARMS))
def test_default_ik_is_ssik_from_the_model_with_the_exact_solver(name, request):
    fixture, solver_name = ARMS[name]
    arm = request.getfixturevalue(fixture)
    assert isinstance(arm.ik_solver, MuJoCoSSIKSolver)
    assert arm.ik_solver.solver_name == solver_name


@pytest.mark.parametrize("name", list(ARMS))
def test_every_solution_reaches_the_pose_exactly(name, request):
    """Solutions come from the model's own kinematics: MuJoCo FK lands on the target to 1e-8."""
    arm = request.getfixturevalue(ARMS[name][0])
    for q in _samples(arm, 20, 0):
        target = _site_pose(arm, q)
        sols = arm.ik_solver.solve_valid(target)
        assert sols
        lo, hi = arm.get_joint_limits()
        for s in sols[:8]:
            assert np.all(s >= lo) and np.all(s <= hi)
            np.testing.assert_allclose(_site_pose(arm, s), target, atol=1e-8)


@pytest.mark.parametrize("name", ["franka", "iiwa14"])
def test_a_seeded_7dof_solve_continues_from_the_seed(name, request):
    """A redundant arm's solutions are a sample of a one-parameter family; seeded, the nearest is the seed itself
    (the self-motion manifold at the seed's coordinate), so Cartesian paths do not jump."""
    arm = request.getfixturevalue(ARMS[name][0])
    for q in _samples(arm, 30, 1):
        sols = arm.ik_solver.solve_valid(_site_pose(arm, q), q_init=q)
        assert np.max(np.abs(sols[0] - q)) < 1e-8


def test_eaik_mode_and_its_arguments_are_deprecated(ur5e_arm):  # noqa: F811
    with pytest.warns(DeprecationWarning, match="n_discretizations"):
        resolve_ik_solver(ur5e_arm, with_ik="auto", n_discretizations=8)
    with pytest.warns(DeprecationWarning, match="fixed_joint_index"):
        resolve_ik_solver(ur5e_arm, with_ik="ssik", fixed_joint_index=4)
    if importlib.util.find_spec("eaik") is not None:
        with pytest.warns(DeprecationWarning, match="with_ik='eaik' is deprecated"):
            resolve_ik_solver(ur5e_arm, with_ik="eaik")


def test_the_default_path_never_warns_or_imports_eaik(ur5e_env):  # noqa: F811
    from mj_manipulator.arms.ur5e import create_ur5e_arm

    sys.modules.pop("eaik", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create_ur5e_arm(ur5e_env)
    assert "eaik" not in sys.modules
