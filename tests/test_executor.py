"""Tests for KinematicExecutor and PhysicsExecutor.

Uses a minimal inline MuJoCo model with a 2-DOF arm and actuators.
"""

import mujoco
import numpy as np
import pytest

from mj_manipulator.executor import KinematicExecutor, PhysicsExecutor
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.trajectory import Trajectory

# Minimal model: 2-joint arm with actuators
_ARM_XML = """
<mujoco model="test_executor">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      </body>
    </body>
    <body name="box" pos="0.5 0 0.5">
      <joint name="box_free" type="free"/>
      <geom type="box" size="0.03 0.03 0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act1" joint="joint1" kp="100"/>
    <position name="act2" joint="joint2" kp="100"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(_ARM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def joint_qpos_indices(model_and_data):
    model, _ = model_and_data
    indices = []
    for name in ["joint1", "joint2"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        indices.append(model.jnt_qposadr[jid])
    return indices


@pytest.fixture
def actuator_ids(model_and_data):
    model, _ = model_and_data
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act2"),
    ]


def _make_trajectory(positions: np.ndarray) -> Trajectory:
    """Helper to create a trajectory from positions array."""
    return Trajectory(
        timestamps=np.linspace(0, 1, len(positions)),
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        joint_names=["joint1", "joint2"],
    )


class TestKinematicExecutor:
    def test_constructs(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        assert ex is not None

    def test_execute_trajectory(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(
            model, data, joint_qpos_indices, control_dt=0.0
        )
        positions = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ])
        traj = _make_trajectory(positions)
        result = ex.execute(traj)
        assert result is True

        # Final position should match trajectory end
        for i, idx in enumerate(joint_qpos_indices):
            assert abs(data.qpos[idx] - 0.3) < 1e-6

    def test_execute_dof_mismatch_raises(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        positions = np.array([[0.0, 0.0, 0.0]])  # 3 DOF, but arm is 2
        traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
            joint_names=["j1", "j2", "j3"],
        )
        with pytest.raises(ValueError, match="DOF"):
            ex.execute(traj)

    def test_set_position(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        ex.set_position(np.array([0.5, -0.5]))
        for i, idx in enumerate(joint_qpos_indices):
            expected = [0.5, -0.5][i]
            assert abs(data.qpos[idx] - expected) < 1e-6

    def test_step_applies_target(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        ex.set_position(np.array([1.0, -1.0]))
        # Overwrite qpos to something different
        for idx in joint_qpos_indices:
            data.qpos[idx] = 0.0
        # Step should restore target
        ex.step()
        for i, idx in enumerate(joint_qpos_indices):
            expected = [1.0, -1.0][i]
            assert abs(data.qpos[idx] - expected) < 1e-6

    def test_with_grasp_manager(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        gm = GraspManager(model, data)
        ex = KinematicExecutor(
            model, data, joint_qpos_indices, grasp_manager=gm
        )
        # Should not crash even with grasp manager attached
        ex.set_position(np.array([0.1, 0.1]))
        ex.step()


class TestPhysicsExecutor:
    def test_constructs(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids
        )
        assert ex is not None
        assert ex.steps_per_control == 4  # 0.008 / 0.002

    def test_set_target_and_step(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids, control_dt=0.002
        )
        ex.set_target(np.array([0.5, -0.5]))
        ex.step()
        # Physics should have stepped — position won't be exact but
        # actuators should be commanding the target
        assert data.ctrl[actuator_ids[0]] != 0.0

    def test_hold(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids
        )
        # Set some position in data
        for idx in joint_qpos_indices:
            data.qpos[idx] = 0.42
        ex.hold()
        np.testing.assert_allclose(ex.target_position, [0.42, 0.42])

    def test_execute_trajectory(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids, control_dt=0.0
        )
        positions = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ])
        traj = _make_trajectory(positions)
        result = ex.execute(traj)
        assert result is True

    def test_get_position(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids
        )
        pos = ex.get_position()
        assert pos.shape == (2,)

    def test_get_velocity(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids
        )
        vel = ex.get_velocity()
        assert vel.shape == (2,)

    def test_tracking_error(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids
        )
        ex.set_target(np.array([1.0, 1.0]))
        error = ex.get_tracking_error()
        assert error.shape == (2,)
        # Error should be target - current
        np.testing.assert_allclose(
            error, [1.0, 1.0] - ex.get_position()
        )

    def test_lookahead_time(
        self, model_and_data, joint_qpos_indices, actuator_ids
    ):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model, data, joint_qpos_indices, actuator_ids,
            lookahead_time=0.2,
        )
        ex.set_target(np.array([1.0, 0.5]), velocity=np.array([0.5, 0.25]))
        ex.step()
        # Command should be position + lookahead * velocity
        expected_cmd = np.array([1.0 + 0.2 * 0.5, 0.5 + 0.2 * 0.25])
        np.testing.assert_allclose(
            [data.ctrl[aid] for aid in actuator_ids],
            expected_cmd,
        )
