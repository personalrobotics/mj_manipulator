# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for SimPerceptionService."""

from __future__ import annotations

import mujoco
import pytest

from mj_manipulator.perception import SimPerceptionService
from mj_manipulator.protocols import PerceptionService

_XML = """
<mujoco>
  <worldbody>
    <body name="can_0" pos="0.5 0 0.1">
      <freejoint/>
      <geom size="0.03 0.06" type="cylinder"/>
    </body>
    <body name="can_1" pos="0.3 0.2 0.1">
      <freejoint/>
      <geom size="0.03 0.06" type="cylinder"/>
    </body>
    <body name="box_0" pos="-0.3 0 0.1">
      <freejoint/>
      <geom size="0.05 0.05 0.05" type="box"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeRegistry:
    def __init__(self):
        self.active_objects: dict[str, bool] = {
            "can_0": True,
            "can_1": True,
            "box_0": False,
        }
        self.objects: dict = {
            "can": {"instances": ["can_0", "can_1"]},
            "box": {"instances": ["box_0"]},
        }

    def hide(self, name: str) -> None:
        self.active_objects[name] = False


class _FakeEnv:
    """Minimal environment stub for perception tests."""

    def __init__(self, model, data, registry=None):
        self.model = model
        self.data = data
        self.registry = registry

    def update(self, object_list, hide_unlisted=None):
        """Stub — in real env this writes qpos + calls mj_forward."""
        pass


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def env(model_and_data):
    model, data = model_and_data
    return _FakeEnv(model, data, registry=_FakeRegistry())


@pytest.fixture
def perception(env):
    return SimPerceptionService(env)


class TestGetPose:
    def test_active_object_returns_pose(self, perception):
        pose = perception.get_pose("can_0")
        assert pose is not None
        assert pose.shape == (4, 4)
        assert abs(pose[0, 3] - 0.5) < 0.01

    def test_hidden_object_returns_none(self, perception):
        assert perception.get_pose("box_0") is None

    def test_unknown_object_returns_none(self, perception):
        assert perception.get_pose("nonexistent") is None

    def test_without_registry(self, model_and_data):
        model, data = model_and_data
        env = _FakeEnv(model, data, registry=None)
        p = SimPerceptionService(env)
        assert p.get_pose("can_0") is not None
        assert p.get_pose("box_0") is not None


class TestRefresh:
    def test_refresh_runs_pipeline(self, env):
        """refresh() should produce tracker updates from ground truth."""
        p = SimPerceptionService(env)
        # Should not raise — runs mock detection → tracker → env.update
        p.refresh()

    def test_refresh_without_registry_is_noop(self, model_and_data):
        model, data = model_and_data
        env = _FakeEnv(model, data, registry=None)
        p = SimPerceptionService(env)
        p.refresh()  # no-op, no crash


class TestProtocol:
    def test_satisfies_protocol(self, perception):
        assert isinstance(perception, PerceptionService)
