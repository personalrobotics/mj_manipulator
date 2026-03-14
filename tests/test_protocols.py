"""Tests for protocol definitions.

Verifies that the protocols are well-formed and that example implementations
satisfy them correctly (using runtime_checkable).
"""

import numpy as np

from mj_manipulator.protocols import Gripper, GraspSource, IKSolver


class MockGripper:
    """Minimal gripper implementation for protocol testing."""

    def __init__(self):
        self._holding = False
        self._held = None

    @property
    def arm_name(self) -> str:
        return "test_arm"

    @property
    def gripper_body_names(self) -> list[str]:
        return ["gripper/left_finger", "gripper/right_finger"]

    @property
    def attachment_body(self) -> str:
        return "gripper/right_finger"

    @property
    def actuator_id(self) -> int | None:
        return 0

    @property
    def ctrl_open(self) -> float:
        return 0.0

    @property
    def ctrl_closed(self) -> float:
        return 255.0

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def held_object(self) -> str | None:
        return self._held

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        pass

    def kinematic_close(self, steps: int = 50) -> str | None:
        self._holding = True
        self._held = "test_object"
        return "test_object"

    def kinematic_open(self) -> None:
        self._holding = False
        self._held = None

    def get_actual_position(self) -> float:
        return 0.0 if not self._holding else 0.7


class MockIKSolver:
    """Minimal IK solver for protocol testing."""

    def solve(self, pose, q_init=None):
        return [np.zeros(6)]

    def solve_valid(self, pose, q_init=None):
        return [np.zeros(6)]


class MockGraspSource:
    """Minimal grasp source for protocol testing."""

    def get_grasps(self, object_name, hand_type):
        return []

    def get_placements(self, destination, object_name):
        return []

    def get_graspable_objects(self):
        return ["mug", "can"]

    def get_place_destinations(self, object_name):
        return ["table", "bin"]


class TestGripperProtocol:
    """Tests for the Gripper protocol."""

    def test_mock_satisfies_protocol(self):
        """MockGripper satisfies the Gripper protocol."""
        gripper = MockGripper()
        assert isinstance(gripper, Gripper)

    def test_gripper_lifecycle(self):
        """Basic open/close lifecycle works."""
        gripper = MockGripper()
        assert not gripper.is_holding
        assert gripper.held_object is None

        result = gripper.kinematic_close()
        assert result == "test_object"
        assert gripper.is_holding
        assert gripper.held_object == "test_object"

        gripper.kinematic_open()
        assert not gripper.is_holding

    def test_attachment_body(self):
        """attachment_body is accessible."""
        gripper = MockGripper()
        assert gripper.attachment_body == "gripper/right_finger"


class TestIKSolverProtocol:
    """Tests for the IKSolver protocol."""

    def test_mock_satisfies_protocol(self):
        """MockIKSolver satisfies the IKSolver protocol."""
        solver = MockIKSolver()
        assert isinstance(solver, IKSolver)

    def test_solve_returns_list(self):
        """solve() returns a list of configurations."""
        solver = MockIKSolver()
        solutions = solver.solve(np.eye(4))
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        assert solutions[0].shape == (6,)


class TestGraspSourceProtocol:
    """Tests for the GraspSource protocol."""

    def test_mock_satisfies_protocol(self):
        """MockGraspSource satisfies the GraspSource protocol."""
        source = MockGraspSource()
        assert isinstance(source, GraspSource)

    def test_get_graspable_objects(self):
        """Can query available objects."""
        source = MockGraspSource()
        objects = source.get_graspable_objects()
        assert "mug" in objects
        assert "can" in objects

    def test_get_place_destinations(self):
        """Can query placement destinations."""
        source = MockGraspSource()
        destinations = source.get_place_destinations("mug")
        assert "table" in destinations
