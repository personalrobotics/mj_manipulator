# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for arm ownership registry."""

import threading

import pytest

from mj_manipulator.ownership import OwnerKind, OwnershipRegistry


@pytest.fixture
def registry():
    return OwnershipRegistry(["left", "right"])


class TestAcquireRelease:
    def test_acquire_idle_arm(self, registry):
        owner = object()
        assert registry.acquire("right", OwnerKind.TRAJECTORY, owner)
        kind, ref = registry.owner_of("right")
        assert kind == OwnerKind.TRAJECTORY
        assert ref is owner

    def test_acquire_busy_arm_fails(self, registry):
        owner1 = object()
        owner2 = object()
        registry.acquire("right", OwnerKind.TRAJECTORY, owner1)
        assert not registry.acquire("right", OwnerKind.TELEOP, owner2)

    def test_release_returns_to_idle(self, registry):
        owner = object()
        registry.acquire("right", OwnerKind.TELEOP, owner)
        registry.release("right", owner)
        kind, ref = registry.owner_of("right")
        assert kind == OwnerKind.IDLE
        assert ref is None

    def test_release_wrong_owner_ignored(self, registry):
        owner = object()
        impostor = object()
        registry.acquire("right", OwnerKind.TRAJECTORY, owner)
        registry.release("right", impostor)
        kind, _ = registry.owner_of("right")
        assert kind == OwnerKind.TRAJECTORY

    def test_unknown_arm_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown arm"):
            registry.acquire("middle", OwnerKind.TELEOP, object())

    def test_arms_are_independent(self, registry):
        owner_l = object()
        owner_r = object()
        registry.acquire("left", OwnerKind.TRAJECTORY, owner_l)
        assert registry.acquire("right", OwnerKind.TELEOP, owner_r)
        assert registry.owner_of("left")[0] == OwnerKind.TRAJECTORY
        assert registry.owner_of("right")[0] == OwnerKind.TELEOP


class TestPreempt:
    def test_preempt_idle(self, registry):
        owner = object()
        registry.preempt("right", OwnerKind.TELEOP, owner)
        kind, ref = registry.owner_of("right")
        assert kind == OwnerKind.TELEOP
        assert ref is owner

    def test_teleop_preempts_trajectory(self, registry):
        traj_owner = object()
        teleop_owner = object()
        registry.acquire("right", OwnerKind.TRAJECTORY, traj_owner)
        registry.preempt("right", OwnerKind.TELEOP, teleop_owner)

        kind, ref = registry.owner_of("right")
        assert kind == OwnerKind.TELEOP
        assert ref is teleop_owner
        assert registry.is_aborted("right")

    def test_lower_priority_cannot_preempt(self, registry):
        owner = object()
        registry.acquire("right", OwnerKind.TELEOP, owner)
        with pytest.raises(ValueError, match="Cannot preempt"):
            registry.preempt("right", OwnerKind.TRAJECTORY, object())

    def test_preempt_does_not_affect_other_arm(self, registry):
        left_owner = object()
        right_traj = object()
        right_teleop = object()
        registry.acquire("left", OwnerKind.TRAJECTORY, left_owner)
        registry.acquire("right", OwnerKind.TRAJECTORY, right_traj)
        registry.preempt("right", OwnerKind.TELEOP, right_teleop)

        assert registry.owner_of("left")[0] == OwnerKind.TRAJECTORY
        assert not registry.is_aborted("left")

    def test_gripper_preempts_teleop(self, registry):
        teleop = object()
        gripper = object()
        registry.acquire("right", OwnerKind.TELEOP, teleop)
        registry.preempt("right", OwnerKind.GRIPPER, gripper)
        assert registry.owner_of("right")[0] == OwnerKind.GRIPPER


class TestAbort:
    def test_per_arm_abort(self, registry):
        registry.set_abort("left")
        assert registry.is_aborted("left")
        assert not registry.is_aborted("right")

    def test_clear_abort(self, registry):
        registry.set_abort("right")
        registry.clear_abort("right")
        assert not registry.is_aborted("right")

    def test_abort_all(self, registry):
        registry.abort_all()
        assert registry.is_aborted("left")
        assert registry.is_aborted("right")

    def test_clear_all(self, registry):
        registry.abort_all()
        registry.clear_all()
        assert not registry.is_aborted("left")
        assert not registry.is_aborted("right")

    def test_acquire_clears_abort(self, registry):
        registry.set_abort("right")
        owner = object()
        registry.acquire("right", OwnerKind.TRAJECTORY, owner)
        assert not registry.is_aborted("right")


class TestThreadSafety:
    def test_concurrent_acquire(self, registry):
        """Only one of N concurrent acquires should succeed."""
        results = []
        barrier = threading.Barrier(10)

        def try_acquire(i):
            barrier.wait()
            ok = registry.acquire("right", OwnerKind.TRAJECTORY, i)
            results.append(ok)

        threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 9
