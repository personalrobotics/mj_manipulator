# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Status HUD overlay for the viser browser viewer.

Shows per-arm status: force, held object, and current action/state.
Works with any robot that has ``arms``, ``grasp_manager``, and
optionally ``_active_context.ownership``.

Created automatically by ``start_console()`` when viser is active.
Primitives update it via ``robot._status_hud.set_action(arm, text)``.

Requires: mj_viser (only imported when the HUD is actually created).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class StatusHud:
    """Compact status overlay on the 3D viewport.

    Reads live state from the robot each frame. Actions set by
    primitives auto-expire after ``ACTION_TIMEOUT`` seconds.
    """

    ACTION_TIMEOUT = 5.0

    def __init__(self, robot, mode: str = "") -> None:
        self._robot = robot
        self._mode = mode
        self._actions: dict[str, tuple[str, float]] = {}

    def name(self) -> str:
        return "StatusHud"

    def set_action(self, arm_name: str, text: str) -> None:
        """Set the current action text for an arm."""
        self._actions[arm_name] = (text, time.monotonic())

    def clear_action(self, arm_name: str) -> None:
        """Clear the action text for an arm."""
        self._actions.pop(arm_name, None)

    def setup(self, gui, viewer) -> None:
        """Initialize the HUD overlay."""
        self._viewer = viewer
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def on_sync(self, viewer) -> None:
        """Update the HUD overlay each frame."""
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def _build_status(self) -> str:
        robot = self._robot
        now = time.monotonic()
        parts = []

        for arm_name, arm in robot.arms.items():
            # Label
            if len(robot.arms) == 1:
                label = arm_name
            else:
                label = arm_name[0].upper()

            # F/T magnitude (safe — skip if no sensor or error)
            force_str = ""
            try:
                if getattr(arm, "has_ft_sensor", False):
                    wrench = arm.get_ft_wrench()
                    if not np.isnan(wrench[0]):
                        force_mag = float(np.linalg.norm(wrench[:3]))
                        force_str = f"[{force_mag:.0f}N] "
            except Exception:
                pass

            # Held object
            held_str = ""
            try:
                held = robot.grasp_manager.get_grasped_by(arm_name)
                if held:
                    held_str = held[0]
            except Exception:
                pass

            # Current state from ownership (teleop / executing)
            action = ""
            try:
                ctx = getattr(robot, "_active_context", None)
                if ctx is not None and hasattr(ctx, "ownership") and ctx.ownership is not None:
                    from mj_manipulator.ownership import OwnerKind

                    kind, _ = ctx.ownership.owner_of(arm_name)
                    if kind == OwnerKind.TELEOP:
                        action = "teleop"
                    elif kind == OwnerKind.TRAJECTORY:
                        action = "executing"
            except Exception:
                pass

            # Timed action from primitives (expires after timeout)
            if not action and arm_name in self._actions:
                text, ts = self._actions[arm_name]
                if now - ts < self.ACTION_TIMEOUT:
                    action = text
                else:
                    del self._actions[arm_name]

            # Compose
            status = f"<b>{label}</b>: {force_str}"
            if held_str:
                status += held_str
            if action:
                if held_str:
                    status += f" | {action}"
                else:
                    status += action

            parts.append(status)

        line = " &nbsp;&nbsp; ".join(parts)
        if self._mode:
            line += f" &nbsp;&nbsp; {self._mode}"
        return line
