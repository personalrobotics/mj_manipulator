# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Scenario runtime wiring.

Discovers user-facing functions in a scenario module and binds the
robot into their first argument so they can be called from the IPython
console as if they were zero-argument (or arg-less) helpers.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from mj_manipulator.robot import ManipulationRobot


def get_user_functions(
    scenario: ModuleType,
    robot: ManipulationRobot,
) -> dict[str, Callable[..., Any]]:
    """Return scenario-defined functions with ``robot`` bound as first arg.

    Scenario functions are written as::

        def sort_all(robot):
            while robot.pickup():
                robot.place("bin")

    This helper wraps each one with :func:`functools.partial` so the
    IPython console can call ``sort_all()`` with no arguments. Functions
    whose first parameter is not named ``robot`` are returned unwrapped
    (they're assumed to be standalone helpers the user defined).

    Only public functions (no leading underscore) defined in the
    scenario module itself (not imports) are returned.
    """
    result: dict[str, Callable[..., Any]] = {}
    for name, obj in inspect.getmembers(scenario, inspect.isfunction):
        if name.startswith("_"):
            continue
        if obj.__module__ != scenario.__name__:
            continue
        try:
            params = list(inspect.signature(obj).parameters.values())
        except (TypeError, ValueError):
            result[name] = obj
            continue
        if params and params[0].name == "robot":
            result[name] = functools.partial(obj, robot)
        else:
            result[name] = obj
    return result


def resolve_spawn_count(scenario: ModuleType | None) -> int | None:
    """Extract ``spawn_count`` from a scenario's scene dict, if present."""
    if scenario is None or not hasattr(scenario, "scene"):
        return None
    scene = scenario.scene
    if not isinstance(scene, dict):
        return None
    return scene.get("spawn_count")
