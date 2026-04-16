# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Scenario system: reusable scene + task definitions.

A **scenario** is a Python module with:

- ``scene`` — a dict describing objects (pool), fixtures (stationary),
  and optional ``spawn_count`` (random subset size)
- Optional user-facing functions (``sort_all``, ``stack_all``, ...)

Example::

    # my_scenario.py
    \"\"\"Pick up cans and drop them in the recycle bin.\"\"\"

    scene = {
        "objects": {"can": 5},
        "fixtures": {"recycle_bin": [[0.3, -0.5, 0.0]]},
    }

    def recycle(robot):
        while robot.pickup():
            robot.place("recycle_bin")
        robot.go_home()

The loader discovers, describes, and loads scenario modules. The runner
binds ``robot`` as the first argument of each scenario function so the
IPython console can call ``recycle()`` with no arguments. The spawn
module scatters objects on a robot-provided worktop surface.

Robots integrate by implementing ``setup_scenario_scene(scene)`` and
``get_worktop_pose()`` on their :class:`~mj_manipulator.RobotBase`
subclass.
"""

from mj_manipulator.scenarios.loader import (
    choose_interactive,
    describe,
    discover,
    load,
)
from mj_manipulator.scenarios.runner import (
    get_user_functions,
    resolve_spawn_count,
)
from mj_manipulator.scenarios.spawn import WorktopPose, scatter_on_surface

__all__ = [
    # Loader
    "discover",
    "load",
    "describe",
    "choose_interactive",
    # Runner
    "get_user_functions",
    "resolve_spawn_count",
    # Spawn
    "scatter_on_surface",
    "WorktopPose",
]
