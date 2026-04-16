# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Recycle soda cans into a yellow tote.

Scenario for the Franka recycling demo. The ``scene`` dict below is
consumed by :mod:`mj_manipulator.scenarios` — the same mechanism
:mod:`geodude.demos` uses — so this file doubles as a template for your
own scenarios. Functions here take ``robot`` as their first argument;
the scenario runner binds it before exposing them to the IPython
console, so users can call ``sort_all()`` with no arguments.

Usage::

    python -m mj_manipulator                      # scenario picker
    python -m mj_manipulator --scenario recycling # this file
"""

# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------
#
# ``objects``     pool of graspable types (type → count). The scenario
#                 system scatters these on ``robot.get_worktop_pose()``.
# ``fixtures``    stationary objects at fixed poses. Preserved across
#                 resets. Each entry is a list of XYZ positions.
# ``spawn_count`` optional — if set, a random subset of the pool is
#                 activated per run.

scene = {
    "objects": {"can": 5},
    "fixtures": {"yellow_tote": [[-0.5, 0.0, 0.0]]},
}


# ---------------------------------------------------------------------------
# User-facing actions
# ---------------------------------------------------------------------------


def sort_all(robot) -> None:
    """Pick up every can and drop it in the yellow tote."""
    while robot.pickup():
        robot.place("yellow_tote")
        robot.go_home()
    robot.go_home()
