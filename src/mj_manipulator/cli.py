# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CLI entry point for ``python -m mj_manipulator``.

Runs a scenario on a Franka Panda. Scenarios live as Python modules
under :mod:`mj_manipulator.demos` — each defines a ``scene`` dict and
optional user-facing functions. See :mod:`mj_manipulator.scenarios`
for the protocol.

Usage::

    python -m mj_manipulator                           # scenario picker
    python -m mj_manipulator --scenario recycling      # run directly
    python -m mj_manipulator --list-scenarios          # print scenarios and exit
    python -m mj_manipulator --no-physics              # kinematic mode
    python -m mj_manipulator --no-viser                # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mj_manipulator import scenarios

# Directory where this package's scenarios live. Users can add their own
# scenarios by placing Python modules here (or pass --scenario <path>).
_SCENARIO_DIR = Path(__file__).parent / "demos"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mj_manipulator",
        description="Run a manipulation scenario on a Franka Panda.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name (e.g. 'recycling') or path to a .py file. If omitted, shows a picker.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument("--no-physics", action="store_true", help="Kinematic mode (no physics stepping)")
    parser.add_argument("--no-viser", action="store_true", help="Disable viser web viewer")
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    # Resolve the scenario module.
    scenario_module = _resolve_scenario(args.scenario)
    if scenario_module is None:
        sys.exit(0)

    scene = getattr(scenario_module, "scene", None) or {}
    name = scenario_module.__name__

    print(f"\nLoading Franka with scenario '{name}'...", flush=True)

    # Build the robot, apply the scene, launch the console.
    from mj_manipulator.demos.franka_setup import build_franka_robot

    robot = build_franka_robot(scene)

    from mj_manipulator.console import start_console

    user_fns = scenarios.get_user_functions(scenario_module, robot)
    extra_ns = dict(user_fns)
    extra_ns["reset"] = lambda: robot.reset(scene)

    start_console(
        robot,
        physics=not args.no_physics,
        viser=not args.no_viser,
        robot_name="Franka",
        extra_ns=extra_ns,
    )


def _resolve_scenario(name_or_path: str | None):
    """Pick a scenario interactively or load the given name."""
    if name_or_path is None:
        return scenarios.choose_interactive([_SCENARIO_DIR])
    try:
        return scenarios.load(name_or_path, search_dirs=[_SCENARIO_DIR])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _list_scenarios() -> None:
    """Print discovered scenarios with their descriptions."""
    found = scenarios.discover([_SCENARIO_DIR])
    if not found:
        print("No scenarios found.")
        return
    print("\nAvailable scenarios:\n")
    for name, path in found.items():
        print(f"  {name:20s} — {scenarios.describe(path)}")
    print()
