#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Headless sweep of grip force × pad friction on the recycling task.

Not a pytest — run directly. For each parameter combo, builds a Franka
robot using the demo's own ``build_franka_robot`` (overriding the grip
parameters), runs ``sort_all``, and counts cans that made it into the
tote.

Usage::

    uv run python tests/grip_sweep.py --cans 3 --reps 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import mujoco
import numpy as np


def run_one(force: float, friction: float, n_cans: int, verbose: bool = False) -> tuple[int, float]:
    """Run a single recycling trial with the given (force, friction)."""
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

    # Monkey-patch the two helpers BEFORE calling build_franka_robot so
    # the demo uses our swept defaults. This keeps the sweep's scene
    # setup identical to the demo's — no drift.
    from mj_manipulator.arms import franka as _franka

    _orig_fix = _franka.fix_franka_grip_force
    _orig_pad = _franka.add_franka_pad_friction

    def _patched_fix(model, target_force=force):
        return _orig_fix(model, target_force=force)

    def _patched_pad(spec, *, sliding_friction=friction, torsional_friction=0.2, **kw):
        return _orig_pad(
            spec,
            sliding_friction=friction,
            torsional_friction=torsional_friction,
            **kw,
        )

    _franka.fix_franka_grip_force = _patched_fix
    _franka.add_franka_pad_friction = _patched_pad
    try:
        from mj_manipulator.demos.franka_setup import build_franka_robot

        scene = {
            "objects": {"can": n_cans},
            "fixtures": {"yellow_tote": [[-0.5, 0.0, 0.0]]},
        }
        robot = build_franka_robot(scene)

        # Count successful pickups that survive to a successful place
        # (sim "places" a can by hiding it inside the tote — the registry
        # marks the instance inactive when place succeeds).
        successes = 0
        t0 = time.monotonic()
        with robot.sim(physics=True, headless=True):
            while robot.pickup():
                if robot.place("yellow_tote"):
                    successes += 1
                robot.go_home()
        wall = time.monotonic() - t0
        return successes, wall
    finally:
        _franka.fix_franka_grip_force = _orig_fix
        _franka.add_franka_pad_friction = _orig_pad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cans", type=int, default=3, help="Cans per trial")
    parser.add_argument("--reps", type=int, default=2, help="Reps per (force, friction) combo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Parameter grid
    forces = [50.0, 70.0, 100.0, 140.0]
    frictions = [1.5, 3.0, 4.0]

    print(f"\nSweeping {len(forces)} × {len(frictions)} × {args.reps} reps, {args.cans} cans/trial\n")
    header = (
        f"{'force(N)':>9} | {'friction':>8} | "
        + " | ".join(f"rep{i + 1}" for i in range(args.reps))
        + " | total/max | time(s)"
    )
    print(header)
    print("-" * len(header))

    for force in forces:
        for friction in frictions:
            per_rep = []
            total_time = 0.0
            for rep in range(args.reps):
                try:
                    in_tote, wall = run_one(force, friction, args.cans, verbose=args.verbose)
                except Exception as e:
                    print(f"  ERROR at force={force}, friction={friction}, rep={rep}: {e}")
                    per_rep.append(-1)
                    continue
                per_rep.append(in_tote)
                total_time += wall
            total = sum(r for r in per_rep if r >= 0)
            max_possible = args.cans * sum(1 for r in per_rep if r >= 0)
            per_rep_str = " | ".join(f"{r:>4}" for r in per_rep)
            print(
                f"{force:>9.0f} | {friction:>8.2f} | {per_rep_str} | {total:>3}/{max_possible:<3} | {total_time:>6.1f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
