#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shim that runs ``python -m mj_manipulator --scenario recycling``.

The scenario source lives at
``src/mj_manipulator/demos/recycling.py`` — read it to see the scene
dict and the user-facing ``sort_all`` function.

The Franka robot setup code is at
``src/mj_manipulator/demos/franka_setup.py`` — read it end-to-end to
see what it takes to bring your own manipulator into this framework.

Usage::

    uv run mjpython demos/recycling.py
"""

from __future__ import annotations

import sys

from mj_manipulator.cli import main

if __name__ == "__main__":
    # Ensure the CLI runs this scenario by default, while still allowing
    # the user to pass extra flags (--no-viser, --no-physics, ...).
    if "--scenario" not in sys.argv:
        sys.argv.insert(1, "--scenario")
        sys.argv.insert(2, "recycling")
    main()
