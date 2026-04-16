# Demos

Runnable demos for mj_manipulator. The capstone is the **recycling
scenario**; the other scripts are short references for individual APIs.

## Quick start

```bash
# Launch the scenario picker (lists everything and prompts)
python -m mj_manipulator

# Run the recycling scenario directly
python -m mj_manipulator --scenario recycling

# Headless (no browser viewer)
python -m mj_manipulator --scenario recycling --no-viser

# Kinematic mode (no physics stepping)
python -m mj_manipulator --scenario recycling --no-physics
```

The top-level `demos/recycling.py` is a thin shim; `python -m mj_manipulator`
is the canonical entry point.

## How the capstone demo is organized

Scenarios are Python modules with a `scene = {...}` dict and optional
user-facing functions. The framework handles discovery, loading, and
wiring. To see what that looks like:

| File | What it does |
|---|---|
| [`../src/mj_manipulator/demos/recycling.py`](../src/mj_manipulator/demos/recycling.py) | The scenario file — `scene` dict + `sort_all(robot)` function |
| [`../src/mj_manipulator/demos/franka_setup.py`](../src/mj_manipulator/demos/franka_setup.py) | Franka-specific robot assembly. Read end-to-end to see what it takes to bring your own arm into the framework |
| [`../src/mj_manipulator/scenarios/`](../src/mj_manipulator/scenarios/) | The scenario system itself — loader, runner, spawn |
| [`../src/mj_manipulator/cli.py`](../src/mj_manipulator/cli.py) | Glue that wires scenarios + robot + console |

To bring your own arm, copy `franka_setup.py` and swap the arm factory,
gripper, home pose, and worktop. To write your own scenario, drop a
`my_scene.py` into `src/mj_manipulator/demos/` with a `scene` dict and
your user-facing functions.

## Reference scripts

Short examples of individual APIs. Each exercises one subsystem.

| Script | What it shows |
|---|---|
| [`ik_solver.py`](ik_solver.py) | EAIK analytical IK: kinematic extraction, multi-config solve, FK round-trip |
| [`arm_planning.py`](arm_planning.py) | CBiRRT motion planning: plan to configuration, plan to pose via TSRs, TOPP-RA retiming |
| [`collision_check.py`](collision_check.py) | CollisionChecker: simple mode, grasp-aware mode, batch validation |

```bash
uv run mjpython demos/<script>.py
```

Use `mjpython` (not plain `python`) on macOS so the viser viewer can
render — it's MuJoCo's main-thread launcher.
