# Demos

Integration demos using real MuJoCo robot models (UR5e, Franka).

Unlike `tests/` (which are automated, mock-based, CI-friendly), these are
standalone scripts that load real models, may open a viewer, and show the
framework working end-to-end.

## Running

```bash
cd mj_manipulator
uv run python demos/<script>.py
```

## Demos by Phase

| Script | Phase | What it shows |
|---|---|---|
| `collision_check.py` | 2 | Load UR5e + Franka, check collisions at various configs |
| `plan_trajectory.py` | 4 | Plan + visualize trajectories for both robots |
| `sim_context.py` | 5 | Execute trajectory, cartesian step, grasp/release with viewer |
| `pickup_place.py` | 6 | Full pickup/place with both robots |
| `franka_e2e.py` | 8 | End-to-end Franka: plan, execute, grasp, teleop, policy |
