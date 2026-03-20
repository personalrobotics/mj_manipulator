# Demos

Integration demos using real MuJoCo robot models (UR5e, Franka Panda).

Unlike `tests/` (which are automated, mock-based, CI-friendly), these are
standalone scripts that load real models and show the framework working
end-to-end.

## Running

```bash
cd mj_manipulator
uv run python demos/<script>.py
```

## Available Demos

| Script | What it shows |
|---|---|
| `recycling.py` | **Capstone demo** — full stack integration: prl_assets models, AssetManager-driven TSR grasping, MjSpec scene composition, GraspManager, SimContext. UR5e + Franka each recycle 3 soda cans into a bin. |
| `ik_solver.py` | EAIK analytical IK: kinematic extraction from MuJoCo, multi-config IK with solution analysis, FK round-trip verification |
| `arm_planning.py` | Motion planning with CBiRRT: plan to configuration, plan to pose (via TSRs), trajectory retiming with TOPP-RA |
| `collision_check.py` | Collision checking: simple mode, grasp-aware mode, batch configuration validation |
| `cartesian_control.py` | CartesianController: teleop via step() at 125 Hz, scripted move() with distance limit, move_to() pose targeting; internal Jacobian and QP solver table |
| `sim_context.py` | SimContext execution layer: batch trajectory execution, streaming joint control, streaming cartesian control — physics and kinematic modes |
