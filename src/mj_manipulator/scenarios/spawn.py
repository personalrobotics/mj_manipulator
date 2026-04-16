# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic object scattering on a worktop surface.

Given a pool of object types and counts, activate them at stable,
non-colliding poses on a surface provided by the robot.

This is the generic shape of what geodude's ``_spawn_manipulable_objects``
and the Franka demo's ``scatter_objects`` were both doing. The robot
supplies the worktop (a 4×4 pose + rectangular size); this module
handles the pool sampling, stable-placement TSR, and collision loop.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorktopPose:
    """A rectangular placement surface in world frame.

    Attributes:
        pose: 4×4 transform of the surface center in world frame.
        size: (width, length) in meters; half-extents along the surface's
            local x and y axes.
    """

    pose: np.ndarray
    size: tuple[float, float]


def scatter_on_surface(
    env,
    objects: dict[str, int],
    fixture_types: set[str],
    *,
    worktop: WorktopPose,
    spawn_count: int | None = None,
    margin: float = 0.05,
    min_separation: float = 0.06,
    max_placement_attempts: int = 50,
) -> list[str]:
    """Scatter graspable objects on a worktop surface.

    Activates objects in ``env.registry`` at stable, non-overlapping
    poses. Skips fixture types (those are placed separately by the
    robot's scene setup).

    Args:
        env: :class:`mj_environment.Environment` with an active registry.
        objects: Pool — object type → count.
        fixture_types: Types to skip (placed as fixtures, not scattered).
        worktop: Placement surface from the robot.
        spawn_count: If given, random-sample this many instances from
            the pool (with replacement). If ``None``, spawn every
            object in the pool.
        margin: Inset from worktop edges (meters).
        min_separation: Minimum XY distance between placed objects.
        max_placement_attempts: Retries per object before giving up.

    Returns:
        Names of the activated objects.
    """
    if env.registry is None:
        raise RuntimeError(
            "env.registry is None — scatter_on_surface requires an Environment "
            "built with an object catalog (e.g. Environment.from_spec with objects_dir)."
        )

    graspable = [(t, n) for t, n in objects.items() if t not in fixture_types]
    if not graspable:
        return []

    if spawn_count is not None:
        available = [t for t, _ in graspable]
        specs = list(Counter(random.choices(available, k=spawn_count)).items())
    else:
        specs = graspable

    # Build an AssetManager + StablePlacer scoped to the worktop.
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import StablePlacer

    assets = AssetManager(str(OBJECTS_DIR))
    placer = StablePlacer(
        max(worktop.size[0] - margin, 0.0),
        max(worktop.size[1] - margin, 0.0),
    )

    activated: list[str] = []
    placed_positions: list[np.ndarray] = []

    for obj_type, count in specs:
        try:
            gp = assets.get(obj_type)["geometric_properties"]
        except (KeyError, TypeError):
            logger.warning("Scatter: no geometry metadata for %s; skipping", obj_type)
            continue

        templates = _templates_for(placer, gp)
        if not templates:
            continue

        for _ in range(count):
            # Sample a stable pose on the worktop surface, avoiding
            # collisions with previously placed objects.
            tsr = templates[0].instantiate(worktop.pose)
            T = tsr.sample()
            for _attempt in range(max_placement_attempts):
                if _far_enough(T[:3, 3], placed_positions, min_separation):
                    break
                T = tsr.sample()

            name = env.registry.activate(obj_type, pos=list(T[:3, 3]))
            placed_positions.append(T[:3, 3].copy())
            activated.append(name)

            # Write orientation from the TSR sample
            _write_orientation(env, name, T)

            # Resolve any residual collisions with the floor or earlier objects
            _nudge_until_clear(env, name, tsr)

    mujoco.mj_forward(env.model, env.data)
    return activated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _templates_for(placer, gp: dict):
    """Build stable-placement templates for an object's geometry."""
    geo = gp.get("type")
    if geo == "cylinder":
        return placer.place_cylinder(gp["radius"], gp["height"])
    if geo == "box":
        return placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
    return []


def _far_enough(pos: np.ndarray, others: list[np.ndarray], min_sep: float) -> bool:
    return all(np.linalg.norm(pos[:2] - o[:2]) > min_sep for o in others)


def _write_orientation(env, body_name: str, T: np.ndarray) -> None:
    """Write the rotation from ``T`` to the freejoint's quat slots."""
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    jnt_id = env.model.body_jntadr[bid]
    qpos_adr = env.model.jnt_qposadr[jnt_id]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat


def _nudge_until_clear(env, body_name: str, tsr, max_attempts: int = 50) -> None:
    """Resample pose until the body has no penetrating contacts."""
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return
    jnt_id = env.model.body_jntadr[bid]
    qpos_adr = env.model.jnt_qposadr[jnt_id]

    mujoco.mj_forward(env.model, env.data)
    for _ in range(max_attempts):
        if not _has_bad_contact(env.model, env.data, bid):
            return
        T = tsr.sample()
        env.data.qpos[qpos_adr : qpos_adr + 3] = T[:3, 3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        mujoco.mj_forward(env.model, env.data)


def _has_bad_contact(model, data, body_id: int) -> bool:
    """True if the body is penetrating any non-world body."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 == body_id or b2 == body_id) and c.dist < 0:
            other = b2 if b1 == body_id else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
            if other_name and other_name != "world":
                return True
    return False
