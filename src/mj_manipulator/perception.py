# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Perception service implementations.

:class:`SimPerceptionService` runs the full perception pipeline in
sim: a mock client reads ground-truth poses from ``data.xpos``,
``AssetManager.resolve_alias()`` canonicalizes type labels, the
:class:`~mj_environment.tracker.ObjectTracker` assigns instance
identities, and ``env.update()`` writes the results to the kinematic
twin. This is the same pipeline hardware uses — the only difference
is what produces the raw detections (ground truth vs camera).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager


class SimPerceptionService:
    """Sim-side perception backed by the full tracker pipeline.

    ``refresh()`` runs the complete perception pipeline:

    1. **Mock detection**: read ground-truth poses from ``data.xpos``
       for active, non-held objects. Emit perception-space type
       labels (the same strings a real detector would produce).
    2. **Alias resolution**: ``AssetManager.resolve_alias()``
       canonicalizes labels (e.g. ``"soda can"`` → ``"can"``).
    3. **Tracker**: ``ObjectTracker.associate()`` matches detections
       to persistent instance slots by type + proximity.
    4. **Write**: ``env.update()`` writes poses to qpos and calls
       ``mj_forward``.

    ``get_pose()`` reads ``data.xpos`` — always current after
    ``refresh()`` calls ``env.update()`` → ``mj_forward``.

    Args:
        env: MuJoCo environment (owns model, data, registry).
        grasp_manager: For excluding held objects from refresh.
        asset_manager: For alias resolution. If None, aliases are
            not resolved (canonical types are used directly).
    """

    def __init__(
        self,
        env: Environment,
        grasp_manager: GraspManager | None = None,
        asset_manager=None,
        fixture_types: set[str] | None = None,
    ):
        self._env = env
        self._model = env.model
        self._data = env.data
        self._registry = getattr(env, "registry", None)
        self._gm = grasp_manager
        self._am = asset_manager
        self._fixture_types = fixture_types or set()

        # Create tracker if we have a registry. No seeding needed —
        # refresh() uses hide_unlisted=True which clears stale objects
        # and re-detects from scratch each call.
        self._tracker = None
        if self._registry is not None:
            from mj_environment.tracker import ObjectTracker

            self._tracker = ObjectTracker(self._registry)

    def refresh(self) -> None:
        """Observe the scene and update the kinematic twin.

        Runs the full perception pipeline — same steps as hardware,
        just with ground-truth detections instead of a camera.
        """
        if self._registry is None or self._tracker is None:
            return

        # Step 1: mock detection — read poses BEFORE clearing state
        raw_detections = self._mock_detect()

        # Step 2: hide all non-held, non-fixture objects and reset
        # the tracker. This frees instance slots so the tracker can
        # re-assign them from fresh detections. Without this,
        # _next_available() sees all slots as active and drops every
        # detection.
        for name, active in list(self._registry.active_objects.items()):
            if not active:
                continue
            if self._gm is not None and self._gm.is_grasped(name):
                continue
            obj_type = self._parse_type(name)
            if obj_type in self._fixture_types:
                continue
            self._registry.hide(name)
        self._tracker.reset()

        # Step 3: alias resolution — canonicalize labels
        detections = []
        for det in raw_detections:
            canonical = self._resolve_type(det["type"])
            if canonical is None:
                continue
            detections.append(
                {
                    "type": canonical,
                    "pos": det["pos"],
                    "quat": det.get("quat", [1, 0, 0, 0]),
                }
            )

        # Step 4: tracker assigns instance identities
        updates = self._tracker.associate(detections)

        # Step 5: preserve objects that hide_unlisted=True would
        # otherwise remove — held objects (camera can't see them)
        # and fixtures (recycle bins, worktops — not detected by
        # perception, placed by the operator at known positions).
        for name, active in self._registry.active_objects.items():
            if not active:
                continue
            if any(u["name"] == name for u in updates):
                continue
            is_held = self._gm is not None and self._gm.is_grasped(name)
            obj_type = self._parse_type(name)
            is_fixture = obj_type in self._fixture_types
            if is_held or is_fixture:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    updates.append(
                        {
                            "name": name,
                            "pos": self._data.xpos[bid].tolist(),
                        }
                    )

        # Step 6: write to kinematic twin — hide_unlisted=True clears
        # objects not in updates (disappeared from the table, stale
        # spawn). Held objects and fixtures are preserved above.
        self._env.update(updates, hide_unlisted=True)

    def get_pose(self, name: str) -> np.ndarray | None:
        """Return 4×4 world-frame pose, or None if not active."""
        if self._registry is not None:
            if not self._registry.active_objects.get(name, False):
                return None

        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None

        T = np.eye(4)
        T[:3, :3] = self._data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self._data.xpos[bid]
        return T

    # -- Internal helpers ----------------------------------------------------

    def _mock_detect(self) -> list[dict]:
        """Read ground-truth poses and emit perception-space labels.

        Produces the same ``[{type, pos, quat}]`` format a real
        camera would. The type label is the first perception alias
        from meta.yaml (if asset_manager is available), otherwise
        the canonical type. This exercises the alias resolution
        path on every sim run.
        """
        detections = []
        for name, active in self._registry.active_objects.items():
            if not active:
                continue
            if self._gm is not None and self._gm.is_grasped(name):
                continue
            canonical_type = self._parse_type(name)
            if canonical_type in self._fixture_types:
                continue
            if canonical_type is None:
                continue
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue

            # Emit a perception-space label (first alias if available)
            label = self._get_perception_label(canonical_type)
            pos = self._data.xpos[bid].tolist()
            quat = self._data.xquat[bid].tolist()  # [w, x, y, z]
            detections.append({"type": label, "pos": pos, "quat": quat})
        return detections

    def _get_perception_label(self, canonical_type: str) -> str:
        """Get the perception-space label for a canonical type.

        If asset_manager is available and the type has perception
        aliases, return the first alias (simulating what a real
        detector would output). Otherwise return the canonical type.
        """
        if self._am is None:
            return canonical_type
        try:
            meta = self._am.get(canonical_type)
            perception = meta.get("perception", {})
            if isinstance(perception, dict):
                aliases = perception.get("aliases", [])
                if aliases and isinstance(aliases[0], str):
                    return aliases[0]
        except (KeyError, TypeError):
            pass
        return canonical_type

    def _resolve_type(self, label: str) -> str | None:
        """Resolve a perception label to canonical type.

        Uses AssetManager.resolve_alias if available, otherwise
        treats the label as a canonical type directly.
        """
        if self._am is not None:
            return self._am.resolve_alias(label)
        return label

    @staticmethod
    def _parse_type(name: str) -> str | None:
        """Extract object type from instance name (e.g. "can_0" → "can")."""
        m = re.match(r"^(.+?)_(\d+)$", name)
        return m.group(1) if m else None
