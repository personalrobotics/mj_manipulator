# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Robotiq 2F-140 gripper implementation.

Supports both physics-mode (actuator-driven) and kinematic-mode (trajectory
replay) operation. The 4-bar linkage geometry is replayed from a pre-recorded
physics trajectory for kinematic mode, ensuring correct joint coupling.

Usage:
    from mj_manipulator.grippers.robotiq import RobotiqGripper

    gripper = RobotiqGripper(model, data, "ur5e", prefix="right_ur5e/gripper/")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.grippers._base import _BaseGripper

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager

# ---------------------------------------------------------------------------
# Robotiq 2F-140 constants
# ---------------------------------------------------------------------------

# Gripper joint names (suffixes, without prefix). Order matches trajectory columns.
_JOINT_SUFFIXES = [
    "left_coupler_joint",
    "left_driver_joint",
    "left_follower_joint",
    "left_spring_link_joint",
    "right_coupler_joint",
    "right_driver_joint",
    "right_follower_joint",
    "right_spring_link_joint",
]

# Body name suffixes for contact detection and collision filtering.
_BODY_SUFFIXES = [
    "base_mount",
    "base",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
]

# Attachment body for kinematic tracking. Using base_mount (gripper center)
# rather than a finger pad — base_mount doesn't move when fingers close,
# so the attachment offset is stable regardless of finger position.
_ATTACHMENT_BODY_SUFFIX = "base_mount"

# Pre-recorded gripper joint trajectory from physics simulation.
# Shape: (101, 8) — 101 waypoints from open (t=0) to closed (t=1), 8 joints.
# Joint order matches _JOINT_SUFFIXES above.
#
# This specific table is for the **Robotiq 2F-140** (geodude_assets
# model). The 2F-85 has inverted joint signs and needs its own table;
# record it via ``scripts/record_gripper_trajectory.py`` and pass via
# ``RobotiqGripper(..., trajectory=...)``.
#
# Recorded on geodude_assets' assembled Geodude model with the 2F-140's
# real 0.7 rad driver travel (personalrobotics/geodude_assets#48), under
# MuJoCo 3.14, from the right gripper at the "ready" keyframe: 200 steps
# open, then ctrl ramped 0 -> 255 over 101 waypoints of 20 steps each, as
# geodude's original recording did, then held at 255 until every finger
# joint varied by at most 1e-6 rad over 200 ms. That open-to-rest path is
# resampled at 101 driver angles evenly spaced from open (-0.0019 rad) to
# closed (-0.6998 rad, the pads together), so waypoint t has driver
# fraction t, as get_actual_position assumes. Without the hold, the ramp
# alone ends 0.046 rad short of closed under the drive's kv damping.
_2F140_TRAJECTORY = np.array(
    [
        [-0.000618, -0.001914, 0.006939, -0.003526, -0.000617, -0.001913, 0.006873, -0.003505],
        [-0.000609, -0.008892, 0.014361, -0.010540, -0.000608, -0.008892, 0.014263, -0.010508],
        [-0.000606, -0.015871, 0.021869, -0.017560, -0.000605, -0.015871, 0.021738, -0.017516],
        [-0.000617, -0.022850, 0.030109, -0.025143, -0.000616, -0.022850, 0.029944, -0.025087],
        [-0.000604, -0.029829, 0.036857, -0.031599, -0.000603, -0.029828, 0.036663, -0.031532],
        [-0.000709, -0.036808, 0.045303, -0.040246, -0.000708, -0.036807, 0.045080, -0.040167],
        [-0.000599, -0.043787, 0.052349, -0.046035, -0.000598, -0.043786, 0.052090, -0.045945],
        [-0.000598, -0.050765, 0.059456, -0.052817, -0.000597, -0.050765, 0.059171, -0.052716],
        [-0.000600, -0.057744, 0.066889, -0.059800, -0.000599, -0.057744, 0.066575, -0.059689],
        [-0.000600, -0.064723, 0.074397, -0.066838, -0.000599, -0.064723, 0.074055, -0.066715],
        [-0.000698, -0.071702, 0.082698, -0.075498, -0.000697, -0.071701, 0.082330, -0.075364],
        [-0.000654, -0.078681, 0.090422, -0.081986, -0.000653, -0.078680, 0.090024, -0.081840],
        [-0.000607, -0.085660, 0.097642, -0.088638, -0.000606, -0.085659, 0.097215, -0.088481],
        [-0.000594, -0.092639, 0.104877, -0.095480, -0.000593, -0.092638, 0.104423, -0.095313],
        [-0.000591, -0.099617, 0.112176, -0.102411, -0.000590, -0.099617, 0.111697, -0.102233],
        [-0.000590, -0.106596, 0.119530, -0.109389, -0.000589, -0.106596, 0.119027, -0.109201],
        [-0.000590, -0.113575, 0.126922, -0.116395, -0.000589, -0.113575, 0.126394, -0.116196],
        [-0.000590, -0.120554, 0.134337, -0.123417, -0.000589, -0.120553, 0.133786, -0.123207],
        [-0.000589, -0.127533, 0.141765, -0.130450, -0.000588, -0.127532, 0.141190, -0.130229],
        [-0.000589, -0.134512, 0.149199, -0.137488, -0.000588, -0.134511, 0.148601, -0.137257],
        [-0.000589, -0.141490, 0.156638, -0.144532, -0.000588, -0.141490, 0.156017, -0.144290],
        [-0.000589, -0.148469, 0.164075, -0.151578, -0.000588, -0.148469, 0.163433, -0.151325],
        [-0.000588, -0.155448, 0.171513, -0.158626, -0.000587, -0.155448, 0.170848, -0.158363],
        [-0.000588, -0.162427, 0.178948, -0.165676, -0.000587, -0.162427, 0.178262, -0.165402],
        [-0.000587, -0.169406, 0.186380, -0.172725, -0.000586, -0.169405, 0.185673, -0.172441],
        [-0.000587, -0.176385, 0.193809, -0.179776, -0.000586, -0.176384, 0.193082, -0.179481],
        [-0.000586, -0.183364, 0.201235, -0.186827, -0.000585, -0.183363, 0.200488, -0.186522],
        [-0.000586, -0.190342, 0.208658, -0.193878, -0.000585, -0.190342, 0.207891, -0.193562],
        [-0.000585, -0.197321, 0.216076, -0.200928, -0.000584, -0.197321, 0.215291, -0.200602],
        [-0.000585, -0.204300, 0.223491, -0.207979, -0.000584, -0.204300, 0.222687, -0.207643],
        [-0.000584, -0.211279, 0.230902, -0.215029, -0.000583, -0.211278, 0.230080, -0.214682],
        [-0.000584, -0.218258, 0.238310, -0.222079, -0.000583, -0.218257, 0.237470, -0.221722],
        [-0.000583, -0.225237, 0.245714, -0.229129, -0.000582, -0.225236, 0.244857, -0.228762],
        [-0.000582, -0.232215, 0.253114, -0.236178, -0.000582, -0.232215, 0.252241, -0.235801],
        [-0.000582, -0.239194, 0.260511, -0.243228, -0.000581, -0.239194, 0.259621, -0.242840],
        [-0.000581, -0.246173, 0.267905, -0.250277, -0.000580, -0.246173, 0.266998, -0.249878],
        [-0.000581, -0.253152, 0.275295, -0.257325, -0.000580, -0.253152, 0.274373, -0.256917],
        [-0.000580, -0.260131, 0.282681, -0.264373, -0.000579, -0.260130, 0.281744, -0.263955],
        [-0.000580, -0.267110, 0.290064, -0.271421, -0.000579, -0.267109, 0.289112, -0.270992],
        [-0.000579, -0.274089, 0.297444, -0.278468, -0.000578, -0.274088, 0.296478, -0.278029],
        [-0.000578, -0.281067, 0.304820, -0.285515, -0.000577, -0.281067, 0.303840, -0.285066],
        [-0.000578, -0.288046, 0.312193, -0.292561, -0.000577, -0.288046, 0.311199, -0.292102],
        [-0.000577, -0.295025, 0.319563, -0.299607, -0.000576, -0.295025, 0.318556, -0.299138],
        [-0.000577, -0.302004, 0.326929, -0.306653, -0.000576, -0.302003, 0.325910, -0.306174],
        [-0.000576, -0.308983, 0.334293, -0.313699, -0.000575, -0.308982, 0.333261, -0.313209],
        [-0.000575, -0.315962, 0.341653, -0.320744, -0.000574, -0.315961, 0.340610, -0.320244],
        [-0.000575, -0.322941, 0.349010, -0.327788, -0.000574, -0.322940, 0.347956, -0.327279],
        [-0.000574, -0.329919, 0.356365, -0.334833, -0.000573, -0.329919, 0.355299, -0.334313],
        [-0.000574, -0.336898, 0.363716, -0.341877, -0.000573, -0.336898, 0.362639, -0.341347],
        [-0.000573, -0.343877, 0.371064, -0.348920, -0.000572, -0.343877, 0.369977, -0.348381],
        [-0.000572, -0.350856, 0.378410, -0.355963, -0.000571, -0.350855, 0.377313, -0.355414],
        [-0.000572, -0.357835, 0.385752, -0.363006, -0.000571, -0.357834, 0.384646, -0.362446],
        [-0.000571, -0.364814, 0.393092, -0.370049, -0.000570, -0.364813, 0.391976, -0.369479],
        [-0.000571, -0.371792, 0.400428, -0.377091, -0.000570, -0.371792, 0.399304, -0.376511],
        [-0.000570, -0.378771, 0.407762, -0.384133, -0.000569, -0.378771, 0.406630, -0.383543],
        [-0.000569, -0.385750, 0.415094, -0.391174, -0.000568, -0.385750, 0.413953, -0.390574],
        [-0.000569, -0.392729, 0.422422, -0.398215, -0.000568, -0.392729, 0.421274, -0.397605],
        [-0.000568, -0.399708, 0.429748, -0.405256, -0.000567, -0.399707, 0.428593, -0.404636],
        [-0.000567, -0.406687, 0.437071, -0.412296, -0.000567, -0.406686, 0.435909, -0.411666],
        [-0.000567, -0.413666, 0.444391, -0.419337, -0.000566, -0.413665, 0.443223, -0.418696],
        [-0.000566, -0.420644, 0.451709, -0.426376, -0.000565, -0.420644, 0.450535, -0.425726],
        [-0.000566, -0.427623, 0.459025, -0.433416, -0.000565, -0.427623, 0.457844, -0.432755],
        [-0.000565, -0.434602, 0.466337, -0.440455, -0.000564, -0.434602, 0.465152, -0.439784],
        [-0.000564, -0.441581, 0.473648, -0.447494, -0.000563, -0.441580, 0.472457, -0.446813],
        [-0.000564, -0.448560, 0.480956, -0.454533, -0.000563, -0.448559, 0.479761, -0.453842],
        [-0.000563, -0.455539, 0.488261, -0.461572, -0.000562, -0.455538, 0.487062, -0.460871],
        [-0.000563, -0.462517, 0.495564, -0.468610, -0.000562, -0.462517, 0.494361, -0.467899],
        [-0.000562, -0.469496, 0.502865, -0.475648, -0.000561, -0.469496, 0.501658, -0.474927],
        [-0.000561, -0.476475, 0.510163, -0.482686, -0.000560, -0.476475, 0.508954, -0.481954],
        [-0.000561, -0.483454, 0.517459, -0.489723, -0.000560, -0.483454, 0.516247, -0.488981],
        [-0.000560, -0.490433, 0.524753, -0.496760, -0.000559, -0.490432, 0.523538, -0.496008],
        [-0.000559, -0.497412, 0.532044, -0.503797, -0.000558, -0.497411, 0.530827, -0.503035],
        [-0.000559, -0.504391, 0.539333, -0.510834, -0.000558, -0.504390, 0.538115, -0.510061],
        [-0.000558, -0.511369, 0.546619, -0.517870, -0.000557, -0.511369, 0.545400, -0.517088],
        [-0.000557, -0.518348, 0.553904, -0.524906, -0.000557, -0.518348, 0.552684, -0.524113],
        [-0.000557, -0.525327, 0.561186, -0.531942, -0.000556, -0.525327, 0.559966, -0.531139],
        [-0.000556, -0.532306, 0.568466, -0.538978, -0.000555, -0.532305, 0.567245, -0.538164],
        [-0.000555, -0.539285, 0.575743, -0.546014, -0.000555, -0.539284, 0.574523, -0.545190],
        [-0.000555, -0.546264, 0.583019, -0.553049, -0.000554, -0.546263, 0.581800, -0.552214],
        [-0.000554, -0.553242, 0.590292, -0.560084, -0.000553, -0.553242, 0.589074, -0.559239],
        [-0.000554, -0.560221, 0.597564, -0.567119, -0.000553, -0.560221, 0.596347, -0.566264],
        [-0.000553, -0.567200, 0.604833, -0.574154, -0.000552, -0.567200, 0.603619, -0.573288],
        [-0.000552, -0.574179, 0.612100, -0.581189, -0.000551, -0.574179, 0.610889, -0.580312],
        [-0.000552, -0.581158, 0.619366, -0.588223, -0.000551, -0.581157, 0.618157, -0.587336],
        [-0.000551, -0.588137, 0.626629, -0.595258, -0.000550, -0.588136, 0.625423, -0.594360],
        [-0.000550, -0.595116, 0.633890, -0.602292, -0.000549, -0.595115, 0.632688, -0.601384],
        [-0.000550, -0.602094, 0.641149, -0.609326, -0.000549, -0.602094, 0.639951, -0.608407],
        [-0.000549, -0.609073, 0.648406, -0.616360, -0.000548, -0.609073, 0.647212, -0.615430],
        [-0.000548, -0.616052, 0.655661, -0.623393, -0.000547, -0.616052, 0.654472, -0.622453],
        [-0.000547, -0.623031, 0.662914, -0.630427, -0.000547, -0.623030, 0.661730, -0.629476],
        [-0.000547, -0.630010, 0.670165, -0.637460, -0.000546, -0.630009, 0.668987, -0.636498],
        [-0.000546, -0.636989, 0.677414, -0.644493, -0.000545, -0.636988, 0.676242, -0.643521],
        [-0.000545, -0.643968, 0.684661, -0.651526, -0.000544, -0.643967, 0.683495, -0.650543],
        [-0.000545, -0.650946, 0.691906, -0.658559, -0.000544, -0.650946, 0.690747, -0.657565],
        [-0.000549, -0.657925, 0.699115, -0.665532, -0.000548, -0.657925, 0.697963, -0.664526],
        [-0.000550, -0.664904, 0.706405, -0.672619, -0.000549, -0.664904, 0.705261, -0.671602],
        [-0.000552, -0.671883, 0.713693, -0.679707, -0.000551, -0.671882, 0.712558, -0.678679],
        [-0.000553, -0.678862, 0.720980, -0.686795, -0.000552, -0.678861, 0.719853, -0.685755],
        [-0.000555, -0.685841, 0.728265, -0.693882, -0.000554, -0.685840, 0.727147, -0.692832],
        [-0.000556, -0.692819, 0.735548, -0.700970, -0.000555, -0.692819, 0.734440, -0.699907],
        [-0.000120, -0.699798, 0.744842, -0.706640, -0.000144, -0.699799, 0.744031, -0.705588],
    ]
)


# ---------------------------------------------------------------------------
# Grip-force fix
# ---------------------------------------------------------------------------


def fix_robotiq_grip_force(
    model: mujoco.MjModel,
    *,
    prefix: str = "",
    target_tendon_force: float = 15.0,
    actuator_name: str = "fingers_actuator",
) -> None:
    """Replace the Robotiq gripper actuator with a **constant-force** grip.

    The menagerie 2F-85 ships a position actuator whose generalized
    force couples linearly to *tendon length*:

        force = gain[0] * ctrl + bias[0] + bias[1] * length + bias[2] * vel
              = 0.3137 * ctrl + 0 + (-100) * length + (-10) * vel

    With ``ctrl=255`` (close command) and the tendon fully extended
    (``length=0.8`` at fully-closed on the 2F-85):

        force = 80 - 80 = 0

    i.e. **grip force vanishes exactly when the fingers are fully
    closed**. The geodude_assets 2F-140 has the same form with its own
    constants: a stiff servo (``gain[0] = 1000 * 0.7 / 255``,
    ``bias = (0, -1000, -300)``, ``forcerange = ±6.3``) whose full close
    targets ``length=0.7``, the 2F-140's driver travel, where its pads meet
    (personalrobotics/geodude_assets#48, #49). It pushes at its full force
    until within about 0.006 of that target, so its force too fades only
    once the fingers have closed on nothing. If the object ever slips through the jaws, the gripper
    relaxes instead of trying to re-close — the same pathology the
    Franka hand had, fixed by :func:`fix_franka_grip_force`.

    Additionally, the menagerie ``forcerange="-5 5"`` clamps peak tendon
    force to 5 N. On the 2F-85 that works out to roughly 70 N of pad
    force (low end of the real Robotiq 20–235 N spec). For heavier or
    smoother objects, that's not enough — cans slip during transport
    when the arm accelerates.

    This helper rewrites ``actuator_name`` so:

    - ``ctrl = 0``   → tendon force = ``-target_tendon_force`` (open)
    - ``ctrl = 255`` → tendon force = ``+target_tendon_force`` (close)
    - Force is independent of tendon length (``biasprm[1] = 0``)
    - ``forcerange`` widens to ``±(target_tendon_force + small margin)``

    The ``bias[2]`` velocity-damping term is preserved (scaled to the
    new bias magnitude) to keep the finger linkage stable against its
    own inertia.

    Args:
        model: Compiled MjModel (modified in place).
        prefix: Actuator-name prefix. Use ``"gripper/"`` if the gripper
            was attached via ``spec.attach(gripper_spec, prefix="gripper/")``.
        target_tendon_force: Peak tendon force [N] at full close.
            Default 15 N — roughly 3× the menagerie default, yielding
            ~200 N pad force on the 2F-85 (middle of the real Robotiq
            spec). Scale down to 5 N for delicate objects, up to 25 N
            for heavy objects.
        actuator_name: Actuator to fix (default ``"fingers_actuator"``,
            which is the name used by both menagerie 2F-85 and
            geodude_assets 2F-140).

    See ``mj_manipulator/docs/grippers.md`` §2 for the full affine-force
    derivation, the mechanical-advantage numbers for each gripper, and
    tuning guidance.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + actuator_name)
    if aid < 0:
        raise RuntimeError(f"fix_robotiq_grip_force: no actuator named {prefix + actuator_name!r} in this model.")

    # Affine force model: force = gain[0]*ctrl + bias[0] + bias[1]*length + bias[2]*vel
    #   force(0)   = bias[0]        = -target  →  bias[0] = -target
    #   force(255) = 255*gain + bias[0] = +target → gain = 2*target/255
    new_bias0 = -target_tendon_force
    new_gain = 2.0 * target_tendon_force / 255.0

    # Preserve damping ratio — scale by the magnitude change in bias[0].
    old_bias1 = model.actuator_biasprm[aid, 1]
    old_bias2 = model.actuator_biasprm[aid, 2]
    damping_scale = abs(new_bias0 / old_bias1) if old_bias1 != 0 else 1.0

    model.actuator_biasprm[aid, 0] = new_bias0
    model.actuator_biasprm[aid, 1] = 0.0  # kill length coupling
    model.actuator_biasprm[aid, 2] = old_bias2 * damping_scale
    model.actuator_gainprm[aid, 0] = new_gain

    # Widen forcerange to let the actuator actually produce target_tendon_force.
    # Add 20 % margin so transient velocity-damping spikes don't hit the clamp.
    margin = 1.2
    model.actuator_forcerange[aid, 0] = -target_tendon_force * margin
    model.actuator_forcerange[aid, 1] = +target_tendon_force * margin


# ---------------------------------------------------------------------------
# RobotiqGripper
# ---------------------------------------------------------------------------


class RobotiqGripper(_BaseGripper):
    """Robotiq 2-finger parallel-jaw gripper (2F-140 or 2F-85).

    Supports the 4-bar linkage Robotiq grippers. In kinematic mode,
    replays a pre-recorded physics trajectory to maintain correct
    linkage geometry. In physics mode, the PhysicsController drives
    the tendon actuator directly.

    The ``prefix`` parameter handles namespacing in multi-robot scenes.
    For a standalone Robotiq model, use ``prefix=""``. For a prefixed
    model (e.g. geodude's ``right_ur5e/gripper/``), pass the full prefix.

    Variant selection:

    - By default, this class behaves as a **2F-140** — it uses the
      2F-140 kinematic trajectory and reports ``hand_type="robotiq_2f140"``
      (which makes grasp sources load the ``Robotiq2F140`` TSR class).
    - To use a **2F-85** (or any other variant), pass a per-variant
      ``trajectory`` and ``hand_type_override``. The 2F-85 trajectory
      lives in :mod:`mj_manipulator.grippers._robotiq_2f85_trajectory`.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        arm_name: Which arm this gripper belongs to.
        prefix: MuJoCo name prefix for all gripper elements.
        grasp_manager: Optional grasp state tracker.
        trajectory: Per-variant kinematic trajectory, shape ``(N, 8)``
            with joint order matching ``_JOINT_SUFFIXES``. ``None``
            uses the built-in 2F-140 trajectory.
        hand_type_override: Reported as ``self.hand_type`` so grasp
            sources pick the right TSR hand class. ``None`` falls back
            to ``"robotiq_2f140"``.
    """

    # Default hand_type is the 2F-140 (backward compat). Instances can
    # override this via the hand_type_override constructor kwarg.
    hand_type: str = "robotiq_2f140"

    # The Robotiq has continuous finger travel, so fully-closed can in
    # principle still hold an extremely thin object — but in practice
    # the objects we grasp are much wider than the verifier's position
    # threshold. Treating fully-closed as "empty" lets the
    # :class:`~mj_manipulator.grasp_verifier.GraspVerifier`'s
    # decisive-negative branch fire immediately when the gripper closed
    # on nothing, which is a crisp, noise-free, motion-independent
    # signal — better than trying to derive the same verdict from F/T
    # readings that bounce around during transport. See
    # personalrobotics/geodude#173 and personalrobotics/mj_manipulator#98.
    empty_at_fully_closed: bool = True

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        prefix: str = "",
        grasp_manager: GraspManager | None = None,
        trajectory: np.ndarray | None = None,
        hand_type_override: str | None = None,
    ):
        # Resolve actuator
        actuator_name = f"{prefix}fingers_actuator"

        # Resolve body names
        body_names = [f"{prefix}{s}" for s in _BODY_SUFFIXES]

        # Attachment body (finger pad that contacts objects)
        attachment_body = f"{prefix}{_ATTACHMENT_BODY_SUFFIX}"

        super().__init__(
            model=model,
            data=data,
            arm_name=arm_name,
            actuator_name=actuator_name,
            gripper_body_names=body_names,
            attachment_body=attachment_body,
            ctrl_open=0.0,
            ctrl_closed=255.0,
            grasp_manager=grasp_manager,
        )

        # Per-instance trajectory and hand_type.
        self._trajectory = trajectory if trajectory is not None else _2F140_TRAJECTORY
        if hand_type_override is not None:
            self.hand_type = hand_type_override

        # Resolve gripper joint qpos indices for kinematic control
        indices = []
        for suffix in _JOINT_SUFFIXES:
            full_name = f"{prefix}{suffix}"
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                full_name,
            )
            if joint_id != -1:
                indices.append(model.jnt_qposadr[joint_id])
        self._joint_qpos_indices = np.array(indices, dtype=int)

    def set_kinematic_position(self, t: float) -> None:
        """Set gripper to interpolated position kinematically.

        Uses the pre-recorded 4-bar linkage trajectory to ensure correct
        geometry at any position.

        Args:
            t: Position from 0.0 (open) to 1.0 (closed).
        """
        if len(self._joint_qpos_indices) == 0:
            return

        t = np.clip(t, 0.0, 1.0)
        trajectory = self._trajectory
        n = len(trajectory) - 1
        idx = t * n
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n)
        alpha = idx - idx_low

        joint_positions = (1 - alpha) * trajectory[idx_low] + alpha * trajectory[idx_high]

        self._data.qpos[self._joint_qpos_indices] = joint_positions
        mujoco.mj_forward(self._model, self._data)

    def _apply_kinematic_position(self, t: float) -> None:
        self.set_kinematic_position(t)

    def get_actual_position(self) -> float:
        """Get actual gripper position (0=open, 1=closed).

        Reads the driver joint position and maps it to [0, 1] by comparing
        to the open/closed trajectory endpoints.
        """
        if len(self._joint_qpos_indices) == 0:
            return 0.0

        # Use driver joint (index 1) as reference — it has the most range.
        driver_idx = 1
        driver_open = self._trajectory[0, driver_idx]
        driver_closed = self._trajectory[-1, driver_idx]

        if abs(driver_closed - driver_open) < 1e-6:
            return 0.0

        driver_pos = self._data.qpos[self._joint_qpos_indices[driver_idx]]
        t = (driver_pos - driver_open) / (driver_closed - driver_open)
        return float(np.clip(t, 0.0, 1.0))
