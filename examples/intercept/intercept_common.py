# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the conditions in the LICENSE
# file at the repository root.

"""Shared, dependency-light building blocks for deploying an Intercept policy.

This module intentionally depends **only** on ``torch`` and the Python standard
library so that it can be imported both inside the Isaac Sim training
environment (Python 3.11 ``.venv``, used by :mod:`export_policy`) and inside the
Crazyswarm2 environment (Python 3.10 ``.venv-crazyswarm``, used by
:mod:`intercept_controller`).

It must **not** import ``omni_drones``, ``torchrl``, ``isaacsim`` or ``rclpy``.

The two pieces of task-specific logic that have to match the training pipeline
bit-for-bit live here:

* :func:`build_observation` reproduces
  ``Intercept._compute_state_and_obs`` in
  ``omni_drones/envs/single/intercept.py``.
* :func:`decode_action_to_ctbr` reproduces the command half of the
  ``PIDRateController`` action transform in
  ``omni_drones/utils/torchrl/transforms.py``.

Keeping these in one small, unit-testable module avoids silent drift between
training and deployment.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# Format identifier written into ``metadata.json`` so the controller can refuse
# to load artifacts produced by an incompatible exporter.
ARTIFACT_VERSION = 1


# ---------------------------------------------------------------------------
# Configuration payloads (serialised into metadata.json alongside the policy)
# ---------------------------------------------------------------------------
@dataclass
class ObsConfig:
    """Observation layout for the Intercept task.

    Mirrors the flags read in ``Intercept.__init__`` / ``_set_specs`` that
    determine the observation composition. The defaults match
    ``cfg/task/Intercept.yaml`` (``obs_dim == 16``).
    """

    use_ab_world_frame: bool = False
    use_rot_speed: bool = False
    use_relative_velocity: bool = False
    obs_dim: int = 16
    action_dim: int = 4

    def expected_obs_dim(self) -> int:
        """Recompute the observation dimension from the layout flags."""
        evader_state_dim = 3  # relative heading
        if self.use_relative_velocity:
            evader_state_dim += 3

        pursuer_state_dim = 3 + 9 + 1  # lin vel + rot matrix + altitude
        if self.use_ab_world_frame:
            pursuer_state_dim += 3 - 1  # full position replaces altitude
        if self.use_rot_speed:
            pursuer_state_dim += 3
        return evader_state_dim + pursuer_state_dim


@dataclass
class CTBRConfig:
    """Parameters needed to decode a raw policy action into a CTBR command.

    Mirrors the attributes read by the ``PIDRateController`` action transform
    and controller. Only the values required to reproduce the *command* (not
    the on-board PID, which the firmware performs itself) are kept.
    """

    target_clip: float = 1.0
    min_thrust_ratio: float = 0.0
    max_thrust_ratio: float = 0.9
    # ``LPF_coef`` is kept for completeness / diagnostics only; it does not
    # influence the command sent to the drone (see decode_action_to_ctbr).
    lpf_coef: float = 1.0
    dt: float = 0.02


@dataclass
class PolicyMetadata:
    """Everything the controller needs besides the TorchScript weights."""

    artifact_version: int
    algo: str
    obs: ObsConfig = field(default_factory=ObsConfig)
    ctbr: CTBRConfig = field(default_factory=CTBRConfig)
    sim_dt: float = 0.02
    # Free-form provenance (checkpoint path, task name, git hash, ...).
    notes: dict = field(default_factory=dict)

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "artifact_version": self.artifact_version,
            "algo": self.algo,
            "obs": dataclasses.asdict(self.obs),
            "ctbr": dataclasses.asdict(self.ctbr),
            "sim_dt": self.sim_dt,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyMetadata":
        return cls(
            artifact_version=int(data["artifact_version"]),
            algo=str(data["algo"]),
            obs=ObsConfig(**data["obs"]),
            ctbr=CTBRConfig(**data["ctbr"]),
            sim_dt=float(data.get("sim_dt", 0.02)),
            notes=dict(data.get("notes", {})),
        )


def save_metadata(metadata: PolicyMetadata, path: str) -> None:
    """Write ``metadata`` to ``path`` as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)


def load_metadata(path: str) -> PolicyMetadata:
    """Read :class:`PolicyMetadata` from a JSON file, validating the version."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = PolicyMetadata.from_dict(data)
    if metadata.artifact_version != ARTIFACT_VERSION:
        raise ValueError(
            f"Incompatible artifact version {metadata.artifact_version} "
            f"(expected {ARTIFACT_VERSION}). Re-run export_policy.py."
        )
    return metadata


# ---------------------------------------------------------------------------
# Math helpers (kept byte-compatible with omni_drones/utils/torch.py)
# ---------------------------------------------------------------------------
def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise along the last dim, matching ``omni_drones`` (eps=1e-6)."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a ``(w, x, y, z)`` quaternion to a rotation matrix.

    Row-major flattening of the returned ``[..., 3, 3]`` matrix matches the
    ``pursuer_rot`` observation component. Copied verbatim from
    ``omni_drones/utils/torch.py`` to avoid importing the (Isaac-coupled)
    package at deploy time.
    """
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    return matrix.unflatten(matrix.dim() - 1, (3, 3))


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------
def build_observation(
    cfg: ObsConfig,
    pursuer_pos: torch.Tensor,
    pursuer_quat_wxyz: torch.Tensor,
    pursuer_lin_vel_world: torch.Tensor,
    evader_pos: torch.Tensor,
    pursuer_ang_vel_world: Optional[torch.Tensor] = None,
    evader_lin_vel_world: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble the Intercept observation vector from raw world-frame states.

    All tensors have a trailing feature dimension and may carry arbitrary
    leading batch dimensions. Rotational quantities follow Isaac conventions:
    the quaternion is ``(w, x, y, z)`` and linear/angular velocities are
    expressed in the **world** frame (as returned by ``drone.get_state()``).

    Args:
        cfg: Observation layout flags (must match the trained policy).
        pursuer_pos: ``[..., 3]`` pursuer position in world frame.
        pursuer_quat_wxyz: ``[..., 4]`` pursuer orientation as ``(w, x, y, z)``.
        pursuer_lin_vel_world: ``[..., 3]`` pursuer linear velocity (world).
        evader_pos: ``[..., 3]`` evader (target) position in world frame.
        pursuer_ang_vel_world: ``[..., 3]`` pursuer angular velocity (world),
            required only when ``cfg.use_rot_speed`` is True.
        evader_lin_vel_world: ``[..., 3]`` evader linear velocity (world),
            required only when ``cfg.use_relative_velocity`` is True.

    Returns:
        ``[..., obs_dim]`` observation tensor.
    """
    evader_rel_hdg = normalize(evader_pos - pursuer_pos)  # (3)
    pursuer_rot = quaternion_to_rotation_matrix(pursuer_quat_wxyz)
    pursuer_rot = pursuer_rot.reshape(*pursuer_rot.shape[:-2], 9)  # (9)

    components = [evader_rel_hdg, pursuer_lin_vel_world, pursuer_rot]

    if cfg.use_ab_world_frame:
        components.append(pursuer_pos)  # (3)
    else:
        components.append(pursuer_pos[..., 2:3])  # altitude only (1)

    if cfg.use_relative_velocity:
        if evader_lin_vel_world is None:
            raise ValueError(
                "use_relative_velocity=True requires evader_lin_vel_world."
            )
        components.append(evader_lin_vel_world - pursuer_lin_vel_world)  # (3)

    if cfg.use_rot_speed:
        if pursuer_ang_vel_world is None:
            raise ValueError(
                "use_rot_speed=True requires pursuer_ang_vel_world."
            )
        components.append(pursuer_ang_vel_world)  # (3)

    obs = torch.cat(components, dim=-1)
    if obs.shape[-1] != cfg.obs_dim:
        raise ValueError(
            f"Assembled observation has dim {obs.shape[-1]} but metadata "
            f"declares obs_dim={cfg.obs_dim}. Check the ObsConfig flags."
        )
    return obs


# ---------------------------------------------------------------------------
# Action decoding: raw policy action -> collective-thrust / body-rate command
# ---------------------------------------------------------------------------
@dataclass
class CTBRCommand:
    """A collective-thrust + body-rate setpoint.

    ``body_rate_deg`` is ``[roll_rate, pitch_rate, yaw_rate]`` in deg/s (the
    unit the on-board rate controller expects). ``thrust_ratio`` is the
    normalised collective thrust in ``[0, 1]``; ``thrust_pwm`` is the same value
    mapped to the firmware's ``[0, 65535]`` motor-command range.
    """

    body_rate_deg: torch.Tensor  # [..., 3]
    thrust_ratio: torch.Tensor   # [..., 1]
    thrust_pwm: torch.Tensor     # [..., 1]


def decode_action_to_ctbr(raw_action: torch.Tensor, cfg: CTBRConfig) -> CTBRCommand:
    """Decode a raw policy action into a CTBR command.

    This reproduces the *command* computation of the ``PIDRateController``
    action transform (``omni_drones/utils/torchrl/transforms.py``). Note that
    in the reference implementation the low-pass filter only affects a logged
    action-error statistic and the ``prev_action`` buffer; the command itself
    is computed from the raw ``tanh`` outputs. The decoding is therefore
    **stateless**, and the on-board firmware performs the rate PID that turns
    this CTBR setpoint into motor commands.

    Args:
        raw_action: ``[..., 4]`` unbounded network output ``[wx, wy, wz, thr]``.
        cfg: Decoding parameters (must match the trained controller).

    Returns:
        A :class:`CTBRCommand`.
    """
    action = torch.tanh(raw_action)  # -> [-1, 1]
    target_rate, target_thrust = action.split([3, 1], dim=-1)

    # Body-rate target in deg/s.
    body_rate_deg = target_rate * 180.0 * cfg.target_clip

    # Collective thrust: map [-1, 1] -> [0, 1] then clamp to the trained range.
    thrust_ratio = torch.clamp(
        (target_thrust + 1.0) / 2.0,
        min=cfg.min_thrust_ratio,
        max=cfg.max_thrust_ratio,
    )
    thrust_pwm = thrust_ratio * (2 ** 16)

    return CTBRCommand(
        body_rate_deg=body_rate_deg,
        thrust_ratio=thrust_ratio,
        thrust_pwm=thrust_pwm,
    )


# Standard artifact file names, referenced by both scripts.
POLICY_TS_FILENAME = "policy_ts.pt"
METADATA_FILENAME = "metadata.json"


def artifact_paths(output_dir: str) -> "tuple[str, str]":
    """Return ``(torchscript_path, metadata_path)`` inside ``output_dir``."""
    return (
        os.path.join(output_dir, POLICY_TS_FILENAME),
        os.path.join(output_dir, METADATA_FILENAME),
    )
