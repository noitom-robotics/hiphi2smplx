"""SMPL-X body-shape preparation for body-only IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_PARENTS,
    SMPLX_JOINT_NAMES,
    Skeleton,
)


@dataclass(frozen=True)
class SmplxBody:
    """Beta-shaped SMPL-X body data used by the IK pipeline.

    Attributes:
        skeleton: Body-only 22-joint rest skeleton.
        rest_joints: Absolute rest-joint positions with shape ``(22, 3)``.
        sole_rotations: Left and right rotations that level the shaped soles.
    """

    skeleton: Skeleton
    rest_joints: np.ndarray
    sole_rotations: tuple[np.ndarray, np.ndarray]


def _sole_pitch_rotation(
    vertices: np.ndarray,
    weights: np.ndarray,
    joint_indices: tuple[int, int],
) -> np.ndarray:
    """Estimate one shaped foot's sole-leveling pitch rotation.

    Args:
        vertices: Shaped SMPL-X vertices with shape ``(V, 3)``.
        weights: Skinning weights with shape ``(V, J)``.
        joint_indices: Ankle and foot joint indices used to select foot vertices.

    Returns:
        A ``(3, 3)`` pitch rotation matrix that levels the estimated sole.
    """
    foot_mask = weights[:, joint_indices].sum(axis=1) > 0.5
    foot = vertices[np.flatnonzero(foot_mask)]
    lower_z, upper_z = np.quantile(foot[:, 2], (0.25, 0.65))
    heel = foot[foot[:, 2] <= lower_z]
    toe = foot[foot[:, 2] >= upper_z]
    heel_floor = np.quantile(heel[:, 1], 0.05)
    toe_floor = np.quantile(toe[:, 1], 0.05)
    heel_z = np.median(heel[heel[:, 1] <= np.quantile(heel[:, 1], 0.15), 2])
    toe_z = np.median(toe[toe[:, 1] <= np.quantile(toe[:, 1], 0.15), 2])
    angle = float(np.arctan2(toe_floor - heel_floor, toe_z - heel_z))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        ),
        dtype=np.float32,
    )


def load_smplx_body(model_path: str | Path, betas: np.ndarray) -> SmplxBody:
    """Load and shape the SMPL-X data needed by body IK.

    Args:
        model_path: Path to an official SMPL-X NPZ model file.
        betas: Shape coefficients. All coefficients supported by the model are
            applied, while the public pipeline supplies the first 10.

    Returns:
        A body-only rest skeleton, absolute joints, and sole rotations.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model has fewer than 22 joints or invalid betas.
    """
    coefficients = np.asarray(betas, dtype=np.float64).reshape(-1)
    if not coefficients.size or not np.isfinite(coefficients).all():
        raise ValueError("betas must contain finite shape coefficients")

    with np.load(Path(model_path), allow_pickle=True) as model:
        template = np.asarray(model["v_template"], dtype=np.float64)
        shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        regressor = model["J_regressor"]
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        parents = np.asarray(model["kintree_table"], dtype=np.int64)[0].copy()

    beta_count = min(coefficients.size, shapedirs.shape[-1])
    vertices = template + np.einsum(
        "vcn,n->vc",
        shapedirs[..., :beta_count],
        coefficients[:beta_count],
    )
    joints = np.asarray(regressor, dtype=np.float64) @ vertices
    parents[parents > 1_000_000] = -1
    body_count = len(SMPLX_BODY_JOINT_NAMES)
    if len(joints) < body_count or len(parents) < body_count:
        raise ValueError("SMPL-X model must provide at least 22 body joints")

    body_joints = joints[:body_count].astype(np.float32)
    body_parents = parents[:body_count]
    if not np.array_equal(body_parents, SMPLX_BODY_PARENTS):
        raise ValueError("SMPL-X model uses an unexpected body hierarchy")
    offsets = body_joints.copy()
    for joint, parent in enumerate(body_parents):
        if parent >= 0:
            offsets[joint] -= body_joints[parent]

    skeleton = Skeleton(SMPLX_BODY_JOINT_NAMES, body_parents, offsets)
    sole_rotations = (
        _sole_pitch_rotation(vertices, weights, (7, 10)),
        _sole_pitch_rotation(vertices, weights, (8, 11)),
    )
    return SmplxBody(
        skeleton=skeleton,
        rest_joints=body_joints,
        sole_rotations=sole_rotations,
    )


def pack_body_pose(body_pose: np.ndarray) -> np.ndarray:
    """Pack body rotations into the standard 55-joint SMPL-X pose layout.

    Args:
        body_pose: Axis-angle rotations with shape ``(F, 24, 3)``. Only the
            first 22 SMPL-X body joints are copied.

    Returns:
        Axis-angle SMPL-X poses with shape ``(F, 55, 3)``. Jaw, eye, and finger
        rotations are initialized to zero.

    Raises:
        ValueError: If the body-pose shape is invalid.
    """
    body = np.asarray(body_pose, dtype=np.float32)
    if body.ndim != 3 or body.shape[1:] != (24, 3):
        raise ValueError(f"body_pose must have shape (F,24,3), got {body.shape}")
    packed = np.zeros((len(body), len(SMPLX_JOINT_NAMES), 3), dtype=np.float32)
    packed[:, : len(SMPLX_BODY_JOINT_NAMES)] = body[:, : len(SMPLX_BODY_JOINT_NAMES)]
    return packed
