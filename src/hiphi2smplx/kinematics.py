"""Format-independent forward-kinematics utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def local_to_global_rotations(
    local_rotations: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Accumulate local joint rotations along a skeleton hierarchy.

    Args:
        local_rotations: Rotation matrices with shape ``(F, J, 3, 3)``.
        parents: Parent indices with shape ``(J,)`` and ``-1`` for the root.

    Returns:
        Global rotation matrices with the same shape and dtype as the input.

    Raises:
        ValueError: If the array shapes do not describe the same skeleton.
    """
    local = np.asarray(local_rotations)
    hierarchy = np.asarray(parents, dtype=np.int64)
    if local.ndim != 4 or local.shape[-2:] != (3, 3):
        raise ValueError(f"local_rotations must have shape (F,J,3,3), got {local.shape}")
    if hierarchy.shape != (local.shape[1],):
        raise ValueError(f"parents must have shape ({local.shape[1]},), got {hierarchy.shape}")

    global_rotations = np.empty_like(local)
    for joint, parent in enumerate(hierarchy):
        if parent < 0:
            global_rotations[:, joint] = local[:, joint]
        else:
            global_rotations[:, joint] = global_rotations[:, parent] @ local[:, joint]
    return global_rotations


def global_to_local_rotations(
    global_rotations: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Convert global joint rotations to parent-relative rotations.

    Args:
        global_rotations: Rotation matrices with shape ``(F, J, 3, 3)``.
        parents: Parent indices with shape ``(J,)`` and ``-1`` for the root.

    Returns:
        Local rotation matrices with the same shape and dtype as the input.

    Raises:
        ValueError: If the array shapes do not describe the same skeleton.
    """
    global_values = np.asarray(global_rotations)
    hierarchy = np.asarray(parents, dtype=np.int64)
    if global_values.ndim != 4 or global_values.shape[-2:] != (3, 3):
        raise ValueError(f"global_rotations must have shape (F,J,3,3), got {global_values.shape}")
    if hierarchy.shape != (global_values.shape[1],):
        raise ValueError(
            f"parents must have shape ({global_values.shape[1]},), got {hierarchy.shape}"
        )

    local_rotations = global_values.copy()
    for joint, parent in enumerate(hierarchy):
        if parent >= 0:
            local_rotations[:, joint] = (
                np.swapaxes(global_values[:, parent], -1, -2) @ global_values[:, joint]
            )
    return local_rotations


def forward_kinematics(
    local_rotations: np.ndarray,
    local_translations: np.ndarray,
    root_positions: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global joint positions and rotations.

    Args:
        local_rotations: Rotation matrices with shape ``(F, J, 3, 3)``.
        local_translations: Parent-relative offsets with shape ``(F, J, 3)``.
        root_positions: World-space root positions with shape ``(F, 3)``.
        parents: Parent indices with shape ``(J,)`` and ``-1`` for the root.

    Returns:
        A pair containing global positions ``(F, J, 3)`` and global rotations
        ``(F, J, 3, 3)``.

    Raises:
        ValueError: If the translation or root-position shapes are invalid.
    """
    local = np.asarray(local_rotations)
    translations = np.asarray(local_translations)
    roots = np.asarray(root_positions)
    expected_translations = local.shape[:2] + (3,)
    if translations.shape != expected_translations:
        raise ValueError(
            f"local_translations must have shape {expected_translations}, got {translations.shape}"
        )
    if roots.shape != (local.shape[0], 3):
        raise ValueError(f"root_positions must have shape ({local.shape[0]},3), got {roots.shape}")

    global_rotations = local_to_global_rotations(local, parents)
    positions = np.empty(expected_translations, dtype=np.result_type(local, translations))
    positions[:, 0] = roots
    for joint, parent in enumerate(np.asarray(parents, dtype=np.int64)):
        if parent >= 0:
            positions[:, joint] = positions[:, parent] + np.einsum(
                "fij,fj->fi",
                global_rotations[:, parent],
                translations[:, joint],
            )
    return positions, global_rotations


def align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimal rotation that aligns one 3D vector with another.

    Args:
        source: Source vector with shape ``(3,)``.
        target: Target vector with shape ``(3,)``.

    Returns:
        A ``(3, 3)`` world-space rotation matrix. Degenerate vectors produce
        the identity rotation.
    """
    source_vector = np.asarray(source, dtype=np.float64)
    target_vector = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source_vector))
    target_norm = float(np.linalg.norm(target_vector))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return np.eye(3, dtype=np.float64)

    source_vector /= source_norm
    target_vector /= target_norm
    cross = np.cross(source_vector, target_vector)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source_vector, target_vector), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3)[int(np.argmin(np.abs(source_vector)))]
        axis = np.cross(source_vector, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    axis = cross / sine
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * axis).as_matrix()
