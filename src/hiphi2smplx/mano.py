"""Public integration contract for optional, externally supplied MANO fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ManoFittingInput:
    """Inputs made available to an external MANO fitter.

    Attributes:
        bvh_path: Original source BVH path.
        model_path: SMPL-X model path used by body fitting.
        betas: Fixed SMPL-X shape coefficients.
        body_poses: Solved body poses with shape ``(F, 55, 3)``.
        translations: Solved translations with shape ``(F, 3)``.
        frame_time: Source-frame duration in seconds.
    """

    bvh_path: Path
    model_path: Path
    betas: np.ndarray
    body_poses: np.ndarray
    translations: np.ndarray
    frame_time: float


@dataclass(frozen=True)
class ManoFittingResult:
    """Finger rotations returned by an external MANO fitter.

    Attributes:
        left_hand_pose: Left finger rotations with shape ``(F, 15, 3)``.
        right_hand_pose: Right finger rotations with shape ``(F, 15, 3)``.
        metadata: Optional plugin-specific metadata retained in memory.
    """

    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ManoFitter(Protocol):
    """Interface implemented by a separately distributed MANO plugin."""

    def fit(self, inputs: ManoFittingInput) -> ManoFittingResult:
        """Fit both hands for one body motion.

        Args:
            inputs: Body motion, model, shape, and timing information.

        Returns:
            Left and right SMPL-X finger rotations.
        """
        ...


def merge_mano_result(
    body_poses: np.ndarray,
    result: ManoFittingResult,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and merge external finger rotations into SMPL-X pose slots.

    Args:
        body_poses: Body-only poses with shape ``(F, 55, 3)``.
        result: Finger rotations returned by an external fitter.

    Returns:
        A copied pose array containing both hands and a plain metadata dict.

    Raises:
        ValueError: If hand poses have invalid shapes or non-finite values.
    """
    poses = np.asarray(body_poses, dtype=np.float32).copy()
    if poses.ndim != 3 or poses.shape[1:] != (55, 3):
        raise ValueError(f"body_poses must have shape (F,55,3), got {poses.shape}")
    expected = (len(poses), 15, 3)
    left = np.asarray(result.left_hand_pose, dtype=np.float32)
    right = np.asarray(result.right_hand_pose, dtype=np.float32)
    if left.shape != expected or right.shape != expected:
        raise ValueError(f"MANO hand poses must each have shape {expected}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("MANO hand poses contain non-finite values")
    poses[:, 25:40] = left
    poses[:, 40:55] = right
    return poses, dict(result.metadata)
