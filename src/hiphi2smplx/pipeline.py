"""Body-only HiPHI BVH to SMPL-X conversion with Mink IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import (
    foot_sole_pitch_offsets,
    load_smplx_rest_joints,
    materialize_bvh_inputs,
)
from .mano import ManoFitter, ManoFittingInput, merge_mano_result
from .mink import MinkPositionConfig, MinkPositionSolver


SMPL22_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)
PIPELINE_VERSION = "hiphi2smplx-mink-v1"


@dataclass(frozen=True)
class ConversionConfig:
    """Settings for the fixed-shape, body-only Mink solver."""

    iterations: int = 10
    dt: float = 0.01
    damping: float = 1e-2
    posture_cost: float = 0.1
    root_posture_cost: float = 0.1
    position_cost: float = 1.0
    ankle_wrist_toe_weight: float = 5.0
    orientation_cost: float = 0.1
    qp_solver: str = "daqp"
    progress: bool = False


@dataclass(frozen=True)
class MotionResult:
    """Converted SMPL-X motion and body-fitting diagnostics."""

    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    fitted_joints: np.ndarray
    residuals: np.ndarray
    final_losses: np.ndarray
    frame_time: float
    source_scale: float
    hand_metadata: dict[str, Any] | None = None


def _global_rotations(local: np.ndarray) -> np.ndarray:
    output = np.empty_like(local)
    for joint, parent in enumerate(SMPL22_PARENTS):
        output[:, joint] = local[:, joint] if parent < 0 else np.einsum(
            "fij,fjk->fik", output[:, parent], local[:, joint]
        )
    return output


def equivalent_foot_targets(
    targets: np.ndarray,
    rest_joints: np.ndarray,
    reference_pose: np.ndarray,
    sole_level_rotations: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Replace BVH toe markers with beta-shaped SMPL-X foot targets."""

    output = np.asarray(targets, dtype=np.float32).copy()
    rest = np.asarray(rest_joints, dtype=np.float32)
    reference = np.asarray(reference_pose, dtype=np.float32)
    expected = (len(output), 22, 3)
    if output.shape != expected or reference.shape != expected:
        raise ValueError(f"targets and reference_pose must both have shape {expected}")
    if rest.shape != (22, 3):
        raise ValueError(f"rest_joints must have shape (22,3), got {rest.shape}")
    if not all(np.isfinite(array).all() for array in (output, rest, reference)):
        raise ValueError("body inputs must contain only finite values")

    local = Rotation.from_rotvec(reference.reshape(-1, 3)).as_matrix()
    global_rotations = _global_rotations(local.reshape(-1, 22, 3, 3))
    for side, (ankle, foot) in enumerate(((7, 10), (8, 11))):
        level = np.asarray(sole_level_rotations[side], dtype=np.float32)
        if level.shape != (3, 3) or not np.isfinite(level).all():
            raise ValueError("sole-level rotations must be finite 3x3 matrices")
        bind_offset = level @ (rest[foot] - rest[ankle])
        output[:, foot] = output[:, ankle] + np.einsum(
            "fij,j->fi", global_rotations[:, ankle], bind_offset
        )
    return output


def _pack_body_pose(body_pose: np.ndarray) -> np.ndarray:
    packed = np.zeros((len(body_pose), 55, 3), dtype=np.float32)
    packed[:, :22] = np.asarray(body_pose, dtype=np.float32)[:, :22]
    return packed


def fit_bvh(
    bvh_path: str | Path,
    model_path: str | Path,
    betas: np.ndarray,
    *,
    config: ConversionConfig = ConversionConfig(),
    max_frames: int | None = None,
    skeleton_map: str = "hiphi",
    mano_fitter: ManoFitter | None = None,
) -> MotionResult:
    """Convert one HiPHI BVH to SMPL-X body parameters.

    Fingers, jaw, and eyes remain zero unless an external MANO fitter is
    supplied. The fitter is an integration interface; no MANO implementation is
    distributed by this package.
    """

    source = Path(bvh_path).resolve()
    model = Path(model_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not model.is_file():
        raise FileNotFoundError(model)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    beta = np.asarray(betas, dtype=np.float32).reshape(-1)
    if beta.size < 10 or not np.isfinite(beta).all():
        raise ValueError("betas must contain at least 10 finite coefficients")
    beta = beta[:10].copy()

    targets, reference, frame_time, source_scale = materialize_bvh_inputs(
        source, model, beta, max_frames, skeleton_map
    )
    rest_joints = load_smplx_rest_joints(model, beta)[:22].astype(np.float32)
    sole = foot_sole_pitch_offsets(beta, model)
    fit_targets = equivalent_foot_targets(
        targets,
        rest_joints,
        reference,
        (sole["left_foot"], sole["right_foot"]),
    )
    solver = MinkPositionSolver(
        rest_joints,
        SMPL22_PARENTS,
        MinkPositionConfig(
            iterations=config.iterations,
            dt=config.dt,
            damping=config.damping,
            posture_cost=config.posture_cost,
            root_posture_cost=config.root_posture_cost,
            position_cost=config.position_cost,
            ankle_wrist_toe_weight=config.ankle_wrist_toe_weight,
            orientation_cost=config.orientation_cost,
            solver=config.qp_solver,
        ),
    )

    frame_indices: Any = range(len(targets))
    if config.progress:
        from tqdm import tqdm

        frame_indices = tqdm(frame_indices, desc="mink-body", leave=False)
    body_pose = np.empty((len(targets), 24, 3), dtype=np.float32)
    translations = np.empty((len(targets), 3), dtype=np.float32)
    fitted = np.empty_like(fit_targets)
    losses = np.empty(len(targets), dtype=np.float32)
    previous_pose: np.ndarray | None = None
    previous_translation: np.ndarray | None = None
    for frame in frame_indices:
        pose_reference = np.zeros((24, 3), dtype=np.float32)
        pose_reference[:22] = reference[frame]
        seed = pose_reference if previous_pose is None else previous_pose
        pose, translation, joints, loss = solver.solve(
            fit_targets[frame],
            seed,
            previous_translation,
            reference_pose=pose_reference,
        )
        body_pose[frame] = pose
        translations[frame] = translation
        fitted[frame] = joints
        losses[frame] = loss
        previous_pose = pose
        previous_translation = translation

    poses = _pack_body_pose(body_pose)
    hand_metadata = None
    if mano_fitter is not None:
        hand_result = mano_fitter.fit(
            ManoFittingInput(
                bvh_path=source,
                model_path=model,
                betas=beta.copy(),
                body_poses=poses.copy(),
                translations=translations.copy(),
                frame_time=float(frame_time),
            )
        )
        poses, hand_metadata = merge_mano_result(poses, hand_result)

    return MotionResult(
        poses=poses,
        trans=translations,
        betas=beta,
        fitted_joints=fitted,
        residuals=np.linalg.norm(fitted - fit_targets, axis=-1).mean(axis=1).astype(np.float32),
        final_losses=losses,
        frame_time=float(frame_time),
        source_scale=float(source_scale),
        hand_metadata=hand_metadata,
    )


def save_result(
    path: str | Path,
    result: MotionResult,
    *,
    source_bvh: str | Path,
    config: ConversionConfig,
) -> None:
    """Write a portable SMPL-X NPZ using the established HiPHI field names."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        poses=result.poses,
        smpl_pose_axis_angle=result.poses,
        trans=result.trans,
        transl=result.trans,
        betas=result.betas,
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(1.0 / result.frame_time, dtype=np.float32),
        fps=np.asarray(1.0 / result.frame_time, dtype=np.float32),
        frame_time=np.asarray(result.frame_time, dtype=np.float64),
        model_type=np.asarray("smplx"),
        num_joints=np.asarray(55, dtype=np.int32),
        coordinate_up=np.asarray("y"),
        output_up=np.asarray("y"),
        fitted_joints=result.fitted_joints,
        per_frame_residuals=result.residuals,
        final_losses=result.final_losses,
        solver=np.asarray("mink"),
        pipeline=np.asarray(PIPELINE_VERSION),
        source_bvh=np.asarray(str(Path(source_bvh).resolve())),
        source_scale=np.asarray(result.source_scale, dtype=np.float32),
        hands_fitted=np.asarray(result.hand_metadata is not None),
        mink_iterations=np.asarray(config.iterations, dtype=np.int32),
        mink_damping=np.asarray(config.damping, dtype=np.float32),
        mink_position_cost=np.asarray(config.position_cost, dtype=np.float32),
        mink_ankle_wrist_toe_weight=np.asarray(
            config.ankle_wrist_toe_weight, dtype=np.float32
        ),
        mink_orientation_cost=np.asarray(config.orientation_cost, dtype=np.float32),
        mink_posture_cost=np.asarray(config.posture_cost, dtype=np.float32),
        mink_root_posture_cost=np.asarray(config.root_posture_cost, dtype=np.float32),
        mink_qp_solver=np.asarray(config.qp_solver),
    )
    temporary.replace(destination)
