"""Single-file HiPHI BVH to SMPL-X conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .beta_fitting import BetaFitConfig, fit_betas
from .bvh import load_bvh
from .ik_solver import (
    MinkPositionConfig,
    MinkPositionSolver,
    equivalent_foot_targets,
    prepare_body_inputs,
)
from .mano import ManoFitter, ManoFittingInput, merge_mano_result
from .skeleton import map_skeleton
from .smplx import load_smplx_body, pack_body_pose


@dataclass(frozen=True)
class ConversionConfig:
    """Settings for fixed-shape body conversion.

    Attributes:
        iterations: Maximum Mink iterations per frame.
        dt: Mink integration time step.
        damping: Levenberg-Marquardt damping for IK tasks.
        posture_cost: Non-root posture regularization cost.
        root_posture_cost: Root orientation and posture cost.
        position_cost: Base joint-position cost.
        ankle_wrist_toe_weight: Extremity position-cost multiplier.
        orientation_cost: Non-root orientation cost.
        qp_solver: qpsolvers backend name.
        progress: Whether to display a frame progress bar.
    """

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
    """Converted SMPL-X motion and in-memory fitting diagnostics.

    Attributes:
        poses: Axis-angle SMPL-X poses with shape ``(F, 55, 3)``.
        trans: SMPL-X translations with shape ``(F, 3)``.
        betas: The 10 fixed SMPL-X shape coefficients.
        fitted_joints: Solved body joints with shape ``(F, 22, 3)``.
        residuals: Mean per-frame joint residuals in metres.
        final_losses: Final weighted solver losses per frame.
        frame_time: Source-frame duration in seconds.
        hand_metadata: Optional metadata returned by an external MANO fitter.
    """

    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    fitted_joints: np.ndarray
    residuals: np.ndarray
    final_losses: np.ndarray
    frame_time: float
    hand_metadata: dict[str, Any] | None = None


def fit_bvh(
    bvh_path: str | Path,
    model_path: str | Path,
    betas: np.ndarray | None = None,
    *,
    beta_fit_config: BetaFitConfig | None = None,
    config: ConversionConfig | None = None,
    max_frames: int | None = None,
    skeleton_map: str = "hiphi",
    mano_fitter: ManoFitter | None = None,
) -> MotionResult:
    """Convert one HiPHI BVH file to fixed-shape SMPL-X parameters.

    Args:
        bvh_path: Path to one source BVH file.
        model_path: Path to an official SMPL-X NPZ model.
        betas: At least 10 finite fixed shape coefficients. Mutually exclusive
            with ``beta_fit_config``.
        beta_fit_config: First-frame beta-fitting settings. Supplying this
            enables beta fitting and is mutually exclusive with ``betas``.
        config: Mink and progress settings. Defaults to ``ConversionConfig()``.
        max_frames: Optional limit on the number of leading frames.
        skeleton_map: Registered exact source-skeleton mapping.
        mano_fitter: Optional external hand-fitting implementation.

    Returns:
        Converted poses, translations, shape, timing, and fitting diagnostics.

    Raises:
        FileNotFoundError: If the BVH or SMPL-X model does not exist.
        ValueError: If an input, mapping, or shape is invalid.
    """
    config = ConversionConfig() if config is None else config
    source_path = Path(bvh_path).resolve()
    model_path = Path(model_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if (betas is None) == (beta_fit_config is None):
        raise ValueError("provide exactly one of betas or beta_fit_config")

    source = load_bvh(source_path, unit_scale=0.01)
    if max_frames is not None:
        source = source.sliced(min(source.num_frames, max_frames))
    source_mapping = map_skeleton(source.skeleton, skeleton_map)

    if beta_fit_config is None:
        beta = np.asarray(betas, dtype=np.float32).reshape(-1)
        if beta.size < 10 or not np.isfinite(beta).all():
            raise ValueError("betas must contain at least 10 finite coefficients")
        beta = beta[:10].copy()
        smplx_body = load_smplx_body(model_path, beta)
    else:
        neutral_body = load_smplx_body(model_path, np.zeros(10, dtype=np.float32))
        neutral_targets, _ = prepare_body_inputs(
            source,
            source_mapping,
            neutral_body.skeleton,
        )
        beta = fit_betas(
            neutral_targets[0],
            neutral_body.rest_joints,
            model_path,
            config=beta_fit_config,
        ).betas
        smplx_body = load_smplx_body(model_path, beta)

    targets, reference = prepare_body_inputs(
        source,
        source_mapping,
        smplx_body.skeleton,
    )
    fit_targets = equivalent_foot_targets(
        targets,
        smplx_body.rest_joints,
        reference,
        smplx_body.sole_rotations,
    )
    solver = MinkPositionSolver(
        smplx_body.rest_joints,
        smplx_body.skeleton.parents,
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

    poses = pack_body_pose(body_pose)
    hand_metadata = None
    if mano_fitter is not None:
        hand_result = mano_fitter.fit(
            ManoFittingInput(
                bvh_path=source_path,
                model_path=model_path,
                betas=beta.copy(),
                body_poses=poses.copy(),
                translations=translations.copy(),
                frame_time=float(source.frame_time),
            )
        )
        poses, hand_metadata = merge_mano_result(poses, hand_result)

    residuals = np.linalg.norm(fitted - fit_targets, axis=-1).mean(axis=1)
    return MotionResult(
        poses=poses,
        trans=translations,
        betas=beta,
        fitted_joints=fitted,
        residuals=residuals.astype(np.float32),
        final_losses=losses,
        frame_time=float(source.frame_time),
        hand_metadata=hand_metadata,
    )


def save_result(path: str | Path, result: MotionResult) -> None:
    """Write an AMASS-style SMPL-X motion NPZ atomically.

    Args:
        path: Destination NPZ path. Parent directories are created as needed.
        result: Converted motion returned by :func:`fit_bvh`.

    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        poses=result.poses.reshape(len(result.poses), -1),
        trans=result.trans,
        betas=result.betas,
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(1.0 / result.frame_time, dtype=np.float32),
    )
    temporary.replace(destination)
