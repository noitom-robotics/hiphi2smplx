"""First-frame SMPL-X beta fitting for the HiPHI conversion pipeline.

Parts of the beta-fitting implementation are adapted from joints2smpl:
https://github.com/wangsen1312/joints2smpl
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import torch

from .kinematics import align_vectors
from .skeleton import SMPLX_BODY_PARENTS

_ALIGNMENT_JOINTS = (2, 1, 17, 16)
_ANGLE_PRIOR_INDICES = (52, 55, 9, 12)
_ANGLE_PRIOR_SIGNS = (1.0, -1.0, -1.0, -1.0)
_SOURCE_FOOT_BIND_OFFSETS = (
    (0.0, -0.0626, 0.1443),
    (0.0, -0.0626, 0.1443),
)
_MEAN_POSE_FILENAME = "neutral_smpl_mean_params.h5"
_POSE_PRIOR_FILENAME = "gmm_08.pkl"


@dataclass(frozen=True)
class BetaFitConfig:
    """Settings for first-frame SMPL-X shape fitting.

    Attributes:
        iterations: LBFGS iterations and body-stage outer iterations.
        step_size: LBFGS learning rate.
        device: Torch device used for fitting.
        data_dir: Optional directory containing ``neutral_smpl_mean_params.h5``
            and ``gmm_08.pkl``. When omitted, the fitter checks the package's
            ``data`` directory. The resources have separate license terms.
    """

    iterations: int = 100
    step_size: float = 1e-2
    device: str = "cuda:0"
    data_dir: str | Path | None = None


@dataclass(frozen=True)
class BetaFitResult:
    """First-frame shape result and in-memory diagnostics.

    Attributes:
        betas: Fitted SMPL-X shape coefficients with shape ``(10,)``.
        pose: Fitted axis-angle pose with shape ``(22, 3)``.
        translation: Fitted global translation with shape ``(3,)``.
        final_loss: Final standalone body-fitting loss.
        mean_joint_error: Final mean 22-joint error in metres.
    """

    betas: np.ndarray
    pose: np.ndarray
    translation: np.ndarray
    final_loss: float
    mean_joint_error: float


def _load_smplx() -> ModuleType:
    """Import the external SMPL-X package.

    Returns:
        The imported ``smplx`` module.

    Raises:
        ModuleNotFoundError: If SMPL-X is not installed.
    """
    try:
        import smplx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("beta fitting requires the 'smplx' package") from exc
    return smplx


def _prepare_target_joints(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
) -> np.ndarray:
    """Apply equivalent-foot correction.

    Args:
        target_joints: Raw first-frame AMASS-22 targets.
        rest_joints: Neutral SMPL-X rest joints with shape ``(22, 3)``.

    Returns:
        A target copy with corrected left and right foot markers.
    """
    targets = np.asarray(target_joints, dtype=np.float32).copy()
    rest = np.asarray(rest_joints, dtype=np.float32)
    if targets.shape != (22, 3) or not np.isfinite(targets).all():
        raise ValueError("target_joints must be a finite array with shape (22,3)")
    if rest.shape != (22, 3) or not np.isfinite(rest).all():
        raise ValueError("rest_joints must be a finite array with shape (22,3)")

    for source_offset, (ankle, foot) in zip(
        _SOURCE_FOOT_BIND_OFFSETS,
        ((7, 10), (8, 11)),
        strict=True,
    ):
        source_bind = np.asarray(source_offset, dtype=np.float32)
        smpl_bind = rest[foot] - rest[ankle]
        correction = align_vectors(source_bind, smpl_bind).astype(np.float32)
        ray = (targets[foot] - targets[ankle]) @ correction.T
        ray *= np.linalg.norm(smpl_bind) / max(np.linalg.norm(ray), 1e-8)
        targets[foot] = targets[ankle] + ray
    return targets


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices.

    Args:
        axis_angle: Tensor whose final dimension is three.

    Returns:
        Rotation matrices with matching leading dimensions.
    """
    theta_squared = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = theta_squared.clamp_min(1e-12).sqrt()
    sine_scale = torch.where(
        theta_squared > 1e-8,
        torch.sin(theta) / theta.clamp_min(1e-8),
        1.0 - theta_squared / 6.0,
    )
    cosine_scale = torch.where(
        theta_squared > 1e-8,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-8),
        0.5 - theta_squared / 24.0,
    )
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(axis_angle.shape[:-1] + (3, 3))
    return identity + sine_scale.unsqueeze(-1) * skew + cosine_scale.unsqueeze(-1) * (skew @ skew)


def _skeleton_fk(
    pose_axis_angle: torch.Tensor,
    root_position: torch.Tensor,
    rest_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the 22-joint differentiable FK.

    Args:
        pose_axis_angle: Axis-angle pose with shape ``(B, 24, 3)``.
        root_position: Root positions with shape ``(B, 3)``.
        rest_offsets: Beta-shaped offsets with shape ``(B, 22, 3)``.

    Returns:
        Global joint positions with shape ``(B, 22, 3)``.
    """
    local = _axis_angle_to_matrix(pose_axis_angle[:, :22])
    global_rotations: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    for joint, parent in enumerate(SMPLX_BODY_PARENTS):
        if parent < 0:
            global_rotation = local[:, joint]
            position = root_position
        else:
            global_rotation = global_rotations[parent] @ local[:, joint]
            position = positions[parent] + torch.einsum(
                "bij,bj->bi",
                global_rotations[parent],
                rest_offsets[:, joint],
            )
        global_rotations.append(global_rotation)
        positions.append(position)
    return torch.stack(positions, dim=1)


class _SmplxBodyAdapter(torch.nn.Module):
    """Expose the 21 SMPL-X body joints through the 69-D SMPL pose API."""

    def __init__(self, body_model: torch.nn.Module) -> None:
        """Store the underlying SMPL-X model."""
        super().__init__()
        self.body_model = body_model

    def forward(
        self,
        *,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        return_full_pose: bool = False,
    ) -> object:
        """Evaluate SMPL-X with all hand and face parameters set to zero."""
        batch_size = body_pose.shape[0]
        zeros = body_pose.new_zeros
        return self.body_model(
            global_orient=global_orient,
            body_pose=body_pose[:, :63],
            betas=betas,
            left_hand_pose=zeros((batch_size, 45)),
            right_hand_pose=zeros((batch_size, 45)),
            jaw_pose=zeros((batch_size, 3)),
            leye_pose=zeros((batch_size, 3)),
            reye_pose=zeros((batch_size, 3)),
            expression=zeros((batch_size, 10)),
            return_full_pose=return_full_pose,
        )


class _SkeletonBodyAdapter(torch.nn.Module):
    """Apply beta through the joint shape basis and use lightweight FK to reduce time cost."""

    def __init__(self, body_model: _SmplxBodyAdapter) -> None:
        """Precompute the affine beta-to-rest-joint map."""
        super().__init__()
        source = body_model.body_model
        regressor = source.J_regressor[:22]
        if getattr(regressor, "is_sparse", False):
            regressor = regressor.to_dense()
        shapedirs = source.shapedirs[..., :10]
        self.register_buffer(
            "joint_template",
            torch.einsum("jv,vc->jc", regressor, source.v_template).detach(),
        )
        self.register_buffer(
            "joint_shape_basis",
            torch.einsum("jv,vck->jck", regressor, shapedirs).detach(),
        )

    def forward(
        self,
        *,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        return_full_pose: bool = False,
    ) -> object:
        """Return beta-shaped joints using the standalone skeleton forward path."""
        pose = torch.cat((global_orient, body_pose), dim=1).reshape(-1, 24, 3)
        rest_joints = self.joint_template.unsqueeze(0) + torch.einsum(
            "jck,bk->bjc",
            self.joint_shape_basis,
            betas[:, :10],
        )
        offsets = torch.stack(
            [
                torch.zeros_like(rest_joints[:, joint])
                if parent < 0
                else rest_joints[:, joint] - rest_joints[:, parent]
                for joint, parent in enumerate(SMPLX_BODY_PARENTS)
            ],
            dim=1,
        )
        joints = _skeleton_fk(pose, rest_joints[:, 0], offsets)
        output = SimpleNamespace(joints=joints, vertices=joints[:, :1])
        if return_full_pose:
            output.full_pose = pose
        return output


class _MaxMixturePrior(torch.nn.Module):
    """Eight-component maximum-mixture pose prior."""

    def __init__(self, prior_path: Path) -> None:
        """Load serialized eight-component GMM parameters."""
        super().__init__()
        with prior_path.open("rb") as handle:
            gmm = pickle.load(handle, encoding="latin1")
        means = gmm["means"].astype(np.float32)
        covariances = gmm["covars"].astype(np.float32)
        precisions = np.stack([np.linalg.inv(value) for value in covariances]).astype(np.float32)
        square_roots = np.asarray([np.sqrt(np.linalg.det(value)) for value in gmm["covars"]])
        normalizer = (2 * np.pi) ** (69 / 2.0)
        weights = np.asarray(gmm["weights"] / (normalizer * (square_roots / square_roots.min())))
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("precisions", torch.tensor(precisions, dtype=torch.float32))
        self.register_buffer(
            "nll_weights",
            torch.tensor(weights, dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, pose: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """Return the minimum negative log likelihood over GMM components."""
        del betas
        difference = pose.unsqueeze(dim=1) - self.means
        product = torch.einsum("mij,bmj->bmi", self.precisions, difference)
        likelihood = 0.5 * (product * difference).sum(dim=-1) - torch.log(self.nll_weights)
        return torch.min(likelihood, dim=1).values


def _gmof(values: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    """Apply the standalone Geman-McClure robust error."""
    squared = values**2
    return (sigma**2 * squared) / (sigma**2 + squared)


def _angle_prior(body_pose: torch.Tensor) -> torch.Tensor:
    """Apply the standalone knee and elbow bending prior."""
    signs = body_pose.new_tensor(_ANGLE_PRIOR_SIGNS)
    return torch.exp(body_pose[:, _ANGLE_PRIOR_INDICES] * signs) ** 2


def _body_fitting_loss(
    body_pose: torch.Tensor,
    preserve_pose: torch.Tensor,
    betas: torch.Tensor,
    model_joints: torch.Tensor,
    camera_translation: torch.Tensor,
    target_joints: torch.Tensor,
    pose_prior: _MaxMixturePrior,
    confidence: torch.Tensor,
    *,
    pose_preserve_weight: float,
) -> torch.Tensor:
    """Compute the first-frame body loss with the preserved fit weights."""
    joint_error = _gmof((model_joints + camera_translation) - target_joints)
    weighted_joint_error = (confidence**2) * joint_error.sum(dim=-1)
    joint_loss = (600.0**2 * weighted_joint_error).sum(dim=-1)
    pose_prior_loss = ((4.78 * 1.5) ** 2) * pose_prior(body_pose, betas)
    angle_prior_loss = (15.2**2) * _angle_prior(body_pose).sum(dim=-1)
    shape_prior_loss = (5.0**2) * (betas**2).sum(dim=-1)
    preserve_loss = (pose_preserve_weight**2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)
    return (
        joint_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + preserve_loss
    ).sum()


def _camera_fitting_loss(
    model_joints: torch.Tensor,
    camera_translation: torch.Tensor,
    estimated_translation: torch.Tensor,
    target_joints: torch.Tensor,
) -> torch.Tensor:
    """Compute the preserved root orientation and translation loss."""
    translated = model_joints + camera_translation
    joint_error = (target_joints[:, _ALIGNMENT_JOINTS] - translated[:, _ALIGNMENT_JOINTS]) ** 2
    translation_error = (100.0**2) * (camera_translation - estimated_translation) ** 2
    return (joint_error + translation_error).sum()


def _create_skeleton_model(model_path: Path, device: torch.device) -> _SkeletonBodyAdapter:
    """Build the lightweight SMPL-X-to-skeleton fitting adapter."""
    smplx = _load_smplx()
    body_model = smplx.SMPLX(
        str(model_path),
        num_betas=10,
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
    ).to(device)
    return _SkeletonBodyAdapter(_SmplxBodyAdapter(body_model)).to(device)


def _resolve_fitting_resources(data_dir: str | Path | None) -> tuple[Path, Path]:
    """Resolve the mean-pose and GMM files used by beta fitting.

    Args:
        data_dir: Optional external resource directory. ``None`` selects the
            package's ``data`` directory.

    Returns:
        Paths to the neutral mean pose and eight-component GMM prior.

    Raises:
        FileNotFoundError: If either resource is missing.
    """
    directory = (
        Path(__file__).resolve().with_name("data")
        if data_dir is None
        else Path(data_dir).resolve()
    )
    mean_path = directory / _MEAN_POSE_FILENAME
    prior_path = directory / _POSE_PRIOR_FILENAME
    missing = [path for path in (mean_path, prior_path) if not path.is_file()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"missing beta-fitting resources in {directory}: {missing_names}; "
            "install licensed package data or set BetaFitConfig.data_dir"
        )
    return mean_path, prior_path


def fit_betas(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    model_path: str | Path,
    *,
    initial_betas: np.ndarray | None = None,
    config: BetaFitConfig | None = None,
) -> BetaFitResult:
    """Fit beta on frame zero with the preserved SMPLify parameters.

    Args:
        target_joints: Raw first-frame targets with shape ``(22, 3)``.
        rest_joints: Neutral SMPL-X rest joints with shape ``(22, 3)``.
        model_path: Path to an official neutral SMPL-X NPZ model.
        initial_betas: Optional 10-dimensional initialization. Defaults to zero.
        config: Optimizer settings and optional resource-directory override.

    Returns:
        Fitted beta, nuisance pose and translation, and diagnostics.

    Raises:
        FileNotFoundError: If a model or SMPLify resource is unavailable.
        RuntimeError: If the requested CUDA device is unavailable.
        ValueError: If an input or numerical setting is invalid.
    """
    config = BetaFitConfig() if config is None else config
    if config.iterations < 1:
        raise ValueError("beta fitting iterations must be positive")
    if not np.isfinite(config.step_size) or config.step_size <= 0.0:
        raise ValueError("beta fitting step_size must be finite and positive")
    model_file = Path(model_path).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    mean_path, prior_path = _resolve_fitting_resources(config.data_dir)

    targets_array = _prepare_target_joints(target_joints, rest_joints)
    beta_array = (
        np.zeros(10, dtype=np.float32)
        if initial_betas is None
        else np.asarray(initial_betas, dtype=np.float32).reshape(-1)
    )
    if beta_array.shape != (10,) or not np.isfinite(beta_array).all():
        raise ValueError("initial_betas must contain exactly 10 finite coefficients")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested beta fitting device is unavailable: {device}")
    model = _create_skeleton_model(model_file, device)
    pose_prior = _MaxMixturePrior(prior_path).to(device)
    with h5py.File(mean_path, "r") as handle:
        initial_pose = torch.from_numpy(handle["pose"][:]).reshape(1, 72).float().to(device)

    targets = torch.as_tensor(targets_array, device=device).unsqueeze(0)
    body_pose = initial_pose[:, 3:].detach().clone()
    global_orient = initial_pose[:, :3].detach().clone()
    betas = torch.as_tensor(beta_array, device=device).reshape(1, 10).detach().clone()
    initial_output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
    )
    estimated_translation = (
        (targets[:, _ALIGNMENT_JOINTS] - initial_output.joints[:, _ALIGNMENT_JOINTS])
        .mean(dim=1, keepdim=True)
        .detach()
    )
    camera_translation = estimated_translation.clone()
    preserve_pose = body_pose.detach().clone()

    global_orient.requires_grad = True
    camera_translation.requires_grad = True
    camera_optimizer = torch.optim.LBFGS(
        (global_orient, camera_translation),
        max_iter=config.iterations,
        lr=config.step_size,
        line_search_fn="strong_wolfe",
    )
    camera_outer_iterations = int(os.environ.get("JOINT2SMPL_CAMERA_OUTER_ITERS", "10"))
    if camera_outer_iterations < 1:
        raise ValueError("JOINT2SMPL_CAMERA_OUTER_ITERS must be >= 1")
    for _ in range(camera_outer_iterations):

        def camera_closure() -> torch.Tensor:
            """Evaluate the camera-stage closure."""
            camera_optimizer.zero_grad()
            current = model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
            )
            loss = _camera_fitting_loss(
                current.joints,
                camera_translation,
                estimated_translation,
                targets,
            )
            loss.backward()
            return loss

        camera_optimizer.step(camera_closure)

    body_pose.requires_grad = True
    betas.requires_grad = True
    body_optimizer = torch.optim.LBFGS(
        (body_pose, betas, global_orient, camera_translation),
        max_iter=config.iterations,
        lr=config.step_size,
        line_search_fn="strong_wolfe",
    )
    confidence = torch.ones(22, dtype=torch.float32, device=device)

    def body_loss(pose_preserve_weight: float) -> tuple[torch.Tensor, object]:
        """Evaluate the standalone body-stage objective."""
        current = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
        )
        loss = _body_fitting_loss(
            body_pose,
            preserve_pose,
            betas,
            current.joints,
            camera_translation,
            targets,
            pose_prior,
            confidence,
            pose_preserve_weight=pose_preserve_weight,
        )
        return loss, current

    body_outer_iterations = int(
        os.environ.get("JOINT2SMPL_BODY_OUTER_ITERS", str(config.iterations))
    )
    if body_outer_iterations < 1:
        raise ValueError("JOINT2SMPL_BODY_OUTER_ITERS must be >= 1")
    for _ in range(body_outer_iterations):

        def body_closure() -> torch.Tensor:
            """Evaluate the body-stage LBFGS closure."""
            body_optimizer.zero_grad()
            loss, _ = body_loss(5.0)
            loss.backward()
            return loss

        body_optimizer.step(body_closure)

    with torch.no_grad():
        final_loss, final_output = body_loss(0.0)
        fitted_joints = final_output.joints + camera_translation
        mean_joint_error = torch.linalg.norm(fitted_joints - targets, dim=-1).mean()
        values = (body_pose, betas, global_orient, camera_translation, final_loss)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError("beta fitting produced non-finite values")
        fitted_pose = torch.cat((global_orient, body_pose), dim=-1).reshape(24, 3)

    return BetaFitResult(
        betas=betas.detach().cpu().numpy()[0].astype(np.float32),
        pose=fitted_pose[:22].detach().cpu().numpy().astype(np.float32),
        translation=camera_translation.detach().cpu().numpy().reshape(3).astype(np.float32),
        final_loss=float(final_loss.detach().cpu()),
        mean_joint_error=float(mean_joint_error.detach().cpu()),
    )
