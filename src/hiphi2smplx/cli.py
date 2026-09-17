"""Command-line interface for HiPHI to SMPL-X conversion."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from .beta_fitting import BetaFitConfig
from .mano import ManoFitter
from .pipeline import ConversionConfig, fit_bvh, save_result
from .skeleton import SOURCE_SKELETON_MAPS


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser for a BVH input, one SMPL-X model, and one NPZ output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--skeleton-map",
        choices=tuple(sorted(SOURCE_SKELETON_MAPS)),
        default="hiphi",
    )
    shape = parser.add_mutually_exclusive_group(required=True)
    shape.add_argument(
        "--betas",
        type=Path,
        help="NPY/NPZ file containing fixed SMPL-X shape coefficients",
    )
    shape.add_argument(
        "--fit-betas",
        action="store_true",
        help="fit SMPL-X shape from the first BVH frame",
    )
    parser.add_argument("--beta-iters", type=int, default=100)
    parser.add_argument("--beta-step-size", type=float, default=1e-2)
    parser.add_argument("--beta-device", default="cuda:0")
    parser.add_argument(
        "--beta-fit-data",
        type=Path,
        help=(
            "optional directory containing the SMPLify mean pose and GMM prior; "
            "defaults to packaged resources"
        ),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--mink-iters", type=int, default=10)
    parser.add_argument("--mink-dt", type=float, default=0.01)
    parser.add_argument("--mink-damping", type=float, default=1e-2)
    parser.add_argument("--position-cost", type=float, default=1.0)
    parser.add_argument("--ankle-wrist-toe-weight", type=float, default=5.0)
    parser.add_argument("--orientation-cost", type=float, default=0.1)
    parser.add_argument("--posture-cost", type=float, default=0.1)
    parser.add_argument("--root-posture-cost", type=float, default=0.1)
    parser.add_argument("--qp-solver", default="daqp")
    parser.add_argument(
        "--mano-fitter",
        metavar="MODULE:ATTRIBUTE",
        help="Optional external ManoFitter; its implementation is not included",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def load_betas(path: Path) -> np.ndarray:
    """Load the first 10 finite shape coefficients from NPY or NPZ.

    Args:
        path: NPY array or NPZ archive containing ``betas`` or ``beta``.

    Returns:
        A copied float32 array with shape ``(10,)``.

    Raises:
        KeyError: If an NPZ archive has no beta field.
        ValueError: If fewer than 10 finite coefficients are available.
    """
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as data:
            key = "betas" if "betas" in data.files else "beta"
            if key not in data.files:
                raise KeyError(f"{path} must contain betas or beta")
            values = data[key]
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 10 or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain at least 10 finite beta coefficients")
    return values[:10].copy()


def load_mano_fitter(spec: str | None) -> ManoFitter | None:
    """Load an optional external MANO fitter from ``MODULE:ATTRIBUTE``.

    Args:
        spec: Import specification, or ``None`` to disable hand fitting.

    Returns:
        A runtime-checkable :class:`ManoFitter` instance, or ``None``.

    Raises:
        ValueError: If the import specification has invalid syntax.
        TypeError: If the loaded object does not implement ``ManoFitter``.
    """
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--mano-fitter must use MODULE:ATTRIBUTE syntax")
    module_name, attribute_name = spec.rsplit(":", 1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    fitter = target() if isinstance(target, type) else target
    if not isinstance(fitter, ManoFitter):
        raise TypeError(f"{spec} does not implement ManoFitter.fit")
    return fitter


def main(argv: list[str] | None = None) -> int:
    """Run one command-line conversion.

    Args:
        argv: Optional argument list. ``None`` reads from ``sys.argv``.

    Returns:
        Process exit code ``0`` after a successful conversion.

    Raises:
        FileNotFoundError: If an input or model file does not exist.
        FileExistsError: If the output exists without ``--overwrite``.
        ValueError: If an output suffix or numeric option is invalid.
    """
    args = build_parser().parse_args(argv)
    input_bvh = args.input_bvh.resolve()
    output = args.output.resolve()
    model_path = args.model_path.resolve()
    if not input_bvh.is_file():
        raise FileNotFoundError(input_bvh)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must be an NPZ file")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")

    betas = None if args.betas is None else load_betas(args.betas.resolve())
    beta_fit_config = None
    if args.fit_betas:
        beta_fit_config = BetaFitConfig(
            iterations=args.beta_iters,
            step_size=args.beta_step_size,
            device=args.beta_device,
            data_dir=(
                None if args.beta_fit_data is None else args.beta_fit_data.resolve()
            ),
        )
    mano_fitter = load_mano_fitter(args.mano_fitter)
    config = ConversionConfig(
        iterations=args.mink_iters,
        dt=args.mink_dt,
        damping=args.mink_damping,
        posture_cost=args.posture_cost,
        root_posture_cost=args.root_posture_cost,
        position_cost=args.position_cost,
        ankle_wrist_toe_weight=args.ankle_wrist_toe_weight,
        orientation_cost=args.orientation_cost,
        qp_solver=args.qp_solver,
        progress=args.progress,
    )
    result = fit_bvh(
        input_bvh,
        model_path,
        betas,
        beta_fit_config=beta_fit_config,
        config=config,
        max_frames=args.max_frames,
        skeleton_map=args.skeleton_map,
        mano_fitter=mano_fitter,
    )
    save_result(output, result)
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
