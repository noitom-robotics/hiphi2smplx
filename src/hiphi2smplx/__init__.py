"""HiPHI BVH to SMPL-X body conversion using Mink IK."""

from .beta_fitting import BetaFitConfig, BetaFitResult, fit_betas
from .mano import ManoFitter, ManoFittingInput, ManoFittingResult
from .pipeline import ConversionConfig, MotionResult, fit_bvh, save_result

__all__ = [
    "BetaFitConfig",
    "BetaFitResult",
    "ConversionConfig",
    "ManoFitter",
    "ManoFittingInput",
    "ManoFittingResult",
    "MotionResult",
    "fit_betas",
    "fit_bvh",
    "save_result",
]

__version__ = "0.1.0"
