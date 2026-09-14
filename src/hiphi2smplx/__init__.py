"""HiPHI BVH to SMPL-X body conversion using Mink IK."""

from .mano import ManoFitter, ManoFittingInput, ManoFittingResult
from .pipeline import ConversionConfig, MotionResult, fit_bvh, save_result

__all__ = [
    "ConversionConfig",
    "ManoFitter",
    "ManoFittingInput",
    "ManoFittingResult",
    "MotionResult",
    "fit_bvh",
    "save_result",
]

__version__ = "0.1.0"
