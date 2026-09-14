"""Interface-only example. No MANO fitting implementation is included."""

from hiphi2smplx.mano import ManoFittingInput, ManoFittingResult


class ExternalManoFitter:
    def fit(self, inputs: ManoFittingInput) -> ManoFittingResult:
        raise NotImplementedError(
            "Install a separately distributed MANO fitter and implement this method."
        )
