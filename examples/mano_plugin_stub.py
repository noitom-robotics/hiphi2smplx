"""Interface-only example. No MANO fitting implementation is included."""

from hiphi2smplx.mano import ManoFittingInput, ManoFittingResult


class ExternalManoFitter:
    """Example boundary for a separately distributed MANO fitter."""

    def fit(self, inputs: ManoFittingInput) -> ManoFittingResult:
        """Fit finger poses for one body-motion result.

        Args:
            inputs: Body motion and source paths supplied by the pipeline.

        Returns:
            Fitted left and right finger rotations.
        """
        raise NotImplementedError(
            "Install a separately distributed MANO fitter and implement this method."
        )
