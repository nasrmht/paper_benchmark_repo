"""Independent multi-output GP mode regressor. Used for FieldwisePCA + IndepSOGP."""
import numpy as np
from typing import Tuple

from LcGP.sogp.indep_mogp import IndependantMultiOutputGP
from LcGP.sogp.kernels.Kernel import RBFKernel

from .base import ModeRegressor


class IndepGPModeRegressor(ModeRegressor):
    """q independent SOGPs for a single latent mode.

    Used by FieldwisePCA + IndepSOGP.  Each of the q outputs (one per
    non-fixed field) has its own independent SOGP.

    Cross-output covariance is assumed zero (diagonal cross-cov matrix).

    Parameters
    ----------
    output_dim   : q, number of independent outputs
    n_restarts   : random restarts per SOGP
    maxiter      : max iterations per optimisation
    var_noise    : initial noise variance
    """

    def __init__(
        self,
        output_dim: int,
        n_restarts: int = 3,
        maxiter: int = 100,
        var_noise: float = 1e-3,
        seed: int = None,
    ):
        self.output_dim = output_dim
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.var_noise = var_noise
        self.seed = seed
        self._model: IndependantMultiOutputGP = None

    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        """Fit output_dim independent SOGPs.

        Parameters
        ----------
        w : (N, q)
        """
        kernel = RBFKernel()
        self._model = IndependantMultiOutputGP(
            output_dim=self.output_dim,
            kernel=kernel,
            var_noise=self.var_noise,
        )
        fit_kw = dict(
            multi_start=(self.n_restarts > 1),
            n_restart=self.n_restarts,
            maxiter=self.maxiter,
        )
        if self.seed is not None:
            fit_kw["seed"] = self.seed
        self._model.fit(X, w, **fit_kw)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, q), var_diag (N_test, q))."""
        mean, var = self._model.predict(X_test, return_cov=True, full_covar=False)
        # var shape from IndependantMultiOutputGP: (q, N_test) → transpose to (N_test, q)
        return mean, var.T

    # predict_with_cross_cov uses the default diagonal implementation from ModeRegressor
