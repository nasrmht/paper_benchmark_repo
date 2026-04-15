"""Single-output GP mode regressor (q=1). Used for ColwisePCA."""
import numpy as np
from typing import Tuple, Dict

from LcGP.sogp.core import so_GPRegression
from LcGP.sogp.kernels.Kernel import RBFKernel

from .base import ModeRegressor


class SOGPModeRegressor(ModeRegressor):
    """One SOGP for a single-output (q=1) latent mode.

    Used by ColwisePCA where each mode produces one scalar weight.

    Parameters
    ----------
    n_restarts   : number of random restarts for hyperparameter optimisation
    maxiter      : max iterations per optimisation run
    var_noise    : initial noise variance
    """

    def __init__(
        self,
        n_restarts: int = 3,
        maxiter: int = 100,
        var_noise: float = 1e-3,
        seed: int = None,
    ):
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.var_noise = var_noise
        self.seed = seed
        self._model: so_GPRegression = None

    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        """Fit a SOGP on (N, d) inputs and (N, 1) weights."""
        kernel = RBFKernel()
        self._model = so_GPRegression(
            kernel=kernel,
            var_noise=self.var_noise,
            noisy_data=True,
        )
        y = w[:, 0] if w.ndim > 1 else w
        fit_kw = dict(
            multi_start=(self.n_restarts > 1),
            n_start=self.n_restarts,
            maxiter=self.maxiter,
        )
        if self.seed is not None:
            fit_kw["seed"] = self.seed
        self._model.fit(X, y, **fit_kw)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, 1), var_diag (N_test, 1))."""
        mean, cov = self._model.predict(X_test, return_cov=True)
        var = np.diag(cov).reshape(-1, 1)
        return mean.reshape(-1, 1), var

    # predict_with_cross_cov inherits the diagonal default (trivially correct for q=1)
