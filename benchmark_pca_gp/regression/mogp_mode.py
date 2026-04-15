"""MOGP-LCM mode regressor. Used for FieldwisePCA + MOGP-LCM."""
import numpy as np
from typing import Tuple, List

from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LMCKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel, Matern52Kernel

from .base import ModeRegressor, _extract_per_point_cov


class MOGPLCMModeRegressor(ModeRegressor):
    """Multi-output GP (LCM kernel) for a single latent mode with q outputs.

    Used by FieldwisePCA + MOGP-LCM.  The q non-fixed-field weights are
    modelled jointly.

    predict_with_cross_cov returns the FULL (q, q) cross-output covariance
    at each test point (not just diagonal), enabling an accurate variance
    estimate for the deduced fixed output.

    Parameters
    ----------
    output_dim   : q, number of outputs (Q-1 non-fixed fields)
    n_kernels    : number of LCM base kernels
    rank         : rank list for LCM (length = n_kernels)
    n_restarts   : number of optimisation restarts
    maxiter      : max iterations per restart
    noise_var    : initial noise variance
    """

    def __init__(
        self,
        output_dim: int,
        n_kernels: int = 2,
        rank: List[int] = None,
        n_restarts: int = 3,
        maxiter: int = 100,
        noise_var: float = 1e-3,
        seed: int = None,
    ):
        self.output_dim = output_dim
        self.n_kernels = n_kernels
        self.rank = rank if rank is not None else [1] * n_kernels
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.noise_var = noise_var
        self.seed = seed
        self._model: MOGPR = None

    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        """Fit a MOGP-LCM on (N, d) inputs and (N, q) weights."""
        base_kernels = [Matern52Kernel(input_dim=X.shape[1]) for _ in range(self.n_kernels)]
        kernel = LMCKernel(
            base_kernels=base_kernels,
            output_dim=self.output_dim,
            rank=self.rank,
        )
        self._model = MOGPR(
            kernel=kernel,
            noise_variance=self.noise_var,
            use_efficient_lik=False,
        )
        fit_kw = dict(n_restarts=self.n_restarts, maxiter=self.maxiter, use_init_pca=True)
        if self.seed is not None:
            fit_kw["seed"] = self.seed
        self._model.fit(X, w, **fit_kw)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, q), var_diag (N_test, q))."""
        mean, var = self._model.predict(X_test, return_cov=True, full_cov=False)
        return mean, var  # var already (N_test, q)

    def predict_with_cross_cov(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, q), cross_cov (N_test, q, q)).

        Extracts per-point cross-output covariance from the full joint
        covariance returned by MOGPR.
        """
        mean, full_cov = self._model.predict(X_test, return_cov=True, full_cov=True)
        n_test = len(X_test)
        K = _extract_per_point_cov(full_cov, n_test, self.output_dim)
        return mean, K
