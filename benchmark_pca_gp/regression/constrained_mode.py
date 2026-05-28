"""Constrained MOGP mode regressor. Used for RowwisePCA + ConstrainedMOGP."""
import numpy as np
from typing import Tuple, List

from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.ConstrainedLMCKernel import LMCKernelConstrained
from LcGP.mogp.kernels.Kernel import RBFKernel, Matern52Kernel

from .base import ModeRegressor, _extract_per_point_cov


class ConstrainedMOGPModeRegressor(ModeRegressor):
    """Constrained MOGP (LCM kernel) for a single latent mode with Q outputs.

    Used by RowwisePCA.  The Q weight outputs satisfy the constraint
        u_norm.T @ w_m = 0
    where u_norm is the constraint vector AFTER normalisation of the weights.

    predict_with_cross_cov returns the FULL (Q, Q) cross-output covariance
    per test point.  This is used to verify constraint satisfaction; for
    RowwisePCA no output needs to be deduced, so the variance of all Q
    fields is obtained directly.

    Parameters
    ----------
    output_dim   : Q, number of constrained outputs
    u_norm       : normalised constraint vector for this mode (length Q)
    n_kernels    : number of LCM base kernels
    latent_dim         : latent_dim list (length = n_kernels)
    n_restarts   : optimisation restarts
    maxiter      : max iterations per restart
    noise_var    : initial noise variance
    """

    def __init__(
        self,
        output_dim: int,
        u_norm: np.ndarray,
        n_kernels: int = 2,
        latent_dim: List[int] = None,
        n_restarts: int = 3,
        maxiter: int = 100,
        noise_var: float = 1e-3,
        seed: int = None,
    ):
        self.output_dim = output_dim
        self.u_norm = np.asarray(u_norm)
        self.n_kernels = n_kernels
        self.latent_dim = latent_dim if latent_dim is not None else [1] * n_kernels
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.noise_var = noise_var
        self.seed = seed
        self._model: MOGPR = None

    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        """Fit constrained MOGP on (N, d) inputs and (N, Q) weights."""
        base_kernels = [Matern52Kernel(input_dim=X.shape[1]) for _ in range(self.n_kernels)]
        kernel = LMCKernelConstrained(
            base_kernels=base_kernels,
            output_dim=self.output_dim,
            u_vector=self.u_norm,
            latent_dim=self.latent_dim,
            seed=self.seed,
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
        """Return (mean (N_test, Q), var_diag (N_test, Q))."""
        mean, var = self._model.predict(X_test, return_cov=True, full_cov=False)
        return mean, var

    def predict_with_cross_cov(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, Q), cross_cov (N_test, Q, Q))."""
        mean, full_cov = self._model.predict(X_test, return_cov=True, full_cov=True)
        n_test = len(X_test)
        K = _extract_per_point_cov(full_cov, n_test, self.output_dim)
        return mean, K
