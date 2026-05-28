"""MOGP-LCM mode regressor using LMCKernel (free L_q, sigma_B=1 fixed)."""
import numpy as np
from typing import Tuple, List

from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LMCKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import Matern52Kernel

from .base import ModeRegressor, _extract_per_point_cov


class MOGPLCMFreeModeRegressor(ModeRegressor):
    """Multi-output GP with fully-free LCM kernel for a single latent mode.

    Identical interface to MOGPLCMModeRegressor but uses LMCKernelFree:
    - All L_q[i,j] elements are free parameters (bounds: -10 to 10).
    - sigma_B is fixed to 1.0 (not estimated).
    - No constraint on L_q[0,0].

    Parameters
    ----------
    output_dim : int   q, number of outputs (Q-1 non-fixed fields)
    n_kernels  : int   number of LCM base kernels
    latent_dim       : list  latent_dim of each B_q
    n_restarts : int   number of optimisation restarts
    maxiter    : int   max iterations per restart
    noise_var  : float initial noise variance
    """

    def __init__(
        self,
        output_dim: int,
        n_kernels: int = 2,
        latent_dim: List[int] = None,
        n_restarts: int = 3,
        maxiter: int = 100,
        noise_var: float = 1e-3,
    ):
        self.output_dim = output_dim
        self.n_kernels = n_kernels
        self.latent_dim = latent_dim if latent_dim is not None else [1] * n_kernels
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.noise_var = noise_var
        self._model: MOGPR = None

    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        base_kernels = [Matern52Kernel(input_dim=X.shape[1]) for _ in range(self.n_kernels)]
        kernel = LMCKernel(
            base_kernels=base_kernels,
            output_dim=self.output_dim,
            latent_dim=self.latent_dim,
        )
        self._model = MOGPR(
            kernel=kernel,
            noise_variance=self.noise_var,
            use_efficient_lik=False,
        )
        self._model.fit(
            X, w,
            n_restarts=self.n_restarts,
            maxiter=self.maxiter,
            use_init_pca=True,
        )

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, var = self._model.predict(X_test, return_cov=True, full_cov=False)
        return mean, var

    def predict_with_cross_cov(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, full_cov = self._model.predict(X_test, return_cov=True, full_cov=True)
        n_test = len(X_test)
        K = _extract_per_point_cov(full_cov, n_test, self.output_dim)
        return mean, K
