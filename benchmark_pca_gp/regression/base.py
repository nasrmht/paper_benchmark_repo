"""Abstract base classes for latent-space GP regressors."""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional


# ---------------------------------------------------------------------------
# Per-point cross-output covariance extraction
# ---------------------------------------------------------------------------

def _extract_per_point_cov(
    full_cov: np.ndarray, n_test: int, q: int
) -> np.ndarray:
    """Extract per-test-point (q, q) covariance blocks from the joint covariance.

    MOGPR.predict(return_cov=True, full_cov=True) returns a matrix of shape
    (N_test * q, N_test * q) where outputs are stacked:
        [output_0_point_0, ..., output_0_point_{N-1},
         output_1_point_0, ..., output_1_point_{N-1}, ...]

    So full_cov[i*N + n, j*N + n]  =  Cov(output_i at point n, output_j at point n).

    Parameters
    ----------
    full_cov : (N_test * q, N_test * q)
    n_test   : number of test points
    q        : number of outputs

    Returns
    -------
    K : (N_test, q, q)
    """
    K = np.zeros((n_test, q, q))
    for i in range(q):
        for j in range(q):
            K[:, i, j] = np.diag(
                full_cov[i * n_test:(i + 1) * n_test,
                         j * n_test:(j + 1) * n_test]
            )
    return K


# ---------------------------------------------------------------------------
# ModeRegressor  (one model per latent mode)
# ---------------------------------------------------------------------------

class ModeRegressor(ABC):
    """GP model for a single latent mode with q outputs.

    Fit: (X_train (N,d), w_train (N,q)) → optimise hyperparameters.
    Predict: X_test (N_test, d) → mean (N_test, q), var_diag (N_test, q).
    predict_with_cross_cov → mean (N_test, q), cross_cov (N_test, q, q).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, w: np.ndarray, **kwargs) -> None:
        """Fit the GP model.

        Parameters
        ----------
        X : (N, d)
        w : (N, q)  latent weights for this mode
        """

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, q), var_diag (N_test, q))."""

    def predict_with_cross_cov(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean (N_test, q), cross_cov (N_test, q, q)).

        Default implementation: diagonal cross-covariance (no cross-output
        correlation).  Subclasses using MOGP override this to return the
        full cross-output covariance.
        """
        mean, var_diag = self.predict(X_test)
        n = len(X_test)
        q = mean.shape[1] if mean.ndim > 1 else 1
        K = np.zeros((n, q, q))
        for i in range(q):
            K[:, i, i] = var_diag[:, i] if var_diag.ndim > 1 else var_diag
        return mean, K

    def latent_q2_rmse(
        self,
        X_test: np.ndarray,
        w_test: np.ndarray,
    ) -> Dict[str, float]:
        """Q2 and RMSE on normalised latent weights (per output).

        Parameters
        ----------
        w_test : (N_test, q)  true normalised weights

        Returns dict with 'q2_per_output' and 'rmse_per_output'.
        """
        mean, _ = self.predict(X_test)
        q = w_test.shape[1] if w_test.ndim > 1 else 1
        mean = mean.reshape(-1, q)
        w_test = w_test.reshape(-1, q)
        q2   = np.zeros(q)
        rmse = np.zeros(q)
        for i in range(q):
            ss_res = np.sum((w_test[:, i] - mean[:, i]) ** 2)
            ss_tot = np.sum((w_test[:, i] - w_test[:, i].mean()) ** 2)
            q2[i]   = 1.0 - ss_res / (ss_tot + 1e-15)
            rmse[i] = np.sqrt(np.mean((w_test[:, i] - mean[:, i]) ** 2))
        return {"q2_per_output": q2, "rmse_per_output": rmse}


# ---------------------------------------------------------------------------
# PerModeRegressor  (wraps M ModeRegressors)
# ---------------------------------------------------------------------------

class PerModeRegressor:
    """Trains and evaluates one ModeRegressor per latent mode.

    Parameters
    ----------
    n_modes       : number of PCA modes M
    mode_factory  : callable with no args returning a fresh ModeRegressor
    fit_kwargs    : extra keyword arguments forwarded to each ModeRegressor.fit()
    """

    def __init__(
        self,
        n_modes: int,
        mode_factory: Callable[[], ModeRegressor],
        fit_kwargs: Optional[Dict] = None,
    ):
        self.n_modes = n_modes
        self.mode_factory = mode_factory
        self.fit_kwargs = fit_kwargs or {}
        self.models: List[ModeRegressor] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, weights_per_mode: List[np.ndarray]) -> None:
        """Fit one model per mode.

        Parameters
        ----------
        X               : (N, d)
        weights_per_mode: List[M] of (N, q_m)
        """
        self.models = [self.mode_factory() for _ in range(self.n_modes)]
        for m, (model, w) in enumerate(zip(self.models, weights_per_mode)):
            model.fit(X, w, **self.fit_kwargs)
        self.is_fitted = True

    def fit_mode(
        self,
        m: int,
        model: ModeRegressor,
        X: np.ndarray,
        w: np.ndarray,
    ) -> None:
        """Fit a single pre-created mode model (used by RowwiseConstrainedModel
        where each mode has its own constraint vector)."""
        model.fit(X, w, **self.fit_kwargs)

    def predict(
        self, X_test: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (means, var_diags) for all modes."""
        means, var_diags = [], []
        for model in self.models:
            m, v = model.predict(X_test)
            means.append(m)
            var_diags.append(v)
        return means, var_diags

    def predict_with_cross_cov(
        self, X_test: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (means, cross_covs) for all modes.

        cross_covs : List[M] of (N_test, q_m, q_m)
        """
        means, covs = [], []
        for model in self.models:
            m, K = model.predict_with_cross_cov(X_test)
            means.append(m)
            covs.append(K)
        return means, covs

    def latent_metrics(
        self,
        X_test: np.ndarray,
        true_weights_per_mode: List[np.ndarray],
    ) -> Dict:
        """Q2 and RMSE for each mode (on normalised weights).

        Returns dict with 'q2' (M, q) and 'rmse' (M, q).
        """
        q2_list, rmse_list = [], []
        for m, model in enumerate(self.models):
            res = model.latent_q2_rmse(X_test, true_weights_per_mode[m])
            q2_list.append(res["q2_per_output"])
            rmse_list.append(res["rmse_per_output"])
        return {
            "latent_q2":   np.array(q2_list),    # (M, q)
            "latent_rmse": np.array(rmse_list),   # (M, q)
        }
