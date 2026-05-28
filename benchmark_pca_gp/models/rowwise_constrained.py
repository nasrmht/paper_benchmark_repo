"""RC model: RowwisePCA + ConstrainedMOGP (one per mode, each with Q outputs)."""
import numpy as np
from typing import List, Optional, Dict

from ..reduction.rowwise import RowwisePCA
from ..regression.constrained_mode import ConstrainedMOGPModeRegressor
from .base import SurrogateModel, normalise_weights_per_mode, denormalise_cross_covs, denormalise_means


class RowwiseConstrainedModel(SurrogateModel):
    """Row-wise PCA + per-mode Constrained MOGP.

    Strategy
    --------
    - No fixed output: the constraint is enforced directly in the GP.
    - For each PCA mode m, the Q weight outputs satisfy
        u_norm_m.T @ w_m = 0
      where ``u_norm_m = u ⊙ std_m`` (element-wise product of the original
      constraint vector and the per-output std of mode m weights).
    - Each mode is modelled by a separate ConstrainedMOGPModeRegressor with
      its own ``u_norm_m`` (created lazily in ``fit()``).

    Parameters
    ----------
    name       : model identifier, e.g. "RC_ConstMOGP_M5"
    n_modes    : number of PCA modes M
    u          : constraint vector of length Q
    n_kernels  : number of LCM base kernels
    latent_dim : LCM latent_dim list (length = n_kernels)
    n_restarts : GP optimisation restarts per mode
    maxiter    : max iterations per restart
    noise_var  : initial noise variance
    """

    def __init__(
        self,
        name: str,
        n_modes: int,
        u: np.ndarray,
        n_kernels: int = 2,
        latent_dim: List[int] = None,
        n_restarts: int = 3,
        maxiter: int = 100,
        noise_var: float = 1e-3,
        seed: int = 42,
    ):
        reducer = RowwisePCA(n_modes=n_modes)
        super().__init__(name=name, reducer=reducer, u_vector=u, fixed_idx=None)
        self.n_kernels = n_kernels
        self.latent_dim = latent_dim if latent_dim is not None else [1] * n_kernels
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.noise_var = noise_var
        self.seed = seed

        # Populated after fit()
        self._mode_models: List[ConstrainedMOGPModeRegressor] = []
        self._std_weights: Optional[np.ndarray] = None   # (M, Q)

    def fit(
        self,
        X_train: np.ndarray,
        fields_train: List[np.ndarray],
        means_train: List[np.ndarray],
    ) -> None:
        Q = len(fields_train)
        # 1. Centre
        fields_c = [f - means_train[i] for i, f in enumerate(fields_train)]
        self._means_train = means_train

        # 2. Fit PCA
        self.reducer.fit(fields_c)
        M = self.reducer.n_modes

        # 3. Encode → per-mode weights (N, Q)
        weights_per_mode = self.reducer.encode_to_per_mode(fields_c)

        # 4. Normalise: std_weights shape (M, Q)
        weights_norm, self._std_weights = normalise_weights_per_mode(weights_per_mode)

        # 5. Build per-mode constraint: u_norm_m = u ⊙ std_m
        #    Then fit one ConstrainedMOGP per mode
        self._mode_models = []
        for m in range(M):
            u_norm_m = (self.u * self._std_weights[m]) / np.linalg.norm(self.u * self._std_weights[m])   # (Q,)
            model = ConstrainedMOGPModeRegressor(
                output_dim=Q,
                u_norm=u_norm_m,
                n_kernels=self.n_kernels,
                latent_dim=self.latent_dim,
                n_restarts=self.n_restarts,
                maxiter=self.maxiter,
                noise_var=self.noise_var,
                seed=self.seed,
            )
            model.fit(X_train, weights_norm[m])
            self._mode_models.append(model)

        self.is_fitted = True

    def predict_fields(self, X_test: np.ndarray) -> Dict[str, List[np.ndarray]]:
        # 1. Predict normalised weights with full cross-cov
        means_norm, cross_covs_norm = [], []
        for model in self._mode_models:
            m_mean, m_cov = model.predict_with_cross_cov(X_test)
            means_norm.append(m_mean)
            cross_covs_norm.append(m_cov)

        # 2. De-normalise
        means_w = denormalise_means(means_norm, self._std_weights)
        cross_covs = denormalise_cross_covs(cross_covs_norm, self._std_weights)

        # 3. Decode
        return self.reducer.decode_from_per_mode(
            means_w, cross_covs, self._means_train,
            fixed_idx=None, u=self.u
        )

    # Expose _regressor for intermediate_metrics compatibility
    @property
    def _regressor_models(self) -> List[ConstrainedMOGPModeRegressor]:
        return self._mode_models

    def get_std_weights(self) -> np.ndarray:
        """(M, Q) normalisation std values."""
        return self._std_weights

    def intermediate_metrics(
        self,
        X_test: np.ndarray,
        fields_test_centered: List[np.ndarray],
        fields_train_centered: List[np.ndarray],
        means_train: List[np.ndarray],
        _weights_test_norm=None,   # ignored — computed internally
    ) -> Dict:
        """PCA reconstruction error + per-mode latent GP Q²/RMSE."""
        result = {}
        result.update(
            self.reducer.reconstruction_error(fields_train_centered, means_train, "train")
        )
        result.update(
            self.reducer.reconstruction_error(fields_test_centered, means_train, "test")
        )
        result.update(
            self.reducer.cumulative_reconstruction_error(
                fields_test_centered, means_train, "test",
                fixed_idx=None, u=None,
            )
        )
        if self.is_fitted and self._mode_models:
            weights_test = self.reducer.encode_to_per_mode(fields_test_centered)
            M = len(weights_test)
            q2_list, rmse_list = [], []
            means_norm = []
            for m in range(M):
                w_norm = weights_test[m] / self._std_weights[m]   # (N_test, Q)
                res = self._mode_models[m].latent_q2_rmse(X_test, w_norm)
                q2_list.append(res["q2_per_output"])
                rmse_list.append(res["rmse_per_output"])
                
                # Predict mean for cumulative error
                m_mean, _ = self._mode_models[m].predict(X_test)
                means_norm.append(m_mean)

            result["latent_q2"]   = np.array(q2_list)   # (M, Q)
            result["latent_rmse"] = np.array(rmse_list)  # (M, Q)

            # Cumulative prediction error
            means_w = denormalise_means(means_norm, self._std_weights)
            cum_pred = self.reducer.cumulative_prediction_error(
                means_w, fields_test_centered, means_train, "test",
                fixed_idx=None, u=None
            )
            result.update(cum_pred)

        return result
