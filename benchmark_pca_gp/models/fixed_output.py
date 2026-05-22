"""Generic fixed-output surrogate model (CI and FM)."""
import numpy as np
from typing import List, Optional, Dict

from ..reduction.base import FieldReducer
from ..regression.base import PerModeRegressor
from .base import (
    SurrogateModel,
    normalise_weights_per_mode,
    denormalise_cross_covs,
    denormalise_means,
)


class FixedOutputModel(SurrogateModel):
    """Generic model with one output deduced from the constraint.

    Supports:
    - CI  : ColwisePCA + PerModeRegressor(SOGPModeRegressor)
    - FM  : FieldwisePCA + PerModeRegressor(MOGPLCMModeRegressor)

    The fixed output (``fixed_idx``) is deduced in the decode step.

    Parameters
    ----------
    name          : model identifier, e.g. "CI_SOGP_M5_p2"
    reducer       : fitted or unfitted FieldReducer
    mode_factory  : callable returning a fresh ModeRegressor
    u             : constraint vector
    fixed_idx     : index of the output deduced from constraint
    n_restarts    : forwarded to each ModeRegressor.fit()
    maxiter       : forwarded to each ModeRegressor.fit()
    """

    def __init__(
        self,
        name: str,
        reducer: FieldReducer,
        mode_factory,
        u: np.ndarray,
        fixed_idx: int,
        fit_kwargs: Optional[Dict] = None,
    ):
        super().__init__(name=name, reducer=reducer, u_vector=u, fixed_idx=fixed_idx)
        self.mode_factory = mode_factory
        self.fit_kwargs = fit_kwargs or {}

        # Populated after fit()
        self._regressor: Optional[PerModeRegressor] = None
        self._std_weights: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        fields_train: List[np.ndarray],
        means_train: List[np.ndarray],
    ) -> None:
        # 1. Centre
        fields_c = [f - means_train[i] for i, f in enumerate(fields_train)]
        self._means_train = means_train

        # 2. Fit reducer (on centred fields, passing fixed_idx so ColwisePCA
        #    can exclude the pivot)
        self.reducer.fit(fields_c)
        M = self.reducer.n_modes

        # 3. Encode → per-mode weights with the pivot excluded
        weights_per_mode = self.reducer.encode_to_per_mode(
            fields_c, exclude_idx=self.fixed_idx
        )

        # 4. Normalise
        weights_norm, self._std_weights = normalise_weights_per_mode(weights_per_mode)

        # 5. Fit per-mode GP regressor
        self._regressor = PerModeRegressor(
            n_modes=M,
            mode_factory=self.mode_factory,
            fit_kwargs=self.fit_kwargs,
        )
        self._regressor.fit(X_train, weights_norm)

        self.is_fitted = True

    def predict_fields(self, X_test: np.ndarray) -> Dict[str, List[np.ndarray]]:
        # 1. Predict normalised weights
        means_norm, cross_covs_norm = self._regressor.predict_with_cross_cov(X_test)

        # 2. De-normalise
        means_w = denormalise_means(means_norm, self._std_weights)
        cross_covs = denormalise_cross_covs(cross_covs_norm, self._std_weights)

        # 3. Decode (reducer handles deduction of fixed output)
        return self.reducer.decode_from_per_mode(
            means_w, cross_covs, self._means_train,
            fixed_idx=self.fixed_idx, u=self.u,
        )

    def get_std_weights(self) -> np.ndarray:
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

        # PCA reconstruction error — we encode the Q-1 non-fixed fields
        # and deduce the fixed field via the constraint (passing fixed_idx and u)
        for split_name, fields_c in [("train", fields_train_centered),
                                      ("test",  fields_test_centered)]:
            Q = len(fields_c)
            weights = self.reducer.encode_to_per_mode(
                fields_c, exclude_idx=self.fixed_idx
            )
            q = weights[0].shape[1]
            n = weights[0].shape[0]
            dummy_covs = [np.zeros((n, q, q)) for _ in weights]
            rec = self.reducer.decode_from_per_mode(
                weights, dummy_covs, means_train,
                fixed_idx=self.fixed_idx, u=self.u,
            )
            rrmse = np.zeros(Q)
            for i in range(Q):
                f_orig = fields_c[i] + means_train[i]
                f_rec  = rec["fields_mean"][i]
                ss_res = np.sum((f_orig - f_rec) ** 2)
                ss_tot = np.sum(f_orig ** 2)
                rrmse[i] = np.sqrt(ss_res / (ss_tot + 1e-15))
            result[f"rrmse_per_field_{split_name}"] = rrmse

        # Cumulative PCA reconstruction (k=1..M modes, fixed field deduced)
        result.update(
            self.reducer.cumulative_reconstruction_error(
                fields_test_centered, means_train, "test",
                fixed_idx=self.fixed_idx, u=self.u,
            )
        )

        # Latent GP metrics per mode
        if self.is_fitted and self._regressor is not None:
            weights_test = self.reducer.encode_to_per_mode(
                fields_test_centered, exclude_idx=self.fixed_idx
            )
            w_norm = [w / self._std_weights[m] for m, w in enumerate(weights_test)]
            latent = self._regressor.latent_metrics(X_test, w_norm)
            result.update(latent)   # 'latent_q2' (M, q) et 'latent_rmse' (M, q)
        return result
