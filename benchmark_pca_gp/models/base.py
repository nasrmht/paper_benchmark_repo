"""Abstract SurrogateModel base and shared utilities."""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..reduction.base import FieldReducer
from ..regression.base import PerModeRegressor


class SurrogateModel(ABC):
    """Abstract surrogate model combining a field reducer and a latent GP regressor.

    Subclasses implement the specific strategy for:
    - how to normalise latent weights before GP training (and store the std)
    - how to combine reducer + regressor in ``fit`` / ``predict_fields``

    Attributes
    ----------
    name        : human-readable identifier
    reducer     : FieldReducer instance
    u           : constraint vector
    fixed_idx   : index of the output deduced from constraint (None for RC)
    is_fitted   : True after fit() has been called
    """

    def __init__(
        self,
        name: str,
        reducer: FieldReducer,
        u_vector: np.ndarray,
        fixed_idx: Optional[int] = None,
    ):
        self.name = name
        self.reducer = reducer
        self.u = np.asarray(u_vector)
        self.fixed_idx = fixed_idx
        self.is_fitted = False
        # Stored after fit
        self._means_train: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        fields_train: List[np.ndarray],
        means_train: List[np.ndarray],
    ) -> None:
        """Fit PCA + GP models.

        Parameters
        ----------
        X_train      : (N_train, d)
        fields_train : List[Q of (N_train, S)] – NOT yet centred
        means_train  : List[Q of (S,)] – precomputed centering means
                       (usually from Dataset.center())
        """

    @abstractmethod
    def predict_fields(
        self, X_test: np.ndarray
    ) -> Dict[str, List[np.ndarray]]:
        """Predict all Q output fields.

        Returns
        -------
        dict with:
            'fields_mean' : List[Q of (N_test, S)]
            'fields_var'  : List[Q of (N_test, S)]
        """

    # ------------------------------------------------------------------
    # Common evaluation helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        fields_test_orig: List[np.ndarray],
        means_train: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Predict and compute final metrics.

        Parameters
        ----------
        fields_test_orig : List[Q of (N_test, S)] – original (un-centred) test fields

        Returns
        -------
        dict with all metrics from ``compute_all_metrics`` plus the raw predictions.
        """
        from ..metrics.metrics import compute_all_metrics
        preds = self.predict_fields(X_test)
        metrics = compute_all_metrics(
            preds, fields_test_orig, means_train, self.u
        )
        metrics["predictions"] = preds
        return metrics

    def intermediate_metrics(
        self,
        X_test: np.ndarray,
        fields_test_centered: List[np.ndarray],
        fields_train_centered: List[np.ndarray],
        means_train: List[np.ndarray],
        weights_test_norm: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute PCA-only and GP latent metrics.

        Parameters
        ----------
        fields_test_centered  : centred test fields  (for PCA error)
        fields_train_centered : centred train fields (for PCA error on train)
        weights_test_norm     : normalised test weights (for GP latent metrics)
                                If None, GP latent metrics are skipped.

        Returns
        -------
        dict with 'pca_rrmse_train', 'pca_rrmse_test', optionally 'latent_q2',
        'latent_rmse'.
        """
        result = {}
        result.update(
            self.reducer.reconstruction_error(
                fields_train_centered, means_train, split_name="train"
            )
        )
        result.update(
            self.reducer.reconstruction_error(
                fields_test_centered, means_train, split_name="test"
            )
        )
        if weights_test_norm is not None and hasattr(self, "_regressor"):
            latent = self._regressor.latent_metrics(X_test, weights_test_norm)
            result.update(latent)
        return result


# ------------------------------------------------------------------
# Normalisation helpers (shared across models)
# ------------------------------------------------------------------

def normalise_weights_per_mode(
    weights_per_mode: List[np.ndarray],
) -> tuple:
    """Normalise latent weights without centering (divide by train std).

    Parameters
    ----------
    weights_per_mode : List[M] of (N, q)  training weights

    Returns
    -------
    weights_norm : List[M] of (N, q)  normalised weights
    std_matrix   : (M, q)             per-mode, per-output std
    """
    M = len(weights_per_mode)
    q = weights_per_mode[0].shape[1]
    std_matrix = np.zeros((M, q))
    weights_norm = []
    for m, w in enumerate(weights_per_mode):
        s = w.std(axis=0)*0.0+w.std()
        s = np.where(s < 1e-12, 1.0, s)   # avoid division by zero
        std_matrix[m] = s
        weights_norm.append(w / s[np.newaxis, :])
    return weights_norm, std_matrix


def denormalise_cross_covs(
    cross_covs: List[np.ndarray],
    std_matrix: np.ndarray,
) -> List[np.ndarray]:
    """Scale cross-covariance matrices back to original units.

    K_denorm[n, i, j] = std[m, i] * std[m, j] * K_norm[n, i, j]

    Parameters
    ----------
    cross_covs  : List[M] of (N_test, q, q)  normalised covariances
    std_matrix  : (M, q)

    Returns
    -------
    List[M] of (N_test, q, q)
    """
    result = []
    for m, K in enumerate(cross_covs):
        s = std_matrix[m]           # (q,)
        outer = np.outer(s, s)      # (q, q)
        result.append(K * outer[np.newaxis, :, :])
    return result


def denormalise_means(
    means_norm: List[np.ndarray],
    std_matrix: np.ndarray,
) -> List[np.ndarray]:
    """Multiply normalised means by their std to recover original-scale weights.

    Parameters
    ----------
    means_norm  : List[M] of (N_test, q)
    std_matrix  : (M, q)
    """
    return [w * std_matrix[m][np.newaxis, :] for m, w in enumerate(means_norm)]
