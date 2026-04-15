"""FI model: FieldwisePCA + IndependentSOGP, trained once for Q scenarios."""
import numpy as np
from typing import List, Optional, Dict

from ..reduction.fieldwise import FieldwisePCA
from ..regression.base import PerModeRegressor, ModeRegressor
from ..regression.indep_gp_mode import IndepGPModeRegressor
from .base import (
    SurrogateModel,
    normalise_weights_per_mode,
    denormalise_cross_covs,
    denormalise_means,
)


class FieldwiseOptimizedModel:
    """FieldwisePCA + IndepSOGP trained once for multiple fixed-output scenarios.

    Optimisation
    ------------
    For FieldwisePCA + independent SOGPs, all Q fields can be encoded and
    their GP models trained independently.  When evaluating scenario p
    (fixed output = p), we simply ignore the GP predictions for field p
    and deduce them from the constraint.  No retraining is needed.

    Usage
    -----
    model = FieldwiseOptimizedModel(n_modes=5, u=u, Q=4, fixed_indices=[0,1,2,3])
    model.fit(X_train, fields_train, means_train)
    # predict for scenario p=2:
    scenario = model.get_scenario(2)
    preds = scenario.predict_fields(X_test)
    # or all at once:
    all_preds = model.predict_all_scenarios(X_test)

    Parameters
    ----------
    n_modes       : number of PCA modes M
    u             : constraint vector of length Q
    Q             : number of output fields
    fixed_indices : list of fixed output indices for which scenarios are built
    n_restarts    : per-SOGP optimisation restarts
    maxiter       : per-SOGP max iterations
    var_noise     : initial noise variance
    """

    def __init__(
        self,
        n_modes: int,
        u: np.ndarray,
        Q: int,
        fixed_indices: List[int],
        n_restarts: int = 3,
        maxiter: int = 100,
        var_noise: float = 1e-3,
        seed: int = None,
    ):
        self.n_modes = n_modes
        self.u = np.asarray(u)
        self.Q = Q
        self.fixed_indices = fixed_indices
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.var_noise = var_noise
        self.seed = seed

        self._reducer = FieldwisePCA(n_modes=n_modes)
        self._all_weights_norm: Optional[List[np.ndarray]] = None  # List[Q] of (N, M)
        self._all_std: Optional[np.ndarray] = None   # (Q, M)
        self._gp_models: Optional[List[List[ModeRegressor]]] = None  # [field][mode]
        self._means_train: Optional[List[np.ndarray]] = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        fields_train: List[np.ndarray],
        means_train: List[np.ndarray],
    ) -> None:
        """Fit Q PCAs and M*Q independent SOGPs.

        Steps:
        1. Centre fields.
        2. Fit Q independent PCAs.
        3. Encode all Q fields → Q weight matrices (N, M).
        4. Normalise per-field per-mode.
        5. Fit one SOGP per (field, mode) pair → M*Q SOGPs total.
        """
        Q, M_req = self.Q, self.n_modes
        fields_c = [fields_train[i] - means_train[i] for i in range(Q)]
        self._means_train = means_train

        # Fit PCA on all fields
        self._reducer.fit(fields_c)
        M = self._reducer.n_modes   # may be clamped

        # Encode all Q fields
        all_weights = self._reducer.encode_all_fields(fields_c)  # List[Q] of (N, M)

        # Normalise per field
        self._all_weights_norm = []
        self._all_std = np.zeros((Q, M))
        for i in range(Q):
            W = all_weights[i]    # (N, M)
            std = W.std(axis=0)
            std = np.where(std < 1e-12, 1.0, std)
            self._all_std[i] = std
            self._all_weights_norm.append(W / std[np.newaxis, :])

        # Fit M*Q SOGPs: self._gp_models[i][m] is the model for field i, mode m
        self._gp_models = []
        for i in range(Q):
            models_i = []
            for m in range(M):
                gp = IndepGPModeRegressor(
                    output_dim=1,
                    n_restarts=self.n_restarts,
                    maxiter=self.maxiter,
                    var_noise=self.var_noise,
                    seed=self.seed,
                )
                w_im = self._all_weights_norm[i][:, m:m+1]   # (N, 1)
                gp.fit(X_train, w_im)
                models_i.append(gp)
            self._gp_models.append(models_i)

        self.is_fitted = True

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_field(
        self, X_test: np.ndarray, field_idx: int
    ) -> tuple:
        """Predict normalised weights for one field → (means_per_mode, covs_per_mode).

        Returns
        -------
        means_norm : List[M] of (N_test, 1)
        covs_norm  : List[M] of (N_test, 1, 1)
        """
        M = self._reducer.n_modes
        means_norm, covs_norm = [], []
        for m in range(M):
            gp = self._gp_models[field_idx][m]
            mean, cov = gp.predict_with_cross_cov(X_test)
            means_norm.append(mean)     # (N_test, 1)
            covs_norm.append(cov)       # (N_test, 1, 1)
        return means_norm, covs_norm

    # ------------------------------------------------------------------
    # Scenario interface
    # ------------------------------------------------------------------

    def get_scenario(self, fixed_idx: int) -> "FieldwiseScenario":
        """Return a SurrogateModel-compatible object for scenario fixed_idx."""
        return FieldwiseScenario(self, fixed_idx)

    def predict_all_scenarios(
        self, X_test: np.ndarray
    ) -> Dict[int, Dict[str, List[np.ndarray]]]:
        """Predict fields for all scenarios without retraining.

        Returns
        -------
        dict mapping fixed_idx → {'fields_mean': ..., 'fields_var': ...}
        """
        return {p: self.get_scenario(p).predict_fields(X_test)
                for p in self.fixed_indices}

    def latent_metrics(
        self,
        X_test: np.ndarray,
        fields_test_centered: List[np.ndarray],
    ) -> Dict:
        """GP latent Q2/RMSE for each field and mode.

        Parameters
        ----------
        fields_test_centered : centred test fields for computing true weights

        Returns
        -------
        dict with 'latent_q2' (Q, M) and 'latent_rmse' (Q, M)
        """
        M = self._reducer.n_modes
        all_weights_test = self._reducer.encode_all_fields(fields_test_centered)
        q2_all   = np.zeros((self.Q, M))
        rmse_all = np.zeros((self.Q, M))
        for i in range(self.Q):
            W_test = all_weights_test[i][:, :M]   # (N_test, M)
            W_test_norm = W_test / self._all_std[i][np.newaxis, :]
            for m in range(M):
                gp = self._gp_models[i][m]
                mean_norm, _ = gp.predict(X_test)
                mean_norm = mean_norm.ravel()
                w_true = W_test_norm[:, m]
                ss_res = np.sum((w_true - mean_norm) ** 2)
                ss_tot = np.sum((w_true - w_true.mean()) ** 2)
                q2_all[i, m]   = 1.0 - ss_res / (ss_tot + 1e-15)
                rmse_all[i, m] = np.sqrt(np.mean((w_true - mean_norm) ** 2))
        return {"latent_q2": q2_all, "latent_rmse": rmse_all}


# ---------------------------------------------------------------------------
# Thin wrapper: makes a scenario look like a SurrogateModel
# ---------------------------------------------------------------------------

class FieldwiseScenario(SurrogateModel):
    """Read-only view of a FieldwiseOptimizedModel for a specific fixed_idx.

    Created by ``FieldwiseOptimizedModel.get_scenario(fixed_idx)``.
    fit() is a no-op since the parent model is already fitted.
    """

    def __init__(self, parent: FieldwiseOptimizedModel, fixed_idx: int):
        super().__init__(
            name=f"FI_IndepSOGP_M{parent._reducer.n_modes}_p{fixed_idx}",
            reducer=parent._reducer,
            u_vector=parent.u,
            fixed_idx=fixed_idx,
        )
        self._parent = parent
        self._means_train = parent._means_train
        self.is_fitted = parent.is_fitted

    def fit(self, X_train, fields_train, means_train) -> None:
        # No-op: parent handles training
        pass

    def predict_fields(self, X_test: np.ndarray) -> Dict[str, List[np.ndarray]]:
        parent = self._parent
        M = parent._reducer.n_modes
        Q = parent.Q
        p = self.fixed_idx
        non_fixed = [i for i in range(Q) if i != p]
        q = len(non_fixed)

        # 1. Predict each non-fixed field's normalised weights
        means_norm_per_mode = []   # List[M] of (N_test, q)
        covs_norm_per_mode  = []   # List[M] of (N_test, q, q)
        for m in range(M):
            means_m_list, covs_m_list = [], []
            for i in non_fixed:
                gp = parent._gp_models[i][m]
                mu, cov = gp.predict_with_cross_cov(X_test)
                means_m_list.append(mu[:, 0])      # (N_test,)
                covs_m_list.append(cov[:, 0, 0])   # (N_test,) diagonal
            means_norm_per_mode.append(
                np.column_stack(means_m_list)      # (N_test, q)
            )
            # Independent SOGPs → diagonal cross-cov
            covs_diag = np.stack(covs_m_list, axis=1)   # (N_test, q)
            K_m = np.zeros((len(X_test), q, q))
            for ii in range(q):
                K_m[:, ii, ii] = covs_diag[:, ii]
            covs_norm_per_mode.append(K_m)

        # 2. Build std_matrix for the non-fixed fields: shape (M, q)
        std_nf = np.column_stack(
            [parent._all_std[i] for i in non_fixed]
        )  # (Q-1, M) → need (M, Q-1)
        # parent._all_std is (Q, M); extract non-fixed rows, transpose
        std_matrix = parent._all_std[non_fixed, :]   # (q, M) — index by list
        std_matrix = std_matrix.T                    # (M, q)

        # 3. De-normalise
        means_w   = denormalise_means(means_norm_per_mode, std_matrix)
        cross_covs = denormalise_cross_covs(covs_norm_per_mode, std_matrix)

        # 4. Decode via FieldwisePCA (handles deduction of fixed output and variance)
        return parent._reducer.decode_from_per_mode(
            means_w, cross_covs, parent._means_train,
            fixed_idx=p, u=parent.u,
        )
