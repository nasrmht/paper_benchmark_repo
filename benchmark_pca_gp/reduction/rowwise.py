"""Row-wise PCA: a single PCA fitted on all Q fields stacked row-wise."""
import numpy as np
from typing import List, Optional, Dict
from sklearn.decomposition import PCA

from .base import FieldReducer, deduce_fixed_output, deduce_fixed_output_var


class RowwisePCA(FieldReducer):
    """PCA on row-stacked fields.

    Stacking: y_stacked = vstack([f1, f2, ..., fQ])  → shape (Q*N, S).
    The shared PCA basis φ_m ∈ R^S captures variance across all fields jointly.

    Per-mode weight structure
    -------------------------
    For mode m and sample n, the Q weights are [W_1[n,m], ..., W_Q[n,m]].
    encode_to_per_mode returns List[M] of (N, Q) arrays.

    Constraint in latent space
    --------------------------
    Since f_i ≈ Σ_m W_i_m φ_m, the constraint Σ_i u_i f_i = 0 implies
    (for each mode m independently):
        Σ_i u_i * W_i_m = 0   ⟺   u.T @ w_m = 0
    This is used by ConstrainedMOGPModeRegressor (with normalised u).
    """

    def __init__(self, n_modes: int):
        super().__init__(n_modes)
        self._pca: Optional[PCA] = None
        self._Q: Optional[int] = None

    def fit(self, fields_centered: List[np.ndarray]) -> None:
        Q = len(fields_centered)
        self._Q = Q
        y_stacked = np.vstack(fields_centered)   # (Q*N, S)
        n_comp = min(self.n_modes, y_stacked.shape[1], y_stacked.shape[0])
        self._pca = PCA(n_components=n_comp)
        self._pca.fit(y_stacked)
        self.n_modes = n_comp   # update in case clamped
        self.is_fitted = True

    def _components(self) -> np.ndarray:
        """PCA components (M, S)."""
        return self._pca.components_

    def encode_to_per_mode(
        self,
        fields: List[np.ndarray],
        exclude_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Encode Q fields → List[M] of (N, Q) weight arrays.

        Parameters
        ----------
        exclude_idx : not used for RowwisePCA (all Q fields always encoded).
        """
        Q = len(fields)
        N = fields[0].shape[0]
        M = self.n_modes

        # Project each field independently with the shared PCA
        W_list = [self._pca.transform(fields[i]) for i in range(Q)]  # Q × (N, M)

        # Re-organise by mode
        weights_per_mode = []
        for m in range(M):
            W_m = np.column_stack([W_list[i][:, m] for i in range(Q)])  # (N, Q)
            weights_per_mode.append(W_m)
        return weights_per_mode

    def decode_from_per_mode(
        self,
        means_w: List[np.ndarray],
        cross_covs: List[np.ndarray],
        means_train: List[np.ndarray],
        fixed_idx: Optional[int] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Decode per-mode weights back to Q fields.

        For RowwisePCA, no output deduction is needed: the constraint is
        enforced directly in the latent-space GP (ConstrainedMOGP).

        Reconstruction:
            f_i_mean[n, t] = Σ_m means_w[m][n, i] * φ_m(t) + mean_i(t)
            Var(f_i[n, t]) = Σ_m φ_m(t)² * cross_covs[m][n, i, i]

        The ``fixed_idx`` argument is accepted but not used (the constrained
        GP already enforces the constraint).
        """
        Q = self._Q
        M = self.n_modes
        phi = self._components()          # (M, S)
        N_test = means_w[0].shape[0]
        S = phi.shape[1]

        fields_mean = [np.zeros((N_test, S)) for _ in range(Q)]
        fields_var  = [np.zeros((N_test, S)) for _ in range(Q)]

        for m in range(len(means_w)):   # support truncated reconstruction (k < M)
            phi_m = phi[m]                   # (S,)
            w_m   = means_w[m]               # (N_test, Q)
            K_m   = cross_covs[m]            # (N_test, Q, Q)

            for i in range(Q):
                # mean contribution
                fields_mean[i] += np.outer(w_m[:, i], phi_m)
                # variance contribution (diagonal of cross-cov)
                fields_var[i]  += np.outer(K_m[:, i, i], phi_m ** 2)

        # Add training means back
        for i in range(Q):
            fields_mean[i] += means_train[i][np.newaxis, :]

        return {
            "fields_mean": fields_mean,
            "fields_var": fields_var,
            "predicted_weights": means_w
        }
