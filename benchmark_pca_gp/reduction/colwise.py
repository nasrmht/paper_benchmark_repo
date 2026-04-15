"""Column-wise PCA: PCA on horizontally concatenated Q-1 fields."""
import numpy as np
from typing import List, Optional, Dict
from sklearn.decomposition import PCA

from .base import FieldReducer, deduce_fixed_output, deduce_fixed_output_var


class ColwisePCA(FieldReducer):
    """PCA on horizontally concatenated Q-1 fields (pivot excluded).

    Stacking: y_concat = hstack([f_0, ..., f_{p-1}, f_{p+1}, ..., f_{Q-1}])
               → shape (N, (Q-1)*S)

    Per-mode weight structure
    -------------------------
    Each PCA mode produces a single scalar weight per sample.
    encode_to_per_mode returns List[M] of (N, 1) arrays.

    Constraint handling
    -------------------
    The pivot field f_p is deduced from:
        f_p_orig = -(1/u_p) * Σ_{i≠p} u_i * f_i_orig
    Variance of f_p uses the diagonal formula (each mode is a single SOGP,
    no cross-output covariance between modes).
    """

    def __init__(self, n_modes: int, fixed_idx: int):
        super().__init__(n_modes)
        self.fixed_idx = fixed_idx
        self._pca: Optional[PCA] = None
        self._Q: Optional[int] = None
        self._S: Optional[int] = None
        self._non_fixed: Optional[List[int]] = None

    def fit(self, fields_centered: List[np.ndarray]) -> None:
        Q = len(fields_centered)
        self._Q = Q
        self._S = fields_centered[0].shape[1]
        self._non_fixed = [i for i in range(Q) if i != self.fixed_idx]

        y_concat = np.hstack([fields_centered[i] for i in self._non_fixed])  # (N, (Q-1)*S)
        n_comp = min(self.n_modes, y_concat.shape[1], y_concat.shape[0])
        self._pca = PCA(n_components=n_comp)
        self._pca.fit(y_concat)
        self.n_modes = n_comp
        self.is_fitted = True

    def encode_to_per_mode(
        self,
        fields: List[np.ndarray],
        exclude_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Encode Q-1 fields → List[M] of (N, 1) weight arrays."""
        y_concat = np.hstack([fields[i] for i in self._non_fixed])  # (N, (Q-1)*S)
        W = self._pca.transform(y_concat)   # (N, M)
        # Each mode yields a 1-d weight; keep shape (N, 1) for interface consistency
        return [W[:, m:m+1] for m in range(self.n_modes)]

    def decode_from_per_mode(
        self,
        means_w: List[np.ndarray],
        cross_covs: List[np.ndarray],
        means_train: List[np.ndarray],
        fixed_idx: Optional[int] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Reconstruct Q fields from scalar per-mode weights.

        Steps:
        1. Reconstruct concatenated (Q-1)*S field via PCA inverse.
        2. Split back into Q-1 individual fields.
        3. Deduce the pivot field from the constraint.
        4. Compute variance via error propagation.
        """
        M = self.n_modes
        S = self._S
        Q = self._Q
        phi = self._pca.components_          # (M, (Q-1)*S)
        N_test = means_w[0].shape[0]

        # Support truncated reconstruction: use only the first k modes
        k = len(means_w)

        # 1. Reconstruct the concatenated mean field (manual to support k < M)
        W_k = np.column_stack([means_w[m][:, 0] for m in range(k)])  # (N, k)
        y_concat_mean = W_k @ phi[:k, :] + self._pca.mean_            # (N, (Q-1)*S)

        # 2. Variance of concatenated field  (modes independent, q=1 → diagonal trivially)
        #    Var(y_concat[n, s]) = Σ_m cross_covs[m][n, 0, 0] * phi_m[s]^2
        var_concat = np.zeros((N_test, (Q - 1) * S))
        for m in range(k):
            var_concat += np.outer(cross_covs[m][:, 0, 0], phi[m] ** 2)

        # 3. Split Q-1 fields (mean + var) and add training means
        fields_mean: List[Optional[np.ndarray]] = [None] * Q
        fields_var:  List[Optional[np.ndarray]] = [None] * Q
        for loc, i in enumerate(self._non_fixed):
            s0, s1 = loc * S, (loc + 1) * S
            fields_mean[i] = y_concat_mean[:, s0:s1] + means_train[i][np.newaxis, :]
            fields_var[i]  = var_concat[:, s0:s1]

        # 4. Deduce the pivot field (mean)
        fields_mean[self.fixed_idx] = deduce_fixed_output(
            fields_mean, self.fixed_idx, u
        )

        # 5. Variance of pivot: for ColwisePCA each mode has q=1 (SOGP),
        #    so cross_covs[m] is already (N, 1, 1); full formula still applies
        #    but reduces to diagonal because there is only one mode per mode.
        #    We use the general formula from base.py.
        #
        # u_excl: sub-vector of u without fixed component
        u_excl = np.array([u[i] for i in self._non_fixed])
        phi_per_mode = [phi[m, 0:S] for m in range(M)]   # component of 1st non-fixed field
        # For colwise: each PCA component spans (Q-1)*S dimensions;
        # the full formula requires the per-non-fixed-field component.
        # We propagate directly via the reconstructed field variances:
        #   Var(f_p) = (1/u_p)^2 * Σ_{i≠p} u_i^2 * Var(f_i)
        # (modes are independent, no cross-field covariance in ColwisePCA+SOGP)
        var_fp = np.zeros_like(fields_var[self._non_fixed[0]])
        for loc, i in enumerate(self._non_fixed):
            var_fp += (u[i] / u[self.fixed_idx]) ** 2 * fields_var[i]
        fields_var[self.fixed_idx] = var_fp

        return {
            "fields_mean": fields_mean,
            "fields_var": fields_var,
            "predicted_weights": means_w
        }
