"""Field-wise PCA: one independent PCA per output field."""
import numpy as np
from typing import List, Optional, Dict
from sklearn.decomposition import PCA

from .base import FieldReducer, deduce_fixed_output, deduce_fixed_output_var


class FieldwisePCA(FieldReducer):
    """Independent PCA per output field.

    Per-mode weight structure
    -------------------------
    For a scenario with fixed output p:
        encode_to_per_mode(fields, exclude_idx=p) returns
        List[M] of (N, Q-1) arrays — each mode m, Q-1 non-fixed outputs.

    Without exclusion (``exclude_idx=None``):
        returns List[M] of (N, Q) arrays.

    Optimisation for IndepSOGP
    --------------------------
    Use ``encode_all_fields`` to obtain all Q weight matrices (N, M) at
    once.  Multiple fixed-output scenarios can then reuse the same encoded
    weights without re-running PCA transform.

    Constraint handling
    -------------------
    Pivot field f_p is deduced using the full cross-output covariance from
    the GP predictions in latent space.
    """

    def __init__(self, n_modes: int):
        super().__init__(n_modes)
        self._pcas: List[Optional[PCA]] = []
        self._Q: Optional[int] = None

    def fit(self, fields_centered: List[np.ndarray]) -> None:
        Q = len(fields_centered)
        self._Q = Q
        self._pcas = []
        actual_modes = self.n_modes
        for i in range(Q):
            n_comp = min(self.n_modes, fields_centered[i].shape[1],
                         fields_centered[i].shape[0])
            pca = PCA(n_components=n_comp)
            pca.fit(fields_centered[i])
            self._pcas.append(pca)
            actual_modes = min(actual_modes, n_comp)
        # Align n_modes to the minimum across all fields
        self.n_modes = actual_modes
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_all_fields(self, fields_centered: List[np.ndarray]) -> List[np.ndarray]:
        """Return Q weight matrices, each of shape (N, M).

        Used by FieldwiseOptimizedModel to train all Q scenarios without
        repeating the PCA transform.
        """
        M = self.n_modes
        return [self._pcas[i].transform(fields_centered[i])[:, :M]
                for i in range(self._Q)]

    def encode_to_per_mode(
        self,
        fields: List[np.ndarray],
        exclude_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Encode fields → List[M] of (N, q) weight arrays.

        Parameters
        ----------
        exclude_idx : if not None, skip this field (pivot scenario).

        Returns
        -------
        List[M] of (N, Q-1) or (N, Q) depending on exclude_idx.
        """
        Q = self._Q
        M = self.n_modes
        active_idx = [i for i in range(Q) if i != exclude_idx]

        W_list = [self._pcas[i].transform(fields[i])[:, :M] for i in active_idx]
        # (N, M) each; organise by mode
        weights_per_mode = []
        for m in range(M):
            W_m = np.column_stack([W_list[k][:, m] for k in range(len(active_idx))])
            weights_per_mode.append(W_m)         # (N, q)
        return weights_per_mode

    def encode_from_all_weights(
        self,
        all_weights: List[np.ndarray],
        exclude_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Organise pre-computed per-field weight matrices by mode.

        Parameters
        ----------
        all_weights : List[Q] of (N, M)  – from encode_all_fields()
        exclude_idx : field to skip

        Returns
        -------
        List[M] of (N, Q-1) or (N, Q)
        """
        Q = self._Q
        M = self.n_modes
        active_idx = [i for i in range(Q) if i != exclude_idx]

        weights_per_mode = []
        for m in range(M):
            W_m = np.column_stack([all_weights[i][:, m] for i in active_idx])
            weights_per_mode.append(W_m)
        return weights_per_mode

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_from_per_mode(
        self,
        means_w: List[np.ndarray],
        cross_covs: List[np.ndarray],
        means_train: List[np.ndarray],
        fixed_idx: Optional[int] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Decode per-mode weights back to Q fields.

        Steps:
        1. Reconstruct Q-1 non-fixed fields (mean + variance).
        2. Deduce the pivot field mean.
        3. Compute pivot variance using the full cross-output covariance formula.
        """
        Q = self._Q
        M = len(means_w)   # support truncated reconstruction (k ≤ self.n_modes)
        non_fixed = [i for i in range(Q) if i != fixed_idx]
        q = len(non_fixed)   # number of non-fixed outputs
        N_test = means_w[0].shape[0]

        # 1. Reconstruct each non-fixed field
        fields_mean: List[Optional[np.ndarray]] = [None] * Q
        fields_var:  List[Optional[np.ndarray]] = [None] * Q

        for loc, field_idx in enumerate(non_fixed):
            phi = self._pcas[field_idx].components_[:M]   # (M, S)
            S   = phi.shape[1]
            f_mean = np.zeros((N_test, S))
            f_var  = np.zeros((N_test, S))
            for m in range(M):
                phi_m = phi[m]                       # (S,)
                f_mean += np.outer(means_w[m][:, loc], phi_m)
                f_var  += np.outer(cross_covs[m][:, loc, loc], phi_m ** 2)
            f_mean += means_train[field_idx][np.newaxis, :]
            fields_mean[field_idx] = f_mean
            fields_var[field_idx]  = f_var

        if fixed_idx is None:
            return {
                "fields_mean": fields_mean,
                "fields_var": fields_var,
                "predicted_weights": means_w
            }

        # 2. Deduce pivot mean
        fields_mean[fixed_idx] = deduce_fixed_output(fields_mean, fixed_idx, u)

        # 3. Deduce pivot variance using full cross-output covariance
        #    φ_m for the pivot's own PCA is NOT needed for the deduction variance;
        #    we use the non-fixed fields' PCA components.
        #    For simplicity we pick the first non-fixed field's S and assume all same S.
        phi_0 = self._pcas[non_fixed[0]].components_[:M]  # (M, S)
        phi_per_mode_vec = [phi_0[m] for m in range(M)]

        # std_per_mode is not available here (it lives in the SurrogateModel);
        # at this stage cross_covs already contains the DE-NORMALISED covariances
        # (de-normalisation is done in the SurrogateModel before calling this).
        u_excl = np.array([u[i] for i in non_fixed])

        # Build per-mode phi vectors for each non-fixed field
        # The formula requires φ_m^2 from each non-fixed field separately,
        # but since each has its own PCA, we use a product approximation:
        # Cov(f_i(t), f_j(t)) ≈ Σ_m φ_m_i(t)*φ_m_j(t) * K_m_denorm[n, i, j]
        # Here φ_m_i is the m-th component of field i.
        # We compute this correctly per (i,j) pair.
        S = phi_0.shape[1]
        var_fp = np.zeros((N_test, S))

        for m in range(M):
            K_m = cross_covs[m]  # (N_test, q, q) — already de-normalised
            for ii, fi in enumerate(non_fixed):
                phi_i = self._pcas[fi].components_[m]   # (S,)
                for jj, fj in enumerate(non_fixed):
                    phi_j = self._pcas[fj].components_[m]  # (S,)
                    coeff = u[fi] * u[fj]
                    cov_ij = K_m[:, ii, jj]               # (N_test,)
                    var_fp += coeff * np.outer(cov_ij, phi_i * phi_j)

        var_fp /= (u[fixed_idx] ** 2)
        fields_var[fixed_idx] = var_fp

        return {
            "fields_mean": fields_mean,
            "fields_var": fields_var,
            "predicted_weights": means_w
        }
