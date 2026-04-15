"""Abstract base class for field dimensionality reducers."""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class FieldReducer(ABC):
    """Abstract PCA-based reducer for Q output fields.

    Workflow
    --------
    1. ``fit(fields_centered)``            – fit PCA model(s) on centred training fields
    2. ``encode_to_per_mode(fields)``      – encode fields → per-mode weight structure
    3. ``decode_from_per_mode(...)``       – decode weights → reconstructed fields
                                             (handles constraint-based deduction of a
                                              fixed output if *fixed_idx* is given)
    4. ``reconstruction_error(...)``       – PCA-only RRMSE on any set of fields

    Parameters
    ----------
    n_modes : number of PCA modes (M)
    """

    def __init__(self, n_modes: int):
        self.n_modes = n_modes
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, fields_centered: List[np.ndarray]) -> None:
        """Fit PCA model(s) on centred training fields.

        Parameters
        ----------
        fields_centered : List[Q arrays of shape (N_train, S)]
            Already centred (mean subtracted) training fields.
        """

    @abstractmethod
    def encode_to_per_mode(
        self,
        fields: List[np.ndarray],
        exclude_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Project fields to latent weights, organised by mode.

        Parameters
        ----------
        fields     : List[Q arrays of shape (N, S)]  (centred)
        exclude_idx: if not None, skip this field index (used by FieldwisePCA).

        Returns
        -------
        weights_per_mode : List[M arrays]
            For RowwisePCA  : each array is (N, Q)
            For ColwisePCA  : each array is (N, 1)
            For FieldwisePCA: each array is (N, Q) or (N, Q-1)
        """

    @abstractmethod
    def decode_from_per_mode(
        self,
        means_w: List[np.ndarray],
        cross_covs: List[np.ndarray],
        means_train: List[np.ndarray],
        fixed_idx: Optional[int],
        u: np.ndarray,
    ) -> Dict[str, List[np.ndarray]]:
        """Reconstruct Q fields from per-mode predicted weights.

        Parameters
        ----------
        means_w   : List[M] of (N_test, q_m)  – predicted weight means
        cross_covs: List[M] of (N_test, q_m, q_m) – predicted per-point cross-output
                    covariance (diagonal for IndepSOGP, full for MOGP-LCM)
        means_train: List[Q of (S,)] – training means added back
        fixed_idx : index of the output deduced from constraint (None = no deduction)
        u         : constraint normal vector

        Returns
        -------
        dict with keys:
            'fields_mean' : List[Q arrays (N_test, S)]
            'fields_var'  : List[Q arrays (N_test, S)]
        """

    # ------------------------------------------------------------------
    # Common utility
    # ------------------------------------------------------------------

    def reconstruction_error(
        self,
        fields_centered: List[np.ndarray],
        means_train: List[np.ndarray],
        split_name: str = "test",
    ) -> Dict[str, Any]:
        """RRMSE of PCA-only reconstruction (no GP).

        Parameters
        ----------
        fields_centered : centred fields (train or test)
        means_train     : training means (to recover original scale)
        split_name      : label for the returned dict keys

        Returns
        -------
        dict with 'rrmse_per_field' array of shape (Q,).
        """
        Q = len(fields_centered)
        rrmse = np.zeros(Q)
        for i in range(Q):
            f_orig = fields_centered[i] + means_train[i]
            # Encode then decode field i using the reducer
            # We use all fields together for consistency
        weights = self.encode_to_per_mode(fields_centered)
        dummy_covs = [
            np.zeros((w.shape[0], w.shape[1], w.shape[1])) for w in weights
        ]
        result = self.decode_from_per_mode(
            weights, dummy_covs, means_train, fixed_idx=None, u=None
        )
        for i in range(Q):
            f_orig = fields_centered[i] + means_train[i]
            f_rec  = result["fields_mean"][i]
            ss_res = np.sum((f_orig - f_rec) ** 2)
            ss_tot = np.sum(f_orig ** 2)
            rrmse[i] = np.sqrt(ss_res / (ss_tot + 1e-15))
        return {f"rrmse_per_field_{split_name}": rrmse}

    def cumulative_reconstruction_error(
        self,
        fields_centered: List[np.ndarray],
        means_train: List[np.ndarray],
        split_name: str = "test",
        fixed_idx: Optional[int] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Q² and RRMSE of PCA reconstruction for k = 1 … M modes (cumulative).

        For each k, reconstructs the fields using only the first k PCA modes
        (with constraint deduction if ``fixed_idx`` is given) and measures
        quality against the original (un-centred) fields.

        Parameters
        ----------
        fields_centered : centred test (or train) fields, List[Q] of (N, S)
        means_train     : training means, List[Q] of (S,)
        split_name      : suffix for the returned dict keys
        fixed_idx       : if not None, deduce this output from the constraint
                          (uses ``u``); the deduced field quality is included.
        u               : constraint vector (required when fixed_idx is not None)

        Returns
        -------
        dict with:
            ``cumulative_q2_{split_name}``    : (M, Q) – Q²   per (k, field)
            ``cumulative_rrmse_{split_name}`` : (M, Q) – RRMSE per (k, field)
        """
        M = self.n_modes
        Q = len(fields_centered)

        weights = self.encode_to_per_mode(fields_centered, exclude_idx=fixed_idx)
        dummy_covs = [
            np.zeros((w.shape[0], w.shape[1], w.shape[1])) for w in weights
        ]

        # Pre-compute per-field reference norms (do not change with k)
        ss_tot_list  = []
        ss_mean_list = []
        for i in range(Q):
            f = fields_centered[i] + means_train[i]
            ss_tot_list.append(np.sum(f ** 2))
            ss_mean_list.append(np.sum((f - f.mean(axis=0, keepdims=True)) ** 2))

        cum_q2    = np.zeros((M, Q))
        cum_rrmse = np.zeros((M, Q))

        for k in range(1, M + 1):
            rec = self.decode_from_per_mode(
                weights[:k], dummy_covs[:k], means_train,
                fixed_idx=fixed_idx, u=u,
            )
            for i in range(Q):
                f_orig = fields_centered[i] + means_train[i]
                f_rec  = rec["fields_mean"][i]
                ss_res = np.sum((f_orig - f_rec) ** 2)
                cum_rrmse[k - 1, i] = np.sqrt(ss_res / (ss_tot_list[i]  + 1e-15))
                cum_q2[k - 1, i]    = 1.0 - ss_res  / (ss_mean_list[i] + 1e-15)

        return {
            f"cumulative_q2_{split_name}":    cum_q2,
            f"cumulative_rrmse_{split_name}": cum_rrmse,
        }

    def cumulative_prediction_error(
        self,
        predicted_weights: List[np.ndarray],
        fields_centered: List[np.ndarray],
        means_train: List[np.ndarray],
        split_name: str = "test",
        fixed_idx: Optional[int] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Q² and RRMSE of field reconstruction using first k predicted modes.

        Parameters
        ----------
        predicted_weights : List[M] of (N, q_m) – the predicted weights per mode
        fields_centered   : List[Q] of (N, S) – true centred fields
        means_train       : List[Q] of (S,)
        split_name        : str
        fixed_idx         : Optional[int], index to deduce
        u                 : Optional[np.ndarray], constraint vector

        Returns
        -------
        dict with `cumulative_q2_pred_{split_name}` and `cumulative_rrmse_pred_{split_name}`
        """
        M = len(predicted_weights)
        Q = len(fields_centered)

        dummy_covs = [
            np.zeros((w.shape[0], w.shape[1], w.shape[1])) for w in predicted_weights
        ]

        ss_tot_list  = []
        ss_mean_list = []
        for i in range(Q):
            f = fields_centered[i] + means_train[i]
            ss_tot_list.append(np.sum(f ** 2))
            ss_mean_list.append(np.sum((f - f.mean(axis=0, keepdims=True)) ** 2))

        cum_q2    = np.zeros((M, Q))
        cum_rrmse = np.zeros((M, Q))

        for k in range(1, M + 1):
            rec = self.decode_from_per_mode(
                predicted_weights[:k], dummy_covs[:k], means_train,
                fixed_idx=fixed_idx, u=u,
            )
            for i in range(Q):
                f_orig = fields_centered[i] + means_train[i]
                f_rec  = rec["fields_mean"][i]
                ss_res = np.sum((f_orig - f_rec) ** 2)
                cum_rrmse[k - 1, i] = np.sqrt(ss_res / (ss_tot_list[i]  + 1e-15))
                cum_q2[k - 1, i]    = 1.0 - ss_res  / (ss_mean_list[i] + 1e-15)

        return {
            f"cumulative_q2_pred_{split_name}":    cum_q2,
            f"cumulative_rrmse_pred_{split_name}": cum_rrmse,
        }

# ------------------------------------------------------------------
# Shared helper: deduce a fixed output from the sum constraint
# ------------------------------------------------------------------

def deduce_fixed_output(
    fields_orig: List[Optional[np.ndarray]],
    fixed_idx: int,
    u: np.ndarray,
) -> np.ndarray:
    """Deduce fields_orig[fixed_idx] from the constraint u.T @ f = 0.

    f_p_orig = -(1/u_p) * Σ_{i≠p} u_i * f_i_orig

    Parameters
    ----------
    fields_orig : List[Q] – already un-centred fields; fields_orig[fixed_idx]
                  must be None (placeholder for the deduced field).
    fixed_idx   : index of the output to deduce
    u           : constraint vector of length Q

    Returns
    -------
    f_p_orig : (N, S) deduced field (in original un-centred units)
    """
    total = None
    for i, f in enumerate(fields_orig):
        if i == fixed_idx:
            continue
        contrib = u[i] * f
        total = contrib if total is None else total + contrib
    return -total / u[fixed_idx]


def deduce_fixed_output_var(
    vars_not_fixed: List[np.ndarray],
    cross_covs_per_mode: List[np.ndarray],
    phi_per_mode: List[np.ndarray],
    std_per_mode: np.ndarray,
    fixed_local_idx: int,
    u_excl: np.ndarray,
    u_p: float,
) -> np.ndarray:
    """Compute Var(f_p) using the full cross-output covariance (no diagonal simplification).

    For each mode m, the per-point cross-output covariance in the *original*
    (de-normalised) weight space is:
        K_m_denorm[n, i, j] = std_m[i] * std_m[j] * K_m_norm[n, i, j]

    Then:
        Var(f_p[n, t]) = (1/u_p)² * Σ_m φ_m(t)² * u_excl.T @ K_m_denorm[n] @ u_excl

    Parameters
    ----------
    vars_not_fixed    : List[q] arrays (N_test, S)  – var of non-fixed reconstructed fields
                        (used only as fallback / ignored here, kept for API symmetry)
    cross_covs_per_mode : List[M] of (N_test, q, q) – normalised cross-cov per mode
    phi_per_mode      : List[M] of (S,) – PCA components for reconstruction
    std_per_mode      : (M, q) – de-normalisation factors per mode and non-fixed output
    fixed_local_idx   : local index of the fixed output within the q-dim weight vector
                        (not used here; included for potential future use)
    u_excl            : (q,) sub-vector of u without the fixed component
    u_p               : u[fixed_idx] scalar

    Returns
    -------
    var_fp : (N_test, S)
    """
    M = len(cross_covs_per_mode)
    N_test = cross_covs_per_mode[0].shape[0]
    S = phi_per_mode[0].shape[0]

    var_fp = np.zeros((N_test, S))

    for m in range(M):
        phi_sq = phi_per_mode[m] ** 2          # (S,)
        K_norm = cross_covs_per_mode[m]        # (N_test, q, q)

        # De-normalise: K_denorm[n, i, j] = std[m, i] * std[m, j] * K_norm[n, i, j]
        std_m = std_per_mode[m]                # (q,)
        outer_std = np.outer(std_m, std_m)     # (q, q)
        K_denorm = K_norm * outer_std[np.newaxis, :, :]  # broadcast (N, q, q)

        # Quadratic form u_excl.T @ K_denorm[n] @ u_excl  → (N_test,)
        quad = np.einsum("nij,i,j->n", K_denorm, u_excl, u_excl)  # (N_test,)

        # Accumulate: (N_test,) outer (S,) → (N_test, S)
        var_fp += np.outer(quad, phi_sq)

    return var_fp / (u_p ** 2)
