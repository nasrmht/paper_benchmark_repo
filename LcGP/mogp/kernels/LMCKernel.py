"""LMC kernel with fully free L_q parameters and fixed sigma_B = 1.

All L_q[i,j] elements are free optimisation parameters (no constraint on any element).
sigma_B is fixed to 1.0 and is NOT a parameter.
Bounds for L_q elements: (-10, 10).
B_q = L_q @ L_q.T  (no scaling by sigma_B).
init_L_from_pca sets all L_q elements from SVD without masking.
"""
import numpy as np
from typing import List, Tuple, Optional

from .Kernel import Kernel


class LMCKernel:
    """
    LCM kernel with fully free co-regionalisation matrices.

    K(x, x') = sum_{q=1}^Q  B_q ⊗ k_q(x, x')

    where  B_q = L_q @ L_q.T  and all elements of L_q are free parameters.

    Parameters
    ----------
    base_kernels : List[Kernel]
        Spatial base kernels k_q(x, x').
    output_dim : int
        Number of outputs (D).
    rank : List[int] or None
        Rank of each B_q.  Defaults to [output_dim]*Q.
    seed : int
        Random seed for initialisation.
    """

    def __init__(
        self,
        base_kernels: List[Kernel],
        output_dim: int,
        rank: Optional[List[int]] = None,
        seed: int = 43,
    ):
        self.base_kernels = base_kernels
        self.output_dim = output_dim

        if rank is None:
            self.rank = [output_dim] * len(base_kernels)
        else:
            if len(rank) != len(base_kernels):
                raise ValueError("len(rank) must match len(base_kernels)")
            self.rank = rank

        # All L_q elements are free
        self.Lq_params: List[np.ndarray] = []
        self._bounds: List[Tuple[float, float]] = []

        np.random.seed(seed)
        start_idx = 0
        for q, r in enumerate(self.rank):
            n_params = output_dim * r  # ALL elements free
            Lq_vec = np.random.randn(n_params) #np.random.uniform(-1.0, 1.0, n_params)
            self.Lq_params.append(Lq_vec)
            self._bounds.extend([(-10.0, 10.0)] * n_params) #(-10.0, 10.0)] * n_params (np.NINF, np.inf)
            start_idx += n_params

        # start_idx_Lq marks the end of Lq params and start of spatial params.
        # Used by MOGPR.fit(use_init_pca=True) to know which params to randomise.
        self.start_idx_Lq = start_idx

        # Add bounds for spatial base kernels
        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)

    # ------------------------------------------------------------------
    # Parameter interface
    # ------------------------------------------------------------------

    @property
    def params(self) -> np.ndarray:
        """All hyperparameters: [Lq_params flattened] + [kernel params]."""
        parts = list(self.Lq_params)
        for kernel in self.base_kernels:
            parts.append(kernel.params)
        return np.concatenate(parts)

    @params.setter
    def params(self, params: np.ndarray):
        start_idx = 0
        for q, r in enumerate(self.rank):
            n_params_Lq = self.output_dim * r
            self.Lq_params[q] = params[start_idx: start_idx + n_params_Lq]
            start_idx += n_params_Lq
        for kernel in self.base_kernels:
            n = kernel.get_n_params()
            kernel.params = params[start_idx: start_idx + n]
            start_idx += n

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds

    def get_n_params(self) -> int:
        return len(self.params)

    # ------------------------------------------------------------------
    # Matrix reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """Return L_q reshaped to (output_dim, rank[q])."""
        r = self.rank[q]
        return self.Lq_params[q].reshape(self.output_dim, r)

    def get_B(self, q: int) -> np.ndarray:
        """Return B_q = L_q @ L_q.T, shape (output_dim, output_dim)."""
        Lq = self._reconstruct_Lq(q)
        return Lq @ Lq.T

    def get_L(self, q: int) -> np.ndarray:
        return self._reconstruct_Lq(q)

    def get_sigma_B(self, q: int) -> float:
        """Fixed at 1.0 (not a parameter)."""
        return 1.0

    # ------------------------------------------------------------------
    # Kernel evaluation
    # ------------------------------------------------------------------

    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Compute the full covariance matrix.

        Parameters
        ----------
        X1 : (n1 * output_dim, input_dim + 1)  last column = output index
        X2 : (n2 * output_dim, input_dim + 1)  if None, X2 = X1
        """
        if X2 is None:
            X2 = X1

        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = int(n1 / self.output_dim)
        n2_q = int(n2 / self.output_dim)

        x1 = X1[:n1_q, :-1]
        x2 = X2[:n2_q, :-1]

        K = np.zeros((n1, n2))
        for q in range(len(self.base_kernels)):
            K_spatial = self.base_kernels[q](x1, x2)
            B_q = self.get_B(q)
            K += np.kron(B_q, K_spatial)
        return K

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Compute dK/d(theta) for all hyperparameters.

        Returns
        -------
        gradients : (n_params, n1, n2)
        """
        if X2 is None:
            X2 = X1

        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = int(n1 / self.output_dim)
        n2_q = int(n2 / self.output_dim)

        x1 = X1[:n1_q, :-1]
        x2 = X2[:n2_q, :-1]

        gradients = np.zeros((self.get_n_params(), n1, n2))
        param_idx = 0

        # --- Gradients w.r.t. L_q[i, j] (all elements free) ---
        for q in range(len(self.base_kernels)):
            K_spatial = self.base_kernels[q](x1, x2)
            Lq = self._reconstruct_Lq(q)

            for i in range(self.output_dim):
                for j in range(self.rank[q]):
                    dB_q = self._compute_dB_q_dL(Lq, i, j)
                    gradients[param_idx] = np.kron(dB_q, K_spatial)
                    param_idx += 1

        # --- Gradients w.r.t. base kernel hyperparameters ---
        for q, kernel in enumerate(self.base_kernels):
            dK_spatial_list = kernel.gradient(x1, x2)
            B_q = self.get_B(q)
            for dK_spatial in dK_spatial_list:
                gradients[param_idx] = np.kron(B_q, dK_spatial)
                param_idx += 1

        return gradients

    def _compute_dB_q_dL(self, Lq: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        dB_q / dL_q[i, j]  =  outer(e_i, L_q[:, j]) + outer(L_q[:, j], e_i)

        where e_i is the i-th unit vector of size output_dim.
        """
        e_i = np.zeros(self.output_dim)
        e_i[i] = 1.0
        col_j = Lq[:, j]
        return np.outer(e_i, col_j) + np.outer(col_j, e_i)

    def _compute_dB(self):
        """
        Calcule la dérivée de B par rapport à tous les paramètres de B (utile en ICM)
        
        Args:
            
            
        Returns:
            Liste des Matrice dB_q de forme (output_dim, output_dim)
        """
        dB = []
        
        for q in range(len(self.base_kernels)):
            # Calculer la matrice de covariance spatiale k_q(x1, x2)
            # Obtenir la matrice L_q complète et sigma_B
            L_q = self._reconstruct_Lq(q)
           # sigma_B = self.sigma_B_params[q]
            
            # Parcourir tous les paramètres (lignes 2 à output_dim-1)
            for i in range(self.output_dim):
                for j in range(self.rank[q]):
                    # Calculer dB_q par rapport à L_q[i,j]
                    dB_q = self._compute_dB_q_dL(L_q, i, j)
                    dB.append(dB_q)
        
        return np.array(dB)

    # ------------------------------------------------------------------
    # PCA initialisation
    # ------------------------------------------------------------------

    def init_L_from_pca(self, Y: np.ndarray):
        """
        Initialise L_q from the PCA of Y (N x output_dim).

        All L_q elements are set from the leading singular vectors of Y.
        No masking: L_q[0, 0] is treated like any other element.
        """
        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

        for q, r in enumerate(self.rank):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])
            # All elements free — store as flat vector (output_dim * r,)
            self.Lq_params[q] = L_pca.flatten()
