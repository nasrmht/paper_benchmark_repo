import numpy as np
from typing import List, Tuple, Optional
from .Kernel import Kernel


class LMCKernelConstrainedBeta:
    """
    LMC kernel with constraint u^T L_q = 0 and estimated scale sigma_B_q.

    K(x, x') = sum_q sigma_B_q * (L_q L_q^T) ⊗ k_q(x, x')

    where:
      - sigma_B_q > 0 is estimated directly (not log-encoded)
      - L_q satisfies u^T L_q = 0 column-wise (first row deduced from others)

    Parameter order: [Lq_params..., sigma_B..., kernel_params...]
    """

    def __init__(
        self,
        base_kernels: List[Kernel],
        output_dim: int,
        u_vector: np.ndarray = None,
        latent_dim: Optional[List[int]] = None,
        seed: int = 43,
    ):
        self.base_kernels = base_kernels
        self.output_dim = output_dim
        self.start_idx_Lq = 0

        if u_vector is None:
            self.u_vector = np.zeros(output_dim)
            self.u_vector[0] = 1.0
        else:
            if len(u_vector) != output_dim:
                raise ValueError(f"u must have length {output_dim}")
            if u_vector[0] == 0:
                raise ValueError("First element of u cannot be zero")
            self.u_vector = np.array(u_vector) / np.linalg.norm(u_vector)

        if latent_dim is None:
            self.latent_dim = [output_dim] * len(base_kernels)
        else:
            if len(latent_dim) != len(base_kernels):
                raise ValueError("len(latent_dim) must equal len(base_kernels)")
            self.latent_dim = latent_dim

        self.Lq_params = []
        self._bounds = []
        np.random.seed(seed)
        start_idx = 0

        for q, r in enumerate(self.latent_dim):
            if self.output_dim > 2:
                n_params = (self.output_dim - 1) * r
                self.Lq_params.append(np.random.randn(n_params))
                self._bounds.extend([(-1.0, 1.0)] * n_params)
                start_idx += n_params
            else:
                pass

        self.start_idx_Lq = start_idx

        # sigma_B: one per kernel, estimated directly, initialised to 1
        self.sigma_B = np.ones(len(self.latent_dim))
        self._bounds.extend([(1.0, 100.0)] * len(self.latent_dim))

        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)

    # ------------------------------------------------------------------
    # params property
    # ------------------------------------------------------------------

    @property
    def params(self) -> np.ndarray:
        parts = list(self.Lq_params) if self.output_dim > 2 else []
        parts.append(self.sigma_B)
        for kernel in self.base_kernels:
            parts.append(kernel.params)
        return np.concatenate(parts) if parts else np.array([])

    @params.setter
    def params(self, params: np.ndarray):
        idx = 0
        if self.output_dim > 2:
            for q, r in enumerate(self.latent_dim):
                n = (self.output_dim - 1) * r
                self.Lq_params[q] = params[idx:idx + n]
                idx += n
        # sigma_B
        n_q = len(self.latent_dim)
        self.sigma_B = params[idx:idx + n_q]
        idx += n_q
        # kernel params
        for kernel in self.base_kernels:
            n = kernel.get_n_params()
            kernel.params = params[idx:idx + n]
            idx += n

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self._bounds

    # ------------------------------------------------------------------
    # L reconstruction and coregionalisation matrix
    # ------------------------------------------------------------------

    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """Reconstruct L_q enforcing u^T L_q = 0 column-wise."""
        r = self.latent_dim[q]
        L_q = np.zeros((self.output_dim, r))

        if self.output_dim > 2:
            L_q[1:, :] = self.Lq_params[q].reshape(self.output_dim - 1, r)

        u = self.u_vector
        for j in range(r):
            if self.output_dim > 2:
                L_q[0, j] = -(1.0 / u[0]) * np.sum(u[1:] * L_q[1:, j])
            else:
                L_q[0, j] = -(1.0 / u[0]) * u[1]
                L_q[1, j] = 1.0
        return L_q

    def get_B(self, q: int) -> np.ndarray:
        """B_q = sigma_B_q * L_q L_q^T."""
        L_q = self._reconstruct_Lq(q)
        return float(self.sigma_B[q]) * (L_q @ L_q.T)

    def get_L(self, q: int) -> np.ndarray:
        return self._reconstruct_Lq(q)

    def get_sigma_B(self, q: int) -> float:
        return float(self.sigma_B[q])

    def verify_constraint(self, q: int, tol: float = 1e-10) -> bool:
        L_q = self._reconstruct_Lq(q)
        return np.all(np.abs(self.u_vector @ L_q) < tol)

    def get_n_params(self) -> int:
        return len(self.params)

    # ------------------------------------------------------------------
    # Kernel evaluation
    # ------------------------------------------------------------------

    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = n1 // self.output_dim
        n2_q = n2 // self.output_dim
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
        if X2 is None:
            X2 = X1
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = n1 // self.output_dim
        n2_q = n2 // self.output_dim
        x1 = X1[:n1_q, :-1]
        x2 = X2[:n2_q, :-1]

        n_params = self.get_n_params()
        gradients = np.zeros((n_params, n1, n2))
        param_idx = 0

        # --- Gradient wrt L_q free parameters ---
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                K_spatial = self.base_kernels[q](x1, x2)
                L_q = self._reconstruct_Lq(q)
                sigma_B_q = float(self.sigma_B[q])
                for i in range(1, self.output_dim):
                    for j in range(self.latent_dim[q]):
                        dB_q = self._compute_dB_q_dL(q, i, j, L_q)
                        gradients[param_idx] = sigma_B_q * np.kron(dB_q, K_spatial)
                        param_idx += 1

        # --- Gradient wrt sigma_B_q ---
        # d/d(sigma_B_q) [sigma_B_q * L@L.T ⊗ K] = L@L.T ⊗ K  (no sigma_B factor)
        for q in range(len(self.base_kernels)):
            K_spatial = self.base_kernels[q](x1, x2)
            L_q = self._reconstruct_Lq(q)
            B_unnorm = L_q @ L_q.T
            gradients[param_idx] = np.kron(B_unnorm, K_spatial)
            param_idx += 1

        # --- Gradient wrt spatial kernel params ---
        for q, kernel in enumerate(self.base_kernels):
            dK_spatial_list = kernel.gradient(x1, x2)
            B_q = self.get_B(q)
            for dK_spatial in dK_spatial_list:
                gradients[param_idx] = np.kron(B_q, dK_spatial)
                param_idx += 1

        return gradients

    def _compute_dB_q_dL(self, q: int, i: int, j: int, L_q: np.ndarray) -> np.ndarray:
        """dB_q_unnorm / dL_q[i, j], accounting for constrained row."""
        dL_q = np.zeros_like(L_q)
        dL_q[i, j] = 1.0
        dL_q[0, j] = -self.u_vector[i] / self.u_vector[0]
        return dL_q @ L_q.T + L_q @ dL_q.T

    def _compute_dB(self):
        """List of dB matrices for all free L parameters."""
        dB = []
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                L_q = self._reconstruct_Lq(q)
                sigma_B_q = float(self.sigma_B[q])
                for i in range(1, self.output_dim):
                    for j in range(self.latent_dim[q]):
                        dB.append(sigma_B_q * self._compute_dB_q_dL(q, i, j, L_q))
        return dB

    def init_L_from_pca(self, Y: np.ndarray):
        """Initialise L_q from PCA of Y, projecting onto u-orthogonal subspace."""
        if self.output_dim <= 2:
            return
        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        for q, r in enumerate(self.latent_dim):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])
            u = self.u_vector / np.linalg.norm(self.u_vector)
            P = np.eye(self.output_dim) - np.outer(u, u)
            L_proj = P @ L_pca
            self.Lq_params[q] = L_proj[1:, :].flatten()
