import numpy as np
from typing import List, Tuple, Optional
from .Kernel import Kernel

class LMCKernelConstrained:
    """
    Implementation of the Linear Coregionalization Model (LMC) kernel with constraint.
    
    The kernel is defined as a sum of Kronecker products:
    K(x, x') = sum_{q=1}^Q B_q ⊗ k_q(x, x')

    where B_q = L_q @ L_q.T (sigma_B is fixed to 1, not estimated), and L_q is constrained such that u.T @ L_q = 0.
    The first element of each column of L_q is set as:
    L_q[0, j] = -sum(u[1:] * L_q[1:, j]) / u[0]

    This constraint guarantees that the dot product between u and the outputs is zero.
    """
    
    def __init__(self, base_kernels: List[Kernel], output_dim: int, u_vector: np.ndarray = None, 
                 latent_dim: Optional[List[int]] = None, seed: int = 43):
        """
        Initializes the constrained LMC kernel.
        
        Args:
            base_kernels: List of base kernels k_q(x, x')
            output_dim: Number of outputs (dimension D)
            u_vector: Constraint vector u of size output_dim. If None, a unit vector will be used.
            latent_dim: List of latent dimensions of each matrix L_q associated with a base kernel. Should match the size of base kernels list
                 If None, all matrices will be of full latent_dim
            seed: Seed for random generation
        """
        self.base_kernels = base_kernels
        self.output_dim = output_dim
        self.start_idx_Lq = 0
        
        # Define the constraint vector u
        if u_vector is None:
            # By default, use a unit vector [1, 0, 0, ...]
            self.u_vector = np.zeros(output_dim)
            self.u_vector[0] = 1.0
        else:
            if len(u_vector) != output_dim:
                raise ValueError(f"The vector u must have a length of {output_dim}")
            if u_vector[0] == 0:
                raise ValueError("The first element of the vector u cannot be zero")
            self.u_vector = np.array(u_vector) / np.linalg.norm(u_vector)  # Normalize u
        
        # If latent_dim is not specified, use full latent_dim (=output_dim)
        if latent_dim is None:
            self.latent_dim = [output_dim] * len(base_kernels)
        else:
            if len(latent_dim) != len(base_kernels):
                raise ValueError("The number of latent dimensions must match the number of base kernels")
            self.latent_dim = latent_dim
        
        # Initialize the L_q matrices (without the first rows which will be computed)
        self.Lq_params = []

        self._bounds = []
        np.random.seed(seed)
        start_idx = 0
        
        # Note: If output_dim <= 1, no constraint is possible in the strict sense (or trivial u*L=0 => L=0)
        # But the original code handled output_dim > 2 specifically.
        # For output_dim=2, the original code uses a specific logic.
        
        for q, r in enumerate(self.latent_dim):
            # For each L_q matrix of size output_dim x r, 
            # we parameterize all elements except the first row L_q[0, :]
            
            if self.output_dim > 2:
                n_params = (self.output_dim - 1) * r
                Lq_vec =  np.random.randn(n_params) #np.random.uniform(-0.5, 0.5, n_params) #
                self.Lq_params.append(Lq_vec)
                self._bounds.extend([(-100.0, 100.0)] * n_params)
                start_idx += n_params
            else:
                pass

        self.start_idx_Lq = start_idx

        # Add bounds for the scale factors sigma_B (positive)
        #self._bounds.extend([(1.0, 10.0)] * len(self.latent_dim))
        # Add bounds for the base kernel parameters
        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)
    
    @property
    def params(self) -> np.ndarray:
        """Returns all hyperparameters of the LMC kernel."""
        parts = list(self.Lq_params) if self.output_dim > 2 else []
        for kernel in self.base_kernels:
            parts.append(kernel.params)
        return np.concatenate(parts) if parts else np.array([])
    
    @params.setter
    def params(self, params: np.ndarray):
        """Sets all hyperparameters of the LMC kernel."""
        start_idx = 0
        if self.output_dim > 2:
            for q, r in enumerate(self.latent_dim):
                n_params_Lq = (self.output_dim - 1) * r
                self.Lq_params[q] = params[start_idx:start_idx + n_params_Lq]
                start_idx += n_params_Lq
        for kernel in self.base_kernels:
            n = kernel.get_n_params()
            kernel.params = params[start_idx:start_idx + n]
            start_idx += n
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Returns the bounds of all hyperparameters."""
        return self._bounds
    
    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """
        Reconstructs the complete L_q matrix by computing the first row
        according to the constraint u.T @ L_q = 0 for each column j.
        """
        r = self.latent_dim[q]
        output_dim = self.output_dim
        
        L_q = np.zeros((output_dim, r))
        
        if output_dim > 2:
            L_q[1:, :] = self.Lq_params[q].reshape(output_dim - 1, r)
        
        u = self.u_vector
        for j in range(r):
            if output_dim > 2:
                # u[0]*L[0,j] + sum(u[1:]*L[1:,j]) = 0
                L_q[0,j] = -(1.0/u[0])*(np.sum(u[1:] * L_q[1:, j]))
            else:
                # u[0]L[0,j] + u[1]L[1,j] = 0
                # Original code for 2D: L[1,j]=1.0 (Fixed), L[0,j] = -u[1]/u[0]
                L_q[0, j] = -(1.0/u[0])*u[1]
                L_q[1, j] = 1.0
        return L_q
    
    def get_B(self, q: int) -> np.ndarray:
        """Returns the coregionalization matrix B_q = L_q @ L_q.T."""
        L_q = self._reconstruct_Lq(q)
        return L_q @ L_q.T
    
    def get_L(self, q: int) -> np.ndarray:
        """Returns the complete L_q matrix."""
        return self._reconstruct_Lq(q)
    
    def get_sigma_B(self, q: int) -> float:
        """Fixed to 1.0 (not estimated)."""
        return 1.0
    
    def verify_constraint(self, q: int, tol: float = 1e-10) -> bool:
        """Verifies that the constraint u.T @ L_q = 0 is satisfied."""
        L_q = self._reconstruct_Lq(q)
        dot_products = self.u_vector @ L_q
        return np.all(np.abs(dot_products) < tol)
    
    def get_n_params(self) -> int:
        return len(self.params)
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        K = np.zeros((n1, n2))
        
        for q in range(len(self.base_kernels)):
            K_spatial = self.base_kernels[q](x1, x2)
            B_q = self.get_B(q)
            K += np.kron(B_q, K_spatial) 
        
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n_params = self.get_n_params()
        n1_q = int(n1/self.output_dim)
        x1 = X1[:n1_q,:-1]
        
        gradients = np.zeros((n_params, n1, n1)) # Assuming X2=X1 for now or square matrices logic from original
        # Note: Original code handled n1 != n2, but here I simplify to what I see. 
        # Actually let's use n1,n2 from args.
        n2 = X2.shape[0]
        n2_q = int(n2/self.output_dim)
        x2 = X2[:n2_q,:-1]
        
        gradients = np.zeros((n_params, n1, n2))
        param_idx = 0
        
        # Gradients wrt L_q (excluding constrained rows)
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                K_spatial = self.base_kernels[q](x1, x2)
                L_q = self._reconstruct_Lq(q)
 
                for i in range(1, self.output_dim):  # Skip row 0 (constrained)
                    for j in range(self.latent_dim[q]):
                        dB_q = self._compute_dB_q_dL(q, i, j, L_q)
                        gradients[param_idx] = np.kron(dB_q, K_spatial)
                        param_idx += 1
                        
        # Gradients wrt sigma_B
        # for q in range(len(self.base_kernels)):
        #     K_spatial = self.base_kernels[q](x1, x2)
        #     dB_q = self._compute_dB_q_dsigma(q)
        #     gradients[param_idx] = np.kron(dB_q, K_spatial)
        #     param_idx += 1
            
        # Gradients wrt spatial kernels
        for q, kernel in enumerate(self.base_kernels):
            dK_spatial_list = kernel.gradient(x1, x2)
            B_q = self.get_B(q)
            for dK_spatial in dK_spatial_list:
                gradients[param_idx] = np.kron(B_q, dK_spatial)
                param_idx += 1
                
        return gradients
 
    def _compute_dB_q_dL(self, q: int, i: int, j: int, L_q: np.ndarray) -> np.ndarray:
        """dB_q / dL_q[i, j], accounting for the constrained row L_q[0, j]."""
        dL_q = np.zeros_like(L_q)
        dL_q[i, j] = 1.0
        # Derivative of constrained row: d(L_q[0,j])/d(L_q[i,j]) = -u[i]/u[0]
        dL_q[0, j] = -(self.u_vector[i]) / self.u_vector[0]
        return dL_q @ L_q.T + L_q @ dL_q.T
 
    def _compute_dB(self):
        """
        Computes the derivative of B with respect to all parameters of B (useful in ICM).
        
        Args:
            
            
        Returns:
            List of dB_q matrices of shape (output_dim, output_dim)
        """
        dB = []
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                L_q = self._reconstruct_Lq(q)
                for i in range(1, self.output_dim):
                    for j in range(self.latent_dim[q]):
                        dB.append(self._compute_dB_q_dL(q, i, j, L_q))
        
        # # The derivative of B with respect to sigma_B is simply B_unit = L_q @ L_q.T
        # for q in range(len(self.base_kernels)):
        #     dB_q = self._compute_dB_q_dsigma(q)
        #     dB.append(dB_q)
        
        return dB
 
 
    def _compute_dB_q_dsigma(self, q: int) -> np.ndarray:
        L_q = self._reconstruct_Lq(q)
        return L_q @ L_q.T
 
    def init_L_from_pca(self, Y: np.ndarray):
        """
        Initializes L_q from the PCA of the data Y (n x p),
        respecting the constraint u^T L = 0.
        """
        # If output_dim <= 2, we don't have free params for L.
        if self.output_dim <= 2:
            return 
 
        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
 
        for q, r in enumerate(self.latent_dim):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])
 
            # Projection to respect the constraint
            u = self.u_vector / np.linalg.norm(self.u_vector)
            P = np.eye(self.output_dim) - np.outer(u, u)
            L_proj = P @ L_pca
 
            # Keep only the free rows (1:)
            self.Lq_params[q] = L_proj[1:, :].flatten()
