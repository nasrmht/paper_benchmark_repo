import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from .Kernel import Kernel


class LMCKernelUnit:
    """
    Implementation of the Linear Coregionalization Model (LMC) kernel.
    
    The kernel is defined as a sum of Kronecker products:
    K(x, x') = sum_{q=1}^Q B_q ⊗ k_q(x, x')
    
    where B_q = sigma_B_q * B_unit_q with B_unit_q having coefficients between -1 and 1,
    and k_q is a spatial kernel.
    
    The first element of each L_q matrix (L_q[0,0]) is set as:
    L_q[0,0] = 1 - sum of the absolute values of the other elements of L_q
    """
    
    def __init__(self, base_kernels: List[Kernel], output_dim: int, latent_dim: Optional[List[int]] = None, seed : int = 43):
        """
        Initializes the LMC kernel.
        
        Args:
            base_kernels: List of base kernels k_q(x, x')
            output_dim: Number of outputs (dimension D)
            latent_dim: List of latent dimensions of each matrix L_q associated with a base kernel. Should match the size of base kernels list
                 If None, all matrices will be of full latent_dim
        """
        self.base_kernels = base_kernels
        self.output_dim = output_dim
        self.start_idx_Lq = 0
        
        # If latent_dim is not specified, use full latent_dim (=output_dim)
        if latent_dim is None:
            self.latent_dim = [output_dim] * len(base_kernels)
        else:
            if len(latent_dim) != len(base_kernels):
                raise ValueError("The number of latent dimensions must match the number of base kernels")
            self.latent_dim = latent_dim
        
        # Initialize the Lq_unit matrices (without the first element which will be computed)
        self.Lq_unit_params = []
        
        # Initialize the scale factors sigma_B for each B_q matrix
        self.sigma_B_params = np.ones(len(self.latent_dim)) #* 0.5  # Initialization at 0.5
        
        self._bounds = []
        np.random.seed(seed)
        start_idx = 0
        for q, r in enumerate(self.latent_dim):
            # For each L_q matrix of size output_dim x r, 
            # we parameterize all elements except the first L_q[0,0]
            n_params = output_dim * r - 1  # -1 to exclude L_q[0,0]
            
            # Initialize the parameters of L_q (all except the first element)
            # with values between -0.5 and 0.5
            Lq_unit_vec = np.random.uniform(-1, 1, n_params)
            self.Lq_unit_params.append(Lq_unit_vec)
            
            # Add bounds for each element of Lq_unit (between -1 and 1)
            self._bounds.extend([(-1.0, 1.0)] * n_params)
            start_idx += n_params
        self.start_idx_Lq = start_idx
        
        
        # Add bounds for the scale factors sigma_B (positive)
        self._bounds.extend([(1.0, 10.0)] * len(self.latent_dim))
        
        # Add bounds for the base kernel parameters
        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)
    
    @property
    def params(self) -> np.ndarray:
        """Returns all hyperparameters of the LMC kernel."""
        # Concatenate parameters of Lq_unit (without L_q[0,0]), sigma_B and base kernel parameters
        params = np.concatenate(self.Lq_unit_params)
        params = np.concatenate([params, self.sigma_B_params])
        for kernel in self.base_kernels:
            params = np.concatenate([params, kernel.params])
        return params
    
    @params.setter
    def params(self, params: np.ndarray):
        """Sets all hyperparameters of the LMC kernel."""
        # Extract parameters of Lq_unit matrices (without L_q[0,0])
        start_idx = 0
        for q, r in enumerate(self.latent_dim):
            n_params_Lq = self.output_dim * r - 1  # -1 to exclude L_q[0,0]
            self.Lq_unit_params[q] = params[start_idx:start_idx + n_params_Lq]
            start_idx += n_params_Lq
        
        # Extract sigma_B parameters
        self.sigma_B_params = params[start_idx:start_idx + len(self.latent_dim)]
        start_idx += len(self.latent_dim)
        
        # Extract base kernel parameters
        for kernel in self.base_kernels:
            n_params_kernel = kernel.get_n_params()
            kernel.params = params[start_idx:start_idx + n_params_kernel]
            start_idx += n_params_kernel
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Returns the bounds of all hyperparameters."""
        return self._bounds
    
    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """
        Reconstructs the complete Lq matrix by computing the first element L_q[0,0]
        
        Args:
            q: Kernel index
            
        Returns:
            Reconstructed Lq matrix of shape (output_dim, latent_dim[q])
        """
        r = self.latent_dim[q]
        #n_params = self.output_dim * r - 1  # Number of parameters without L_q[0,0]
        
        # Create a full matrix for Lq
        Lq = np.zeros((self.output_dim, r))
        
        # Fill the matrix with the parameters (except the first element)
        param_idx = 0
        for i in range(self.output_dim):
            for j in range(r):
                if i == 0 and j == 0:
                    continue  # Skip the first element, it will be computed later
                Lq[i, j] = self.Lq_unit_params[q][param_idx]
                param_idx += 1
        
        # Compute the first element L_q[0,0] = 1 - sum of the absolute values of the others
        Lq[0, 0] = 1.0 - np.sum(np.abs(Lq.flatten()[1:]))
        
        return Lq
    
    def get_B(self, q: int) -> np.ndarray:
        """
        Returns the coregionalization matrix B_q = sigma_B_q * (Lq_unit @ Lq_unit.T).
        
        Args:
            q: Kernel index
            
        Returns:
            B_q matrix of shape (output_dim, output_dim)
        """
        Lq = self._reconstruct_Lq(q)
        B_unit = Lq @ Lq.T
        return self.sigma_B_params[q] * B_unit
    
    def get_L(self, q: int) -> np.ndarray:
        """
        Returns the complete Lq matrix (including the computed first element).
        
        Args:
            q: Kernel index
            
        Returns:
            Lq matrix of shape (output_dim, latent_dim[q])
        """
        return self._reconstruct_Lq(q)
    
    def get_sigma_B(self, q: int) -> float:
        """
        Returns the scale factor sigma_B for the q-th kernel.
        
        Args:
            q: Kernel index
            
        Returns:
            Scale factor sigma_B_q
        """
        return self.sigma_B_params[q]
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Computes the complete covariance matrix K((x1,d1), (x2,d2)) for all inputs and outputs.
        
        Args:
            X1: Matrix of shape (n1 * output_dim, input_dim + 1)
                The last column indicates the output index (0 to output_dim-1)
            X2: Matrix of shape (n2 * output_dim, input_dim + 1), if None, X2 = X1
            
        Returns:
            Covariance matrix of shape (n1 * output_dim, n2 * output_dim)
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 
        

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        # Initialize the covariance matrix
        K = np.zeros((n1, n2))
        
        # Construct the covariance matrix by blocks
        for q in range(len(self.base_kernels)):
            # Compute the spatial covariance matrix k_q(x1, x2)
            K_spatial = self.base_kernels[q](x1, x2)
            #K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            
            # Get the coregionalization matrix B_q
            B_q = self.get_B(q)
            
            # For each pair of output indices, add the contribution to K
            K += np.kron(B_q, K_spatial) 
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Computes the gradients of the covariance matrix with respect to all hyperparameters.
        
        Args:
            X1: Matrix of shape (n1 * output_dim, input_dim + 1)
            X2: Matrix of shape (n2 * output_dim, input_dim + 1), if None, X2 = X1
            
        Returns:
            List of matrices, each of shape (n1 * output_dim, n2 * output_dim)
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        gradients = np.zeros((self.get_n_params(), n1, n2))
        
        param_idx = 0
        
        # Gradients with respect to the elements of Lq_unit (except L_q[0,0])
        for q in range(len(self.base_kernels)):
            # Compute the spatial covariance matrix k_q(x1, x2)
            #K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            K_spatial = self.base_kernels[q](x1, x2)
            
            # Get the complete Lq matrix
            Lq = self._reconstruct_Lq(q)
            sigma_B = self.sigma_B_params[q]
            
            # Compute the gradients with respect to the elements of Lq
            for d in range(self.output_dim):
                for jj in range(self.latent_dim[q]):
                    if d == 0 and jj == 0:
                        continue  # Skip L_q[0,0] which is not a free parameter
                    
                    # Get dB_q with respect to L_q[d,jj]
                    dB_q = self._compute_dB_q_dL(q, d, jj, Lq, sigma_B)
                    
                    # Apply Kronecker product to get dK
                    gradients[param_idx] =np.kron(dB_q,K_spatial) 
                    param_idx += 1
        
        # Gradients with respect to the scale factors sigma_B
        for q in range(len(self.base_kernels)):
            # Compute the spatial covariance matrix k_q(x1, x2)
            K_spatial = self.base_kernels[q](x1, x2)
            # K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            
            # Get dB_q with respect to sigma_B
            dB_q = self._compute_dB_q_dsigma(q)
            
            # Apply Kronecker product to get dK
            gradients[param_idx] = np.kron(dB_q,K_spatial) 
            param_idx += 1
        
        # Gradients with respect to the hyperparameters of the base kernels
        for q, kernel in enumerate(self.base_kernels):
            # Compute the gradients of the spatial kernel
            dK_spatial_list = kernel.gradient(x1, x2)
            
            # Get the coregionalization matrix B_q
            B_q = self.get_B(q)
            
            # For each hyperparameter of the kernel, compute the full gradient
            for dK_spatial in dK_spatial_list:
                # Apply Kronecker product to get dK
                gradients[param_idx] = np.kron(B_q, dK_spatial) 
                param_idx += 1
        
        return gradients

    def _compute_dB_q_dL(self, q: int, d: int, jj: int, Lq: np.ndarray, sigma_B: float) -> np.ndarray:
        """
        Computes the derivative of B_q with respect to the element L_q[d,jj]
        
        Args:
            q: Index of the base kernel
            d: Row index in the Lq matrix
            jj: Column index in the Lq matrix
            Lq: Reconstructed Lq matrix
            sigma_B: Scale factor for B_q
            
        Returns:
            dB_q matrix of shape (output_dim, output_dim)
        """
        # Compute the derivative of Lq with respect to the current parameter
        dLq = np.zeros_like(Lq)
        dLq[d, jj] = 1.0
        
        # The derivative of the first element L_q[0,0] with respect to the parameter L_q[d,jj]
        # is -sign(L_q[d,jj]) since L_q[0,0] = 1 - sum(abs(other elements))
        dLq[0, 0] = -np.sign(Lq[d, jj])
        
        # Compute the derivative of B_unit with respect to the parameter
        dB_unit = dLq @ Lq.T + Lq @ dLq.T
        
        # Apply the scale factor sigma_B
        dB_q = sigma_B * dB_unit
        
        return dB_q
    
    def _compute_dB(self):
        """
        Computes the derivative of B with respect to all parameters of B (useful in ICM)
        
        Args:
            
            
        Returns:
            List of dB_q matrices of shape (output_dim, output_dim)
        """
        dB = []
        
        for q in range(len(self.base_kernels)):
            # Compute the spatial covariance matrix k_q(x1, x2)
            # Get the complete L_q matrix and sigma_B
            L_q = self._reconstruct_Lq(q)
            sigma_B = self.sigma_B_params[q]
            
            # Loop over all parameters (rows 2 to output_dim-1)
            for i in range(self.output_dim):
                for j in range(self.latent_dim[q]):
                    if i == 0 and j == 0:
                        continue
                    # Compute dB_q with respect to L_q[i,j]
                    dB_q = self._compute_dB_q_dL(q, i, j, L_q, sigma_B)
                    dB.append(dB_q)
    
        # The derivative of B with respect to sigma_B is simply B_unit = L_q @ L_q.T
        for q in range(len(self.base_kernels)):
            dB_q = self._compute_dB_q_dsigma(q)
            dB.append(dB_q)
        
        return np.array(dB)

    def _compute_dB_q_dsigma(self, q: int) -> np.ndarray:
        """
        Computes the derivative of B_q with respect to sigma_B
        
        Args:
            q: Index of the base kernel
            
        Returns:
            dB_q matrix of shape (output_dim, output_dim)
        """
        # Get the complete Lq matrix
        Lq = self._reconstruct_Lq(q)
        
        # The derivative of B_q with respect to sigma_B is simply B_unit = Lq @ Lq.T
        B_unit = Lq @ Lq.T
        
        return B_unit
    
    def get_n_params(self) -> int:
        """Returns the total number of hyperparameters."""
        return len(self.params)
    
    def init_L_from_pca(self, Y: np.ndarray):
        """
        Initializes L_q from the PCA of the data Y (n x p),
        respecting the constraint u^T L = 0
        """
        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

        for q, r in enumerate(self.latent_dim):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])

            # Keep only the free rows (1:)
            mask = np.ones(L_pca.shape, dtype=bool)
            mask[0, 0] = False

            vec = L_pca[mask]
            self.Lq_unit_params[q] = vec.flatten() #[1:, :].
