import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass

class Kernel:
    """Base class for covariance kernels."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self._params = np.array([])
        self._bounds = []
        
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Computes the covariance matrix between X1 and X2."""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """Computes the gradient of the covariance matrix with respect to the hyperparameters."""
        raise NotImplementedError("Subclasses must implement gradient")
    
    @property
    def params(self) -> np.ndarray:
        """Returns the current hyperparameters."""
        return self._params
    
    @params.setter
    def params(self, params: np.ndarray):
        """Sets the hyperparameters."""
        if params.shape != self._params.shape:
            raise ValueError(f"Shapes do not match: {params.shape} vs {self._params.shape}")
        self._params = params
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Returns the bounds of the hyperparameters for optimization."""
        return self._bounds
    
    def get_n_params(self) -> int:
        """Returns the number of hyperparameters."""
        return len(self._params)


class RBFKernel(Kernel):
    """RBF (Radial Basis Function) or Gaussian kernel, with support for anisotropy."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initializes the RBF kernel.
        
        Args:
            input_dim: Input dimension
            lengthscale: Initial length scale. If None, initialized to 1.0
                         If scalar, used for all dimensions if ARD=False,
                         or as initial value for all dimensions if ARD=True
                         If array, must have a length equal to input_dim for ARD=True
            ARD: If True, uses a different length scale for each dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Default initialization
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 for all dimensions
        elif np.isscalar(lengthscale):
            # Single scalar provided
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Array of lengthscales provided
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"For ARD=True, lengthscale must have {input_dim} elements")
            elif not ARD and len(lengthscale) > 1:
                print(f"Warning: ARD=False but lengthscale has {len(lengthscale)} elements. Only the first one will be used.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Computes the covariance matrix K(X1, X2) for the anisotropic RBF kernel.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            Covariance matrix of shape (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # For the anisotropic case (ARD=True)
        if self.ARD:
            # Weight each dimension by its lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Compute the weighted squared Euclidean distance
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
        else:
            # Isotropic case (ARD=False)
            lengthscale = lengthscales[0]
            
            # Compute the squared Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist_sq = dist_sq / (lengthscale**2)
        
        # Apply the RBF kernel
        K = variance * np.exp(-0.5 * dist_sq)
        return K
    
        
    #     return K_gradient
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Computes the derivatives of the covariance matrix with respect to the hyperparameters.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            List of matrices, each of shape (n1, n2) corresponding to the derivatives
            with respect to each hyperparameter
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Compute the covariance matrix
        K = self.__call__(X1, X2)
        
        gradients = []
        
        if self.ARD:
            # Anisotropic case: one gradient per dimension
            for d in range(self.input_dim):
                # Compute the squared difference for this dimension
                diff_d = X1[:, d:d+1] - X2[:, d:d+1].T
                sq_diff_d = diff_d**2
                
                # Gradient with respect to log(lengthscale) for this dimension
                dK_dlog_lengthscale_d = K * (sq_diff_d / (lengthscales[d]**2))
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Isotropic case: a single gradient
            lengthscale = lengthscales[0]
            
            # Compute the squared Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            
            # Gradient with respect to log(lengthscale)
            dK_dlog_lengthscale = K * (dist_sq / (lengthscale**2))
            gradients.append(dK_dlog_lengthscale)
        
        return gradients


class Matern52Kernel(Kernel):
    """Matern 5/2 kernel with anisotropy support."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initializes the Matern 5/2 kernel.
        
        Args:
            input_dim: Input dimension
            lengthscale: Initial length scale. If None, initialized to 1.0
                         If scalar, used for all dimensions if ARD=False,
                         or as initial value for all dimensions if ARD=True
                         If array, must have a length equal to input_dim for ARD=True
            ARD: If True, uses a different length scale for each dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Default initialization
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 for all dimensions
        elif np.isscalar(lengthscale):
            # Single scalar provided
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Array of lengthscales provided
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"For ARD=True, lengthscale must have {input_dim} elements")
            elif not ARD and len(lengthscale) > 1:
                print(f"Warning: ARD=False but lengthscale has {len(lengthscale)} elements. Only the first one will be used.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Computes the covariance matrix K(X1, X2) for the anisotropic Matern 5/2 kernel.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            Covariance matrix of shape (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # For the anisotropic case (ARD=True)
        if self.ARD:
            # Weight each dimension by its lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Compute the weighted squared Euclidean distance
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 5/2 kernel
            scaled_dist = np.sqrt(5) * dist
        else:
            # Isotropic case (ARD=False)
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 5/2 kernel
            scaled_dist = np.sqrt(5) * dist / lengthscale
        
        K = variance * (1.0 + scaled_dist + scaled_dist**2/3.0) * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Computes the derivatives of the covariance matrix with respect to the hyperparameters.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            List of matrices, each of shape (n1, n2) corresponding to the derivatives
            with respect to each hyperparameter
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Anisotropic case: one gradient per dimension
            for d in range(self.input_dim):
                # Create copies of X1 and X2 matrices for this dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Zero out all dimensions except the one of interest
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Compute the Euclidean distance for this dimension only
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Compute the global Euclidean distance (all dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                 
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Compute the Matern 5/2 kernel
                scaled_dist = np.sqrt(5) * dist
                base = (1.0 + scaled_dist + scaled_dist**2/3.0) * np.exp(-scaled_dist)
                
                # Compute the gradient contribution for this dimension
                # For Matern 5/2, the gradient is more complex
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        (5/3) * dist_d_sq * (1 + scaled_dist),
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Isotropic case: a single gradient
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 5/2 kernel
            scaled_dist = np.sqrt(5) * dist / lengthscale
            
            # Gradient with respect to log(lengthscale)
            dK_dlog_lengthscale = variance * np.exp(-scaled_dist) * (
                scaled_dist**2 * (scaled_dist + 1) / 3.0
            )
            gradients.append(dK_dlog_lengthscale)
        
        return gradients
    




class Matern32Kernel(Kernel):
    """Matern 3/2 kernel with anisotropy support."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initializes the Matern 3/2 kernel.
        
        Args:
            input_dim: Input dimension
            lengthscale: Initial length scale. If None, initialized to 1.0
                         If scalar, used for all dimensions if ARD=False,
                         or as initial value for all dimensions if ARD=True
                         If array, must have a length equal to input_dim for ARD=True
            ARD: If True, uses a different length scale for each dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Default initialization
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 for all dimensions
        elif np.isscalar(lengthscale):
            # Single scalar provided
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Array of lengthscales provided
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"For ARD=True, lengthscale must have {input_dim} elements")
            elif not ARD and len(lengthscale) > 1:
                print(f"Warning: ARD=False but lengthscale has {len(lengthscale)} elements. Only the first one will be used.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Computes the covariance matrix K(X1, X2) for the anisotropic Matern 3/2 kernel.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            Covariance matrix of shape (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # For the anisotropic case (ARD=True)
        if self.ARD:
            # Weight each dimension by its lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Compute the weighted squared Euclidean distance
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 3/2 kernel
            scaled_dist = np.sqrt(3) * dist
        else:
            # Isotropic case (ARD=False)
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 3/2 kernel
            scaled_dist = np.sqrt(3) * dist / lengthscale
        
        K = variance * (1.0 + scaled_dist) * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Computes the derivatives of the covariance matrix with respect to the hyperparameters.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            List of matrices, each of shape (n1, n2) corresponding to the derivatives
            with respect to each hyperparameter
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Anisotropic case: one gradient per dimension
            for d in range(self.input_dim):
                # Create copies of X1 and X2 matrices for this dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Zero out all dimensions except the one of interest
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Compute the Euclidean distance for this dimension only
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Compute the global Euclidean distance (all dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Compute the Matern 3/2 kernel
                scaled_dist = np.sqrt(3) * dist
                base = (1.0 + scaled_dist) * np.exp(-scaled_dist)
                
                # Compute the gradient contribution for this dimension
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        3.0 * dist_d_sq,
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Isotropic case: a single gradient
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 3/2 kernel
            scaled_dist = np.sqrt(3) * dist / lengthscale
            
            # Gradient with respect to log(lengthscale)
            dK_dlog_lengthscale = variance * scaled_dist**2 * np.exp(-scaled_dist)
            gradients.append(dK_dlog_lengthscale)
        
        return gradients


class Matern12Kernel(Kernel):
    """Matern 1/2 kernel with anisotropy support."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initializes the Matern 1/2 kernel.
        
        Args:
            input_dim: Input dimension
            lengthscale: Initial length scale. If None, initialized to 1.0
                         If scalar, used for all dimensions if ARD=False,
                         or as initial value for all dimensions if ARD=True
                         If array, must have a length equal to input_dim for ARD=True
            ARD: If True, uses a different length scale for each dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Default initialization
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 for all dimensions
        elif np.isscalar(lengthscale):
            # Single scalar provided
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Array of lengthscales provided
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"For ARD=True, lengthscale must have {input_dim} elements")
            elif not ARD and len(lengthscale) > 1:
                print(f"Warning: ARD=False but lengthscale has {len(lengthscale)} elements. Only the first one will be used.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Computes the covariance matrix K(X1, X2) for the anisotropic Matern 1/2 kernel.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            Covariance matrix of shape (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # For the anisotropic case (ARD=True)
        if self.ARD:
            # Weight each dimension by its lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Compute the weighted squared Euclidean distance
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 1/2 kernel (exponential)
            scaled_dist = dist  # For Matern 1/2, we directly use the distance
        else:
            # Isotropic case (ARD=False)
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 1/2 kernel (exponential)
            scaled_dist = dist / lengthscale
        
        K = variance * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Computes the derivatives of the covariance matrix with respect to the hyperparameters.
        
        Args:
            X1: Matrix of shape (n1, input_dim)
            X2: Matrix of shape (n2, input_dim), if None, X2 = X1
            
        Returns:
            List of matrices, each of shape (n1, n2) corresponding to the derivatives
            with respect to each hyperparameter
        """
        if X2 is None:
            X2 = X1
            
        # Extract parameters
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Anisotropic case: one gradient per dimension
            for d in range(self.input_dim):
                # Create copies of X1 and X2 matrices for this dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Zero out all dimensions except the one of interest
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Compute the Euclidean distance for this dimension only
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Compute the global Euclidean distance (all dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Compute the Matern 1/2 kernel (exponential)
                scaled_dist = dist
                
                # Compute the gradient contribution for this dimension
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        dist_d_sq / dist,
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Isotropic case: a single gradient
            lengthscale = lengthscales[0]
            
            # Compute the Euclidean distance
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Compute the Matern 1/2 kernel (exponential)
            scaled_dist = dist / lengthscale
            
            # Gradient with respect to log(lengthscale)
            dK_dlog_lengthscale = variance * scaled_dist * np.exp(-scaled_dist)
            gradients.append(dK_dlog_lengthscale)
        
        return gradients