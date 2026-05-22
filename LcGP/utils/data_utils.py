import numpy as np
from typing import Optional, Tuple
from scipy.linalg import cholesky
from numpy.linalg import svd

def prepare_data(X: np.ndarray, y: Optional[np.ndarray], output_dim: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prepares data for fitting or prediction.

    Args:
        X: Input matrix (n, input_dim)
        y: Output matrix (n, output_dim), optional
        output_dim: Number of model outputs

    Returns:
        X_stacked, y_stacked: Transformed data
    """
    n, input_dim = X.shape
    X_stacked = np.zeros((n * output_dim, input_dim + 1))

    for d in range(output_dim):
        X_stacked[d*n:(d+1)*n, :input_dim] = X
        X_stacked[d*n:(d+1)*n, input_dim] = d
    
    y_stacked = y.flatten(order='F') if y is not None else None
    return X_stacked, y_stacked

def eig_via_cholesky_svd(A):
    # Lower Cholesky
    L = cholesky(A, lower=True)
    # SVD
    U, S_diag, Vt = svd(L)
    # Sort eigenvalues and eigenvectors in ascending order
    idx_sort = np.argsort(S_diag**2)  # eigenvalues = S^2
    eigvals = S_diag[idx_sort] ** 2
    eigvecs = U[:, idx_sort]
    return eigvals, eigvecs

def compute_kernel_eigendecomposition(kernel, X: np.ndarray):
    """
    Computes the eigendecomposition of the covariance matrices to accelerate computations.
    
    Args:
        kernel: Kernel object
        X: Transformed input matrix (n * output_dim, input_dim + 1)

    Returns:
        U_C, S_C, U_R, S_R: Eigenvectors and eigenvalues matrices
    """
    n = X.shape[0] // kernel.output_dim

    if len(kernel.base_kernels) > 1:
        print("Warning: decomposition only implemented for Q=1")
        return None, None, None, None

    B = kernel.get_B(0)
    X_spatial = X[:n, :-1]
    K_spatial = kernel.base_kernels[0](X_spatial)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_K, eigvecs_K = np.linalg.eigh(K_spatial)

    return eigvecs_B, eigvals_B, eigvecs_K, eigvals_K