import numpy as np
from typing import Optional, Tuple
from scipy.linalg import cholesky
from numpy.linalg import svd

def prepare_data(X: np.ndarray, y: Optional[np.ndarray], output_dim: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prépare les données pour l'ajustement ou la prédiction.

    Args:
        X: Matrice d'entrée (n, input_dim)
        y: Matrice de sortie (n, output_dim), facultatif
        output_dim: Nombre de sorties du modèle

    Returns:
        X_stacked, y_stacked: Données transformées
    """
    n, input_dim = X.shape
    X_stacked = np.zeros((n * output_dim, input_dim + 1))

    for d in range(output_dim):
        X_stacked[d*n:(d+1)*n, :input_dim] = X
        X_stacked[d*n:(d+1)*n, input_dim] = d
    
    y_stacked = y.flatten(order='F') if y is not None else None
    return X_stacked, y_stacked

def eig_via_cholesky_svd(A):
    # Cholesky (inférieure)
    L = cholesky(A, lower=True)
    # SVD
    U, S_diag, Vt = svd(L)
    # Ordonner les valeurs propres et vecteurs propres par ordre **croissant**
    idx_sort = np.argsort(S_diag**2)  # valeurs propres = S^2
    eigvals = S_diag[idx_sort] ** 2
    eigvecs = U[:, idx_sort]
    return eigvals, eigvecs

def compute_kernel_eigendecomposition(kernel, X: np.ndarray):
    """
    Calcule la décomposition en valeurs propres des matrices de covariance pour accélérer les calculs.
    
    Args:
        kernel: Objet du noyau
        X: Matrice d'entrée transformée (n * output_dim, input_dim + 1)

    Returns:
        U_C, S_C, U_R, S_R: Matrices propres et valeurs propres
    """
    n = X.shape[0] // kernel.output_dim

    if len(kernel.base_kernels) > 1:
        print("Avertissement: décomposition uniquement implémentée pour Q=1")
        return None, None, None, None

    B = kernel.get_B(0)
    X_spatial = X[:n, :-1]
    K_spatial = kernel.base_kernels[0](X_spatial)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_K, eigvecs_K = np.linalg.eigh(K_spatial)

    return eigvecs_B, eigvals_B, eigvecs_K, eigvals_K