import numpy as np
from scipy import linalg
import copy
def compute_log_likelihood_naive(kernel_,params,X: np.ndarray, y: np.ndarray, log_noise_variance: float) -> float:
    """
        Calcule la log-vraisemblance en utilisant la méthode naïve.
        
        Args:
            X: Matrice d'entrée préparée de forme (n * output_dim, input_dim + 1)
            y: Vecteur de sortie préparé de forme (n * output_dim,)
            
        Returns:
            La log-vraisemblance négative (pour la minimisation)
    """
    n_total = y.shape[0]
    kernel = copy.deepcopy(kernel_)
    kernel.params = params[:-1]
    noise_variance = np.exp(params[-1])
    K  = kernel(X)
    K_noisy = K + noise_variance * np.eye(n_total)

    try:
        L = linalg.cholesky(K_noisy, lower=True)
        alpha = linalg.solve_triangular(L, y, lower=True)
        alpha = linalg.solve_triangular(L.T, alpha, lower=False)
        log_likelihood = -0.5 * np.dot(y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n_total * np.log(2 * np.pi)
        return -log_likelihood, L, alpha
    except np.linalg.LinAlgError:
        return 1e10, L, alpha


def compute_log_likelihood_gradient_naive(kernel_, params, X: np.ndarray, y: np.ndarray, log_noise_variance: float):
    n_total = X.shape[0]
    kernel = copy.deepcopy(kernel_)
    kernel.params = params[:-1]
    K = kernel(X)
    noise_variance = np.exp(params[-1])
    K_noisy = K + noise_variance * np.eye(n_total)
    
    try:
        L = linalg.cholesky(K_noisy, lower=True)
        alpha = linalg.solve_triangular(L, y, lower=True)
        alpha = linalg.solve_triangular(L.T, alpha, lower=False)
        
        K_inv = linalg.cho_solve((L, True), np.eye(n_total))
        dL_dK = 0.5 * (np.outer(alpha, alpha) - K_inv)
        
        dK_dtheta = kernel.gradient(X)
        grad = np.zeros(len(kernel.params) + 1)
        
        for i, dK in enumerate(dK_dtheta):
            grad[i] = np.sum(dL_dK * dK)
        
        dK_dnoise = np.eye(n_total) * noise_variance
        grad[-1] = np.sum(dL_dK * dK_dnoise)
        
        return -grad
    except np.linalg.LinAlgError:
        return np.ones(len(kernel.params) + 1) * 1e10