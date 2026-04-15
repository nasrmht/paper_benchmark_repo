import numpy as np
from scipy import linalg
from ...utils.data_utils import compute_kernel_eigendecomposition
import copy


def compute_log_likelihood_efficient(kernel_, params, X: np.ndarray, y: np.ndarray, log_noise_variance: float, eigendecomp_computed):
    n = X.shape[0] // kernel_.output_dim
    kernel = copy.deepcopy(kernel_)
    kernel.params = params[:-1]
    d = kernel.output_dim
    
    # if not eigendecomp_computed:
    #     eigendecomp_computed = compute_kernel_eigendecomposition(kernel, X)
    
    # if eigendecomp_computed is None :
    #     raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood grad is not possible")
    
    U_C, S_C, U_R, S_R = eigendecomp_computed #eigendecomp['U_C'], eigendecomp['S_C'], eigendecomp['U_R'], eigendecomp['S_R']
    noise_variance = np.exp(params[-1])
    
    Y = y.reshape(n, d, order='F')
    Y_rotated = U_R.T @ Y @ U_C
    
    try:
        S_kron = np.kron(S_C, S_R) + noise_variance
        log_det = np.sum(np.log(S_kron))
        S_kron_inv = 1.0 / S_kron
        
        Y_tilde = Y_rotated.flatten(order='F') * S_kron_inv
        quad_form = np.dot(Y_rotated.flatten(order='F'), Y_tilde)
        
        log_likelihood = -0.5 * quad_form - 0.5 * log_det - 0.5 * n * d * np.log(2 * np.pi)
        
        return -log_likelihood, S_kron_inv, Y_tilde
    except np.linalg.LinAlgError:
        return 1e10, None, None


def compute_log_likelihood_gradient_efficient(kernel_, params, X: np.ndarray, y: np.ndarray, log_noise_variance: float, eigendecomp_computed):
    n = X.shape[0] // kernel_.output_dim
    kernel = copy.deepcopy(kernel_)
    kernel.params = params[:-1]
    d = kernel.output_dim
    
    # if not eigendecomp_computed:
    #     eigendecomp_computed = compute_kernel_eigendecomposition(kernel, X)
    
    # if eigendecomp_computed is None :
    #     raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood grad is not possible")
    
    U_C, S_C, U_R, S_R = eigendecomp_computed #eigendecomp['U_C'], eigendecomp['S_C'], eigendecomp['U_R'], eigendecomp['S_R']
    noise_variance = np.exp(params[-1])
    
    Y = y.reshape(n, d, order='F')
    Y_rotated = U_R.T @ Y @ U_C
    
    try:
        S_kron = np.kron(S_C, S_R) + noise_variance
        S_kron_inv = 1.0 / S_kron
        Y_tilde = Y_rotated.flatten(order='F') * S_kron_inv
        Yt_reshaped = Y_tilde.reshape(n, d, order='F')
        S_k_i_reshaped = S_kron_inv.reshape(n, d, order='F')
        
        temp1 = U_R @ Yt_reshaped
        temp2 = np.dot(S_k_i_reshaped, S_C)
        dL_dK_x = (0.5 * temp1 * S_C) @ temp1.T - 0.5 * (U_R * temp2) @ U_R.T
        
        temp11 = U_C @ Yt_reshaped.T
        temp22 = np.dot(S_k_i_reshaped.T, S_R)
        dL_dC = 0.5 * (temp11 * S_R) @ temp11.T - 0.5 * (U_C * temp22) @ U_C.T
        
        grad = np.zeros(len(kernel.params) + 1)
        dK_dtheta_x = [grad_x for bk in kernel.base_kernels for grad_x in bk.gradient(X[:n, :-1])]
        dB_dll = kernel._compute_dB()
        
        for i, dK in enumerate(dB_dll):
            grad[i] = np.sum(dL_dC * dK)
        
        for i, dK in enumerate(dK_dtheta_x, start=len(dB_dll)):
            grad[i] = np.sum(dL_dK_x * dK)
        
        dL_dsigma2 = -0.5 * S_kron_inv.sum() * noise_variance + 0.5 * np.sum(np.square(Y_tilde)) * noise_variance
        grad[-1] = dL_dsigma2
        
        return -grad
    except np.linalg.LinAlgError:
        return np.ones(len(kernel.params) + 1) * 1e10