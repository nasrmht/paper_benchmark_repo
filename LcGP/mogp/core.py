import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from scipy.stats.qmc import LatinHypercube 
from joblib import Parallel, delayed
import time
from .kernels.LMCKernel import LMCKernel
from ..utils.data_utils import compute_kernel_eigendecomposition, prepare_data
from .likelihoods.log_likelihood_efficient import compute_log_likelihood_efficient, compute_log_likelihood_gradient_efficient
from .likelihoods.log_likelihood_naive import compute_log_likelihood_naive, compute_log_likelihood_gradient_naive


class MOGPR:
    """
    Multi-Output Gaussian Process Regression (MOGPR) model using
    the Linear Model of Coregionalization (LMC).
    """
    
    def __init__(self, kernel: LMCKernel, noise_variance: float = 1e-6, use_efficient_lik: bool = True,
                 parallel: bool = True, n_jobs: int = -1, verbose: int = 0):
        """
        Initializes the MOGPR model.
        
        Args:
            kernel: LMC kernel
            noise_variance: Observation noise variance (sigma²)
            use_efficient_lik: Whether to use the efficient method to compute log-likelihood
                             (only for simple kernels with Q=1)
            parallel: Enable parallelization for sampling and other operations
            n_jobs: Number of jobs for parallelization (-1 = all cores)
            verbose: Verbosity level for information display
        """
        self.kernel = kernel
        self.log_noise_variance = np.log(noise_variance)
        self.use_efficient_lik = use_efficient_lik
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
        # Parallelization configuration
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # For efficient storage after fitting
        self.L = None  # Cholesky factor of the covariance matrix
        self.alpha = None  # Solution of the linear system
        self.eigendecomp_computed = False
        self.U_C = None  # Eigenvectors of the coregionalization matrix
        self.S_C = None  # Eigenvalues of the coregionalization matrix
        self.U_R = None  # Eigenvectors of the spatial covariance matrix
        self.S_R = None  # Eigenvalues of the spatial covariance matrix
        self.S_kron_inv = None
        self.Ytilde = None
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return prepare_data(X, y, self.kernel.output_dim)
    
    
    def _compute_kernel_eigendecomposition(self, X: np.ndarray):
        res = compute_kernel_eigendecomposition(self.kernel, X)
        self.U_C, self.S_C, self.U_R, self.S_R = res
        self.eigendecomp_computed = True
        return res
    
    
    def _compute_log_likelihood_naive(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        nll, L, alpha =  compute_log_likelihood_naive(self.kernel, params, X, y, self.log_noise_variance)
        self.L = L
        self.alpha = alpha
        return nll
    
    
    def _compute_log_likelihood_gradient_naive(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        return compute_log_likelihood_gradient_naive(self.kernel, params, X, y, self.log_noise_variance)
    
    
    def _compute_log_likelihood_efficient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        eigendecomp_computed = None
        if not self.eigendecomp_computed:
            eigendecomp_computed = self._compute_kernel_eigendecomposition(X)
        else : 
            eigendecomp_computed = self.U_C, self.S_C, self.U_R, self.S_R
        
        if eigendecomp_computed is None :
            raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood is not possible")
        
        nll, self.S_kron_inv, self.Ytilde = compute_log_likelihood_efficient(self.kernel, params, X, y, self.log_noise_variance, eigendecomp_computed)
        return nll

    
    def _compute_log_likelihood_gradient_efficient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        if not self.eigendecomp_computed:
            eigendecomp_computed = compute_kernel_eigendecomposition(self.kernel, X)
        else : 
            eigendecomp_computed = self.U_C, self.S_C, self.U_R, self.S_R

        if eigendecomp_computed is None :
            raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood grad is not possible")

        return compute_log_likelihood_gradient_efficient(self.kernel, params, X, y, self.log_noise_variance, eigendecomp_computed)
    
    
    def _compute_log_likelihood_function(self, params: np.ndarray) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            params: Vector of concatenated parameters [kernel_params, log_noise_variance]
            
        Returns:
            Negative log-likelihood
        """
        # Extract kernel parameters and noise variance
        kernel_params = params[:-1]
        log_noise_variance = params[-1]
        
        # Update parameters
        self.kernel.params = kernel_params
        self.log_noise_variance = log_noise_variance
        
        # Reset precomputed decompositions
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        
        # Compute log-likelihood
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            res = self._compute_log_likelihood_efficient(params, self.X_train, self.y_train)
            return res
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            res = self._compute_log_likelihood_efficient(params, self.X_train, self.y_train)
            return res
        else:
            res = self._compute_log_likelihood_naive(params, self.X_train, self.y_train)
            return res
    
    def _compute_log_likelihood_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Gradient of the objective function for hyperparameter optimization.
        
        Args:
            params: Vector of concatenated parameters [kernel_params, log_noise_variance]
            
        Returns:
            Gradient of the negative log-likelihood
        """
        # Extract kernel parameters and noise variance
        kernel_params = params[:-1]
        log_noise_variance = params[-1]
        
        # Update parameters
        self.kernel.params = kernel_params
        self.log_noise_variance = log_noise_variance
        
        # Reset precomputed decompositions
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        
        # Compute log-likelihood gradient
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            return self._compute_log_likelihood_gradient_efficient(params, self.X_train, self.y_train)
        else : 
            return self._compute_log_likelihood_gradient_naive(params, self.X_train, self.y_train)

    def _optimize_single_start(self, initial_params, bounds, optimizer, maxiter):
        """
        Optimizes hyperparameters starting from a single initial point.
        
        Args:
            initial_params: Initial parameters
            bounds: Bounds for optimization
            optimizer: Optimization method
            maxiter: Maximum number of iterations
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        
        if optimizer in ['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG'] and hasattr(self, '_compute_log_likelihood_gradient'):
            # Methods using the gradient
            result = minimize(
                self._compute_log_likelihood_function,
                initial_params,
                method=optimizer,
                jac=self._compute_log_likelihood_gradient,
                bounds=bounds if optimizer == 'L-BFGS-B' else None,
                options={'maxiter': maxiter, 'disp': self.verbose > 0}
            )
        else:
            # Methods not using the gradient
            result = minimize(
                self._compute_log_likelihood_function,
                initial_params,
                method=optimizer,
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': self.verbose > 0}
            )
        
        duration = time.time() - start_time
        
        return result, duration
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: str = 'L-BFGS-B', 
            n_restarts: int = 1, maxiter: int = 100, verbose: bool = False, use_grad: bool = True, 
            theta_lb=None, theta_ub=None, seed=42, parallel_opt=None, use_init_pca: bool = True) -> 'MOGPR':
        """
        Fits the model to the training data by optimizing hyperparameters.
        
        Args:
            X: Input matrix of shape (n, input_dim)
            y: Output matrix of shape (n, output_dim)
            optimizer: Optimization method to use ('L-BFGS-B', 'BFGS', etc.)
            n_restarts: Number of random restarts for optimization
            maxiter: Maximum number of iterations for optimization
            verbose: Print information during fitting
            use_grad: Use analytical gradient for optimization
            theta_lb, theta_ub: Lower and upper bounds for the kernel hyperparameters
            seed: Random seed for reproducibility
            parallel_opt: Enable parallelization for optimization (if None, uses self.parallel)
            use_init_pca: Use PCA initialization for the coregionalization matrix
            
        Returns:
            self: The fitted model
        """
        # Update verbosity level
        self.verbose = verbose
        
        # Determine if optimization should be parallelized
        parallel_opt = self.parallel if parallel_opt is None else parallel_opt
        
        # Prepare the data
        X_stacked, y_stacked = self._prepare_data(X, y)
        self.X_train = X_stacked
        self.y_train = y_stacked
        
        # Get bounds for all parameters
        for j in range(len(self.kernel.base_kernels)):
            len_theta = self.kernel.base_kernels[j].get_n_params()
            Theta_lb = theta_lb if theta_lb is not None else 1e-3 * np.ones(len_theta)
            
            if theta_ub is not None:
                Theta_ub = theta_ub
            else:
                if X.ndim > 1:
                    Theta_ub = 10 * (np.max(X, axis=0) - np.min(X, axis=0))
                else:
                    Theta_ub = 10 * (np.max(X) - np.min(X))
            
            custom_bounds = np.vstack([np.log(Theta_lb), np.log(Theta_ub)]).T
            self.kernel.base_kernels[j]._bounds = [tuple(sous_bounds) for sous_bounds in custom_bounds]

        bounds = self.kernel.bounds + [(-10.0, 10.0)]  # Adding bounds for log_noise_variance

        # Initialize best log-likelihood to a high value
        best_nll = np.inf
        best_params = None
        np.random.seed(seed)
        
        # Generating starting points
        initial_params = []
        
        if not use_init_pca:
            # Standard method: 1st point = current params, others = complete LHS
            initial_params.append(np.concatenate([self.kernel.params, [self.log_noise_variance]]))
            
            if n_restarts > 1:
                bounds_low = np.array([b[0] for b in self.kernel.bounds])
                bounds_high = np.array([b[1] for b in self.kernel.bounds])
                n_params = len(bounds_low)
                
                lhs = LatinHypercube(d=n_params, seed=seed)
                lhs_samples_normalized = lhs.random(n=n_restarts-1)
                initial_kernel_params = bounds_low + lhs_samples_normalized * (bounds_high - bounds_low)
                
                noise_low, noise_high = -10.0, 0.0
                lhs_noise = LatinHypercube(d=1, seed=seed)
                noise_lhs_normalized = lhs_noise.random(n=n_restarts-1)
                initial_log_noise = noise_low + noise_lhs_normalized * (noise_high - noise_low)
                
                initial_params_lhs = np.hstack([initial_kernel_params, initial_log_noise])
                for p in initial_params_lhs:
                    initial_params.append(p)
        else:
            # PCA method: Initialize L via PCA on y, randomize the rest
            if not hasattr(self.kernel, "init_L_from_pca"):
                raise ValueError("Kernel does not support PCA initialization")
            
            # Init L from PCA using original y (expected by init_L_from_pca)
            self.kernel.init_L_from_pca(y)
            
            current_kernel_params = self.kernel.params.copy()
            
            # Indices to separate L (fixed by PCA) from the others (to randomize)
            # Hypothesis: kernel.params = [Lq_unit, sigma_B, spatial]
            # start_idx_Lq marks the end of Lq_unit
            idx_spatial_start = self.kernel.start_idx_Lq
            idx_spatial_end = len(current_kernel_params)
            
            spatial_bounds = bounds[idx_spatial_start:idx_spatial_end]
            noise_bounds = [(-10.0, 0.0)]
            
            # Restart 0: PCA + current params (for sigma_B and spatial)
            p0 = np.concatenate([current_kernel_params, [self.log_noise_variance]])
            initial_params.append(p0)
            
            if n_restarts > 1:
                n_spatial = idx_spatial_end - idx_spatial_start
                # LHS on (sigma_B + spatial) + noise
                lhs = LatinHypercube(d=n_spatial + 1, seed=seed)
                samples = lhs.random(n=n_restarts - 1)
                
                lb = np.array([b[0] for b in spatial_bounds] + [noise_bounds[0][0]])
                ub = np.array([b[1] for b in spatial_bounds] + [noise_bounds[0][1]])
                
                lhs_params = lb + samples * (ub - lb)
                
                for k in range(n_restarts - 1):
                    params = p0.copy()
                    # Replace the (sigma_B + spatial) part with LHS
                    params[idx_spatial_start:idx_spatial_end] = lhs_params[k, :-1]
                    # Replace noise
                    params[-1] = lhs_params[k, -1]
                    initial_params.append(params)
        
        # Parallel or sequential optimization
        if parallel_opt and n_restarts > 1:
            if self.verbose > 0:
                print(f"Parallel execution of optimization with {n_restarts} starting points")
            
            try:
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._optimize_single_start)(params, bounds, optimizer, maxiter)
                    for params in initial_params
                )
                
                # Extract results and durations
                opt_results = [res[0] for res in results]
                durations = [res[1] for res in results]
                
                # Find the best result
                best_idx = np.argmin([res.fun for res in opt_results])
                best_result = opt_results[best_idx]
                best_params = best_result.x
                best_nll = best_result.fun
                
                if self.verbose > 0:
                    for i, (res, dur) in enumerate(zip(opt_results, durations)):
                        print(f"Run {i+1}/{n_restarts}: nLL = {res.fun:.6f}, duration = {dur:.2f}s")
                    print(f"Best run: {best_idx+1}/{n_restarts}, nLL = {best_nll:.6f}")
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error during parallelization: {str(e)}. Using sequential optimization.")
                
                # Fallback to sequential optimization
                for i, init_params in enumerate(initial_params):
                    if self.verbose > 0:
                        print(f"Optimization {i+1}/{n_restarts}")
                    
                    result, duration = self._optimize_single_start(init_params, bounds, optimizer, maxiter)
                    
                    if self.verbose > 0:
                        print(f"Run {i+1}/{n_restarts}: nLL = {result.fun:.6f}, duration = {duration:.2f}s")
                    
                    if result.fun < best_nll:
                        best_nll = result.fun
                        best_params = result.x
        else:
            # Sequential optimization
            for i, init_params in enumerate(initial_params):
                if self.verbose > 0:
                    print(f"Optimization {i+1}/{n_restarts}")
                
                result, duration = self._optimize_single_start(init_params, bounds, optimizer, maxiter)
                
                if self.verbose > 0:
                    print(f"Run {i+1}/{n_restarts}: nLL = {result.fun:.6f}, duration = {duration:.2f}s")
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
        
        # Set optimal parameters
        if best_params is not None:
            self.kernel.params = best_params[:-1]
            self.log_noise_variance = best_params[-1]
            
            if self.verbose > 0:
                print(f"Optimal hyperparameters: {best_params[:-1]}, log_noise = {best_params[-1]}")
        
        # Recalculate with the best parameters
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            self._compute_kernel_eigendecomposition(X_stacked)
            self._compute_log_likelihood_efficient(best_params, X_stacked, y_stacked)
        else:
            self._compute_log_likelihood_naive(best_params, X_stacked, y_stacked)
        
        self.is_fitted = True
        return self
    def predict(self, X_test: np.ndarray, return_cov: bool = True, full_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance/covariance for test points.
        
        Args:
            X_test: Test points, matrix of shape (n_test, input_dim)
            return_cov: If True, returns the covariance matrix
            full_cov: If True and return_cov=True, returns the full covariance matrix,
                    otherwise returns only variances
            
        Returns:
            y_pred: Predicted mean of shape (n_test, output_dim)
            var_pred: Predicted variance of shape (n_test, output_dim) if full_cov=False,
                     otherwise covariance matrix of shape (n_test * output_dim, n_test * output_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before making predictions.")
        
        # Prepare the test data
        X_test_stacked, _ = self._prepare_data(X_test)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        n = int(self.X_train.shape[0]/output_dim)
        
        if not self.use_efficient_lik:
            # Compute covariance between training and test points
            K_star = self.kernel(self.X_train, X_test_stacked)
            # Compute the predicted mean
            f_pred = K_star.T @ self.alpha
        else : 
            X_train_spatial = self.X_train[:n, :-1]
            K_x_star = self.kernel.base_kernels[0](X_test, X_train_spatial)
            A = self.kernel.get_B(0).dot(self.U_C)
            B = K_x_star.dot(self.U_R)
            f_pred = B.dot(self.Ytilde.reshape(n, output_dim, order='F')).dot(A.T).flatten(order='F')
            
        # Reshape the predicted mean
        y_pred = f_pred.reshape(output_dim, n_test).T
        
        # Compute the predicted variance
        if return_cov:
            if not self.use_efficient_lik:
                if self.L is None:
                    # Compute Cholesky decomposition if necessary
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
            
                # Solve the linear system K_noisy^(-1) @ K_star
                v = linalg.solve_triangular(self.L, K_star, lower=True)
                
                # Compute the predictive covariance
                K_test_test = self.kernel(X_test_stacked)
                
                #if full_cov:
                    # Full covariance
                var_pred = K_test_test - v.T @ v
            else:
                if self.L is None:
                    # Compute Cholesky decomposition if necessary
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
            
                # Solve the linear system K_noisy^(-1) @ K_star
                K_star = self.kernel(self.X_train, X_test_stacked)
                v = linalg.solve_triangular(self.L, K_star, lower=True)
                
                # Compute the predictive covariance
                K_test_test = self.kernel(X_test_stacked)
                
                #if full_cov:
                    # Full covariance
                var_pred = K_test_test - v.T @ v
                # else:
                #     # Diagonal only (variances)
                #     var_pred = np.diag(K_test_test) - np.diag(v.T@v)
                #     var_pred = var_pred.reshape(output_dim, n_test).T
            # else:
                # X_train_spatial = self.X_train[:n, :-1]
                # # K_x_star = self.kernel.base_kernels[0](X_test, X_train_spatial)
                # # A = self.kernel.get_B(0).dot(self.U_C)
                # # B = K_x_star.dot(self.U_R)
 
                # # k_C_xx = np.diag(self.kernel.get_B(0))
                # # k_R_xx = np.diag(self.kernel.base_kernels[0](X_test))
                
                # K_star = self.kernel(self.X_train, X_test_stacked)
                # K = self.kernel(self.X_train)
                # noise_variance = np.exp(self.log_noise_variance)
                # K_noisy = K + noise_variance * np.eye(K.shape[0])
                # self.L = linalg.cholesky(K_noisy, lower=True)
 
                # v = linalg.solve_triangular(self.L, K_star, lower=True)
                # K_xx_star = self.kernel(X_test_stacked, X_test_stacked)
                # #print("v shape kron : ", K_star.shape)
                # var_pred = K_xx_star-v.T@v
                # BA = np.kron(A, B)
                # var_pred = (np.kron(k_C_xx, k_R_xx) - np.sum(BA**2*self.S_kron_inv, 1) + np.exp(self.log_noise_variance)).reshape(output_dim, n_test).T
        else:
            # By default, return only predicted variances
            if not self.use_efficient_lik:
                if self.L is None:
                    # Compute Cholesky decomposition if necessary
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
                
                # Solve the linear system K_noisy^(-1) @ K_star
                v = linalg.solve_triangular(self.L, K_star, lower=True)
 
                # Compute predicted variances (diagonal of covariance)
                K_test_test = self.kernel(X_test_stacked, X_test_stacked)
                K_test_diag = np.diag(K_test_test)
                # for i in range(X_test_stacked.shape[0]):
                #     print("i : ", i)
                #     print("X_test_stacked[i:i+1] : ", X_test_stacked[i:i+1])
                #     K_test_diag[i] = self.kernel(X_test_stacked[i:i+1], X_test_stacked[i:i+1])[0, 0]
                
                var_pred = K_test_diag - np.diag(v.T@v)
                var_pred = var_pred.reshape(n_test, output_dim, order='F')#.T 
            else:
                X_train_spatial = self.X_train[:n, :-1]
                K_x_star = self.kernel.base_kernels[0](X_test, X_train_spatial)
                A = self.kernel.get_B(0).dot(self.U_C)
                B = K_x_star.dot(self.U_R)
 
                k_C_xx = np.diag(self.kernel.get_B(0))                    # (D,)
                k_R_xx = np.diag(self.kernel.base_kernels[0](X_test))     # (n_test,)
                # S_kron_inv layout: S_kron_inv[p*N + r] = 1/(S_C[p]*S_R[r] + σ²)
                # Reshape to (N, D) with Fortran order: S_mat[r, p] = S_kron_inv[p*N + r]
                S_mat = self.S_kron_inv.reshape(n, output_dim, order='F') # (N, D)
                # reduction[r*, p*] = Σ_{r'',p''} B[r*,r'']² · A[p*,p'']² · S_mat[r'',p'']
                reduction = (B ** 2) @ S_mat @ (A ** 2).T                 # (n_test, D)
                var_pred = (np.outer(k_R_xx, k_C_xx)
                            - reduction
                            + np.exp(self.log_noise_variance))

        return y_pred, var_pred
    
    def _generate_single_sample(self, X_test, y_mean, y_cov, random_state):
        """
        Generates a single sample from the predictive distribution.
        
        Args:
            X_test: Test points
            y_mean: Predicted mean
            y_cov: Predicted covariance
            random_state: Random seed
            
        Returns:
            A sample from the distribution
        """
        rng = np.random.RandomState(random_state)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Reshape the mean
        y_mean_flat = y_mean.T.flatten(order='F')
        
        # Generate a sample
        sample_flat = rng.multivariate_normal(y_mean_flat, y_cov)
        
        # Reshape the sample
        sample = sample_flat.reshape(output_dim, n_test, order='F').T
        
        return sample
    
    def sample_y(self, X_test: np.ndarray, n_samples: int = 10, random_state: Optional[int] = None,
                 parallel: Optional[bool] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Samples from the predictive distribution, with support for parallelization.
        
        Args:
            X_test: Test points, matrix of shape (n_test, input_dim)
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            parallel: Enable parallelization (if None, uses self.parallel)
            batch_size: Batch size for parallelization
            
        Returns:
            samples: Samples of shape (n_test, output_dim, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before sampling.")
        
        start_time = time.time()
        
        # Determine if sampling should be parallelized
        parallel = self.parallel if parallel is None else parallel
        
        # Predict mean and covariance
        y_mean, y_cov = self.predict(X_test, return_cov=True, full_cov=True)
        print("y_cov shape : ", y_cov.shape)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Configure the main random generator
        if random_state is not None:
            main_rng = np.random.RandomState(random_state)
            # Generate seeds for each sample
            seeds = main_rng.randint(0, 2**32 - 1, size=n_samples)
        else:
            seeds = [None] * n_samples
        
        # Determine batch size if necessary
        if parallel and batch_size is None:
            # Determine batch size based on the number of samples and cores
            if self.n_jobs <= 0:
                import multiprocessing
                n_cores = multiprocessing.cpu_count()
            else:
                n_cores = min(self.n_jobs, n_samples)
                
            batch_size = max(1, n_samples // n_cores)
        
        if self.verbose > 0:
            print(f"Generating {n_samples} samples for {n_test} test points...")
        
        if parallel and n_samples > 1:
            try:
                # Parallel generation of samples
                if self.verbose > 0:
                    print(f"Parallel sampling with {self.n_jobs} jobs")
                
                # Use joblib to parallelize
                samples_list = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._generate_single_sample)(X_test, y_mean, y_cov, seed)
                    for seed in seeds
                )
                
                # Combine the samples
                samples = np.stack(samples_list, axis=2)
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error during parallelization: {str(e)}. Using sequential sampling.")
                
                # Fallback to sequential generation
                samples = np.zeros((n_test, output_dim, n_samples))
                for i, seed in enumerate(seeds):
                    samples[:, :, i] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
        else:
            # Sequential generation of samples
            samples = np.zeros((n_test, output_dim, n_samples))
            for i, seed in enumerate(seeds):
                samples[:, :, i] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
        
        if self.verbose > 0:
            duration = time.time() - start_time
            print(f"Sampling completed in {duration:.2f} seconds")
        
        return samples
    
    def sample_y_batch(self, X_test: np.ndarray, n_samples: int = 10, batch_size: int = 10, 
                       random_state: Optional[int] = None) -> np.ndarray:
        """
        Samples from the predictive distribution using a batched approach.
        This method is useful for large datasets where generating all samples
        simultaneously could exceed available memory.
        
        Args:
            X_test: Test points, matrix of shape (n_test, input_dim)
            n_samples: Total number of samples to generate
            batch_size: Number of samples per batch
            random_state: Random seed for reproducibility
            
        Returns:
            samples: Samples of shape (n_test, output_dim, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before sampling.")
        
        start_time = time.time()
        
        # Predict mean and covariance
        y_mean, y_cov = self.predict(X_test, return_cov=True, full_cov=True)
        
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Initialize the output array
        samples = np.zeros((n_test, output_dim, n_samples))
        
        # Configure the main random generator
        if random_state is not None:
            main_rng = np.random.RandomState(random_state)
        else:
            main_rng = np.random.RandomState()
        
        # Calculate the number of batches
        n_batches = int(np.ceil(n_samples / batch_size))
        
        if self.verbose > 0:
            print(f"Generating {n_samples} samples in {n_batches} batches of {batch_size}")
        
        # Generate samples in batches
        for i in range(n_batches):
            # Determine the start and end index for this batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            if self.verbose > 1:
                print(f"Generating batch {i+1}/{n_batches} ({current_batch_size} samples)")
            
            # Generate seeds for this batch
            batch_seeds = main_rng.randint(0, 2**32 - 1, size=current_batch_size)
            
            # Generate samples for this batch
            if self.parallel:
                batch_samples = Parallel(n_jobs=self.n_jobs, verbose=max(0, self.verbose-1))(
                    delayed(self._generate_single_sample)(X_test, y_mean, y_cov, seed)
                    for seed in batch_seeds
                )
                # Convert list to array
                batch_samples = np.stack(batch_samples, axis=2)
            else:
                batch_samples = np.zeros((n_test, output_dim, current_batch_size))
                for j, seed in enumerate(batch_seeds):
                    batch_samples[:, :, j] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
            
            # Store the samples of this batch
            samples[:, :, start_idx:end_idx] = batch_samples
        
        if self.verbose > 0:
            duration = time.time() - start_time
            print(f"Batched sampling completed in {duration:.2f} seconds")
        
        return samples
    
    def log_marginal_likelihood(self) -> float:
        """
        Computes the marginal log-likelihood of the model with current hyperparameters.
        
        Returns:
            log_likelihood: Marginal log-likelihood
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before computing log-likelihood.")
        
        # Construct params vector
        params = np.concatenate([self.kernel.params, [self.log_noise_variance]])
        
        # Use the efficient method if possible
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1 and self.eigendecomp_computed:
            return -self._compute_log_likelihood_efficient(params, self.X_train, self.y_train)
        else:
            return -self._compute_log_likelihood_naive(params, self.X_train, self.y_train)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the marginal log-likelihood on test data.
        
        Args:
            X: Input matrix of shape (n, input_dim)
            y: Output matrix of shape (n, output_dim)
            
        Returns:
            log_likelihood: Marginal log-likelihood
        """
        X_stacked, y_stacked = self._prepare_data(X, y)
        
        # Temporarily save training data
        X_train_orig, y_train_orig = self.X_train, self.y_train
        
        # Use test data
        self.X_train, self.y_train = X_stacked, y_stacked
        
        # Reset precomputed decompositions
        L_orig, alpha_orig = self.L, self.alpha
        eigendecomp_computed_orig = self.eigendecomp_computed
        self.L, self.alpha = None, None
        self.eigendecomp_computed = False
        
        # Construct params vector
        params = np.concatenate([self.kernel.params, [self.log_noise_variance]])
        
        # Compute log-likelihood
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            score = -self._compute_log_likelihood_efficient(params, X_stacked, y_stacked)
        else:
            score = -self._compute_log_likelihood_naive(params, X_stacked, y_stacked)
        
        # Restore training data and decompositions
        self.X_train, self.y_train = X_train_orig, y_train_orig
        self.L, self.alpha = L_orig, alpha_orig
        self.eigendecomp_computed = eigendecomp_computed_orig
        
        return score