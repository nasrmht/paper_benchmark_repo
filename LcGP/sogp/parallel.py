import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cholesky
from scipy.stats.qmc import LatinHypercube
from .kernels.Kernel import RBFKernel  # Single_Output_GP.
from joblib import Parallel, delayed
import time


class P_so_GPRegression:
    def __init__(self, kernel=None, mean_prior='zero', var_noise=1e-13, noisy_data=True, use_kernel_grad=False, 
                 optimizer='L-BFGS-B', parallel=False, n_jobs=-1, verbose=0):
        """
        Initializes the Gaussian process regression model with parallelization support.
        
        :param kernel: Kernel function that takes parameters and inputs, and returns a covariance matrix.
        :param mean_prior: Prior mean type ('zero' or 'constant').
        :param var_noise: Noise variance (used if noisy_data=True).
        :param noisy_data: Boolean to perform interpolation (False) or regression (True).
        :param use_kernel_grad: Use the analytical gradient of the kernel.
        :param optimizer: Optimizer to use for hyperparameter optimization.
        :param parallel: Enable parallelization for multi-start.
        :param n_jobs: Number of jobs for parallelization (-1 = all cores).
        :param verbose: Verbosity level for displaying information during optimization.
        """
        self.optimizer = optimizer
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.mean_prior = mean_prior
        self.mean_p = None
        self.use_kernel_grad = use_kernel_grad
        self.var_noise = var_noise
        self.sigma_k = 1.0
        self.noisy_data = noisy_data
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.alpha_one_y = None
        self.cholesky_K = None
        
        if self.noisy_data:
            self.hyperparameters = (
                self.kernel.hyperparams + [self.sigma_k, self.var_noise] if isinstance(self.kernel.hyperparams, list)
                else [self.kernel.hyperparams, self.sigma_k, self.var_noise]
            )
        else: 
            self.hyperparameters = (self.kernel.hyperparams if isinstance(self.kernel.hyperparams, list) 
                                    else [self.kernel.hyperparams])
    
    def _initialize_hyperparameters(self, num_params, kernel_params_0, var_noise_0, custom_bounds, multi_start, n_start):
        """
        Initializes the hyperparameters randomly or via LHS if multi-start is enabled.
        
        :param num_params: Total number of hyperparameters.
        :param kernel_params_0: Initial values for the kernel parameters.
        :param var_noise_0: Initial value for the noise variance.
        :param custom_bounds: Initialization bounds.
        :param multi_start: Enable multiple initializations.
        :param n_start: Number of starting points if multi_start is enabled.
        :return: Array of initial points for optimization.
        """
        n_params_k = self.X_train.shape[1] if self.X_train.ndim > 1 else 1
        np.random.seed(42)
        
        if multi_start:
            # Efficient generation of initial points with LHS
            lhs = LatinHypercube(d=num_params, seed=42)
            sampled_points = lhs.random(n=n_start)  # n_start points in [0,1]^d
            
            # Scaling LHS points to the specified bounds
            # Optimization: vectorized computation instead of a loop
            init_params_list = custom_bounds[:, 0] + sampled_points * (custom_bounds[:, 1] - custom_bounds[:, 0])
            
            # Reshape to get (n_start, num_params)
            init_params_list = init_params_list.reshape(n_start, num_params)
        else:
            # Simple initialization without multi-start
            kernel_params_0 = (np.array(kernel_params_0) if kernel_params_0 is not None 
                              else np.random.uniform(custom_bounds[:n_params_k, 0], custom_bounds[:n_params_k, 1], size=n_params_k))
            
            if self.noisy_data:
                var_noise_0 = var_noise_0 if var_noise_0 is not None else np.random.uniform(custom_bounds[-1, 0], custom_bounds[-1, 1])
                init_params = np.append(kernel_params_0, var_noise_0)
            else:
                init_params = kernel_params_0
                
            init_params_list = init_params.reshape(1, -1)  # Reshape to ensure shape consistency
            
        return init_params_list
    
    def n_log_marginal_likelihood(self, params):
        """
        Computes the negative marginal log-likelihood.
        
        :param params: Kernel and model hyperparameters.
        :return: Negative marginal log-likelihood.
        """
        n = self.X_train.shape[0]
        
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            self.kernel.hyperparams = kernel_params
            var_noise = params[-1] 
        else:
            kernel_params = np.exp(params)
            self.kernel.hyperparams = kernel_params
            var_noise = self.var_noise
        
        # Compute the covariance matrix
        K = self.kernel(self.X_train) + var_noise * np.eye(n)
        K = (K + K.T) / 2  # Ensure numerical symmetry
        
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # In case Cholesky fails, return a high value
            return 1e10
        
        one_1 = np.ones((n, 1))
        
        # Efficient calculation with solve_triangular
        alpha_y = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
        
        # Compute the prior mean
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            alpha_one = solve_triangular(L.T, solve_triangular(L, one_1, lower=True))
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else:
            raise ValueError(f"Error: mean_prior can only be defined as 'constant' or 'zero' !!")
        
        # Optimized calculation of the log-likelihood
        y_centered = self.y_train - mean_hat * one_1
        alpha_one_y = solve_triangular(L.T, solve_triangular(L, y_centered, lower=True))
        L_T_y = solve_triangular(L, y_centered, lower=True)
        sigma_k2 = (L_T_y.T @ L_T_y) / n  # Mean of the explained variance
        
        # Optimized calculation of log-likelihood
        n_log_l = 0.5 * n * np.log(sigma_k2) + 0.5 * n + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        
        # Save for prediction
        self.cholesky_K = L
        self.sigma_k = np.sqrt(sigma_k2)
        self.alpha_one_y = alpha_one_y
        self.mean_p = mean_hat
        
        return n_log_l
    
    def n_log_marginal_likelihood_grad(self, params):
        """
        Computes the gradient of the negative marginal log-likelihood.
        
        :param params: Kernel and model hyperparameters.
        :return: Gradient of the negative marginal log-likelihood.
        """
        n = self.y_train.shape[0]
        
        # Configure parameters depending on the model
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            self.kernel.hyperparams = kernel_params
            var_noise = params[-1]
        else:
            kernel_params = np.exp(params)
            self.kernel.hyperparams = kernel_params
            var_noise = self.var_noise
        
        n_params_K = kernel_params.shape[0]
        
        # Efficient computation of the covariance matrix and decomposition
        K = self.kernel(self.X_train) + var_noise * np.eye(n)
        
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # Return a zero gradient if Cholesky fails
            grad_size = n_params_K + 1 if self.noisy_data else n_params_K
            return np.zeros(grad_size)
        
        # Optimized computation of K_inv
        K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(n), lower=True))
        
        # Compute the prior mean
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            one_1 = np.ones((n, 1))
            alpha_y = K_inv @ self.y_train
            alpha_one = K_inv @ one_1
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else:
            raise ValueError("Error: mean_prior can only be defined as 'constant' or 'zero' !!")
        
        # Efficient computation of y_centered
        one_1 = np.ones_like(self.y_train)
        y_centered = self.y_train - mean_hat * one_1
        
        # Optimized computation of alpha_one_y
        alpha_one_y = K_inv @ y_centered
        
        # Compute sigma_k2
        sigma_k2 = (y_centered.T @ alpha_one_y) / n
        
        # Common term for gradient computation
        term_commun = 1.0 / sigma_k2
        
        # Efficient computation of alpha_alpha_T
        alpha_alpha_T = np.outer(alpha_one_y, alpha_one_y)
        
        # Optimized common term
        term_c = 0.5 * (alpha_alpha_T * term_commun * n - K_inv)
        
        # Compute partial derivatives with respect to the kernel hyperparameters
        dK_dTheta = self.kernel.grad_K(self.X_train)
        
        # Pre-allocation for gradients
        grad_theta_list = np.zeros(n_params_K)
        
        # Vectorized computation of gradients
        for i, dK_dtheta in enumerate(np.rollaxis(dK_dTheta, 2)):
            # Optimized computation of the gradient
            grad_theta_list[i] = -np.sum(term_c * dK_dtheta)
        
        # Apply scale factor for log-transformed parameters
        grad_theta_list *= kernel_params
        
        # Add gradient for the noise variance if necessary
        if self.noisy_data:
            # Gradient for the noise variance (no logarithmic scale)
            dK_dvar = np.eye(n)
            grad_var = -np.sum(term_c * dK_dvar)
            
            # Combine gradients
            grad_Cov = np.append(grad_theta_list, grad_var)
        else:
            grad_Cov = grad_theta_list
        
        return grad_Cov
    
    def _optimize_single_start(self, init_params, bounds):
        """
        Optimizes hyperparameters starting from a given initial point.
        
        :param init_params: Initial parameters.
        :param bounds: Bounds for optimization.
        :return: Optimization result and associated parameters.
        """
        start_time = time.time()
        
        if self.use_kernel_grad:
            # Optimization with analytical gradient
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                jac=self.n_log_marginal_likelihood_grad,
                method=self.optimizer,
                bounds=bounds,
                options={'maxiter': 200}
            )
        else:
            # Optimization without analytical gradient
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                method=self.optimizer,
                bounds=bounds,
                options={'maxiter': 200}
            )
        
        # Duration of optimization
        duration = time.time() - start_time
        
        # Return results and associated parameters
        return results, self.cholesky_K, self.alpha_one_y, self.sigma_k, self.mean_p, duration
    
    def _optimize_hyperparameters(self, init_k_params, init_var_noise, multi_start, n_start, theta_lb, theta_ub, var_lb, var_ub):
        """
        Optimizes hyperparameters with support for parallel multi-start.
        
        :param init_k_params: Initial values for the kernel parameters.
        :param init_var_noise: Initial value for the noise variance.
        :param multi_start: Enable multiple starting points.
        :param n_start: Number of starting points if multi_start is enabled.
        :param theta_lb, theta_ub: Lower and upper bounds for the kernel hyperparameters.
        :param var_lb, var_ub: Lower and upper bounds for the noise variance.
        :return: Optimization result with the best hyperparameters.
        """
        # Determine the number of kernel parameters
        if init_k_params is not None:
            len_kern_hyp = len(init_k_params)
        else:
            len_kern_hyp = self.X_train.shape[1] if self.X_train.ndim > 1 else 1
        
        # Total number of parameters (kernel + noise if applicable)
        num_params = len_kern_hyp + 1 if self.noisy_data else len_kern_hyp
        
        # Bounds for the kernel hyperparameters
        Theta_lb = theta_lb if theta_lb is not None else 1e-3 * np.ones(len_kern_hyp)
        
        if theta_ub is not None:
            Theta_ub = theta_ub
        else:
            if self.X_train.ndim > 1:
                Theta_ub = 10 * (np.max(self.X_train, axis=0) - np.min(self.X_train, axis=0))
            else:
                Theta_ub = 10 * (np.max(self.X_train) - np.min(self.X_train)) * np.ones(len_kern_hyp)
                
        # Bounds for the noise variance
        var_lb = var_lb if var_lb is not None else 1e-9 * np.var(self.y_train)
        var_ub = var_ub if var_ub is not None else np.var(self.y_train)
        
        # Prepare the bounds for optimization
        custom_bounds = np.vstack([np.log(Theta_lb), np.log(Theta_ub)]).T
        
        if self.noisy_data:
            custom_bounds = np.vstack([custom_bounds, np.array([var_lb, var_ub])])
        
        bounds = [tuple(sous_bounds) for sous_bounds in custom_bounds]
        
        # Parameter initialization
        init_params_list = self._initialize_hyperparameters(num_params, init_k_params, init_var_noise, 
                                                           custom_bounds, multi_start, n_start)
        
        if self.verbose > 0:
            print(f"Optimization with {init_params_list.shape[0]} starting points")
        
        # Parallel optimization if requested and multiple starting points
        if multi_start and self.parallel and init_params_list.shape[0] > 1:
            try:
                if self.verbose > 0:
                    print(f"Parallel execution with {self.n_jobs} jobs")
                
                # Parallel execution with joblib
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._optimize_single_start)(params, bounds) 
                    for params in init_params_list
                )
                
                # Select the best result
                sorted_results = sorted(results, key=lambda res: res[0].fun)
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
                
                if self.verbose > 0:
                    for i, (res, _, _, _, _, dur) in enumerate(sorted_results):
                        print(f"Run {i+1}: nLL = {res.fun:.6f}, duration = {dur:.2f}s")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Parallelization error: {str(e)}. Fallback to sequential execution.")
                
                # Sequential execution in case of failure
                results = [self._optimize_single_start(params, bounds) for params in init_params_list]
                sorted_results = sorted(results, key=lambda res: res[0].fun)
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
        else:
            # Sequential execution
            if self.verbose > 0 and multi_start and init_params_list.shape[0] > 1:
                print("Sequential execution of optimizations")
                
            results = []
            for i, params in enumerate(init_params_list):
                res, chol, alpha, sigma, mean, dur = self._optimize_single_start(params, bounds)
                results.append((res, chol, alpha, sigma, mean, dur))
                
                if self.verbose > 0:
                    print(f"Run {i+1}: nLL = {res.fun:.6f}, duration = {dur:.2f}s")
            
            # Select the best result
            sorted_results = sorted(results, key=lambda res: res[0].fun)
            best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
        
        if self.verbose > 0:
            print(f"Best optimization: nLL = {best_result.fun:.6f}, duration = {duration:.2f}s")
            
            if self.noisy_data:
                kernel_params = np.exp(best_result.x[:-1])
                noise_var = best_result.x[-1]
                print(f"Optimal hyperparameters: kernels = {kernel_params}, noise = {noise_var}")
            else:
                kernel_params = np.exp(best_result.x)
                print(f"Optimal hyperparameters: kernels = {kernel_params}")
        
        # Update the state variables
        self.cholesky_K = best_cholesky_K
        self.alpha_one_y = best_alpha
        self.sigma_k = best_sigma_k
        self.mean_p = best_mean
        
        return best_result
    
    def fit(self, X, y, multi_start=False, n_start=5, theta_0=None, var_noise_0=None, 
            theta_lb=None, theta_ub=None, var_lb=None, var_ub=None):
        """
        Fits the model to the training data with parallel hyperparameter optimization.
        
        :param X: Training inputs.
        :param y: Training outputs.
        :param multi_start: Enable multiple starting points.
        :param n_start: Number of starting points if multi_start is enabled.
        :param theta_0: Initial values for the kernel hyperparameters.
        :param var_noise_0: Initial value for the noise variance.
        :param theta_lb, theta_ub: Bounds for the kernel hyperparameters.
        :param var_lb, var_ub: Bounds for the noise variance.
        :return: self
        """
        # Input formatting
        if X.ndim == 1:
            self.X_train = X.reshape(-1, 1)
        else:
            self.X_train = X
            
        self.y_train = y.reshape(-1, 1)
        
        # Start timing
        start_time = time.time()
        
        # Hyperparameter optimization
        opt_result = self._optimize_hyperparameters(
            theta_0, var_noise_0, multi_start, n_start, 
            theta_lb, theta_ub, var_lb, var_ub
        )
        
        # Update the hyperparameters
        if self.noisy_data:
            self.kernel.hyperparams = np.exp(opt_result.x[:-1])
            self.var_noise = opt_result.x[-1]
            self.hyperparameters = [np.exp(opt_result.x[:-1]), self.sigma_k**2, opt_result.x[-1]]
        else:
            self.kernel.hyperparams = np.exp(opt_result.x)
            self.hyperparameters = [np.exp(opt_result.x), self.sigma_k**2]
        
        # Display total fitting time
        if self.verbose > 0:
            print(f"Total fitting time: {time.time() - start_time:.2f}s")
        
        return self
    
    def predict(self, X_star, return_cov=False):
        """
        Predicts outputs for new inputs.
        
        :param X_star: New inputs.
        :param return_cov: If True, returns the full covariance matrix, otherwise returns the variance.
        :return: Predictive mean and variance/covariance.
        """
        # Input formatting
        if X_star.ndim == 1:
            X_star = X_star.reshape(-1, 1)
            
        # Compute covariance between training points and new points
        K_star = self.kernel(self.X_train, X_star)
        
        # Retrieve saved parameters
        L = self.cholesky_K
        mean_hat = self.mean_p
        alpha_one_y = self.alpha_one_y
        
        # Compute the predictive mean
        mean = mean_hat + np.dot(K_star.T, alpha_one_y)
        
        # Compute the predictive variance
        v = solve_triangular(L, K_star, lower=True)
        K_star_star = self.kernel(X_star, X_star)
        
        if return_cov:
            # Compute the full covariance matrix
            y_cov = (K_star_star - np.dot(v.T, v)) * self.sigma_k**2
            return mean.ravel(), y_cov
        else:
            # Compute the predictive variance (covariance diagonal)
            y_var = np.maximum(0, np.diag(K_star_star - np.dot(v.T, v)) * self.sigma_k**2)
            return mean.ravel(), y_var