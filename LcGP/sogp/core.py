import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cholesky, cho_solve
from scipy.stats.qmc import LatinHypercube
from .kernels.Kernel import RBFKernel, MaternKernel #Single_Output_GP.
from joblib import Parallel, delayed
import copy
#import time

class so_GPRegression:
    def __init__(self, kernel=None, mean_prior = 'zero', var_noise=1e-13, noisy_data = True, use_kernel_grad = True, optimizer='L-BFGS-B', parallel=True, verbose=False):
        """
        Initializes the Gaussian process regression model.
        :param kernel: Kernel function that takes parameters and inputs, and returns a covariance matrix.
        :param optimizer: Optimizer to use for hyperparameter optimization.
        :noisy_data: boolean to perform interpolation.
        :param var_noise: noise variance.
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
        self.verbose = verbose
        
        self.alpha_one_y = None
        self.cholesky_K = None
        
        if self.noisy_data:
            self.hyperparameters = (
                self.kernel.hyperparams + [self.sigma_k, self.var_noise] if isinstance(self.kernel.hyperparams, list)
                else [self.kernel.hyperparams, self.sigma_k, self.var_noise]
            )
        else : 
            self.hyperparameters = (self.kernel.hyperparams if isinstance(self.kernel.hyperparams, list) 
                                    else [self.kernel.hyperparams])


    def _initialize_hyperparameters(self, num_params, kernel_params_0, var_noise_0, custom_bounds, multi_start, n_start, seed=10):
        """
        Initializes hyperparameters randomly if no value is provided.
        And if multi-start, initializes n_start hyperparameters via LHS.

        custom_bounds: initialization bounds (smaller than the optimization bounds)
        """
     
        n_params_k = self.X_train.shape[1] # len(self.kernel.hyperparams)

        if multi_start:
            # Generate initial points with LHS
            lhs = LatinHypercube(d=num_params, seed=seed)
            sampled_points = lhs.random(n=n_start)  # n_start points in [0,1]^d

            # Scale LHS points to the specified bounds
            init_params_list = custom_bounds[:, 0] + sampled_points * (custom_bounds[:, 1] - custom_bounds[:, 0])
        else:
            kernel_params_0 = (np.array(kernel_params_0) if kernel_params_0 is not None 
                               else np.random.uniform(custom_bounds[:n_params_k,0],(custom_bounds[:n_params_k,1]),size=n_params_k ).tolist())
            if self.noisy_data==True:
                var_noise_0= var_noise_0 if var_noise_0 is not None else np.random.uniform(custom_bounds[-1, 0],custom_bounds[-1, 1])
                init_params_list = np.append(kernel_params_0,var_noise_0).reshape(1,-1)
            else : 
                init_params_list = np.array(kernel_params_0).reshape(1,-1)
        return init_params_list
    

    def _compute_state(self, params):
        """
        Helper method to compute the state (Cholesky, alpha, etc.) for a given set of params.
        Does NOT modify self.
        """
        n = self.X_train.shape[0]
        # Use a local copy of kernel to avoid race conditions
        kernel_ = copy.deepcopy(self.kernel)
        
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            kernel_.hyperparams = kernel_params
            var_noise = params[-1] 
        else:
            kernel_params = np.exp(params)
            kernel_.hyperparams = kernel_params
            var_noise = self.var_noise
        
        K = kernel_(self.X_train) + var_noise * np.eye(n)
        K = (K + K.T) / 2
        L = cholesky(K, lower=True)
        
        one_1 = np.ones_like(self.y_train)
        alpha_y = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
        
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            alpha_one = solve_triangular(L.T, solve_triangular(L, one_1, lower=True))
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else:
            raise ValueError("Error: mean_prior can only be defined as 'constant' or 'zero' !!")
            
        alpha_one_y = solve_triangular(L.T, solve_triangular(L, self.y_train - mean_hat * one_1, lower=True))
        sigma_k2 = ((self.y_train - mean_hat * one_1).T @ alpha_one_y) / n
        sigma_k = np.sqrt(sigma_k2)
        
        return L, sigma_k, alpha_one_y, mean_hat

    def n_log_marginal_likelihood(self, params):
        """
        Computes the negative marginal log-likelihood.
        Thread-safe: does not modify self.
        """
        n = self.X_train.shape[0]
        # Use a local copy of kernel to avoid race conditions
        kernel_ = copy.deepcopy(self.kernel)
        
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            kernel_.hyperparams = kernel_params
            var_noise = params[-1] 
        else:
            kernel_params = np.exp(params)
            kernel_.hyperparams = kernel_params
            var_noise = self.var_noise
        
        K = kernel_(self.X_train) + var_noise * np.eye(n)
        K = (K + K.T) / 2
        L = cholesky(K, lower=True)

        one_1 = np.ones_like(self.y_train)
        alpha_y = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
        
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            alpha_one = solve_triangular(L.T, solve_triangular(L, one_1, lower=True))
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else :
            raise ValueError("Error: mean_prior can only be defined as 'constant' or 'zero' !!")
            
        alpha_one_y = solve_triangular(L.T, solve_triangular(L, self.y_train - mean_hat * one_1, lower=True))
        sigma_k2 = ((self.y_train - mean_hat * one_1).T @ alpha_one_y) / n
        
        n_log_l = 0.5 * n * np.log(sigma_k2) + 0.5 * n + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

        # Do NOT save state to self here. State is computed separately.
        return n_log_l.item() if isinstance(n_log_l, np.ndarray) else n_log_l
    
    def n_log_marginal_likelihood_grad(self, params):
        """
        Computes the gradient of the negative marginal log-likelihood.
        Thread-safe: does not modify self.
        """
        n = self.y_train.shape[0]
        # Use a local copy of kernel to avoid race conditions
        kernel_ = copy.deepcopy(self.kernel)
        
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            kernel_.hyperparams = kernel_params
            var_noise = params[-1]
        else:
            kernel_params = np.exp(params)
            kernel_.hyperparams = kernel_params
            var_noise = self.var_noise
        
        n_params_K = kernel_params.shape[0]
        
        # Compute the covariance matrix and Cholesky decomposition
        K = kernel_(self.X_train) + var_noise * np.eye(n)
        L = cholesky(K, lower=True)
        
        # Compute K_inv
        K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(n), lower=True))

        # Compute the prior mean
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            alpha_y = K_inv @ self.y_train
            alpha_one = K_inv @ np.ones_like(self.y_train)
            mean_hat = (np.ones_like(self.y_train).T @ alpha_y) / (np.ones_like(self.y_train).T @ alpha_one)
        else:
            raise ValueError("Error: mean_prior can only be defined as 'constant' or 'zero' !!")
        
        y_centered = self.y_train - mean_hat * np.ones_like(self.y_train)
        alpha_one_y = solve_triangular(L.T, solve_triangular(L, y_centered, lower=True))
        
        L_T_y = solve_triangular(L, y_centered, lower=True)
        sigma_k2 = L_T_y.T @ L_T_y
        term_commun = 1.0 / sigma_k2
        
        # Compute the gradient for all hyperparameters
        dK_dTheta = kernel_.grad_K(self.X_train)
        
        grad_theta_list = np.zeros(n_params_K)
        alpha_alpha_T = alpha_one_y @ alpha_one_y.T
        term_c = 0.5 * (alpha_alpha_T * term_commun * n - K_inv)
        
        for i, dK_dtheta in enumerate(np.rollaxis(dK_dTheta, 2)):
            term = term_c * dK_dtheta
            grad = -np.sum(term)
            grad_theta_list[i] = grad
        
        # Apply scale factor for the kernel parameters
        grad_theta_list *= kernel_params
        
        # Add the gradient for the noise variance if necessary
        if self.noisy_data:
            dK_dvar = np.eye(n)
            term = term_c * dK_dvar
            grad_var = -np.sum(term)
            grad_Cov = np.append(grad_theta_list, grad_var)
        else:
            grad_Cov = grad_theta_list
        
        return grad_Cov


    def _optimize_hyperparameters(self, init_k_params,init_var_noise, multi_start, n_start, theta_lb, theta_ub, var_lb,var_ub, maxiter=100,seed=13):
        """
        Optimizes hyperparameters by maximizing the marginal log-likelihood with parallel multi-start.

        :param initial_params: Initial parameters for optimization.
        :param multi_start: Enables or disables multi-start.
        :param n_start: Number of initializations if multi_start is enabled.
        :param n_jobs: Number of parallel jobs (-1 uses all available cores).
        :return: Optimization result with the best hyperparameters.
        """
        
        len_kern_hyp = 0 # #self.kernel.hyperparams.shape[0]
        if self.noisy_data :
            if init_k_params is not None : 
                len_kern_hyp = len(init_k_params)
                num_params = len_kern_hyp+1
            else:
                len_kern_hyp = self.X_train.shape[1]  ## By default, the number of lengthscales to initialize is the input dimension
                num_params = len_kern_hyp+1
        else : 
            if init_k_params is not None : 
                len_kern_hyp = len(init_k_params)
                num_params = len_kern_hyp
            else :
                len_kern_hyp = self.X_train.shape[1]  ## By default, the number of lengthscales to initialize is the input dimension
                num_params = len_kern_hyp
        

        # Specific bounds for each hyperparameter

        Theta_lb = theta_lb if theta_lb is not None else 1e-3*np.ones(len_kern_hyp) #(max(1e-2, min_distance(self.X_train)/np.sqrt(self.X_train.shape[1])))
        Theta_ub = theta_ub if theta_ub is not None else  10*(np.max(self.X_train, axis=0) - np.min(self.X_train,axis=0)) #max_distance(self.X_train) #/np.sqrt(self.X_train.shape[1])  #
        if len_kern_hyp==1 and theta_ub is None:
            Theta_ub = 10*(np.max(self.X_train) - np.min(self.X_train))

        
        var_lb = var_lb if var_lb is not None else 1e-9*np.var(self.y_train)
        var_ub = var_ub if var_ub is not None else np.var(self.y_train)

        custom_bounds = np.vstack([np.log(Theta_lb), np.log(Theta_ub)]).T
        if self.noisy_data:
            custom_bounds = np.vstack([custom_bounds, np.array([var_lb, var_ub])])
        
        
        bounds = [tuple(sous_bounds) for sous_bounds in custom_bounds]
        init_params_list = self._initialize_hyperparameters(num_params, init_k_params,init_var_noise, custom_bounds,multi_start,n_start,seed)
        #print("init param list : ", init_params_list)
        # Optimization function to execute in parallel
        def optimize_single_wgrad(init_params):
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                jac=self.n_log_marginal_likelihood_grad,
                method=self.optimizer,
                bounds= bounds, 
                options={'maxiter': maxiter, 'disp': self.verbose}
            )
            # Compute state for the optimized params
            L, sigma_k, alpha_one_y, mean_p = self._compute_state(results.x)
            return results, L, alpha_one_y, sigma_k, mean_p
        
        def optimize_single_wo_grad(init_params):
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                method=self.optimizer,
                bounds= bounds, 
                options={'maxiter': maxiter, 'disp': self.verbose}
            )
            L, sigma_k, alpha_one_y, mean_p = self._compute_state(results.x)
            return results, L, alpha_one_y, sigma_k, mean_p
        
        
        optimize_func = optimize_single_wgrad if self.use_kernel_grad else optimize_single_wo_grad
        # Case where there is only one set of initial parameters or if parallel is False
        if init_params_list.shape[0] == 1 or not self.parallel:
            if init_params_list.shape[0] == 1:
                # A single set of parameters, no need for parallelization
                results = [optimize_func(init_params_list[0])] #[0]
            else:
                # Sequential execution of multiple parameter sets
                results = [optimize_func(params) for params in init_params_list]
            # Select the best result
            best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean = min(
                results, key=lambda res: res[0].fun)
        else:
            # Parallel execution when parallel=True and multiple parameter sets
            try:
                # Determine the number of jobs (use all available cores by default)
                n_jobs = -1 #self.n_jobs if hasattr(self, 'n_jobs') else -1
                # Parallel execution
                results = Parallel(n_jobs=n_jobs, verbose=self.verbose if hasattr(self, 'verbose') else 0)(
                    delayed(optimize_func)(params) for params in init_params_list
                )
                # Select the best result
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean = min(
                    results, key=lambda res: res[0].fun)             
            except ImportError:
                # Fallback if joblib is not available
                print("Warning: joblib is not available. Sequential execution.")
                results = [optimize_func(params) for params in init_params_list]
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean = min(
                    results, key=lambda res: res[0].fun)
        
        #print("nll : ", best_result)

        # Update the main instance variables (self)
        self.cholesky_K = best_cholesky_K
        self.alpha_one_y = best_alpha
        self.sigma_k = best_sigma_k
        self.mean_p = best_mean

        return best_result
    

    def fit(self, X, y, multi_start=True, n_start=5, theta_0 = None, var_noise_0=None, theta_lb=None, theta_ub=None, var_lb=None,var_ub=None, hyperparamet_optimize=True, maxiter=100, seed=13):
        """
        Fits the model to the training data.

        :param X: Training inputs.
        :param y: Training outputs.
        """
        self.X_train = X
        
        self.y_train = y.reshape(-1, 1)

        # Hyperparameter optimization
        if hyperparamet_optimize:
            opt_result = self._optimize_hyperparameters(theta_0,var_noise_0, multi_start, n_start, theta_lb, theta_ub, var_lb, var_ub,maxiter,seed)
            if self.noisy_data:
                self.hyperparameters = [np.exp(opt_result.x[:-1]), self.sigma_k**2, opt_result.x[-1]]
                self.var_noise = opt_result.x[-1]
                self.kernel.hyperparams = np.exp(opt_result.x[:-1])
            else:
                self.hyperparameters = [np.exp(opt_result.x), self.sigma_k**2]
                self.kernel.hyperparams = np.exp(opt_result.x)

    def predict(self, X_star, return_cov=False):
        """
        Predicts outputs for new inputs.

        :param X_star: New inputs.
        :return: Mean and variance of predictions.
        """
        K_star = self.kernel(self.X_train, X_star)
        K_star_star = self.kernel(X_star, X_star)
        
        L = self.cholesky_K 
        mean_hat = self.mean_p 
        alpha_one_y = self.alpha_one_y 
        mean =mean_hat+np.dot(K_star.T, alpha_one_y)
        v = solve_triangular(L, K_star, lower=True)
        
        if return_cov:
            y_cov = (K_star_star - np.dot(v.T, v))*self.sigma_k**2
            return mean.ravel(),y_cov
        else :
            y_var = np.diag((1 - np.dot(v.T, v))*self.sigma_k**2)
        #variance = (1 - np.dot(v.T, v))*self.sigma_k**2  
            return mean.ravel(), y_var
