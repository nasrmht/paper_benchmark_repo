import numpy as np
from .core import so_GPRegression

class IndependantMultiOutputGP:
    def __init__(self, output_dim, kernel=None, var_noise=1e-13, parallel=True):
        """
        Gaussian process regression model for multiple independent outputs.
        :param output_dim: Number of independent outputs (p).
        :param kernel: Kernel to use for each GP.
        :param var_noise: Noise variance for each GP.
        """
        self.output_dim = output_dim
        self.models = [so_GPRegression(kernel=kernel, var_noise=var_noise, use_kernel_grad=True, parallel=parallel) for _ in range(output_dim)]
    
    def fit(self, X, Y, multi_start=True, n_restart=5, maxiter=100, seed=42):
        """
        Trains each GP independently.
        :param X: Inputs (n, d).
        :param Y: Outputs (n, p).
        """
        for i in range(self.output_dim):
            self.models[i].fit(X, Y[:, i], multi_start=multi_start, n_start=n_restart, maxiter=maxiter, seed=seed)
    
    def predict(self, X_test, return_cov=True, full_covar=False):
        """
        Predicts Y* for all outputs independently.
        :param X_test: New inputs (m, d).
        :return: Mean (m, p) and, if return_cov=True, Variance (m, p).
        """
        means, variances = [], []
        for model in self.models:
            mean, var = model.predict(X_test, return_cov=return_cov)
            means.append(mean)
            if full_covar:
                variances.append(var)
            else :
                variances.append(np.diag(var))
        
        if return_cov:
            return np.column_stack(means), np.asarray(variances)
        return np.column_stack(means)
    
    def get_likelihoods(self):
        """ Returns the log-likelihood of each GP. """
        return [model.n_log_marginal_likelihood(model.hyperparameters) for model in self.models]
    
    def get_params(self):
        """ Returns the hyperparameters of each GP. """
        return [model.hyperparameters for model in self.models]
    
    def sample_functions(self, X_test, n_samples=1):
        """
        Generates simultaneous samples for all independent outputs.
        :param X_test: New inputs (m, d).
        :param n_samples: Number of samples to generate.
        :return: Samples (m, p, n_samples).
        """
        np.random.seed(42)
        m = X_test.shape[0]  # Number of test points
        p = len(self.models)  # Number of outputs
        
        samples = np.zeros((n_samples, m, p))  # (n_samples, m, p)
        
        for i, model in enumerate(self.models):
            mean, var = model.predict(X_test, return_cov=True)
            z = np.random.randn(m,n_samples)  # Independent random variable for each output
            # Generation of samples from the multivariate normal distribution
            #trajectories = np.random.multivariate_normal(mean, var, size=n_samples)
            
            L = np.linalg.cholesky(var+ 1e-6*np.eye(m))
            samples[:, :, i] =  (mean.reshape(-1,1) + L@z).T #np.sqrt(var) * z
        
        return samples.transpose(1, 2, 0)  # Reorder to (m, p, n_samples)
