import numpy as np
from .core import so_GPRegression

class IndependantMultiOutputGP:
    def __init__(self, output_dim, kernel=None, var_noise=1e-13, parallel=True):
        """
        Modèle de régression par processus gaussiens pour plusieurs sorties indépendantes.
        :param output_dim: Nombre de sorties indépendantes (p).
        :param kernel: Noyau à utiliser pour chaque GP.
        :param var_noise: Variance du bruit pour chaque GP.
        """
        self.output_dim = output_dim
        self.models = [so_GPRegression(kernel=kernel, var_noise=var_noise, use_kernel_grad=True, parallel=parallel) for _ in range(output_dim)]
    
    def fit(self, X, Y, multi_start=True, n_restart=5, maxiter=100, seed=42):
        """
        Entraîne chaque GP indépendamment.
        :param X: Entrées (n, d).
        :param Y: Sorties (n, p).
        """
        for i in range(self.output_dim):
            self.models[i].fit(X, Y[:, i], multi_start=multi_start, n_start=n_restart, maxiter=maxiter, seed=seed)
    
    def predict(self, X_test, return_cov=True, full_covar=False):
        """
        Prédit Y* pour toutes les sorties indépendamment.
        :param X_test: Nouvelles entrées (m, d).
        :return: Moyenne (m, p) et, si return_cov=True, Variance (m, p).
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
        """ Retourne la log-vraisemblance de chaque GP. """
        return [model.n_log_marginal_likelihood(model.hyperparameters) for model in self.models]
    
    def get_params(self):
        """ Retourne les hyperparamètres de chaque GP. """
        return [model.hyperparameters for model in self.models]
    
    def sample_functions(self, X_test, n_samples=1):
        """
        Génère des échantillons simultanés pour toutes les sorties indépendantes.
        :param X_test: Nouvelles entrées (m, d).
        :param n_samples: Nombre d'échantillons à générer.
        :return: Échantillons (m, p, n_samples).
        """
        np.random.seed(42)
        m = X_test.shape[0]  # Nombre de points de test
        p = len(self.models)  # Nombre de sorties
        
        samples = np.zeros((n_samples, m, p))  # (n_samples, m, p)
        
        for i, model in enumerate(self.models):
            mean, var = model.predict(X_test, return_cov=True)
            z = np.random.randn(m,n_samples)  # Variable aléatoire indépendante pour chaque sortie
            # Génération des échantillons de la loi normale multivariée
            #trajectories = np.random.multivariate_normal(mean, var, size=n_samples)
            
            L = np.linalg.cholesky(var+ 1e-6*np.eye(m))
            samples[:, :, i] =  (mean.reshape(-1,1) + L@z).T #np.sqrt(var) * z
        
        return samples.transpose(1, 2, 0)  # Reordonner en (m, p, n_samples)
