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
        Initialise le modèle de régression par processus gaussien avec support de parallélisation.
        
        :param kernel: Fonction noyau qui prend des paramètres et des entrées, et retourne une matrice de covariance.
        :param mean_prior: Type de moyenne a priori ('zero' ou 'constant').
        :param var_noise: Variance du bruit (utilisée si noisy_data=True).
        :param noisy_data: Booléen pour faire de l'interpolation (False) ou régression (True).
        :param use_kernel_grad: Utiliser le gradient analytique du noyau.
        :param optimizer: Optimiseur à utiliser pour l'optimisation des hyperparamètres.
        :param parallel: Activer la parallélisation pour multi-start.
        :param n_jobs: Nombre de jobs pour la parallélisation (-1 = tous les cœurs).
        :param verbose: Niveau de verbosité pour l'affichage des informations pendant l'optimisation.
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
        Initialise les hyperparamètres aléatoirement ou par LHS si multi-start est activé.
        
        :param num_params: Nombre total d'hyperparamètres.
        :param kernel_params_0: Valeurs initiales pour les paramètres du noyau.
        :param var_noise_0: Valeur initiale pour la variance du bruit.
        :param custom_bounds: Bornes d'initialisation.
        :param multi_start: Activer l'initialisation multiple.
        :param n_start: Nombre de points de départ si multi_start est activé.
        :return: Array de points initiaux pour l'optimisation.
        """
        n_params_k = self.X_train.shape[1] if self.X_train.ndim > 1 else 1
        np.random.seed(42)
        
        if multi_start:
            # Génération efficace des points initiaux avec LHS
            lhs = LatinHypercube(d=num_params, seed=42)
            sampled_points = lhs.random(n=n_start)  # n_start points dans [0,1]^d
            
            # Mise à l'échelle des points LHS dans les bornes spécifiées
            # Optimisation: calcul vectorisé au lieu d'une boucle
            init_params_list = custom_bounds[:, 0] + sampled_points * (custom_bounds[:, 1] - custom_bounds[:, 0])
            
            # Reshape pour obtenir (n_start, num_params)
            init_params_list = init_params_list.reshape(n_start, num_params)
        else:
            # Initialisation simple sans multi-start
            kernel_params_0 = (np.array(kernel_params_0) if kernel_params_0 is not None 
                              else np.random.uniform(custom_bounds[:n_params_k, 0], custom_bounds[:n_params_k, 1], size=n_params_k))
            
            if self.noisy_data:
                var_noise_0 = var_noise_0 if var_noise_0 is not None else np.random.uniform(custom_bounds[-1, 0], custom_bounds[-1, 1])
                init_params = np.append(kernel_params_0, var_noise_0)
            else:
                init_params = kernel_params_0
                
            init_params_list = init_params.reshape(1, -1)  # Reshape pour assurer la cohérence de forme
            
        return init_params_list
    
    def n_log_marginal_likelihood(self, params):
        """
        Calcule la log-vraisemblance marginale négative.
        
        :param params: Hyperparamètres du noyau et du modèle.
        :return: Log-vraisemblance marginale négative.
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
        
        # Calcul de la matrice de covariance
        K = self.kernel(self.X_train) + var_noise * np.eye(n)
        K = (K + K.T) / 2  # Assurer la symétrie numérique
        
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # En cas d'échec de Cholesky, retourner une valeur élevée
            return 1e10
        
        one_1 = np.ones((n, 1))
        
        # Calcul efficace avec solve_triangular
        alpha_y = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
        
        # Calcul de la moyenne a priori
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            alpha_one = solve_triangular(L.T, solve_triangular(L, one_1, lower=True))
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else:
            raise ValueError(f"Erreur : mean_prior ne peut être défini que comme 'constant' ou 'zero' !!")
        
        # Calcul optimisé de la log-vraisemblance
        y_centered = self.y_train - mean_hat * one_1
        alpha_one_y = solve_triangular(L.T, solve_triangular(L, y_centered, lower=True))
        L_T_y = solve_triangular(L, y_centered, lower=True)
        sigma_k2 = (L_T_y.T @ L_T_y) / n  # Moyenne de la variance expliquée
        
        # Calcul optimisé de log-likelihood
        n_log_l = 0.5 * n * np.log(sigma_k2) + 0.5 * n + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        
        # Sauvegarde pour la prédiction
        self.cholesky_K = L
        self.sigma_k = np.sqrt(sigma_k2)
        self.alpha_one_y = alpha_one_y
        self.mean_p = mean_hat
        
        return n_log_l
    
    def n_log_marginal_likelihood_grad(self, params):
        """
        Calcule le gradient de la log-vraisemblance marginale négative.
        
        :param params: Hyperparamètres du noyau et du modèle.
        :return: Gradient de la log-vraisemblance marginale négative.
        """
        n = self.y_train.shape[0]
        
        # Configuration des paramètres selon le modèle
        if self.noisy_data:
            kernel_params = np.exp(params[:-1])
            self.kernel.hyperparams = kernel_params
            var_noise = params[-1]
        else:
            kernel_params = np.exp(params)
            self.kernel.hyperparams = kernel_params
            var_noise = self.var_noise
        
        n_params_K = kernel_params.shape[0]
        
        # Calcul efficace de la matrice de covariance et décomposition
        K = self.kernel(self.X_train) + var_noise * np.eye(n)
        
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # Retourner un gradient nul en cas d'échec de Cholesky
            grad_size = n_params_K + 1 if self.noisy_data else n_params_K
            return np.zeros(grad_size)
        
        # Calcul optimisé de K_inv
        K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(n), lower=True))
        
        # Calcul de la moyenne a priori
        if self.mean_prior == 'zero':
            mean_hat = 0.0
        elif self.mean_prior == 'constant':
            one_1 = np.ones((n, 1))
            alpha_y = K_inv @ self.y_train
            alpha_one = K_inv @ one_1
            mean_hat = (one_1.T @ alpha_y) / (one_1.T @ alpha_one)
        else:
            raise ValueError("Erreur : mean_prior ne peut être défini que comme 'constant' ou 'zero' !!")
        
        # Calcul efficace de y_centered
        one_1 = np.ones_like(self.y_train)
        y_centered = self.y_train - mean_hat * one_1
        
        # Calcul optimisé de alpha_one_y
        alpha_one_y = K_inv @ y_centered
        
        # Calcul du sigma_k2
        sigma_k2 = (y_centered.T @ alpha_one_y) / n
        
        # Terme commun pour le calcul des gradients
        term_commun = 1.0 / sigma_k2
        
        # Calcul efficace de alpha_alpha_T
        alpha_alpha_T = np.outer(alpha_one_y, alpha_one_y)
        
        # Terme commun optimisé
        term_c = 0.5 * (alpha_alpha_T * term_commun * n - K_inv)
        
        # Calcul des dérivées partielles par rapport aux hyperparamètres du noyau
        dK_dTheta = self.kernel.grad_K(self.X_train)
        
        # Pré-allocation pour les gradients
        grad_theta_list = np.zeros(n_params_K)
        
        # Calcul vectorisé des gradients
        for i, dK_dtheta in enumerate(np.rollaxis(dK_dTheta, 2)):
            # Calcul optimisé du gradient
            grad_theta_list[i] = -np.sum(term_c * dK_dtheta)
        
        # Application du facteur d'échelle pour les paramètres log-transformés
        grad_theta_list *= kernel_params
        
        # Ajout du gradient pour la variance du bruit si nécessaire
        if self.noisy_data:
            # Gradient pour la variance du bruit (pas d'échelle logarithmique)
            dK_dvar = np.eye(n)
            grad_var = -np.sum(term_c * dK_dvar)
            
            # Combinaison des gradients
            grad_Cov = np.append(grad_theta_list, grad_var)
        else:
            grad_Cov = grad_theta_list
        
        return grad_Cov
    
    def _optimize_single_start(self, init_params, bounds):
        """
        Optimise les hyperparamètres à partir d'un point de départ.
        
        :param init_params: Paramètres initiaux.
        :param bounds: Bornes pour l'optimisation.
        :return: Résultat d'optimisation et paramètres associés.
        """
        start_time = time.time()
        
        if self.use_kernel_grad:
            # Optimisation avec gradient analytique
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                jac=self.n_log_marginal_likelihood_grad,
                method=self.optimizer,
                bounds=bounds,
                options={'maxiter': 200}
            )
        else:
            # Optimisation sans gradient analytique
            results = minimize(
                self.n_log_marginal_likelihood,
                init_params,
                method=self.optimizer,
                bounds=bounds,
                options={'maxiter': 200}
            )
        
        # Durée de l'optimisation
        duration = time.time() - start_time
        
        # Retourner les résultats et les paramètres associés
        return results, self.cholesky_K, self.alpha_one_y, self.sigma_k, self.mean_p, duration
    
    def _optimize_hyperparameters(self, init_k_params, init_var_noise, multi_start, n_start, theta_lb, theta_ub, var_lb, var_ub):
        """
        Optimise les hyperparamètres avec support pour la parallélisation multi-start.
        
        :param init_k_params: Valeurs initiales pour les paramètres du noyau.
        :param init_var_noise: Valeur initiale pour la variance du bruit.
        :param multi_start: Activer l'initialisation multiple.
        :param n_start: Nombre de points de départ si multi_start est activé.
        :param theta_lb, theta_ub: Bornes inférieures et supérieures pour les hyperparamètres du noyau.
        :param var_lb, var_ub: Bornes inférieures et supérieures pour la variance du bruit.
        :return: Résultat d'optimisation avec les meilleurs hyperparamètres.
        """
        # Détermination du nombre de paramètres du noyau
        if init_k_params is not None:
            len_kern_hyp = len(init_k_params)
        else:
            len_kern_hyp = self.X_train.shape[1] if self.X_train.ndim > 1 else 1
        
        # Nombre total de paramètres (noyau + bruit si applicable)
        num_params = len_kern_hyp + 1 if self.noisy_data else len_kern_hyp
        
        # Bornes pour les hyperparamètres du noyau
        Theta_lb = theta_lb if theta_lb is not None else 1e-3 * np.ones(len_kern_hyp)
        
        if theta_ub is not None:
            Theta_ub = theta_ub
        else:
            if self.X_train.ndim > 1:
                Theta_ub = 10 * (np.max(self.X_train, axis=0) - np.min(self.X_train, axis=0))
            else:
                Theta_ub = 10 * (np.max(self.X_train) - np.min(self.X_train)) * np.ones(len_kern_hyp)
                
        # Bornes pour la variance du bruit
        var_lb = var_lb if var_lb is not None else 1e-9 * np.var(self.y_train)
        var_ub = var_ub if var_ub is not None else np.var(self.y_train)
        
        # Préparation des bornes pour l'optimisation
        custom_bounds = np.vstack([np.log(Theta_lb), np.log(Theta_ub)]).T
        
        if self.noisy_data:
            custom_bounds = np.vstack([custom_bounds, np.array([var_lb, var_ub])])
        
        bounds = [tuple(sous_bounds) for sous_bounds in custom_bounds]
        
        # Initialisation des paramètres
        init_params_list = self._initialize_hyperparameters(num_params, init_k_params, init_var_noise, 
                                                          custom_bounds, multi_start, n_start)
        
        if self.verbose > 0:
            print(f"Optimisation avec {init_params_list.shape[0]} points de départ")
        
        # Optimisation parallèle si demandée et plusieurs points de départ
        if multi_start and self.parallel and init_params_list.shape[0] > 1:
            try:
                if self.verbose > 0:
                    print(f"Exécution parallèle avec {self.n_jobs} jobs")
                
                # Exécution parallèle avec joblib
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._optimize_single_start)(params, bounds) 
                    for params in init_params_list
                )
                
                # Sélection du meilleur résultat
                sorted_results = sorted(results, key=lambda res: res[0].fun)
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
                
                if self.verbose > 0:
                    for i, (res, _, _, _, _, dur) in enumerate(sorted_results):
                        print(f"Run {i+1}: nLL = {res.fun:.6f}, durée = {dur:.2f}s")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Erreur de parallélisation: {str(e)}. Repli sur exécution séquentielle.")
                
                # Exécution séquentielle en cas d'échec
                results = [self._optimize_single_start(params, bounds) for params in init_params_list]
                sorted_results = sorted(results, key=lambda res: res[0].fun)
                best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
        else:
            # Exécution séquentielle
            if self.verbose > 0 and multi_start and init_params_list.shape[0] > 1:
                print("Exécution séquentielle des optimisations")
                
            results = []
            for i, params in enumerate(init_params_list):
                res, chol, alpha, sigma, mean, dur = self._optimize_single_start(params, bounds)
                results.append((res, chol, alpha, sigma, mean, dur))
                
                if self.verbose > 0:
                    print(f"Run {i+1}: nLL = {res.fun:.6f}, durée = {dur:.2f}s")
            
            # Sélection du meilleur résultat
            sorted_results = sorted(results, key=lambda res: res[0].fun)
            best_result, best_cholesky_K, best_alpha, best_sigma_k, best_mean, duration = sorted_results[0]
        
        if self.verbose > 0:
            print(f"Meilleure optimisation: nLL = {best_result.fun:.6f}, durée = {duration:.2f}s")
            
            if self.noisy_data:
                kernel_params = np.exp(best_result.x[:-1])
                noise_var = best_result.x[-1]
                print(f"Hyperparamètres optimaux: kernels = {kernel_params}, bruit = {noise_var}")
            else:
                kernel_params = np.exp(best_result.x)
                print(f"Hyperparamètres optimaux: kernels = {kernel_params}")
        
        # Mise à jour des variables d'état
        self.cholesky_K = best_cholesky_K
        self.alpha_one_y = best_alpha
        self.sigma_k = best_sigma_k
        self.mean_p = best_mean
        
        return best_result
    
    def fit(self, X, y, multi_start=False, n_start=5, theta_0=None, var_noise_0=None, 
            theta_lb=None, theta_ub=None, var_lb=None, var_ub=None):
        """
        Ajuste le modèle aux données d'entraînement avec optimisation parallèle des hyperparamètres.
        
        :param X: Entrées d'entraînement.
        :param y: Sorties d'entraînement.
        :param multi_start: Activer l'initialisation multiple.
        :param n_start: Nombre de points de départ si multi_start est activé.
        :param theta_0: Valeurs initiales pour les hyperparamètres du noyau.
        :param var_noise_0: Valeur initiale pour la variance du bruit.
        :param theta_lb, theta_ub: Bornes pour les hyperparamètres du noyau.
        :param var_lb, var_ub: Bornes pour la variance du bruit.
        :return: self
        """
        # Formatage des entrées
        if X.ndim == 1:
            self.X_train = X.reshape(-1, 1)
        else:
            self.X_train = X
            
        self.y_train = y.reshape(-1, 1)
        
        # Début du chronométrage
        start_time = time.time()
        
        # Optimisation des hyperparamètres
        opt_result = self._optimize_hyperparameters(
            theta_0, var_noise_0, multi_start, n_start, 
            theta_lb, theta_ub, var_lb, var_ub
        )
        
        # Mise à jour des hyperparamètres
        if self.noisy_data:
            self.kernel.hyperparams = np.exp(opt_result.x[:-1])
            self.var_noise = opt_result.x[-1]
            self.hyperparameters = [np.exp(opt_result.x[:-1]), self.sigma_k**2, opt_result.x[-1]]
        else:
            self.kernel.hyperparams = np.exp(opt_result.x)
            self.hyperparameters = [np.exp(opt_result.x), self.sigma_k**2]
        
        # Affichage du temps total d'ajustement
        if self.verbose > 0:
            print(f"Temps total d'ajustement: {time.time() - start_time:.2f}s")
        
        return self
    
    def predict(self, X_star, return_cov=False):
        """
        Prédit les sorties pour de nouvelles entrées.
        
        :param X_star: Nouvelles entrées.
        :param return_cov: Si True, retourne la matrice de covariance complète, sinon retourne la variance.
        :return: Moyenne et variance/covariance des prédictions.
        """
        # Formatage des entrées
        if X_star.ndim == 1:
            X_star = X_star.reshape(-1, 1)
            
        # Calcul de la covariance entre les points d'entraînement et les nouveaux points
        K_star = self.kernel(self.X_train, X_star)
        
        # Récupération des paramètres sauvegardés
        L = self.cholesky_K
        mean_hat = self.mean_p
        alpha_one_y = self.alpha_one_y
        
        # Calcul de la moyenne prédictive
        mean = mean_hat + np.dot(K_star.T, alpha_one_y)
        
        # Calcul de la variance prédictive
        v = solve_triangular(L, K_star, lower=True)
        K_star_star = self.kernel(X_star, X_star)
        
        if return_cov:
            # Calcul de la matrice de covariance complète
            y_cov = (K_star_star - np.dot(v.T, v)) * self.sigma_k**2
            return mean.ravel(), y_cov
        else:
            # Calcul de la variance prédictive (diagonale de la covariance)
            y_var = np.maximum(0, np.diag(K_star_star - np.dot(v.T, v)) * self.sigma_k**2)
            return mean.ravel(), y_var