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
    Modèle de Régression par Processus Gaussien Multi-Sorties (MOGPR) utilisant
    le modèle de corégionalisation linéaire (LMC).
    """
    
    def __init__(self, kernel: LMCKernel, noise_variance: float = 1e-6, use_efficient_lik: bool = True,
                 parallel: bool = True, n_jobs: int = -1, verbose: int = 0):
        """
        Initialise le modèle MOGPR.
        
        Args:
            kernel: Noyau LMC
            noise_variance: Variance du bruit d'observation (sigma²)
            use_efficient_lik: Utiliser la méthode efficace pour calculer la log-vraisemblance
                             (uniquement pour les noyaux simples avec Q=1)
            parallel: Activer la parallélisation pour l'échantillonnage et autres opérations
            n_jobs: Nombre de jobs pour la parallélisation (-1 = tous les cœurs)
            verbose: Niveau de verbosité pour l'affichage des informations
        """
        self.kernel = kernel
        self.log_noise_variance = np.log(noise_variance)
        self.use_efficient_lik = use_efficient_lik
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
        # Configuration de la parallélisation
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Pour stockage efficace après ajustement
        self.L = None  # Facteur de Cholesky de la matrice de covariance
        self.alpha = None  # Solution du système linéaire
        self.eigendecomp_computed = False
        self.U_C = None  # Vecteurs propres de la matrice de corégionalisation
        self.S_C = None  # Valeurs propres de la matrice de corégionalisation
        self.U_R = None  # Vecteurs propres de la matrice de covariance spatiale
        self.S_R = None  # Valeurs propres de la matrice de covariance spatiale
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
        Fonction objectif pour l'optimisation des hyperparamètres.
        
        Args:
            params: Vecteur de paramètres concaténés [kernel_params, log_noise_variance]
            
        Returns:
            La log-vraisemblance négative
        """
        # Extraire les paramètres du noyau et la variance du bruit
        kernel_params = params[:-1]
        log_noise_variance = params[-1]
        
        # Mettre à jour les paramètres
        self.kernel.params = kernel_params
        self.log_noise_variance = log_noise_variance
        
        # Réinitialiser les décompositions précalculées
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        
        # Calculer la log-vraisemblance
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
        Gradient de la fonction objectif pour l'optimisation des hyperparamètres.
        
        Args:
            params: Vecteur de paramètres concaténés [kernel_params, log_noise_variance]
            
        Returns:
            Le gradient de la log-vraisemblance négative
        """
        # Extraire les paramètres du noyau et la variance du bruit
        kernel_params = params[:-1]
        log_noise_variance = params[-1]
        
        # Mettre à jour les paramètres
        self.kernel.params = kernel_params
        self.log_noise_variance = log_noise_variance
        
        # Réinitialiser les décompositions précalculées
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        
        # Calculer le gradient de la log-vraisemblance
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            return self._compute_log_likelihood_gradient_efficient(params, self.X_train, self.y_train)
        else : 
            return self._compute_log_likelihood_gradient_naive(params, self.X_train, self.y_train)

    def _optimize_single_start(self, initial_params, bounds, optimizer, maxiter):
        """
        Optimise les hyperparamètres à partir d'un point de départ.
        
        Args:
            initial_params: Paramètres initiaux
            bounds: Bornes pour l'optimisation
            optimizer: Méthode d'optimisation
            maxiter: Nombre maximum d'itérations
            
        Returns:
            Résultat de l'optimisation
        """
        start_time = time.time()
        
        if optimizer in ['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG'] and hasattr(self, '_compute_log_likelihood_gradient'):
            # Méthodes utilisant le gradient
            result = minimize(
                self._compute_log_likelihood_function,
                initial_params,
                method=optimizer,
                jac=self._compute_log_likelihood_gradient,
                bounds=bounds if optimizer == 'L-BFGS-B' else None,
                options={'maxiter': maxiter, 'disp': self.verbose > 0}
            )
        else:
            # Méthodes n'utilisant pas le gradient
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
        Ajuste le modèle aux données d'entraînement en optimisant les hyperparamètres.
        
        Args:
            X: Matrice d'entrée de forme (n, input_dim)
            y: Matrice de sortie de forme (n, output_dim)
            optimizer: Méthode d'optimisation à utiliser ('L-BFGS-B', 'BFGS', etc.)
            n_restarts: Nombre de redémarrages aléatoires pour l'optimisation
            maxiter: Nombre maximum d'itérations pour l'optimisation
            verbose: Afficher des informations pendant l'ajustement
            use_grad: Utiliser le gradient analytique pour l'optimisation
            theta_lb, theta_ub: Bornes inférieures et supérieures pour les hyperparamètres du noyau
            seed: Graine aléatoire pour la reproductibilité
            parallel_opt: Activer la parallélisation pour l'optimisation (si None, utilise self.parallel)
            use_init_pca: Utiliser l'initialisation PCA pour la matrice de corégionalisation
            
        Returns:
            self: Le modèle ajusté
        """
        # Mettre à jour le niveau de verbosité
        self.verbose = verbose
        
        # Déterminer si l'optimisation doit être parallélisée
        parallel_opt = self.parallel if parallel_opt is None else parallel_opt
        
        # Préparer les données
        X_stacked, y_stacked = self._prepare_data(X, y)
        self.X_train = X_stacked
        self.y_train = y_stacked
        
        # Obtenir les bornes pour tous les paramètres
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

        bounds = self.kernel.bounds + [(-10.0, 10.0)]  # Ajout des bornes pour log_noise_variance

        # Initialiser la meilleure log-vraisemblance à une valeur élevée
        best_nll = np.inf
        best_params = None
        #np.random.seed(seed)
        
        # Génération des points de départ
        initial_params = []
        
        if not use_init_pca:
            # Méthode standard: 1er point = params actuels, autres = LHS complet
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
            # Méthode PCA: Initialiser L via PCA sur y, randomiser le reste
            if not hasattr(self.kernel, "init_L_from_pca"):
                raise ValueError("Kernel does not support PCA initialization")
            
            # Init L from PCA using original y (expected by init_L_from_pca)
            self.kernel.init_L_from_pca(y)
            
            current_kernel_params = self.kernel.params.copy()
            
            # Indices pour séparer L (fixé par PCA) des autres (à randomiser)
            # Hypothèse: kernel.params = [Lq_unit, sigma_B, spatial]
            # start_idx_Lq marque la fin de Lq_unit
            idx_spatial_start = self.kernel.start_idx_Lq
            idx_spatial_end = len(current_kernel_params)
            
            spatial_bounds = bounds[idx_spatial_start:idx_spatial_end]
            noise_bounds = [(-10.0, 0.0)]
            
            # Restart 0: PCA + params courants (pour sigma_B et spatial)
            p0 = np.concatenate([current_kernel_params, [self.log_noise_variance]])
            initial_params.append(p0)
            
            if n_restarts > 1:
                n_spatial = idx_spatial_end - idx_spatial_start
                # LHS sur (sigma_B + spatial) + noise
                lhs = LatinHypercube(d=n_spatial + 1, seed=seed)
                samples = lhs.random(n=n_restarts - 1)
                
                lb = np.array([b[0] for b in spatial_bounds] + [noise_bounds[0][0]])
                ub = np.array([b[1] for b in spatial_bounds] + [noise_bounds[0][1]])
                
                lhs_params = lb + samples * (ub - lb)
                
                for k in range(n_restarts - 1):
                    params = p0.copy()
                    # Remplacer la partie (sigma_B + spatial) par LHS
                    params[idx_spatial_start:idx_spatial_end] = lhs_params[k, :-1]
                    # Remplacer le bruit
                    params[-1] = lhs_params[k, -1]
                    initial_params.append(params)
        
        # Optimisation parallèle ou séquentielle
        if parallel_opt and n_restarts > 1:
            if self.verbose > 0:
                print(f"Exécution parallèle de l'optimisation avec {n_restarts} points de départ")
            
            try:
                results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._optimize_single_start)(params, bounds, optimizer, maxiter)
                    for params in initial_params
                )
                
                # Extraire les résultats et les durées
                opt_results = [res[0] for res in results]
                durations = [res[1] for res in results]
                
                # Trouver le meilleur résultat
                best_idx = np.argmin([res.fun for res in opt_results])
                best_result = opt_results[best_idx]
                best_params = best_result.x
                best_nll = best_result.fun
                
                if self.verbose > 0:
                    for i, (res, dur) in enumerate(zip(opt_results, durations)):
                        print(f"Run {i+1}/{n_restarts}: nLL = {res.fun:.6f}, durée = {dur:.2f}s")
                    print(f"Meilleur run: {best_idx+1}/{n_restarts}, nLL = {best_nll:.6f}")
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Erreur lors de la parallélisation: {str(e)}. Utilisation de l'optimisation séquentielle.")
                
                # Repli sur l'optimisation séquentielle
                for i, init_params in enumerate(initial_params):
                    if self.verbose > 0:
                        print(f"Optimisation {i+1}/{n_restarts}")
                    
                    result, duration = self._optimize_single_start(init_params, bounds, optimizer, maxiter)
                    
                    if self.verbose > 0:
                        print(f"Run {i+1}/{n_restarts}: nLL = {result.fun:.6f}, durée = {duration:.2f}s")
                    
                    if result.fun < best_nll:
                        best_nll = result.fun
                        best_params = result.x
        else:
            # Optimisation séquentielle
            for i, init_params in enumerate(initial_params):
                if self.verbose > 0:
                    print(f"Optimisation {i+1}/{n_restarts}")
                
                result, duration = self._optimize_single_start(init_params, bounds, optimizer, maxiter)
                
                if self.verbose > 0:
                    print(f"Run {i+1}/{n_restarts}: nLL = {result.fun:.6f}, durée = {duration:.2f}s")
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
        
        # Définir les paramètres optimaux
        if best_params is not None:
            self.kernel.params = best_params[:-1]
            self.log_noise_variance = best_params[-1]
            
            if self.verbose > 0:
                print(f"Hyperparamètres optimaux: {best_params[:-1]}, log_noise = {best_params[-1]}")
        
        # Calculer à nouveau avec les meilleurs paramètres
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            self._compute_kernel_eigendecomposition(X_stacked)
            self._compute_log_likelihood_efficient(best_params, X_stacked, y_stacked)
        else:
            self._compute_log_likelihood_naive(best_params, X_stacked, y_stacked)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_test: np.ndarray, return_cov: bool = True, full_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit la moyenne et la variance pour les points de test.
        
        Args:
            X_test: Points de test, matrice de forme (n_test, input_dim)
            return_cov: Si True, retourne la matrice de covariance complète
            full_cov: Si True et return_cov=True, retourne la matrice de covariance complète,
                    sinon retourne uniquement les variances
            
        Returns:
            y_pred: Moyenne prédite de forme (n_test, output_dim)
            var_pred: Variance prédite de forme (n_test, output_dim) si full_cov=False,
                     sinon matrice de covariance de forme (n_test * output_dim, n_test * output_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant de faire des prédictions.")
        
        # Préparer les données de test
        X_test_stacked, _ = self._prepare_data(X_test)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        n = int(self.X_train.shape[0]/output_dim)
        
        if not self.use_efficient_lik:
            # Calculer la covariance entre les points d'entraînement et de test
            K_star = self.kernel(self.X_train, X_test_stacked)
            # Calculer la moyenne prédite
            f_pred = K_star.T @ self.alpha
        else : 
            X_train_spatial = self.X_train[:n, :-1]
            K_x_star = self.kernel.base_kernels[0](X_test, X_train_spatial)
            A = self.kernel.get_B(0).dot(self.U_C)
            B = K_x_star.dot(self.U_R)
            f_pred = B.dot(self.Ytilde.reshape(n, output_dim, order='F')).dot(A.T).flatten(order='F')
            
        # Reformater la moyenne prédite
        y_pred = f_pred.reshape(output_dim, n_test).T
        
        # Calculer la variance prédite
        if return_cov:
            if not self.use_efficient_lik:
                if self.L is None:
                    # Calculer la décomposition de Cholesky si nécessaire
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
            
                # Résoudre le système K_noisy^(-1) @ K_star
                v = linalg.solve_triangular(self.L, K_star, lower=True)
                
                # Calculer la covariance prédictive
                K_test_test = self.kernel(X_test_stacked)
                
                #if full_cov:
                    # Covariance complète
                var_pred = K_test_test - v.T @ v
            else:
                if self.L is None:
                    # Calculer la décomposition de Cholesky si nécessaire
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
            
                # Résoudre le système K_noisy^(-1) @ K_star
                K_star = self.kernel(self.X_train, X_test_stacked)
                v = linalg.solve_triangular(self.L, K_star, lower=True)
                
                # Calculer la covariance prédictive
                K_test_test = self.kernel(X_test_stacked)
                
                #if full_cov:
                    # Covariance complète
                var_pred = K_test_test - v.T @ v
                # else:
                #     # Diagonale seulement (variances)
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
            # Par défaut, retourner seulement les variances prédites
            if not self.use_efficient_lik:
                if self.L is None:
                    # Calculer la décomposition de Cholesky si nécessaire
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)
                
                # Résoudre le système K_noisy^(-1) @ K_star
                v = linalg.solve_triangular(self.L, K_star, lower=True)

                # Calculer les variances prédites (diagonale de la covariance)
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
        Génère un seul échantillon à partir de la distribution prédictive.
        
        Args:
            X_test: Points de test
            y_mean: Moyenne prédite
            y_cov: Covariance prédite
            random_state: Graine aléatoire
            
        Returns:
            Un échantillon de la distribution
        """
        rng = np.random.RandomState(random_state)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Reformater la moyenne
        y_mean_flat = y_mean.T.flatten(order='F')
        
        # Générer un échantillon
        sample_flat = rng.multivariate_normal(y_mean_flat, y_cov)
        
        # Reformater l'échantillon
        sample = sample_flat.reshape(output_dim, n_test, order='F').T
        
        return sample
    
    def sample_y(self, X_test: np.ndarray, n_samples: int = 10, random_state: Optional[int] = None,
                 parallel: Optional[bool] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Échantillonne à partir de la distribution prédictive, avec support pour la parallélisation.
        
        Args:
            X_test: Points de test, matrice de forme (n_test, input_dim)
            n_samples: Nombre d'échantillons à générer
            random_state: Graine aléatoire pour la reproductibilité
            parallel: Activer la parallélisation (si None, utilise self.parallel)
            batch_size: Nombre d'échantillons par lot pour la parallélisation
            
        Returns:
            samples: Échantillons de forme (n_test, output_dim, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant d'échantillonner.")
        
        start_time = time.time()
        
        # Déterminer si l'échantillonnage doit être parallélisé
        parallel = self.parallel if parallel is None else parallel
        
        # Prédire la moyenne et la covariance
        y_mean, y_cov = self.predict(X_test, return_cov=True, full_cov=True)
        print("y_cov shape : ", y_cov.shape)
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Configurer le générateur aléatoire principal
        if random_state is not None:
            main_rng = np.random.RandomState(random_state)
            # Générer des graines pour chaque échantillon
            seeds = main_rng.randint(0, 2**32 - 1, size=n_samples)
        else:
            seeds = [None] * n_samples
        
        # Déterminer la taille des lots si nécessaire
        if parallel and batch_size is None:
            # Déterminer la taille des lots en fonction du nombre d'échantillons et de cœurs
            if self.n_jobs <= 0:
                import multiprocessing
                n_cores = multiprocessing.cpu_count()
            else:
                n_cores = min(self.n_jobs, n_samples)
                
            batch_size = max(1, n_samples // n_cores)
        
        if self.verbose > 0:
            print(f"Génération de {n_samples} échantillons pour {n_test} points de test...")
        
        if parallel and n_samples > 1:
            try:
                # Génération parallèle des échantillons
                if self.verbose > 0:
                    print(f"Échantillonnage parallèle avec {self.n_jobs} jobs")
                
                # Utiliser joblib pour paralléliser
                samples_list = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._generate_single_sample)(X_test, y_mean, y_cov, seed)
                    for seed in seeds
                )
                
                # Combiner les échantillons
                samples = np.stack(samples_list, axis=2)
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Erreur lors de la parallélisation: {str(e)}. Utilisation de l'échantillonnage séquentiel.")
                
                # Repli sur la génération séquentielle
                samples = np.zeros((n_test, output_dim, n_samples))
                for i, seed in enumerate(seeds):
                    samples[:, :, i] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
        else:
            # Génération séquentielle des échantillons
            samples = np.zeros((n_test, output_dim, n_samples))
            for i, seed in enumerate(seeds):
                samples[:, :, i] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
        
        if self.verbose > 0:
            duration = time.time() - start_time
            print(f"Échantillonnage terminé en {duration:.2f} secondes")
        
        return samples
    
    def sample_y_batch(self, X_test: np.ndarray, n_samples: int = 10, batch_size: int = 10, 
                       random_state: Optional[int] = None) -> np.ndarray:
        """
        Échantillonne à partir de la distribution prédictive en utilisant une approche par lots.
        Cette méthode est utile pour les grands ensembles de données où générer tous les échantillons
        simultanément pourrait dépasser la mémoire disponible.
        
        Args:
            X_test: Points de test, matrice de forme (n_test, input_dim)
            n_samples: Nombre total d'échantillons à générer
            batch_size: Nombre d'échantillons par lot
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            samples: Échantillons de forme (n_test, output_dim, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant d'échantillonner.")
        
        start_time = time.time()
        
        # Prédire la moyenne et la covariance
        y_mean, y_cov = self.predict(X_test, return_cov=True, full_cov=True)
        
        n_test = X_test.shape[0]
        output_dim = self.kernel.output_dim
        
        # Initialiser le tableau de sortie
        samples = np.zeros((n_test, output_dim, n_samples))
        
        # Configurer le générateur aléatoire principal
        if random_state is not None:
            main_rng = np.random.RandomState(random_state)
        else:
            main_rng = np.random.RandomState()
        
        # Calculer le nombre de lots
        n_batches = int(np.ceil(n_samples / batch_size))
        
        if self.verbose > 0:
            print(f"Génération de {n_samples} échantillons en {n_batches} lots de {batch_size}")
        
        # Générer les échantillons par lots
        for i in range(n_batches):
            # Déterminer l'indice de début et de fin pour ce lot
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            if self.verbose > 1:
                print(f"Génération du lot {i+1}/{n_batches} ({current_batch_size} échantillons)")
            
            # Générer des graines pour ce lot
            batch_seeds = main_rng.randint(0, 2**32 - 1, size=current_batch_size)
            
            # Générer des échantillons pour ce lot
            if self.parallel:
                batch_samples = Parallel(n_jobs=self.n_jobs, verbose=max(0, self.verbose-1))(
                    delayed(self._generate_single_sample)(X_test, y_mean, y_cov, seed)
                    for seed in batch_seeds
                )
                # Convertir la liste en tableau
                batch_samples = np.stack(batch_samples, axis=2)
            else:
                batch_samples = np.zeros((n_test, output_dim, current_batch_size))
                for j, seed in enumerate(batch_seeds):
                    batch_samples[:, :, j] = self._generate_single_sample(X_test, y_mean, y_cov, seed)
            
            # Stocker les échantillons de ce lot
            samples[:, :, start_idx:end_idx] = batch_samples
        
        if self.verbose > 0:
            duration = time.time() - start_time
            print(f"Échantillonnage par lots terminé en {duration:.2f} secondes")
        
        return samples
    
    def log_marginal_likelihood(self) -> float:
        """
        Calcule la log-vraisemblance marginale du modèle avec les hyperparamètres actuels.
        
        Returns:
            log_likelihood: Log-vraisemblance marginale
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant de calculer la log-vraisemblance.")
        
        # Construct params vector
        params = np.concatenate([self.kernel.params, [self.log_noise_variance]])
        
        # Utiliser la méthode efficace si possible
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1 and self.eigendecomp_computed:
            return -self._compute_log_likelihood_efficient(params, self.X_train, self.y_train)
        else:
            return -self._compute_log_likelihood_naive(params, self.X_train, self.y_train)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Retourne la log-vraisemblance marginale sur les données de test.
        
        Args:
            X: Matrice d'entrée de forme (n, input_dim)
            y: Matrice de sortie de forme (n, output_dim)
            
        Returns:
            log_likelihood: Log-vraisemblance marginale
        """
        X_stacked, y_stacked = self._prepare_data(X, y)
        
        # Sauvegarder temporairement les données d'entraînement
        X_train_orig, y_train_orig = self.X_train, self.y_train
        
        # Utiliser les données de test
        self.X_train, self.y_train = X_stacked, y_stacked
        
        # Réinitialiser les décompositions précalculées
        L_orig, alpha_orig = self.L, self.alpha
        eigendecomp_computed_orig = self.eigendecomp_computed
        self.L, self.alpha = None, None
        self.eigendecomp_computed = False
        
        # Construct params vector
        params = np.concatenate([self.kernel.params, [self.log_noise_variance]])
        
        # Calculer la log-vraisemblance
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            score = -self._compute_log_likelihood_efficient(params, X_stacked, y_stacked)
        else:
            score = -self._compute_log_likelihood_naive(params, X_stacked, y_stacked)
        
        # Restaurer les données d'entraînement et les décompositions
        self.X_train, self.y_train = X_train_orig, y_train_orig
        self.L, self.alpha = L_orig, alpha_orig
        self.eigendecomp_computed = eigendecomp_computed_orig
        
        return score