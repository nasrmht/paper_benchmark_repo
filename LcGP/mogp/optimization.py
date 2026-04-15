import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from scipy.stats.qmc import LatinHypercube 
from .kernels.LMCKernel import LMCKernel
from ..utils.data_utils import compute_kernel_eigendecomposition, prepare_data
from .likelihoods.log_likelihood_efficient import compute_log_likelihood_efficient, compute_log_likelihood_gradient_efficient
from .likelihoods.log_likelihood_naive import compute_log_likelihood_naive, compute_log_likelihood_gradient_naive
#from ..utils import efficient_alpha_kronecker
from joblib import Parallel, delayed


class MOGPR:
    """
    Modèle de Régression par Processus Gaussien Multi-Sorties (MOGPR) utilisant
    le modèle de corégionalisation linéaire (LMC).
    """
    
    def __init__(self, kernel: LMCKernel, noise_variance: float = 1e-6, use_efficient_lik: bool = True):
        """
        Initialise le modèle MOGPR.
        
        Args:
            kernel: Noyau LMC
            noise_variance: Variance du bruit d'observation (sigma²)
            use_efficient_lik: Utiliser la méthode efficace pour calculer la log-vraisemblance
                             (uniquement pour les noyaux simples avec Q=1)
        """
        self.kernel = kernel
        #self.kernel_0 = kernel
        self.log_noise_variance = np.log(noise_variance)
        self.use_efficient_lik = use_efficient_lik
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
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
    
    
    def _compute_log_likelihood_naive(self, params, X: np.ndarray, y: np.ndarray):
        nll, L, alpha =  compute_log_likelihood_naive(self.kernel,params,X, y, self.log_noise_variance)
        self.L = L
        self.alpha = alpha
        return nll
    
    
    def _compute_log_likelihood_gradient_naive(self,params, X: np.ndarray, y: np.ndarray):
        return compute_log_likelihood_gradient_naive(self.kernel,params,X, y, self.log_noise_variance)
    
    
    def _compute_log_likelihood_efficient(self,params, X: np.ndarray, y: np.ndarray):

        eigendecomp_computed = None
        if not self.eigendecomp_computed:
            eigendecomp_computed = self._compute_kernel_eigendecomposition(X)
        else : 
            eigendecomp_computed = self.U_C, self.S_C, self.U_R, self.S_R
        
        if eigendecomp_computed is None :
            raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood is not possible")
        
        nll, self.S_kron_inv, self.Ytilde = compute_log_likelihood_efficient(self.kernel,params,X, y, self.log_noise_variance, eigendecomp_computed)
        return nll

    
    def _compute_log_likelihood_gradient_efficient(self,params, X: np.ndarray, y: np.ndarray):
        if not self.eigendecomp_computed:
            eigendecomp_computed = compute_kernel_eigendecomposition(self.kernel, X)
        else : 
            eigendecomp_computed = self.U_C, self.S_C, self.U_R, self.S_R


        if eigendecomp_computed is None :
            raise ValueError("Eigendecomp is not possible for base kernel with len > 1, therefore efficient_likelihood grad is not possible")

        return compute_log_likelihood_gradient_efficient(self.kernel,params, X, y, self.log_noise_variance, eigendecomp_computed)
    
    
    def _compute_log_likelihood_function(self, params: np.ndarray) -> float:
        """
        Fonction objectif pour l'optimisation des hyperparamètres.
        
        Args:
            params: Vecteur de paramètres concaténés [kernel_params, log_noise_variance]
            
        Returns:
            La log-vraisemblance négative
        """
        # # Extraire les paramètres du noyau et la variance du bruit
        # kernel_params = params[:-1]
        # log_noise_variance = params[-1]
        
        #print("params nll:", params)
        # Mettre à jour les paramètres
        # self.kernel.params = kernel_params
        # self.log_noise_variance = log_noise_variance
        
        # Réinitialiser les décompositions précalculées
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        # Calculer la log-vraisemblance
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            res = self._compute_log_likelihood_efficient(params,self.X_train, self.y_train)
            #print("res : ",res)
            return res
        else:
            res = self._compute_log_likelihood_naive(params,self.X_train, self.y_train)
            #print("res : ",res)
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
    #     kernel_params = params[:-1]
    #     log_noise_variance = params[-1]
        
    #    # print("nll params : ",params)
    #     # Mettre à jour les paramètres
    #     self.kernel.params = kernel_params
    #     self.log_noise_variance = log_noise_variance
        
        # Réinitialiser les décompositions précalculées
        self.L = None
        self.alpha = None
        self.eigendecomp_computed = False
        # Calculer le gradient de la log-vraisemblance
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            return self._compute_log_likelihood_gradient_efficient(params,self.X_train, self.y_train)
        else : 
            return self._compute_log_likelihood_gradient_naive(params,self.X_train, self.y_train)
    
    def _run_single_optimization(self, initial_params, bounds, optimizer, maxiter, use_grad, verbose):
        """
        Méthode auxiliaire exécutée par les workers parallèles.
        """
       # print("initial_params : ", initial_params[:4], "len : ", len(initial_params))
        if optimizer in ['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG']:
            # Méthodes utilisant le gradient
            result = minimize(
                self._compute_log_likelihood_function,
                initial_params,
                method=optimizer,
                jac=self._compute_log_likelihood_gradient if use_grad else None,
                bounds=bounds if optimizer == 'L-BFGS-B' else None,
                options={'maxiter': maxiter, 'disp': verbose}
            )
        else:
            # Méthodes n'utilisant pas le gradient
            result = minimize(
                self._compute_log_likelihood_function,
                initial_params,
                method=optimizer,
                bounds=bounds,
                options={'disp': verbose}
            )
        return result

    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: str = 'L-BFGS-B', 
            n_restarts: int = 1, maxiter: int = 100, verbose: bool = True, 
            use_grad: bool = True, theta_lb=None, theta_ub=None, 
            seed=42, hyp_optimize=True, n_jobs: int = -1,use_init_pca: bool=True) -> 'MOGPR':
        """
        Ajuste le modèle avec parallélisation.
        Args:
            n_jobs: Nombre de cœurs à utiliser (-1 pour tous les cœurs disponibles)
        """
        # Préparer les données
        X_stacked, y_stacked = self._prepare_data(X, y)
        self.X_train = X_stacked
        self.y_train = y_stacked

        # Si pas d'optimisation, on marque juste comme fitté et on sort
        if not hyp_optimize:
            self.is_fitted = True
            return self

        # --- 1. Configuration des bornes (Identique à ton code) ---
        for j in range(len(self.kernel.base_kernels)):
            len_theta = self.kernel.base_kernels[j].get_n_params()
            Theta_lb = theta_lb if theta_lb is not None else 1e-3*np.ones(len_theta)
            Theta_ub = theta_ub if theta_ub is not None else 10*(np.max(X, axis=0) - np.min(X,axis=0)) 
            if len_theta == 1 and theta_ub is None:
                Theta_ub = 10*(np.max(X) - np.min(X))
            
            custom_bounds = np.vstack([np.log(Theta_lb), np.log(Theta_ub)]).T
            self.kernel.base_kernels[j]._bounds = [tuple(sous_bounds) for sous_bounds in custom_bounds]

        bounds = self.kernel.bounds + [(-10.0, 10.0)]  # Bornes log_noise
        #print("Overall bounds: ", bounds)
        # --- 2. Génération de TOUS les points de départ ---
        #np.random.seed(seed)
        rng = np.random.default_rng(seed)
        start_points_list = []
        if use_init_pca==False:
            # Point de départ 1 : Paramètres actuels
            current_params = np.concatenate([self.kernel.params, np.array([self.log_noise_variance])])
            start_points_list.append(current_params)

            # Points de départ suivants : LHS
            if n_restarts > 1:
                bounds_low = np.array([b[0] for b in self.kernel.bounds])
                bounds_high = np.array([b[1] for b in self.kernel.bounds])
                n_params = len(bounds_low)

                # LHS Kernel params
                lhs = LatinHypercube(d=n_params, seed=seed)
                lhs_samples = lhs.random(n=n_restarts-1)
                initial_kernel_params = bounds_low + lhs_samples * (bounds_high - bounds_low)

                # LHS Noise params
                noise_low, noise_high = -10.0, 0.0
                lhs_noise = LatinHypercube(d=1, seed=seed)
                noise_samples = lhs_noise.random(n=n_restarts-1)
                initial_log_noise = noise_low + noise_samples * (noise_high - noise_low)

                # Concaténation et ajout à la liste
                initial_params_lhs = np.hstack([initial_kernel_params, initial_log_noise])
                for params in initial_params_lhs:
                    start_points_list.append(params)
        else :
            if not hasattr(self.kernel, "init_L_from_pca"):
                raise ValueError("Kernel does not support PCA initialization")
            
            self.kernel.init_L_from_pca(y)

            kernel_params = self.kernel.params.copy()

            # Indices
            idx_L_start = 0
            idx_L_end = self.kernel.start_idx_Lq   # exclus
            idx_spatial_start = self.kernel.start_idx_Lq
            idx_spatial_end = len(kernel_params)

            # bounds = self.kernel.bounds
            
            spatial_bounds = bounds[idx_spatial_start:idx_spatial_end]
            noise_bounds = [(-10.0, 0.0)]

            # --- Restart 0 : PCA + params courants ---
            p0 = np.concatenate([kernel_params, [self.log_noise_variance]])
            start_points_list.append(p0)

            if n_restarts > 1:
                n_spatial = idx_spatial_end - idx_spatial_start

                lhs = LatinHypercube(d=n_spatial + 1, seed=seed)
                samples = lhs.random(n=n_restarts - 1)

                # Bornes
                lb = np.array([b[0] for b in spatial_bounds] + [noise_bounds[0][0]])
                ub = np.array([b[1] for b in spatial_bounds] + [noise_bounds[0][1]])

                lhs_params = lb + samples * (ub - lb)

                for k in range(n_restarts - 1):
                    params = p0.copy()

                    # Remplir UNIQUEMENT les hyperparams spatiaux
                    params[idx_spatial_start:idx_spatial_end] = lhs_params[k, :-1]

                    # Bruit
                    params[-1] = lhs_params[k, -1]

                    start_points_list.append(params)
            
 
        # --- 3. Exécution Parallèle ---
        if verbose:
            print(f"Lancement de l'optimisation avec {n_restarts} redémarrages sur {n_jobs} jobs...")

        # Utilisation de joblib pour paralléliser
        # Note: verbose à l'intérieur des workers peut être confus, on le désactive souvent
        worker_verbose = verbose if n_jobs == 1 else False 
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_optimization)(
                params, bounds, optimizer, maxiter, use_grad, worker_verbose
            ) 
            for params in start_points_list
        )

        # --- 4. Sélection du meilleur résultat ---
        # On cherche celui qui a la plus petite valeur de fonction (fun = Negative Log Likelihood)
        best_result = min(results, key=lambda res: res.fun)
        
        best_nll = best_result.fun
      #  print("seed = ", seed, "best_nll = ", best_nll) #, " start_points list : ", start_points_list) "best_nll = ", [res.fun for res in results],
        best_params = best_result.x

        if verbose:
            print(f"Meilleure NLL trouvée : {best_nll}")

        # --- 5. Finalisation (Identique à ton code) ---
        if best_params is not None:
            self.kernel.params = best_params[:-1]
            self.log_noise_variance = best_params[-1]
        
        # Recalcul final des matrices avec les meilleurs paramètres
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            self._compute_kernel_eigendecomposition(X_stacked)
            self._compute_log_likelihood_efficient(best_params,X_stacked, y_stacked)
        else:
            self._compute_log_likelihood_naive(best_params,X_stacked, y_stacked)
        
        self.is_fitted = True
        return self
    
    
    
    def predict(self, X_test: np.ndarray, return_cov: bool = False, full_cov: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
            K_x_star = self.kernel.base_kernels[0](X_test,X_train_spatial)
            A = self.kernel.get_B(0).dot(self.U_C)
            B = K_x_star.dot(self.U_R)
            f_pred = B.dot(self.Ytilde.reshape(n, output_dim, order='F')).dot(A.T).flatten(order='F')
            
        
        # Reformater la moyenne prédite
        y_pred = f_pred.reshape(output_dim, n_test).T #, order='F'
        
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
                
                if full_cov:
                    # Covariance complète
                    var_pred = K_test_test - v.T @ v
                else:
                    # Diagonale seulement (variances)
                    var_pred = np.diag(K_test_test) - np.diag(v.T@v)
                    var_pred = var_pred.reshape(output_dim, n_test).T #, order='F').T
            else:
                X_train_spatial = self.X_train[:n, :-1]
                K_x_star = self.kernel.base_kernels[0](X_test,X_train_spatial)
                
                #print("shape K_x star : ", K_star.shape)
                
                A = self.kernel.get_B(0).dot(self.U_C)
                B = K_x_star.dot(self.U_R)

                k_C_xx = np.diag(self.kernel.get_B(0))
                k_R_xx = np.diag(self.kernel.base_kernels[0](X_test)) # 2.Kdiag(X2new)
                if full_cov:
                   # K_star = np.kron(self.kernel.get_B(0), K_x_star)
                    K_star = self.kernel(self.X_train, X_test_stacked)
                    K = self.kernel(self.X_train)
                    noise_variance = np.exp(self.log_noise_variance)
                    K_noisy = K + noise_variance * np.eye(K.shape[0])
                    self.L = linalg.cholesky(K_noisy, lower=True)

                    v = linalg.solve_triangular(self.L, K_star, lower=True)
                    K_xx_star = self.kernel(X_test_stacked, X_test_stacked)
                    #print("v shape kron : ", K_star.shape)
                    var_pred = K_xx_star-v.T@v
                    #f_pred = B.dot(self.Ytilde.reshape(n, output_dim, order='F')).dot(A.T).flatten(order='F')
                else :
                    BA = np.kron(A, B)
                    var_pred = (np.kron(k_C_xx, k_R_xx) - np.sum(BA**2*self.S_kron_inv, 1) + np.exp(self.log_noise_variance)).reshape(n_test, output_dim,order='F').T
        
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
                K_test_diag = np.diag(self.kernel(X_test_stacked,X_test_stacked) ) #np.zeros(X_test_stacked.shape[0])
                # print("shape x_teststack : ",X_test_stacked)
                # for i in range(X_test_stacked.shape[0]):
                #     K_test_diag[i] = self.kernel(X_test_stacked[i:i+1], X_test_stacked[i:i+1])#[0, 0]


                
                
                var_pred = K_test_diag - np.diag(v.T@v)
                var_pred = var_pred.reshape(output_dim, n_test).T 
            else :
                X_train_spatial = self.X_train[:n, :-1]
                K_x_star = self.kernel.base_kernels[0](X_test,X_train_spatial)
                A = self.kernel.get_B(0).dot(self.U_C)
                B = K_x_star.dot(self.U_R)

                k_C_xx = np.diag(self.kernel.get_B(0))
                k_R_xx = np.diag(self.kernel.base_kernels[0](X_test)) # 
                BA = np.kron(B, A)
                var_pred = (np.kron(k_C_xx, k_R_xx) - np.sum(BA**2*self.S_kron_inv, 1) + np.exp(self.log_noise_variance)).reshape(output_dim, n_test, order='F').T

        return y_pred, var_pred
    
    def sample_y(self, X_test: np.ndarray, n_samples: int = 10, random_state: Optional[int] = 42) -> np.ndarray:
        """
        Échantillonne à partir de la distribution prédictive.
        
        Args:
            X_test: Points de test, matrice de forme (n_test, input_dim)
            n_samples: Nombre d'échantillons à générer
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            samples: Échantillons de forme (n_test, output_dim, n_samples)
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant d'échantillonner.")
        
        # Définir le générateur aléatoire
        #rng = np.random.RandomState(random_state)
        np.random.seed(random_state)
        n = self.X_train.shape[0]
        X_train_spatial = self.X_train[:n, :-1]
        n_test = X_test.shape[0]
        output_dim =self.kernel.output_dim
        rang = self.kernel.rank[0]
        B = self.kernel.get_L(0)*np.sqrt(self.kernel.get_sigma_B(0))
        #B_p = np.linalg.cholesky(self.kernel.get_B(0)+1e-6*np.eye(output_dim))
        #print("sigma B : ", self.kernel.get_sigma_B(0))
        L = linalg.cholesky(self.kernel.base_kernels[0](X_train_spatial)+1e-6*np.eye(n),lower=True)

        K_star = self.kernel.base_kernels[0](X_test, X_train_spatial).T
        v = linalg.solve_triangular(L, K_star, lower=True)
        K_xx_star = self.kernel.base_kernels[0](X_test, X_test)

        #print("K_star shape : ", K_star.shape)
        var_pred = K_xx_star-v.T@v

        LL = np.linalg.cholesky(var_pred+1e-6*np.eye(n_test))
        # Prédire la moyenne et la covariance
        y_mean, y_cov = self.predict(X_test, return_cov=False)

        #Z = np.random.randn(n_test, rang, n_samples)
        samples = np.zeros((n_test,self.kernel.output_dim, n_samples))
        for i in range(n_samples):
             #print("shape zL :",  (Z[:,:,i].T@L.T).shape)
            z_i = np.random.randn(n_test, rang)
            samples[:,:,i] = y_mean + (B@ (LL@z_i).T).T  #Z[:,:,i]

        #samples = np.einsum('tn,npi,pi->npi', L.T, Z, B) # +B[...,np.newaxis]@Z[...,:]@L[...,np.newaxis].T
        #samples = np.einsum('or,nri,nt->noi', B, Z, L.T)

        
        # n_test = X_test.shape[0]
        # output_dim = self.kernel.output_dim
        
        # # Reformater la moyenne
        # y_mean_flat = y_mean.T.flatten(order='F')
        
        # # Générer des échantillons à partir de la distribution normale multivariée
        # samples_flat = rng.multivariate_normal(y_mean_flat, y_cov, n_samples).T
        
        # # Reformater les échantillons de forme (n_test * output_dim, n_samples)
        # # à (n_test, output_dim, n_samples)
        # samples = samples_flat.reshape(output_dim, n_test, n_samples, order='F').transpose(1, 0, 2)
        
        return samples
    
    def log_marginal_likelihood(self) -> float:
        """
        Calcule la log-vraisemblance marginale du modèle avec les hyperparamètres actuels.
        
        Returns:
            log_likelihood: Log-vraisemblance marginale
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être ajusté avant de calculer la log-vraisemblance.")
        
        # Utiliser la méthode efficace si possible
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1 and self.eigendecomp_computed:
            return -self._compute_log_likelihood_efficient(self.X_train, self.y_train)
        else:
            return -self._compute_log_likelihood_naive(self.X_train, self.y_train)
    
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
        
        # Calculer la log-vraisemblance
        if self.use_efficient_lik and len(self.kernel.base_kernels) == 1:
            score = -self._compute_log_likelihood_efficient(X_stacked, y_stacked)
        else:
            score = -self._compute_log_likelihood_naive(X_stacked, y_stacked)
        
        # Restaurer les données d'entraînement et les décompositions
        self.X_train, self.y_train = X_train_orig, y_train_orig
        self.L, self.alpha = L_orig, alpha_orig
        self.eigendecomp_computed = eigendecomp_computed_orig
        
        return score