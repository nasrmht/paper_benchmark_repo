import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import qmc
from joblib import Parallel, delayed

class FastLMCKernel:
    """
    Gère la structure triangulaire inférieure (Cholesky) pour A.
    A = L (Lower Triangular)
    """
    def __init__(self, n_features, p_components, kernel_type='rbf'):
        self.d = n_features
        self.p = p_components
        self.kernel_type = kernel_type
        
        # Nombre d'éléments dans la partie triangulaire inférieure (inclus diagonale)
        self.n_L_params = (self.p * (self.p + 1)) // 2

    def get_parameter_bounds(self, X_min, X_max):
        """
        Définit les bornes :
        - Elements diagonaux de L : [1e-5, 10.0] (Positivité = Inversibilité)
        - Elements hors-diagonale de L : [-10.0, 10.0]
        - Log_Lengthscales : Logarithme des bornes physiques
        """
        bounds_L = []
        
        # On parcourt la matrice L ligne par ligne
        for i in range(self.p):
            for j in range(i + 1):
                if i == j:
                    # Diagonale : Doit être strictement positive
                    bounds_L.append((1e-5, 5.0))
                else:
                    # Hors-diagonale (Triangle inférieur)
                    bounds_L.append((-5.0, 5.0))
        
        # Bornes pour les LOG-lengthscales
        ranges = X_max - X_min
        ranges[ranges == 0] = 1.0 
        
        bounds_log_lengthscales = []
        for _ in range(self.p): 
            for dim in range(self.d):
                lower_phys = 1e-3
                upper_phys = 2.0 * ranges[dim]
                bounds_log_lengthscales.append((np.log(lower_phys), np.log(upper_phys)))
                
        return bounds_L + bounds_log_lengthscales

    def unpack_params(self, theta):
        """
        Reconstruit la matrice L (Triangulaire Inf) et les lengthscales.
        """
        # 1. Reconstruction de L
        L_params = theta[:self.n_L_params]
        L = np.zeros((self.p, self.p))
        
        idx = 0
        for i in range(self.p):
            for j in range(i + 1):
                L[i, j] = L_params[idx]
                idx += 1
                
        # 2. Extraction des lengthscales (Log -> Exp)
        log_ls_params = theta[self.n_L_params:]
        lengthscales = np.exp(log_ls_params).reshape(self.p, self.d)
        
        return L, lengthscales

class FastLMCNll:
    """Calcul de la NLL O(p) avec structure Triangulaire"""
    def __init__(self, kernel, sparsity_lambda=0.0):
        self.kernel = kernel
        self.lam = sparsity_lambda

    def _compute_kernel_matrix(self, X, lengthscales_j):
        # Mise à l'échelle des données
        X_scaled = X / lengthscales_j
        
        # Calcul des distances au carré (vecteur condensé)
        dists_sq_vec = pdist(X_scaled, metric='sqeuclidean')
        
        # Calcul du noyau (vecteur condensé)
        if self.kernel.kernel_type == 'rbf':
            K_vec = np.exp(-0.5 * dists_sq_vec)
        elif self.kernel.kernel_type == 'matern32':
            d_vec = np.sqrt(dists_sq_vec)
            sqrt3 = np.sqrt(3.0)
            K_vec = (1.0 + sqrt3 * d_vec) * np.exp(-sqrt3 * d_vec)
            
        # Transformation en matrice carrée
        K_mat = squareform(K_vec)
        
        # --- CORRECTION CRUCIALE ---
        # squareform met 0 sur la diagonale par défaut. 
        # Or, k(x, x) = 1. On doit remplir la diagonale.
        np.fill_diagonal(K_mat, 1.0)
        
        return K_mat

    def compute(self, theta, X, Y):
        # 1. Déballage
        # Pas besoin de try/except sur unpack, c'est déterministe
        L, lengthscales = self.kernel.unpack_params(theta)
            
        # 2. Inversion de L (Triangulaire)
        # L est triangulaire inférieure. Son déterminant est le produit de la diagonale.
        # Pour le log-det global, on a besoin de 2*n*log|det(L)|
        diag_L = np.diag(L)
        
        # Sécurité : si un élément diag est trop proche de 0 (malgré les bornes)
        # if np.any(diag_L <= 1e-9): 
        #     return 1e15
            
        logdet_L = np.sum(np.log(diag_L))
        
        # W_hat = L^-1 * Y.T
        # solve_triangular est O(p^2) vs O(p^3) pour inv standard
        #try:
        W_hat = spl.solve_triangular(L, Y.T, lower=True)
        # except np.linalg.LinAlgError:
        #     return 1e15

        # 3. Somme sur les processus latents
        log_det_Rs = 0
        quad_form_sum = 0
        n = X.shape[0]
        p = Y.shape[1]
       # print('p : ', theta)
        for j in range(p):
            #print('j : ', j)
            #print("lengthscales for process", j, ":", lengthscales[j, :])
            R_j = self._compute_kernel_matrix(X, lengthscales[j, :])
            R_j[np.diag_indices_from(R_j)] += 1e-6 
            #print("R_j :", R_j)
            
            # try:
            chol_R = spl.cholesky(R_j, lower=True)
            log_det_Rs += 2 * np.sum(np.log(np.diag(chol_R)))
            w_j = W_hat[j, :]
            alpha = spl.cho_solve((chol_R, True), w_j)
            quad_form_sum += w_j @ alpha
            # except np.linalg.LinAlgError:
            #     return 1e15

        const = n * p * np.log(2 * np.pi)
        
        # log(det(Sigma)) = 2*n*log(det(L)) + sum(log(det(R_j)))
        total_log_det = 2 * n * logdet_L + log_det_Rs
        
        nll = 0.5 * (const + total_log_det + quad_form_sum)
        
        # Sparsité sur L (Lasso)
        if self.lam > 0:
            nll += self.lam * np.sum(np.abs(L))
            
        return nll

class FastSparseLMC:
    def __init__(self, p_components, kernel_type='rbf', sparsity_lambda=0.0, n_restarts=5, use_init_heuristic=True, n_jobs=1,seed = 42):
        self.p = p_components
        self.kernel_type = kernel_type
        self.lam = sparsity_lambda
        self.n_restarts = n_restarts
        self.use_init_heuristic = use_init_heuristic # Renommé pour généralité (c'était use_pca)
        self.n_jobs = n_jobs
        self.seed = seed
        
        self.kernel = None 
        self.best_params_ = None
        self.cached_inv_R_ = []
        self.cached_alpha_ = []
        self.L_hat_ = None # On stocke L maintenant
        self.phis_hat_ = None

    def _initialize_hyperparams(self, n_starts, bounds):
        n_params = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        sampler = qmc.LatinHypercube(d=n_params, optimization="random-cd", rng=self.seed)
        sample = sampler.random(n=n_starts)
        initial_thetas = qmc.scale(sample, lower_bounds, upper_bounds)
        
        # --- INITIALISATION INTELLIGENTE (CHOLESKY EMPIRIQUE) ---
        if self.use_init_heuristic and self.Y_train_ is not None:
            try:
                # 1. Calcul de la covariance empirique des sorties Y
                # (p, p)
                cov_emp = np.cov(self.Y_train_.T)
                
                # Jitter pour garantir que la cov empirique est définie positive
                # (nécessaire si n < p ou données très corrélées)
                cov_emp += np.eye(self.p) * 1e-4
                
                # 2. Décomposition de Cholesky : Cov = L_init * L_init.T
                # C'est exactement la structure que nous cherchons !
                L_init = np.linalg.cholesky(cov_emp)
                
                # Clipping pour rester dans les bornes [-10, 10]
                L_init = np.clip(L_init, -10.0, 10.0)
                # Diagonale positive
                np.fill_diagonal(L_init, np.maximum(np.diag(L_init), 1e-4))
                
                # 3. Packing dans le vecteur theta
                L_flat = []
                for i in range(self.p):
                    for j in range(i + 1):
                        L_flat.append(L_init[i, j])
                L_flat = np.array(L_flat)
                
                # Appliquer à tous les starts (on garde la diversité des lengthscales)
                # Les params de L sont au début du vecteur
                n_L = len(L_flat)
                for i in range(n_starts):
                    initial_thetas[i, :n_L] = L_flat
                    
            except np.linalg.LinAlgError:
                print("Warning: Cholesky init failed, using LHS random.")

        return initial_thetas, list(zip(lower_bounds, upper_bounds))

    def _optimize_hyperparams(self, initial_theta, bounds):
        #try:
        res = minimize(
            self.nll_engine.compute,
            initial_theta,
            args=(self.X_train_, self.Y_train_),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200} #, 'ftol': 1e-9}
        )
        return res
        # except Exception:
        #     return None

    def fit(self, X, Y):
        # Recommandation à l'utilisateur
        if np.max(np.abs(Y)) > 100:
            print("ATTENTION: Vos données Y ont une grande amplitude.")
            print("Il est fortement recommandé de normaliser Y (StandardScaler) avant le fit")
            print("car les bornes de la matrice L sont fixées à [-10, 10].")

        self.X_train_ = X
        self.Y_train_ = Y
        n_features = X.shape[1]
        
        self.kernel = FastLMCKernel(n_features, self.p, self.kernel_type)
        self.nll_engine = FastLMCNll(self.kernel, self.lam)
        
        # Bornes
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        bounds_def = self.kernel.get_parameter_bounds(X_min, X_max)
        
        
        # Init
        starts, bounds = self._initialize_hyperparams(self.n_restarts, bounds_def)
        
        # Optim
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._optimize_hyperparams)(theta, bounds) for theta in starts
        )
        
        valid_results = [r for r in results if r is not None and r.success]
        if not valid_results: valid_results = [r for r in results if r is not None]
        if not valid_results: raise RuntimeError("Optim failed.")
            
        best_res = min(valid_results, key=lambda x: x.fun)
        self.best_params_ = best_res.x
        self.final_nll_ = best_res.fun
        
        # --- MISE EN CACHE ---
        self.L_hat_, self.phis_hat_ = self.kernel.unpack_params(self.best_params_)
        
        # W_obs = L^-1 * Y.T
        W_obs = spl.solve_triangular(self.L_hat_, self.Y_train_.T, lower=True)
        
        self.cached_inv_R_ = []
        self.cached_alpha_ = []
        
        for j in range(self.p):
            ls = self.phis_hat_[j, :]
            X_scaled = self.X_train_ / ls
            dists = squareform(pdist(X_scaled, metric='sqeuclidean'))
            
            if self.kernel_type == 'rbf': K_base = np.exp(-0.5 * dists)
            elif self.kernel_type == 'matern32':
                d = np.sqrt(dists); sqrt3 = np.sqrt(3.0)
                K_base = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
            
            np.fill_diagonal(K_base, 1.0)
            # Robust Cholesky Jitter
            jitter = 1e-6
            L_chol = None
            for _ in range(5):
                try:
                    K_mat = K_base.copy()
                    K_mat[np.diag_indices_from(K_mat)] += jitter
                    L_chol = spl.cholesky(K_mat, lower=True)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            
            if L_chol is None: raise np.linalg.LinAlgError("Cholesky failed.")
            self.cached_inv_R_.append(L_chol)
            self.cached_alpha_.append(spl.cho_solve((L_chol, True), W_obs[j, :]))

    def predict(self, X_new, return_cov=False):
        return self._predict_impl(X_new, return_cov)

    def _predict_impl(self, X_new, return_cov):
        if self.L_hat_ is None: raise Exception("Not fitted")
        n_new = X_new.shape[0]
        W_means = np.zeros((self.p, n_new))
        
        if return_cov: W_covs = []
        else: W_vars = np.zeros((self.p, n_new))
            
        for j in range(self.p):
            ls = self.phis_hat_[j, :]
            L_chol = self.cached_inv_R_[j]
            alpha = self.cached_alpha_[j]
            
            X1_s = X_new / ls; X2_s = self.X_train_ / ls
            dists_c = cdist(X1_s, X2_s, metric='sqeuclidean')
            
            if self.kernel_type == 'rbf': K_trans = np.exp(-0.5 * dists_c)
            else:
                d = np.sqrt(dists_c); sqrt3 = np.sqrt(3.0)
                K_trans = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
            
            f_star = K_trans @ alpha
            W_means[j, :] = f_star
            
            v = spl.solve_triangular(L_chol, K_trans.T, lower=True)
            
            if return_cov:
                X_s = X_new / ls
                d_ss = pdist(X_s, metric='sqeuclidean')
                if self.kernel_type == 'rbf': K_ss = squareform(np.exp(-0.5 * d_ss))
                else:
                    d = np.sqrt(d_ss); K_ss = squareform((1.0 + sqrt3*d)*np.exp(-sqrt3*d))
                K_ss[np.diag_indices_from(K_ss)] += 1.0
                
                cov_j = K_ss - v.T @ v
                W_covs.append(cov_j)
            else:
                k_diag = 1.0 + 1e-6
                var_j = np.maximum(k_diag - np.sum(v**2, axis=0), 1e-9)
                W_vars[j, :] = var_j
        
        # Projection : V = L * W
        V_mean = (self.L_hat_ @ W_means).T
        
        if not return_cov:
            # Var(V) = (L^2) * Var(W)
            L_sq = self.L_hat_**2
            V_var_T = L_sq @ W_vars
            return V_mean, V_var_T.T
        else:
            Full_Cov = np.zeros((n_new * self.p, n_new * self.p))
            for j in range(self.p):
                C_W = W_covs[j]
                # Partie spatiale du latent j
                # Produit Kronecker avec la structure de L pour ce latent
                l_j = self.L_hat_[:, j].reshape(-1, 1)
                Full_Cov += np.kron(l_j @ l_j.T, C_W)
            return V_mean, Full_Cov