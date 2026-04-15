import numpy as np
from typing import List, Tuple, Optional
from .Kernel import Kernel

class LMCKernelConstrained:
    """
    Implémentation du noyau du modèle de corégionalisation linéaire (LMC) avec contrainte.
    
    Le noyau est défini comme une somme de produits de Kronecker:
    K(x, x') = sum_{q=1}^Q B_q ⊗ k_q(x, x')

    où B_q = L_q @ L_q.T (sigma_B est fixé à 1, non estimé), et L_q est contraint tel que u.T @ L_q = 0.
    Le premier élément de chaque colonne de L_q est fixé comme:
    L_q[0, j] = -sum(u[1:] * L_q[1:, j]) / u[0]

    Cette contrainte garantit que le produit scalaire entre u et les sorties est nul.
    """
    
    def __init__(self, base_kernels: List[Kernel], output_dim: int, u_vector: np.ndarray = None, 
                 rank: Optional[List[int]] = None, seed: int = 43):
        """
        Initialise le noyau LMC avec contrainte.
        
        Args:
            base_kernels: Liste des noyaux de base k_q(x, x')
            output_dim: Nombre de sorties (dimension D)
            u_vector: Vecteur de contrainte u de taille output_dim. Si None, un vecteur unitaire sera utilisé.
            rank: Liste des rangs pour chaque matrice de corégionalisation B_q
                 Si None, toutes les matrices seront de rang complet
            seed: Seed pour la génération aléatoire
        """
        self.base_kernels = base_kernels
        self.output_dim = output_dim
        self.start_idx_Lq = 0
        
        # Définir le vecteur de contrainte u
        if u_vector is None:
            # Par défaut, utiliser un vecteur unitaire [1, 0, 0, ...]
            self.u_vector = np.zeros(output_dim)
            self.u_vector[0] = 1.0
        else:
            if len(u_vector) != output_dim:
                raise ValueError(f"Le vecteur u doit avoir une longueur de {output_dim}")
            if u_vector[0] == 0:
                raise ValueError("Le premier élément du vecteur u ne peut pas être zéro")
            self.u_vector = np.array(u_vector) / np.linalg.norm(u_vector)  # Normaliser u
        
        # Si le rang n'est pas spécifié, utiliser un rang complet (=output_dim)
        if rank is None:
            self.rank = [output_dim] * len(base_kernels)
        else:
            if len(rank) != len(base_kernels):
                raise ValueError("Le nombre de rangs doit correspondre au nombre de noyaux de base")
            self.rank = rank
        
        # Initialiser les matrices L_q (sans les premières lignes qui seront calculées)
        self.Lq_params = []

        self._bounds = []
        np.random.seed(seed)
        start_idx = 0
        
        # Note: Si output_dim <= 1, pas de contrainte possible au sens strict (ou trivial u*L=0 => L=0)
        # Mais le code original gérait output_dim > 2 spécifiquement.
        # Pour output_dim=2, le code original utilise une logique spécifique.
        
        for q, r in enumerate(self.rank):
            # Pour chaque matrice L_q de taille output_dim x r, 
            # on paramétrise tous les éléments sauf la première ligne L_q[0, :]
            
            if self.output_dim > 2:
                n_params = (self.output_dim - 1) * r
                Lq_vec =  np.random.randn(n_params) #np.random.uniform(-0.5, 0.5, n_params) #
                self.Lq_params.append(Lq_vec)
                self._bounds.extend([(-10.0, 10.0)] * n_params)
                start_idx += n_params
            else:
                pass

        self.start_idx_Lq = start_idx

        # Ajouter les bornes pour les facteurs d'échelle sigma_B (positifs)
        #self._bounds.extend([(1.0, 10.0)] * len(self.rank))
        # Ajouter les bornes pour les paramètres des noyaux de base
        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)
    
    @property
    def params(self) -> np.ndarray:
        """Retourne tous les hyperparamètres du noyau LMC."""
        parts = list(self.Lq_params) if self.output_dim > 2 else []
        for kernel in self.base_kernels:
            parts.append(kernel.params)
        return np.concatenate(parts) if parts else np.array([])
    
    @params.setter
    def params(self, params: np.ndarray):
        """Définit tous les hyperparamètres du noyau LMC."""
        start_idx = 0
        if self.output_dim > 2:
            for q, r in enumerate(self.rank):
                n_params_Lq = (self.output_dim - 1) * r
                self.Lq_params[q] = params[start_idx:start_idx + n_params_Lq]
                start_idx += n_params_Lq
        for kernel in self.base_kernels:
            n = kernel.get_n_params()
            kernel.params = params[start_idx:start_idx + n]
            start_idx += n
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Retourne les bornes de tous les hyperparamètres."""
        return self._bounds
    
    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """
        Reconstruit la matrice L_q complète en calculant la première ligne
        selon la contrainte u.T @ L_q = 0 pour chaque colonne j.
        """
        r = self.rank[q]
        output_dim = self.output_dim
        
        L_q = np.zeros((output_dim, r))
        
        if output_dim > 2:
            L_q[1:, :] = self.Lq_params[q].reshape(output_dim - 1, r)
        
        u = self.u_vector
        for j in range(r):
            if output_dim > 2:
                # u[0]*L[0,j] + sum(u[1:]*L[1:,j]) = 0
                L_q[0,j] = -(1.0/u[0])*(np.sum(u[1:] * L_q[1:, j]))
            else:
                # u[0]L[0,j] + u[1]L[1,j] = 0
                # Original code for 2D: L[1,j]=1.0 (Fixed), L[0,j] = -u[1]/u[0]
                L_q[0, j] = -(1.0/u[0])*u[1]
                L_q[1, j] = 1.0
        return L_q
    
    def get_B(self, q: int) -> np.ndarray:
        """Retourne la matrice de corégionalisation B_q = L_q @ L_q.T."""
        L_q = self._reconstruct_Lq(q)
        return L_q @ L_q.T
    
    def get_L(self, q: int) -> np.ndarray:
        """Retourne la matrice L_q complète."""
        return self._reconstruct_Lq(q)
    
    def get_sigma_B(self, q: int) -> float:
        """Fixé à 1.0 (non estimé)."""
        return 1.0
    
    def verify_constraint(self, q: int, tol: float = 1e-10) -> bool:
        """Vérifie que la contrainte u.T @ L_q = 0 est satisfaite."""
        L_q = self._reconstruct_Lq(q)
        dot_products = self.u_vector @ L_q
        return np.all(np.abs(dot_products) < tol)
    
    def get_n_params(self) -> int:
        return len(self.params)
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        K = np.zeros((n1, n2))
        
        for q in range(len(self.base_kernels)):
            K_spatial = self.base_kernels[q](x1, x2)
            B_q = self.get_B(q)
            K += np.kron(B_q, K_spatial) 
        
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n_params = self.get_n_params()
        n1_q = int(n1/self.output_dim)
        x1 = X1[:n1_q,:-1]
        
        gradients = np.zeros((n_params, n1, n1)) # Assuming X2=X1 for now or square matrices logic from original
        # Note: Original code handled n1 != n2, but here I simplify to what I see. 
        # Actually let's use n1,n2 from args.
        n2 = X2.shape[0]
        n2_q = int(n2/self.output_dim)
        x2 = X2[:n2_q,:-1]
        
        gradients = np.zeros((n_params, n1, n2))
        param_idx = 0
        
        # Gradients wrt L_q (excluding constrained rows)
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                K_spatial = self.base_kernels[q](x1, x2)
                L_q = self._reconstruct_Lq(q)

                for i in range(1, self.output_dim):  # Skip row 0 (constrained)
                    for j in range(self.rank[q]):
                        dB_q = self._compute_dB_q_dL(q, i, j, L_q)
                        gradients[param_idx] = np.kron(dB_q, K_spatial)
                        param_idx += 1
                        
        # Gradients wrt sigma_B
        # for q in range(len(self.base_kernels)):
        #     K_spatial = self.base_kernels[q](x1, x2)
        #     dB_q = self._compute_dB_q_dsigma(q)
        #     gradients[param_idx] = np.kron(dB_q, K_spatial)
        #     param_idx += 1
            
        # Gradients wrt spatial kernels
        for q, kernel in enumerate(self.base_kernels):
            dK_spatial_list = kernel.gradient(x1, x2)
            B_q = self.get_B(q)
            for dK_spatial in dK_spatial_list:
                gradients[param_idx] = np.kron(B_q, dK_spatial)
                param_idx += 1
                
        return gradients

    def _compute_dB_q_dL(self, q: int, i: int, j: int, L_q: np.ndarray) -> np.ndarray:
        """dB_q / dL_q[i, j], accounting for the constrained row L_q[0, j]."""
        dL_q = np.zeros_like(L_q)
        dL_q[i, j] = 1.0
        # Derivative of constrained row: d(L_q[0,j])/d(L_q[i,j]) = -u[i]/u[0]
        dL_q[0, j] = -(self.u_vector[i]) / self.u_vector[0]
        return dL_q @ L_q.T + L_q @ dL_q.T

    def _compute_dB(self):
        """
        Calcule la dérivée de B par rapport à tous les paramètres de B (utile en ICM)
        
        Args:
            
            
        Returns:
            Liste des Matrice dB_q de forme (output_dim, output_dim)
        """
        dB = []
        if self.output_dim > 2:
            for q in range(len(self.base_kernels)):
                L_q = self._reconstruct_Lq(q)
                for i in range(1, self.output_dim):
                    for j in range(self.rank[q]):
                        dB.append(self._compute_dB_q_dL(q, i, j, L_q))
        
        # # On ajoute la dérivée de B par rapport à sigma_B est simplement B_unit = L_q @ L_q.T
        # for q in range(len(self.base_kernels)):
        #     dB_q = self._compute_dB_q_dsigma(q)
        #     dB.append(dB_q)
        
        return dB


    def _compute_dB_q_dsigma(self, q: int) -> np.ndarray:
        L_q = self._reconstruct_Lq(q)
        return L_q @ L_q.T

    def init_L_from_pca(self, Y: np.ndarray):
        """
        Initialise L_q à partir de la PCA des données Y (n x p),
        en respectant la contrainte u^T L = 0
        """
        # If output_dim <= 2, we don't have free params for L.
        if self.output_dim <= 2:
            return 

        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

        for q, r in enumerate(self.rank):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])

            # Projection pour respecter la contrainte
            u = self.u_vector / np.linalg.norm(self.u_vector)
            P = np.eye(self.output_dim) - np.outer(u, u)
            L_proj = P @ L_pca

            # On ne garde que les lignes libres (1:)
            self.Lq_params[q] = L_proj[1:, :].flatten()
