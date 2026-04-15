import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from .Kernel import Kernel


class LMCKernelUnit:
    """
    Implémentation du noyau du modèle de corégionalisation linéaire (LMC).
    
    Le noyau est défini comme une somme de produits de Kronecker:
    K(x, x') = sum_{q=1}^Q B_q ⊗ k_q(x, x')
    
    où B_q = sigma_B_q * B_unit_q avec B_unit_q ayant des coefficients entre -1 et 1,
    et k_q est un noyau spatial.
    
    Le premier élément de chaque matrice L_q (L_q[0,0]) est fixé comme:
    L_q[0,0] = 1 - somme des valeurs absolues des autres éléments de L_q
    """
    
    def __init__(self, base_kernels: List[Kernel], output_dim: int, rank: Optional[List[int]] = None, seed : int = 43):
        """
        Initialise le noyau LMC.
        
        Args:
            base_kernels: Liste des noyaux de base k_q(x, x')
            output_dim: Nombre de sorties (dimension D)
            rank: Liste des rangs pour chaque matrice de corégionalisation B_q
                 Si None, toutes les matrices seront de rang complet
        """
        self.base_kernels = base_kernels
        self.output_dim = output_dim
        self.start_idx_Lq = 0
        
        # Si le rang n'est pas spécifié, utiliser un rang complet (=output_dim)
        if rank is None:
            self.rank = [output_dim] * len(base_kernels)
        else:
            if len(rank) != len(base_kernels):
                raise ValueError("Le nombre de rangs doit correspondre au nombre de noyaux de base")
            self.rank = rank
        
        # Initialiser les matrices Lq_unit (sans le premier élément qui sera calculé)
        self.Lq_unit_params = []
        
        # Initialiser les facteurs d'échelle sigma_B pour chaque matrice B_q
        self.sigma_B_params = np.ones(len(self.rank)) #* 0.5  # Initialisation à 0.5
        
        self._bounds = []
        np.random.seed(seed)
        start_idx = 0
        for q, r in enumerate(self.rank):
            # Pour chaque matrice L_q de taille output_dim x r, 
            # on paramétrise tous les éléments sauf le premier L_q[0,0]
            n_params = output_dim * r - 1  # -1 pour exclure L_q[0,0]
            
            # Initialiser les paramètres de L_q (tous sauf le premier élément)
            # avec des valeurs entre -0.5 et 0.5
            Lq_unit_vec = np.random.uniform(-1, 1, n_params)
            self.Lq_unit_params.append(Lq_unit_vec)
            
            # Ajouter les bornes pour chaque élément de Lq_unit (entre -1 et 1)
            self._bounds.extend([(-1.0, 1.0)] * n_params)
            start_idx += n_params
        self.start_idx_Lq = start_idx
        
        
        # Ajouter les bornes pour les facteurs d'échelle sigma_B (positifs)
        self._bounds.extend([(1.0, 10.0)] * len(self.rank))
        
        # Ajouter les bornes pour les paramètres des noyaux de base
        for kernel in self.base_kernels:
            self._bounds.extend(kernel.bounds)
    
    @property
    def params(self) -> np.ndarray:
        """Retourne tous les hyperparamètres du noyau LMC."""
        # Concaténer les paramètres de Lq_unit (sans L_q[0,0]), sigma_B et les paramètres des noyaux de base
        params = np.concatenate(self.Lq_unit_params)
        params = np.concatenate([params, self.sigma_B_params])
        for kernel in self.base_kernels:
            params = np.concatenate([params, kernel.params])
        return params
    
    @params.setter
    def params(self, params: np.ndarray):
        """Définit tous les hyperparamètres du noyau LMC."""
        # Extraire les paramètres des matrices Lq_unit (sans L_q[0,0])
        start_idx = 0
        for q, r in enumerate(self.rank):
            n_params_Lq = self.output_dim * r - 1  # -1 pour exclure L_q[0,0]
            self.Lq_unit_params[q] = params[start_idx:start_idx + n_params_Lq]
            start_idx += n_params_Lq
        
        # Extraire les paramètres sigma_B
        self.sigma_B_params = params[start_idx:start_idx + len(self.rank)]
        start_idx += len(self.rank)
        
        # Extraire les paramètres des noyaux de base
        for kernel in self.base_kernels:
            n_params_kernel = kernel.get_n_params()
            kernel.params = params[start_idx:start_idx + n_params_kernel]
            start_idx += n_params_kernel
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Retourne les bornes de tous les hyperparamètres."""
        return self._bounds
    
    def _reconstruct_Lq(self, q: int) -> np.ndarray:
        """
        Reconstruit la matrice Lq complète en calculant le premier élément L_q[0,0]
        
        Args:
            q: Indice du noyau
            
        Returns:
            Matrice Lq reconstruite de forme (output_dim, rank[q])
        """
        r = self.rank[q]
        #n_params = self.output_dim * r - 1  # Nombre de paramètres sans L_q[0,0]
        
        # Créer une matrice pleine pour Lq
        Lq = np.zeros((self.output_dim, r))
        
        # Remplir la matrice avec les paramètres (sauf le premier élément)
        param_idx = 0
        for i in range(self.output_dim):
            for j in range(r):
                if i == 0 and j == 0:
                    continue  # Sauter le premier élément, on le calculera après
                Lq[i, j] = self.Lq_unit_params[q][param_idx]
                param_idx += 1
        
        # Calculer le premier élément L_q[0,0] = 1 - somme des valeurs absolues des autres
        Lq[0, 0] = 1.0 - np.sum(np.abs(Lq.flatten()[1:]))
        
        return Lq
    
    def get_B(self, q: int) -> np.ndarray:
        """
        Retourne la matrice de corégionalisation B_q = sigma_B_q * (Lq_unit @ Lq_unit.T).
        
        Args:
            q: Indice du noyau
            
        Returns:
            Matrice B_q de forme (output_dim, output_dim)
        """
        Lq = self._reconstruct_Lq(q)
        B_unit = Lq @ Lq.T
        return self.sigma_B_params[q] * B_unit
    
    def get_L(self, q: int) -> np.ndarray:
        """
        Retourne la matrice Lq complète (y compris le premier élément calculé).
        
        Args:
            q: Indice du noyau
            
        Returns:
            Matrice Lq de forme (output_dim, rank[q])
        """
        return self._reconstruct_Lq(q)
    
    def get_sigma_B(self, q: int) -> float:
        """
        Retourne le facteur d'échelle sigma_B pour le q-ième noyau.
        
        Args:
            q: Indice du noyau
            
        Returns:
            Facteur d'échelle sigma_B_q
        """
        return self.sigma_B_params[q]
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calcule la matrice de covariance complète K((x1,d1), (x2,d2)) pour toutes les entrées et sorties.
        
        Args:
            X1: Matrice de forme (n1 * output_dim, input_dim + 1)
                La dernière colonne indique l'indice de sortie (0 à output_dim-1)
            X2: Matrice de forme (n2 * output_dim, input_dim + 1), si None, X2 = X1
            
        Returns:
            Matrice de covariance de forme (n1 * output_dim, n2 * output_dim)
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 
        

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        # Initialiser la matrice de covariance
        K = np.zeros((n1, n2))
        
        # Construire la matrice de covariance par blocs
        for q in range(len(self.base_kernels)):
            # Calculer la matrice de covariance spatiale k_q(x1, x2)
            K_spatial = self.base_kernels[q](x1, x2)
            #K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            
            # Obtenir la matrice de corégionalisation B_q
            B_q = self.get_B(q)
            
            # Pour chaque paire d'indices de sortie, ajouter la contribution à K
            K += np.kron(B_q, K_spatial) 
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Calcule les gradients de la matrice de covariance par rapport à tous les hyperparamètres.
        
        Args:
            X1: Matrice de forme (n1 * output_dim, input_dim + 1)
            X2: Matrice de forme (n2 * output_dim, input_dim + 1), si None, X2 = X1
            
        Returns:
            Liste de matrices, chacune de forme (n1 * output_dim, n2 * output_dim)
        """
        if X2 is None:
            X2 = X1
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        n1_q = int(n1/self.output_dim)
        n2_q = int(n2/self.output_dim) 

        x1 = X1[:n1_q,:-1]
        x2 = X2[:n2_q,:-1]
        
        gradients = np.zeros((self.get_n_params(), n1, n2))
        
        param_idx = 0
        
        # Gradients par rapport aux éléments de Lq_unit (sauf L_q[0,0])
        for q in range(len(self.base_kernels)):
            # Calculer la matrice de covariance spatiale k_q(x1, x2)
            #K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            K_spatial = self.base_kernels[q](x1, x2)
            
            # Obtenir la matrice Lq complète
            Lq = self._reconstruct_Lq(q)
            sigma_B = self.sigma_B_params[q]
            
            # Calculer les gradients par rapport aux éléments de Lq
            for d in range(self.output_dim):
                for jj in range(self.rank[q]):
                    if d == 0 and jj == 0:
                        continue  # Sauter L_q[0,0] qui n'est pas un paramètre libre
                    
                    # Obtenir dB_q par rapport à L_q[d,jj]
                    dB_q = self._compute_dB_q_dL(q, d, jj, Lq, sigma_B)
                    
                    # Appliquer produit de Kronecker pour obtenir dK
                    gradients[param_idx] =np.kron(dB_q,K_spatial) 
                    param_idx += 1
        
        # Gradients par rapport aux facteurs d'échelle sigma_B
        for q in range(len(self.base_kernels)):
            # Calculer la matrice de covariance spatiale k_q(x1, x2)
            K_spatial = self.base_kernels[q](x1, x2)
            # K_spatial = self.base_kernels[q](X1_spatial, X2_spatial)
            
            # Obtenir dB_q par rapport à sigma_B
            dB_q = self._compute_dB_q_dsigma(q)
            
            # Appliquer produit de Kronecker pour obtenir dK
            gradients[param_idx] = np.kron(dB_q,K_spatial) 
            param_idx += 1
        
        # Gradients par rapport aux hyperparamètres des noyaux de base
        for q, kernel in enumerate(self.base_kernels):
            # Calculer les gradients du noyau spatial
            dK_spatial_list = kernel.gradient(x1, x2)
            
            # Obtenir la matrice de corégionalisation B_q
            B_q = self.get_B(q)
            
            # Pour chaque hyperparamètre du noyau, calculer le gradient complet
            for dK_spatial in dK_spatial_list:
                # Appliquer produit de Kronecker pour obtenir dK
                gradients[param_idx] = np.kron(B_q, dK_spatial) 
                param_idx += 1
        
        return gradients

    def _compute_dB_q_dL(self, q: int, d: int, jj: int, Lq: np.ndarray, sigma_B: float) -> np.ndarray:
        """
        Calcule la dérivée de B_q par rapport à l'élément L_q[d,jj]
        
        Args:
            q: Index du noyau de base
            d: Index de ligne dans la matrice Lq
            jj: Index de colonne dans la matrice Lq
            Lq: Matrice Lq reconstruite
            sigma_B: Facteur d'échelle pour B_q
            
        Returns:
            Matrice dB_q de forme (output_dim, output_dim)
        """
        # Calculer la dérivée de Lq par rapport au paramètre courant
        dLq = np.zeros_like(Lq)
        dLq[d, jj] = 1.0
        
        # La dérivée du premier élément L_q[0,0] par rapport au paramètre L_q[d,jj]
        # est -sign(L_q[d,jj]) car L_q[0,0] = 1 - somme(abs(autres éléments))
        dLq[0, 0] = -np.sign(Lq[d, jj])
        
        # Calculer la dérivée de B_unit par rapport au paramètre
        dB_unit = dLq @ Lq.T + Lq @ dLq.T
        
        # Appliquer le facteur d'échelle sigma_B
        dB_q = sigma_B * dB_unit
        
        return dB_q
    
    def _compute_dB(self):
        """
        Calcule la dérivée de B par rapport à tous les paramètres de B (utile en ICM)
        
        Args:
            
            
        Returns:
            Liste des Matrice dB_q de forme (output_dim, output_dim)
        """
        dB = []
        
        for q in range(len(self.base_kernels)):
            # Calculer la matrice de covariance spatiale k_q(x1, x2)
            # Obtenir la matrice L_q complète et sigma_B
            L_q = self._reconstruct_Lq(q)
            sigma_B = self.sigma_B_params[q]
            
            # Parcourir tous les paramètres (lignes 2 à output_dim-1)
            for i in range(self.output_dim):
                for j in range(self.rank[q]):
                    if i == 0 and j == 0:
                        continue
                    # Calculer dB_q par rapport à L_q[i,j]
                    dB_q = self._compute_dB_q_dL(q, i, j, L_q, sigma_B)
                    dB.append(dB_q)
    
        # On ajoute la dérivée de B par rapport à sigma_B est simplement B_unit = L_q @ L_q.T
        for q in range(len(self.base_kernels)):
            dB_q = self._compute_dB_q_dsigma(q)
            dB.append(dB_q)
        
        return np.array(dB)

    def _compute_dB_q_dsigma(self, q: int) -> np.ndarray:
        """
        Calcule la dérivée de B_q par rapport à sigma_B
        
        Args:
            q: Index du noyau de base
            
        Returns:
            Matrice dB_q de forme (output_dim, output_dim)
        """
        # Obtenir la matrice Lq complète
        Lq = self._reconstruct_Lq(q)
        
        # La dérivée de B_q par rapport à sigma_B est simplement B_unit = Lq @ Lq.T
        B_unit = Lq @ Lq.T
        
        return B_unit
    
    def get_n_params(self) -> int:
        """Retourne le nombre total d'hyperparamètres."""
        return len(self.params)
    
    def init_L_from_pca(self, Y: np.ndarray):
        """
        Initialise L_q à partir de la PCA des données Y (n x p),
        en respectant la contrainte u^T L = 0
        """
        Yc = Y - Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

        for q, r in enumerate(self.rank):
            L_pca = Vt.T[:, :r] * np.sqrt(S[:r])

            # On ne garde que les lignes libres (1:)
            mask = np.ones(L_pca.shape, dtype=bool)
            mask[0, 0] = False

            vec = L_pca[mask]
            self.Lq_unit_params[q] = vec.flatten() #[1:, :].
