import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass

class Kernel:
    """Classe de base pour les noyaux de covariance."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self._params = np.array([])
        self._bounds = []
        
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Calcule la matrice de covariance entre X1 et X2."""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """Calcule le gradient de la matrice de covariance par rapport aux hyperparamètres."""
        raise NotImplementedError("Subclasses must implement gradient")
    
    @property
    def params(self) -> np.ndarray:
        """Retourne les hyperparamètres actuels."""
        return self._params
    
    @params.setter
    def params(self, params: np.ndarray):
        """Définit les hyperparamètres."""
        if params.shape != self._params.shape:
            raise ValueError(f"Shapes do not match: {params.shape} vs {self._params.shape}")
        self._params = params
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Retourne les bornes des hyperparamètres pour l'optimisation."""
        return self._bounds
    
    def get_n_params(self) -> int:
        """Retourne le nombre d'hyperparamètres."""
        return len(self._params)


class RBFKernel(Kernel):
    """Noyau RBF (Radial Basis Function) ou gaussien, avec support pour anisotropie."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initialise le noyau RBF.
        
        Args:
            input_dim: Dimension des entrées
            lengthscale: Échelle de longueur initiale. Si None, initialisé à 1.0
                         Si c'est un scalaire, utilisé pour toutes les dimensions si ARD=False,
                         ou comme valeur initiale pour toutes les dimensions si ARD=True
                         Si c'est un tableau, doit avoir une longueur égale à input_dim pour ARD=True
            ARD: Si True, utilise un lengthscale différent pour chaque dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Initialisation par défaut
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 pour toutes les dimensions
        elif np.isscalar(lengthscale):
            # Scalaire unique fourni
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Tableau de lengthscales fourni
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"Pour ARD=True, lengthscale doit avoir {input_dim} éléments")
            elif not ARD and len(lengthscale) > 1:
                print(f"Attention: ARD=False mais lengthscale a {len(lengthscale)} éléments. Seul le premier sera utilisé.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calcule la matrice de covariance K(X1, X2) pour le noyau RBF anisotropique.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Matrice de covariance de forme (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Pour le cas anisotropique (ARD=True)
        if self.ARD:
            # Pondérer chaque dimension par son lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Calculer la distance euclidienne carrée pondérée
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
        else:
            # Cas isotropique (ARD=False)
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne carrée
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist_sq = dist_sq / (lengthscale**2)
        
        # Appliquer le noyau RBF
        K = variance * np.exp(-0.5 * dist_sq)
        return K
    
        
    #     return K_gradient
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Calcule les dérivées de la matrice de covariance par rapport aux hyperparamètres.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Liste de matrices, chacune de forme (n1, n2) correspondant aux dérivées
            par rapport à chaque hyperparamètre
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Calculer la matrice de covariance
        K = self.__call__(X1, X2)
        
        gradients = []
        
        if self.ARD:
            # Cas anisotropique: un gradient par dimension
            for d in range(self.input_dim):
                # Calculer la différence carrée pour cette dimension
                diff_d = X1[:, d:d+1] - X2[:, d:d+1].T
                sq_diff_d = diff_d**2
                
                # Gradient par rapport à log(lengthscale) pour cette dimension
                dK_dlog_lengthscale_d = K * (sq_diff_d / (lengthscales[d]**2))
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Cas isotropique: un seul gradient
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne carrée
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            
            # Gradient par rapport à log(lengthscale)
            dK_dlog_lengthscale = K * (dist_sq / (lengthscale**2))
            gradients.append(dK_dlog_lengthscale)
        
        return gradients


class Matern52Kernel(Kernel):
    """Noyau Matern 5/2 avec support pour anisotropie."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initialise le noyau Matern 5/2.
        
        Args:
            input_dim: Dimension des entrées
            lengthscale: Échelle de longueur initiale. Si None, initialisé à 1.0
                         Si c'est un scalaire, utilisé pour toutes les dimensions si ARD=False,
                         ou comme valeur initiale pour toutes les dimensions si ARD=True
                         Si c'est un tableau, doit avoir une longueur égale à input_dim pour ARD=True
            ARD: Si True, utilise un lengthscale différent pour chaque dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Initialisation par défaut
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 pour toutes les dimensions
        elif np.isscalar(lengthscale):
            # Scalaire unique fourni
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Tableau de lengthscales fourni
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"Pour ARD=True, lengthscale doit avoir {input_dim} éléments")
            elif not ARD and len(lengthscale) > 1:
                print(f"Attention: ARD=False mais lengthscale a {len(lengthscale)} éléments. Seul le premier sera utilisé.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calcule la matrice de covariance K(X1, X2) pour le noyau Matern 5/2 anisotropique.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Matrice de covariance de forme (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Pour le cas anisotropique (ARD=True)
        if self.ARD:
            # Pondérer chaque dimension par son lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Calculer la distance euclidienne pondérée
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 5/2
            scaled_dist = np.sqrt(5) * dist
        else:
            # Cas isotropique (ARD=False)
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 5/2
            scaled_dist = np.sqrt(5) * dist / lengthscale
        
        K = variance * (1.0 + scaled_dist + scaled_dist**2/3.0) * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Calcule les dérivées de la matrice de covariance par rapport aux hyperparamètres.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Liste de matrices, chacune de forme (n1, n2) correspondant aux dérivées
            par rapport à chaque hyperparamètre
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Cas anisotropique: un gradient par dimension
            for d in range(self.input_dim):
                # Créer des copies des matrices X1 et X2 pour cette dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Mettre à zéro toutes les dimensions sauf celle d'intérêt
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Calculer la distance euclidienne pour cette dimension uniquement
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Calculer la distance euclidienne globale (toutes dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                 
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Calcul du noyau Matern 5/2
                scaled_dist = np.sqrt(5) * dist
                base = (1.0 + scaled_dist + scaled_dist**2/3.0) * np.exp(-scaled_dist)
                
                # Calculer la contribution au gradient pour cette dimension
                # Pour Matern 5/2, le gradient est plus complexe
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        (5/3) * dist_d_sq * (1 + scaled_dist),
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Cas isotropique: un seul gradient
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 5/2
            scaled_dist = np.sqrt(5) * dist / lengthscale
            
            # Gradient par rapport à log(lengthscale)
            dK_dlog_lengthscale = variance * np.exp(-scaled_dist) * (
                scaled_dist**2 * (scaled_dist + 1) / 3.0
            )
            gradients.append(dK_dlog_lengthscale)
        
        return gradients
    




class Matern32Kernel(Kernel):
    """Noyau Matern 3/2 avec support pour anisotropie."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initialise le noyau Matern 3/2.
        
        Args:
            input_dim: Dimension des entrées
            lengthscale: Échelle de longueur initiale. Si None, initialisé à 1.0
                         Si c'est un scalaire, utilisé pour toutes les dimensions si ARD=False,
                         ou comme valeur initiale pour toutes les dimensions si ARD=True
                         Si c'est un tableau, doit avoir une longueur égale à input_dim pour ARD=True
            ARD: Si True, utilise un lengthscale différent pour chaque dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Initialisation par défaut
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 pour toutes les dimensions
        elif np.isscalar(lengthscale):
            # Scalaire unique fourni
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Tableau de lengthscales fourni
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"Pour ARD=True, lengthscale doit avoir {input_dim} éléments")
            elif not ARD and len(lengthscale) > 1:
                print(f"Attention: ARD=False mais lengthscale a {len(lengthscale)} éléments. Seul le premier sera utilisé.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calcule la matrice de covariance K(X1, X2) pour le noyau Matern 3/2 anisotropique.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Matrice de covariance de forme (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Pour le cas anisotropique (ARD=True)
        if self.ARD:
            # Pondérer chaque dimension par son lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Calculer la distance euclidienne pondérée
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 3/2
            scaled_dist = np.sqrt(3) * dist
        else:
            # Cas isotropique (ARD=False)
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 3/2
            scaled_dist = np.sqrt(3) * dist / lengthscale
        
        K = variance * (1.0 + scaled_dist) * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Calcule les dérivées de la matrice de covariance par rapport aux hyperparamètres.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Liste de matrices, chacune de forme (n1, n2) correspondant aux dérivées
            par rapport à chaque hyperparamètre
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Cas anisotropique: un gradient par dimension
            for d in range(self.input_dim):
                # Créer des copies des matrices X1 et X2 pour cette dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Mettre à zéro toutes les dimensions sauf celle d'intérêt
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Calculer la distance euclidienne pour cette dimension uniquement
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Calculer la distance euclidienne globale (toutes dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Calcul du noyau Matern 3/2
                scaled_dist = np.sqrt(3) * dist
                base = (1.0 + scaled_dist) * np.exp(-scaled_dist)
                
                # Calculer la contribution au gradient pour cette dimension
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        3.0 * dist_d_sq,
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Cas isotropique: un seul gradient
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 3/2
            scaled_dist = np.sqrt(3) * dist / lengthscale
            
            # Gradient par rapport à log(lengthscale)
            dK_dlog_lengthscale = variance * scaled_dist**2 * np.exp(-scaled_dist)
            gradients.append(dK_dlog_lengthscale)
        
        return gradients


class Matern12Kernel(Kernel):
    """Noyau Matern 1/2 avec support pour anisotropie."""
    
    def __init__(self, input_dim: int, lengthscale=None, ARD=True):
        """
        Initialise le noyau Matern 1/2.
        
        Args:
            input_dim: Dimension des entrées
            lengthscale: Échelle de longueur initiale. Si None, initialisé à 1.0
                         Si c'est un scalaire, utilisé pour toutes les dimensions si ARD=False,
                         ou comme valeur initiale pour toutes les dimensions si ARD=True
                         Si c'est un tableau, doit avoir une longueur égale à input_dim pour ARD=True
            ARD: Si True, utilise un lengthscale différent pour chaque dimension (Automatic Relevance Determination)
        """
        super().__init__(input_dim)
        
        self.ARD = ARD
        len_lengthscale = input_dim if ARD else 1
        
        if lengthscale is None:
            # Initialisation par défaut
            self._params = np.zeros(len_lengthscale)  # log(1.0) = 0 pour toutes les dimensions
        elif np.isscalar(lengthscale):
            # Scalaire unique fourni
            self._params = np.ones(len_lengthscale) * np.log(lengthscale)
        else:
            # Tableau de lengthscales fourni
            if ARD and len(lengthscale) != input_dim:
                raise ValueError(f"Pour ARD=True, lengthscale doit avoir {input_dim} éléments")
            elif not ARD and len(lengthscale) > 1:
                print(f"Attention: ARD=False mais lengthscale a {len(lengthscale)} éléments. Seul le premier sera utilisé.")
                self._params = np.array([np.log(lengthscale[0])])
            else:
                self._params = np.log(np.array(lengthscale))
        
        self._bounds = [(-10.0, 10.0)] * len_lengthscale
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Calcule la matrice de covariance K(X1, X2) pour le noyau Matern 1/2 anisotropique.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Matrice de covariance de forme (n1, n2)
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        # Pour le cas anisotropique (ARD=True)
        if self.ARD:
            # Pondérer chaque dimension par son lengthscale
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            
            # Calculer la distance euclidienne pondérée
            X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 1/2 (exponentiel)
            scaled_dist = dist  # Pour Matern 1/2, on utilise directement la distance
        else:
            # Cas isotropique (ARD=False)
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 1/2 (exponentiel)
            scaled_dist = dist / lengthscale
        
        K = variance * np.exp(-scaled_dist)
        return K
    
    def gradient(self, X1: np.ndarray, X2: np.ndarray = None) -> List[np.ndarray]:
        """
        Calcule les dérivées de la matrice de covariance par rapport aux hyperparamètres.
        
        Args:
            X1: Matrice de forme (n1, input_dim)
            X2: Matrice de forme (n2, input_dim), si None, X2 = X1
            
        Returns:
            Liste de matrices, chacune de forme (n1, n2) correspondant aux dérivées
            par rapport à chaque hyperparamètre
        """
        if X2 is None:
            X2 = X1
            
        # Extraire les paramètres
        variance = 1.0
        lengthscales = np.exp(self._params)
        
        gradients = []
        
        if self.ARD:
            # Cas anisotropique: un gradient par dimension
            for d in range(self.input_dim):
                # Créer des copies des matrices X1 et X2 pour cette dimension
                X1_copy = X1.copy()
                X2_copy = X2.copy()
                
                # Mettre à zéro toutes les dimensions sauf celle d'intérêt
                for dim in range(self.input_dim):
                    if dim != d:
                        X1_copy[:, dim] = 0
                        X2_copy[:, dim] = 0
                
                # Calculer la distance euclidienne pour cette dimension uniquement
                X1_d_scaled = X1_copy / lengthscales[d]
                X2_d_scaled = X2_copy / lengthscales[d]
                
                X1_d_norm = np.sum(X1_d_scaled**2, axis=1).reshape(-1, 1)
                X2_d_norm = np.sum(X2_d_scaled**2, axis=1).reshape(1, -1)
                dist_d_sq = X1_d_norm + X2_d_norm - 2.0 * np.dot(X1_d_scaled, X2_d_scaled.T)
                
                # Calculer la distance euclidienne globale (toutes dimensions)
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                
                X1_norm = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
                dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1_scaled, X2_scaled.T)
                dist = np.sqrt(np.maximum(dist_sq, 1e-36))
                
                # Calcul du noyau Matern 1/2 (exponentiel)
                scaled_dist = dist
                
                # Calculer la contribution au gradient pour cette dimension
                with np.errstate(divide='ignore', invalid='ignore'):
                    grad_coef = np.where(
                        dist > 1e-6,
                        dist_d_sq / dist,
                        0.0
                    )
                
                dK_dlog_lengthscale_d = variance * np.exp(-scaled_dist) * grad_coef
                gradients.append(dK_dlog_lengthscale_d)
        else:
            # Cas isotropique: un seul gradient
            lengthscale = lengthscales[0]
            
            # Calculer la distance euclidienne
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_norm + X2_norm - 2.0 * np.dot(X1, X2.T)
            dist = np.sqrt(np.maximum(dist_sq, 1e-36))
            
            # Calcul du noyau Matern 1/2 (exponentiel)
            scaled_dist = dist / lengthscale
            
            # Gradient par rapport à log(lengthscale)
            dK_dlog_lengthscale = variance * scaled_dist * np.exp(-scaled_dist)
            gradients.append(dK_dlog_lengthscale)
        
        return gradients