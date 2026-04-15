import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


class RBFKernel:
    def __init__(self, length_scale=None):
        self.length_scale = length_scale
        self.hyperparams = length_scale
    
    def __call__(self, X1, X2=None):
        
        length_scale = self.hyperparams
        if X2 is None:
            dists = pdist(X1 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X1 / length_scale, X2 / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            
        return K
    
    def grad_K(self, X1):
        """ Calcul du gradient du noyau par rapport au paramètre length_scale """
        # We need to recompute the pairwise dimension-wise distances
        K = self(X1)
        length_scale = self.hyperparams
        K_gradient = (X1[:, np.newaxis, :] - X1[np.newaxis, :, :]) ** 2 / (
            length_scale**2
        )
        K_gradient *= K[..., np.newaxis]
        #print("K_grad shape : ", X1[:, np.newaxis, :].shape)
        if length_scale.shape[0]==1:
            K_gradient = K_gradient[:,:,0]
            K_gradient = K_gradient[..., np.newaxis]
        return K_gradient
    

class MaternKernel:
    def __init__(self, length_scale=1.0, nu=1.5):
        """
        nu : paramètre de lissage (smoothness). 
             Valeurs supportées : 0.5, 1.5 (par défaut), 2.5
        """
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu doit être 0.5, 1.5 ou 2.5")
        
        self.length_scale = length_scale
        self.hyperparams = length_scale
        self.nu = nu
    
    def __call__(self, X1, X2=None):
        length_scale = self.hyperparams
        
        # Calcul de la distance Euclidienne (non carrée pour Matérn)
        # On divise par length_scale avant pour gérer l'anisotropie (ARD)
        if X2 is None:
            dists = pdist(X1 / length_scale, metric="euclidean")
            # Conversion vecteur -> matrice carrée
            dists = squareform(dists)
        else:
            dists = cdist(X1 / length_scale, X2 / length_scale, metric="euclidean")
        
        # Application des formules selon nu
        if self.nu == 0.5:
            K = np.exp(-dists)
            
        elif self.nu == 1.5:
            sqrt_3 = np.sqrt(3)
            K = (1 + sqrt_3 * dists) * np.exp(-sqrt_3 * dists)
            
        elif self.nu == 2.5:
            sqrt_5 = np.sqrt(5)
            # Formule: (1 + sqrt(5)*d + 5/3 * d^2) * exp(-sqrt(5)*d)
            K = (1 + sqrt_5 * dists + (5.0 / 3.0) * dists**2) * np.exp(-sqrt_5 * dists)
            
        # Sur la diagonale, la distance est 0, donc K=1. 
        # C'est implicite avec les formules, mais on peut forcer pour la précision num.
        if X2 is None:
            np.fill_diagonal(K, 1)
            
        return K
    
    def grad_K(self, X1):
        """ 
        Calcul du gradient du noyau par rapport au paramètre length_scale.
        Note: Comme dans votre RBF, cela correspond au gradient w.r.t log(length_scale).
        """
        length_scale = self.hyperparams
        
        # 1. Recalcul des distances Euclidiennes (D)
        dists = pdist(X1 / length_scale, metric="euclidean")
        dists = squareform(dists)
        
        # 2. Préparation du tenseur "différences au carré normalisées"
        # C'est le terme (x_i - x_j)^2 / l^2
        # Shape : (n_samples, n_samples, n_dims)
        diff_sq_normalized = (X1[:, np.newaxis, :] - X1[np.newaxis, :, :]) ** 2 / (
            length_scale**2
        )
        
        # 3. Calcul du facteur multiplicatif selon nu
        # Le gradient s'écrit souvent : K_grad = prefactor * diff_sq_normalized
        
        if self.nu == 0.5:
            # Attention à la division par zéro sur la diagonale
            # Grad = exp(-D) * (1/D) * diff_sq_normalized
            # On utilise un masque pour éviter le warning, la diagonale sera 0 de toute façon
            with np.errstate(divide='ignore', invalid='ignore'):
                prefactor = np.exp(-dists) / dists
            prefactor[dists == 0] = 0  # Fix diag
            
        elif self.nu == 1.5:
            # Grad = 3 * exp(-sqrt(3)*D) * diff_sq_normalized
            sqrt_3 = np.sqrt(3)
            prefactor = 3 * np.exp(-sqrt_3 * dists)
            
        elif self.nu == 2.5:
            # Grad = 5/3 * (1 + sqrt(5)*D) * exp(-sqrt(5)*D) * diff_sq_normalized
            sqrt_5 = np.sqrt(5)
            prefactor = (5.0 / 3.0) * (1 + sqrt_5 * dists) * np.exp(-sqrt_5 * dists)

        # Broadcasting du préfacteur scalaire (N, N) sur le tenseur (N, N, D)
        K_gradient = diff_sq_normalized * prefactor[..., np.newaxis]

        # Gestion du cas scalaire vs vecteur (comme dans votre RBF)
        if hasattr(length_scale, 'shape') and length_scale.shape[0] == 1:
            K_gradient = K_gradient[:, :, 0]
            K_gradient = K_gradient[..., np.newaxis]
            
        return K_gradient