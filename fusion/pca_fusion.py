"""
PCA-based fusion strategy.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict

from .base import BaseFusionStrategy, FusionStrategyRegistry


class PCAFusion(BaseFusionStrategy):
    """
    Strategy 3: PCA-Based Dimensionality Reduction
    
    Concatenates features, applies PCA for dimensionality reduction,
    then computes distances in the reduced space.
    """
    
    def __init__(self, feature_dims: Dict[str, int], 
                 variance_retained: float = 0.95,
                 n_components: int = 256,
                 device: str = "auto"):
        """
        Initialize PCA fusion.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            variance_retained: Fraction of variance to retain (0.95-0.99) - used if n_components is None
            n_components: Fixed number of PCA components (default: 256 for consistency)
                         Will be automatically reduced if data has fewer samples
            device: Device to run on
        """
        super().__init__(feature_dims, device)
        self.variance_retained = variance_retained
        self.n_components = n_components
        self.pca = None
        self.scaler = None
        self._output_dim = n_components  # Fixed output dimension (may be adjusted)
        self.is_fitted = False
        
    def initialize(self) -> None:
        """Initialize the fusion strategy."""
        # Use fixed number of components for consistency
        # Note: actual components may be adjusted based on data size
        self.pca = PCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        self.is_initialized = True
    
    def fit(self, features: Dict[str, np.ndarray]) -> None:
        """
        Fit PCA on training data.
        
        Args:
            features: Training feature arrays
        """
        if not self.is_initialized:
            self.initialize()
        
        # Concatenate and normalize features
        concatenated_features = self._concatenate_features(features)
        
        # Standardize features
        standardized_features = self.scaler.fit_transform(concatenated_features)
        
        # Adjust n_components if data has fewer samples
        n_samples, n_features = standardized_features.shape
        max_components = min(n_samples, n_features)
        
        if self.n_components > max_components:
            print(f"Warning: Reducing PCA components from {self.n_components} to {max_components} "
                  f"(min of n_samples={n_samples}, n_features={n_features})")
            self.n_components = max_components
            self._output_dim = max_components
            # Reinitialize PCA with adjusted components
            self.pca = PCA(n_components=self.n_components)
        
        # Fit PCA with adjusted number of components
        self.pca.fit(standardized_features)
        self.is_fitted = True
        
        print(f"PCA fitted: {self.n_components} components (adjusted) retain "
              f"{np.sum(self.pca.explained_variance_ratio_):.3f} variance")
    
    def _concatenate_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all features with normalization."""
        self._validate_features(features)
        
        # Normalize each feature type
        normalized_features = []
        
        for name in sorted(features.keys()):  # Sort for consistent ordering
            feat = features[name]
            
            # L2 normalization per feature vector
            feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
            normalized_features.append(feat_norm)
        
        # Concatenate all features
        concatenated = np.concatenate(normalized_features, axis=1)
        
        return concatenated
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features using PCA dimensionality reduction.
        If PCA is not fitted yet, fit it on the current data.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            PCA-reduced feature array of shape (N, n_components) - adjusted dimension if needed
        """
        # Concatenate and normalize features
        concatenated_features = self._concatenate_features(features)
        
        # If not fitted, fit on the current data
        if not self.is_fitted:
            if not self.is_initialized:
                self.initialize()
            
            # Standardize features
            standardized_features = self.scaler.fit_transform(concatenated_features)
            
            # Adjust n_components if data has fewer samples
            n_samples, n_features = standardized_features.shape
            max_components = min(n_samples, n_features)
            
            if self.n_components > max_components:
                print(f"Warning: Reducing PCA components from {self.n_components} to {max_components} "
                      f"(min of n_samples={n_samples}, n_features={n_features})")
                self.n_components = max_components
                self._output_dim = max_components
                # Reinitialize PCA with adjusted components
                self.pca = PCA(n_components=self.n_components)
            
            # Fit PCA on the current data
            self.pca.fit(standardized_features)
            self.is_fitted = True
            
            # Transform using the fitted PCA
            reduced_features = self.pca.transform(standardized_features)
            
            print(f"PCA auto-fitted: {self.n_components} components (adjusted) retain "
                  f"{np.sum(self.pca.explained_variance_ratio_):.3f} variance")
        else:
            # Use already fitted PCA
            standardized_features = self.scaler.transform(concatenated_features)
            reduced_features = self.pca.transform(standardized_features)
        
        return reduced_features
    
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances in the PCA-reduced space.
        
        Args:
            fused_features: PCA-reduced features of shape (N, reduced_dim)
            
        Returns:
            Distance array of shape (N-1,)
        """
        if len(fused_features) < 2:
            return np.array([0.0])
        
        # Compute frame-to-frame differences
        diff = fused_features[1:] - fused_features[:-1]
        
        # Compute Euclidean distances
        distances = np.linalg.norm(diff, axis=1)
        
        return distances
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """Get the cumulative explained variance ratio."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted first")
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    @property
    def name(self) -> str:
        """Return the name of the fusion strategy."""
        return "pca"
    
    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features (always fixed)."""
        return self._output_dim


# Register the strategy
FusionStrategyRegistry.register("pca", PCAFusion)