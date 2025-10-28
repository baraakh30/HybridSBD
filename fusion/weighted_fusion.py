"""
Weighted sum fusion strategy with genetic algorithm optimization.
"""

import numpy as np
from typing import Dict, Optional, List
from deap import base, creator, tools, algorithms
import random

from .base import BaseFusionStrategy, FusionStrategyRegistry


class WeightedFusion(BaseFusionStrategy):
    """
    Strategy 2: Weighted Sum of Individual Distances
    
    Computes individual distances for each feature type and combines them
    using optimized weights via genetic algorithm.
    """
    
    def __init__(self, feature_dims: Dict[str, int], 
                 optimization_method: str = "genetic",
                 manual_weights: Optional[Dict[str, float]] = None,
                 ga_population: int = 50,
                 ga_generations: int = 100,
                 device: str = "auto"):
        """
        Initialize weighted fusion.
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            optimization_method: Weight optimization method ('genetic', 'manual')
            manual_weights: Manual weights if not using optimization
            ga_population: Genetic algorithm population size
            ga_generations: Number of GA generations
            device: Device to run on
        """
        super().__init__(feature_dims, device)
        self.optimization_method = optimization_method
        self.manual_weights = manual_weights
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        
        # Initialize weights
        self.weights = {}
        self.feature_names = sorted(feature_dims.keys())
        # Output dim is the sum of all feature dimensions (concatenation before weighting)
        self._output_dim = sum(feature_dims.values())
        
        # For genetic algorithm
        self.ground_truth_boundaries = None
        self.training_features = None
        
    def initialize(self) -> None:
        """Initialize the fusion strategy."""
        if self.optimization_method == "manual" and self.manual_weights:
            self.weights = self.manual_weights.copy()
        else:
            # Initialize with equal weights
            num_features = len(self.feature_names)
            for name in self.feature_names:
                self.weights[name] = 1.0 / num_features
        
        self.is_initialized = True
    
    def set_training_data(self, features: Dict[str, np.ndarray], 
                         boundaries: List[int]) -> None:
        """
        Set training data for weight optimization.
        
        Args:
            features: Training feature arrays
            boundaries: Ground truth boundary frame indices
        """
        self.training_features = features
        self.ground_truth_boundaries = boundaries
    
    def optimize_weights(self) -> None:
        """Optimize weights using genetic algorithm."""
        if self.training_features is None or self.ground_truth_boundaries is None:
            print("Warning: No training data set for weight optimization")
            return
        
        if self.optimization_method != "genetic":
            return
        
        print("Optimizing weights using genetic algorithm...")
        
        # Setup DEAP for genetic algorithm
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Gene generator (weight values between 0 and 1)
        toolbox.register("weight", random.uniform, 0.0, 1.0)
        
        # Individual generator (list of weights)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.weight, len(self.feature_names))
        
        # Population generator
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation function
        toolbox.register("evaluate", self._evaluate_weights)
        
        # Genetic operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run genetic algorithm
        population = toolbox.population(n=self.ga_population)
        
        algorithms.eaSimple(
            population, toolbox,
            cxpb=0.7, mutpb=0.3,
            ngen=self.ga_generations,
            verbose=False
        )
        
        # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        
        # Normalize weights
        weight_sum = sum(best_individual)
        normalized_weights = [w / weight_sum for w in best_individual]
        
        # Update weights
        for i, name in enumerate(self.feature_names):
            self.weights[name] = normalized_weights[i]
        
        print(f"Optimized weights: {self.weights}")
    
    def _evaluate_weights(self, individual: List[float]) -> tuple:
        """
        Evaluate a set of weights using F1-score.
        
        Args:
            individual: List of weight values
            
        Returns:
            Tuple containing fitness score
        """
        # Normalize weights
        weight_sum = sum(individual)
        if weight_sum == 0:
            return (0.0,)
        
        normalized_weights = [w / weight_sum for w in individual]
        
        # Temporarily set weights
        old_weights = self.weights.copy()
        for i, name in enumerate(self.feature_names):
            self.weights[name] = normalized_weights[i]
        
        try:
            # Compute distances with these weights
            distances = self._compute_weighted_distances(self.training_features)
            
            # Simple threshold-based boundary detection for evaluation
            threshold = np.mean(distances) + 1.5 * np.std(distances)
            detected_boundaries = np.where(distances > threshold)[0]
            
            # Compute F1-score
            f1_score = self._compute_f1_score(detected_boundaries, self.ground_truth_boundaries)
            
        except Exception:
            f1_score = 0.0
        finally:
            # Restore original weights
            self.weights = old_weights
        
        return (f1_score,)
    
    def _compute_f1_score(self, detected: np.ndarray, ground_truth: List[int], 
                         tolerance: int = 5) -> float:
        """
        Compute F1-score for boundary detection.
        
        Args:
            detected: Detected boundary indices
            ground_truth: Ground truth boundary indices
            tolerance: Temporal tolerance in frames
            
        Returns:
            F1-score
        """
        if len(detected) == 0 and len(ground_truth) == 0:
            return 1.0
        
        if len(detected) == 0 or len(ground_truth) == 0:
            return 0.0
        
        # Count true positives
        tp = 0
        for gt_boundary in ground_truth:
            for det_boundary in detected:
                if abs(det_boundary - gt_boundary) <= tolerance:
                    tp += 1
                    break
        
        # Compute precision and recall
        precision = tp / len(detected) if len(detected) > 0 else 0.0
        recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        # Compute F1-score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features using weighted concatenation.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Weighted concatenated feature array of shape (N, total_dim)
        """
        if not self.is_initialized:
            self.initialize()
        
        self._validate_features(features)
        
        # Normalize and weight each feature type before concatenation
        weighted_features = []
        
        for name in sorted(features.keys()):  # Sort for consistent ordering
            feat = features[name]
            
            # L2 normalization per feature vector
            feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
            
            # Apply weight by scaling the feature
            weight = self.weights.get(name, 1.0 / len(features))
            weighted_feat = feat_norm * weight
            weighted_features.append(weighted_feat)
        
        # Concatenate all weighted features
        fused = np.concatenate(weighted_features, axis=1)
        
        return fused
    
    def _compute_weighted_distances(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted sum of individual distances."""
        self._validate_features(features)
        
        # Compute individual distances for each feature type
        individual_distances = {}
        
        for name in self.feature_names:
            if name in features:
                feat = features[name]
                
                if len(feat) < 2:
                    individual_distances[name] = np.array([0.0])
                else:
                    # Normalize features
                    feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
                    
                    # Compute frame-to-frame differences
                    diff = feat_norm[1:] - feat_norm[:-1]
                    
                    # Compute Euclidean distances
                    distances = np.linalg.norm(diff, axis=1)
                    individual_distances[name] = distances
        
        # Combine distances using weights
        if not individual_distances:
            return np.array([0.0])
        
        # Get the length of distance arrays (should be same for all)
        distance_length = len(list(individual_distances.values())[0])
        weighted_distances = np.zeros(distance_length)
        
        for name in self.feature_names:
            if name in individual_distances:
                weight = self.weights.get(name, 0.0)
                weighted_distances += weight * individual_distances[name]
        
        return weighted_distances
    
    def compute_distances(self, fused_features: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances from weighted fused features.
        
        Args:
            fused_features: Weighted fused features of shape (N, total_dim)
            
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
    
    @property
    def name(self) -> str:
        """Return the name of the fusion strategy."""
        return "weighted"
    
    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the fused features."""
        return self._output_dim


# Register the strategy
FusionStrategyRegistry.register("weighted", WeightedFusion)