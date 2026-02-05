"""
Post-Processing Module for Shot Boundary Detection

GOAL: Improve gradual F1 from 0.58-0.62 to 0.70+ WITHOUT retraining

STRATEGIES:
1. Temporal smoothing (graduals are smooth, cuts are sharp)
2. Duration-based refinement (graduals last 5-30 frames, cuts are instant)
3. Confidence-based adaptive thresholds (different thresholds for cut vs gradual)
4. Local context analysis (neighborhood consistency)
5. Multi-frame consensus (graduals should agree across frames)

PUBLICATION QUALITY:
- Ablation study ready (can enable/disable each component)
- Statistically grounded (median filter, moving averages)
- Interpretable (each step has clear motivation)
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from typing import Dict, List, Tuple


class ShotBoundaryPostProcessor:
    """
    Post-processing for shot boundary detection
    
    Focuses on improving gradual transition detection through:
    - Temporal smoothing
    - Duration analysis
    - Adaptive thresholding
    - Consensus voting
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Post-processing configuration
        """
        self.config = config or {}
        
        # Thresholds
        self.boundary_threshold_cut = self.config.get('boundary_threshold_cut', 0.70)
        self.boundary_threshold_gradual = self.config.get('boundary_threshold_gradual', 0.55)
        
        # Smoothing
        self.enable_temporal_smoothing = self.config.get('enable_temporal_smoothing', True)
        self.smoothing_window = self.config.get('smoothing_window', 5)
        
        # Duration-based refinement
        self.enable_duration_refinement = self.config.get('enable_duration_refinement', True)
        self.min_gradual_duration = self.config.get('min_gradual_duration', 5)
        self.max_cut_duration = self.config.get('max_cut_duration', 3)
        
        # Consensus voting
        self.enable_consensus = self.config.get('enable_consensus', True)
        self.consensus_window = self.config.get('consensus_window', 7)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.5)
        
        # Type refinement
        self.enable_type_refinement = self.config.get('enable_type_refinement', True)
        self.type_confidence_threshold = self.config.get('type_confidence_threshold', 0.6)
        
        print("Post-Processor Configuration:")
        print(f"  Boundary threshold (cut):     {self.boundary_threshold_cut:.2f}")
        print(f"  Boundary threshold (gradual): {self.boundary_threshold_gradual:.2f}")
        print(f"  Temporal smoothing:           {self.enable_temporal_smoothing} (window={self.smoothing_window})")
        print(f"  Duration refinement:          {self.enable_duration_refinement}")
        print(f"  Consensus voting:             {self.enable_consensus} (window={self.consensus_window})")
        print(f"  Type refinement:              {self.enable_type_refinement}")
    
    def process(self, 
                boundary_logits: torch.Tensor,
                type_logits: torch.Tensor,
                return_intermediate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply post-processing to model predictions
        
        Args:
            boundary_logits: [B, T, 1] boundary detection logits
            type_logits: [B, T, 2] type classification logits (cut vs gradual)
            return_intermediate: If True, return intermediate results
            
        Returns:
            refined_boundaries: [B, T] binary boundary predictions
            refined_types: [B, T] type predictions (0=none, 1=cut, 2=gradual)
        """
        B, T, _ = boundary_logits.shape
        device = boundary_logits.device
        
        # Convert to probabilities
        boundary_probs = torch.sigmoid(boundary_logits).squeeze(-1)  # [B, T]
        type_probs = F.softmax(type_logits, dim=-1)  # [B, T, 2]
        
        # Storage for intermediate results
        intermediate = {} if return_intermediate else None
        
        # Process each sample in batch
        refined_boundaries_list = []
        refined_types_list = []
        
        for b in range(B):
            b_probs = boundary_probs[b].cpu().numpy()  # [T]
            t_probs = type_probs[b].cpu().numpy()  # [T, 2]
            
            # Step 1: Temporal smoothing
            if self.enable_temporal_smoothing:
                b_probs_smooth = self._temporal_smoothing(b_probs)
                t_probs_smooth = self._temporal_smoothing_2d(t_probs)
            else:
                b_probs_smooth = b_probs
                t_probs_smooth = t_probs
            
            if return_intermediate:
                intermediate[f'sample_{b}_boundary_raw'] = b_probs
                intermediate[f'sample_{b}_boundary_smooth'] = b_probs_smooth
            
            # Step 2: Adaptive thresholding (different thresholds for cut vs gradual)
            boundaries_init, types_init = self._adaptive_thresholding(
                b_probs_smooth, t_probs_smooth
            )
            
            if return_intermediate:
                intermediate[f'sample_{b}_boundaries_init'] = boundaries_init
                intermediate[f'sample_{b}_types_init'] = types_init
            
            # Step 3: Duration-based refinement
            if self.enable_duration_refinement:
                boundaries_refined, types_refined = self._duration_refinement(
                    boundaries_init, types_init, b_probs_smooth, t_probs_smooth
                )
            else:
                boundaries_refined = boundaries_init
                types_refined = types_init
            
            if return_intermediate:
                intermediate[f'sample_{b}_boundaries_duration'] = boundaries_refined
                intermediate[f'sample_{b}_types_duration'] = types_refined
            
            # Step 4: Consensus voting
            if self.enable_consensus:
                boundaries_consensus, types_consensus = self._consensus_voting(
                    boundaries_refined, types_refined, b_probs_smooth, t_probs_smooth
                )
            else:
                boundaries_consensus = boundaries_refined
                types_consensus = types_refined
            
            if return_intermediate:
                intermediate[f'sample_{b}_boundaries_consensus'] = boundaries_consensus
                intermediate[f'sample_{b}_types_consensus'] = types_consensus
            
            # Step 5: Type refinement based on confidence
            if self.enable_type_refinement:
                types_final = self._type_refinement(
                    types_consensus, t_probs_smooth, boundaries_consensus
                )
            else:
                types_final = types_consensus
            
            # Convert back to tensors
            refined_boundaries_list.append(torch.from_numpy(boundaries_consensus).to(device))
            refined_types_list.append(torch.from_numpy(types_final).to(device))
        
        # Stack batches
        refined_boundaries = torch.stack(refined_boundaries_list, dim=0)  # [B, T]
        refined_types = torch.stack(refined_types_list, dim=0)  # [B, T]
        
        if return_intermediate:
            return refined_boundaries, refined_types, intermediate
        
        return refined_boundaries, refined_types
    
    def _temporal_smoothing(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to probabilities
        
        Uses median filter (preserves edges better than mean)
        """
        # Median filter (better for preserving sharp cuts)
        smoothed = median_filter(probs, size=self.smoothing_window, mode='nearest')
        
        # Slight averaging to smooth graduals
        smoothed = uniform_filter1d(smoothed, size=3, mode='nearest')
        
        return smoothed
    
    def _temporal_smoothing_2d(self, probs: np.ndarray) -> np.ndarray:
        """Apply smoothing to 2D probability array (type probabilities)"""
        T, C = probs.shape
        smoothed = np.zeros_like(probs)
        
        for c in range(C):
            smoothed[:, c] = self._temporal_smoothing(probs[:, c])
        
        # Re-normalize
        smoothed = smoothed / (smoothed.sum(axis=1, keepdims=True) + 1e-8)
        
        return smoothed
    
    def _adaptive_thresholding(self, 
                               boundary_probs: np.ndarray,
                               type_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use different thresholds for cuts vs graduals
        
        Key insight: Graduals have lower boundary confidence but higher type confidence
        """
        T = len(boundary_probs)
        boundaries = np.zeros(T, dtype=np.int32)
        types = np.zeros(T, dtype=np.int32)
        
        # Determine likely type (0=cut, 1=gradual)
        type_prediction = np.argmax(type_probs, axis=1)
        type_confidence = np.max(type_probs, axis=1)
        
        for t in range(T):
            # If high confidence gradual → use lower threshold
            if type_prediction[t] == 1 and type_confidence[t] > 0.5:
                threshold = self.boundary_threshold_gradual
                if boundary_probs[t] > threshold:
                    boundaries[t] = 1
                    types[t] = 2  # Gradual
            # Otherwise use standard threshold for cuts
            else:
                threshold = self.boundary_threshold_cut
                if boundary_probs[t] > threshold:
                    boundaries[t] = 1
                    types[t] = 1  # Cut
        
        return boundaries, types
    
    def _duration_refinement(self,
                            boundaries: np.ndarray,
                            types: np.ndarray,
                            boundary_probs: np.ndarray,
                            type_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine based on transition duration
        
        Logic:
        - Cuts should be 1-3 frames (instant)
        - Graduals should be 5-30 frames (smooth)
        - If duration doesn't match type → reconsider
        """
        T = len(boundaries)
        refined_boundaries = boundaries.copy()
        refined_types = types.copy()
        
        # Find transition regions
        regions = self._extract_regions(boundaries, types)
        
        for region in regions:
            start, end, region_type = region
            duration = end - start + 1
            center = (start + end) // 2
            
            # Get average probabilities in region
            avg_boundary_prob = np.mean(boundary_probs[start:end+1])
            avg_type_probs = np.mean(type_probs[start:end+1], axis=0)
            
            # Case 1: Labeled as cut but duration is long → might be gradual
            if region_type == 1 and duration > self.max_cut_duration:
                # Check if gradual type probability is reasonable
                if avg_type_probs[1] > 0.3:  # Gradual probability > 30%
                    refined_types[start:end+1] = 2  # Change to gradual
            
            # Case 2: Labeled as gradual but duration is short → might be cut
            elif region_type == 2 and duration < self.min_gradual_duration:
                # Check if it's actually a sharp cut
                if avg_type_probs[0] > 0.4:  # Cut probability > 40%
                    refined_types[start:end+1] = 1  # Change to cut
                # Or if boundary prob is very high → sharp cut
                elif avg_boundary_prob > 0.8:
                    refined_types[start:end+1] = 1
            
            # Case 3: Very short detection (1-2 frames) with low confidence → remove
            elif duration <= 2 and avg_boundary_prob < 0.6:
                refined_boundaries[start:end+1] = 0
                refined_types[start:end+1] = 0
        
        return refined_boundaries, refined_types
    
    def _consensus_voting(self,
                         boundaries: np.ndarray,
                         types: np.ndarray,
                         boundary_probs: np.ndarray,
                         type_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-frame consensus for gradual transitions
        
        Graduals should have consistent predictions across multiple frames
        """
        T = len(boundaries)
        refined_boundaries = boundaries.copy()
        refined_types = types.copy()
        
        half_window = self.consensus_window // 2
        
        for t in range(T):
            # Get neighborhood
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Count votes in neighborhood
            neighbor_boundaries = boundaries[start:end]
            neighbor_types = types[start:end]
            
            # If current frame is boundary
            if boundaries[t] == 1:
                # Check consensus
                boundary_consensus = np.mean(neighbor_boundaries)
                
                # If isolated detection (low consensus) → might be noise
                if boundary_consensus < self.consensus_threshold:
                    # Check if probability is strong enough to override
                    if boundary_probs[t] < 0.75:
                        refined_boundaries[t] = 0
                        refined_types[t] = 0
                
                # If labeled as gradual, check type consensus
                if types[t] == 2:
                    gradual_votes = np.sum(neighbor_types == 2)
                    total_votes = np.sum(neighbor_boundaries == 1)
                    
                    if total_votes > 0:
                        gradual_ratio = gradual_votes / total_votes
                        
                        # If most neighbors are cuts → reconsider
                        if gradual_ratio < 0.3 and type_probs[t][0] > 0.4:
                            refined_types[t] = 1  # Change to cut
        
        return refined_boundaries, refined_types
    
    def _type_refinement(self,
                        types: np.ndarray,
                        type_probs: np.ndarray,
                        boundaries: np.ndarray) -> np.ndarray:
        """
        Final type refinement based on confidence
        """
        refined_types = types.copy()
        
        for t in range(len(types)):
            if boundaries[t] == 0:
                refined_types[t] = 0
                continue
            
            # Get type confidence
            type_confidence = np.max(type_probs[t])
            predicted_type = np.argmax(type_probs[t])
            
            # If very confident → trust the model
            if type_confidence > self.type_confidence_threshold:
                refined_types[t] = predicted_type + 1  # 0→1 (cut), 1→2 (gradual)
        
        return refined_types
    
    def _extract_regions(self, boundaries: np.ndarray, types: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract contiguous boundary regions"""
        regions = []
        
        i = 0
        while i < len(boundaries):
            if boundaries[i] == 1:
                start = i
                region_type = types[i]
                
                # Extend region
                while i < len(boundaries) and boundaries[i] == 1 and types[i] == region_type:
                    i += 1
                
                end = i - 1
                regions.append((start, end, region_type))
            else:
                i += 1
        
        return regions
    
    def get_config_summary(self) -> str:
        """Get human-readable configuration summary"""
        summary = []
        summary.append("Post-Processing Configuration:")
        summary.append(f"  Adaptive Thresholds:")
        summary.append(f"    Cut:     {self.boundary_threshold_cut:.2f}")
        summary.append(f"    Gradual: {self.boundary_threshold_gradual:.2f}")
        
        if self.enable_temporal_smoothing:
            summary.append(f"  Temporal Smoothing: ON (window={self.smoothing_window})")
        else:
            summary.append(f"  Temporal Smoothing: OFF")
        
        if self.enable_duration_refinement:
            summary.append(f"  Duration Refinement: ON")
            summary.append(f"    Min gradual: {self.min_gradual_duration} frames")
            summary.append(f"    Max cut: {self.max_cut_duration} frames")
        else:
            summary.append(f"  Duration Refinement: OFF")
        
        if self.enable_consensus:
            summary.append(f"  Consensus Voting: ON (window={self.consensus_window})")
        else:
            summary.append(f"  Consensus Voting: OFF")
        
        if self.enable_type_refinement:
            summary.append(f"  Type Refinement: ON (threshold={self.type_confidence_threshold})")
        else:
            summary.append(f"  Type Refinement: OFF")
        
        return "\n".join(summary)


def create_post_processor(config: Dict = None) -> ShotBoundaryPostProcessor:
    """
    Factory function for post-processor
    
    Recommended configs:
    
    CONSERVATIVE (high precision, lower recall):
    {
        'boundary_threshold_cut': 0.75,
        'boundary_threshold_gradual': 0.60,
        'min_gradual_duration': 7,
        'consensus_threshold': 0.6
    }
    
    AGGRESSIVE (high recall, lower precision):
    {
        'boundary_threshold_cut': 0.65,
        'boundary_threshold_gradual': 0.50,
        'min_gradual_duration': 4,
        'consensus_threshold': 0.4
    }
    
    BALANCED (recommended for publication):
    {
        'boundary_threshold_cut': 0.70,
        'boundary_threshold_gradual': 0.55,
        'min_gradual_duration': 5,
        'max_cut_duration': 3,
        'consensus_threshold': 0.5
    }
    """
    return ShotBoundaryPostProcessor(config)


if __name__ == '__main__':
    # Test post-processor
    print("Testing Post-Processor...")
    
    # Create dummy predictions
    B, T = 2, 128
    boundary_logits = torch.randn(B, T, 1)
    type_logits = torch.randn(B, T, 2)
    
    # Create post-processor
    config = {
        'boundary_threshold_cut': 0.70,
        'boundary_threshold_gradual': 0.55,
        'enable_temporal_smoothing': True,
        'enable_duration_refinement': True,
        'enable_consensus': True
    }
    
    post_processor = create_post_processor(config)
    
    print("\n" + post_processor.get_config_summary())
    
    # Process
    print("\nProcessing predictions...")
    boundaries, types = post_processor.process(boundary_logits, type_logits)
    
    print(f"\nOutput shapes:")
    print(f"  Boundaries: {boundaries.shape}")
    print(f"  Types: {types.shape}")
    
    # Count predictions
    n_boundaries = boundaries.sum().item()
    n_cuts = (types == 1).sum().item()
    n_graduals = (types == 2).sum().item()
    
    print(f"\nPrediction counts:")
    print(f"  Total boundaries: {n_boundaries}")
    print(f"  Cuts: {n_cuts}")
    print(f"  Graduals: {n_graduals}")
    
    print("\n✓ Post-processor test passed!")
