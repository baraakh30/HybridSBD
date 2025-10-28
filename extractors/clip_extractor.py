"""
CLIP Feature Extractor.
"""

import torch
import numpy as np
import clip
from PIL import Image
from typing import Union
import logging

from .base import BaseFeatureExtractor, FeatureExtractorRegistry


class CLIPFeatureExtractor(BaseFeatureExtractor):
    """
    CLIP Feature Extractor.
    
    Extracts 512-dimensional semantic features using OpenAI's CLIP model.
    Supports frame sampling to reduce computational cost for long videos.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto", 
                 frame_sampling_rate: int = 1, interpolation_method: str = "linear"):
        """
        Initialize CLIP feature extractor.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run on
            frame_sampling_rate: Process every Nth frame (1 = all frames, 5 = every 5th frame)
            interpolation_method: Method to interpolate features ('linear', 'nearest', 'cubic')
        """
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.frame_sampling_rate = max(1, frame_sampling_rate)
        self.interpolation_method = interpolation_method
        
        # Model dimensions mapping
        self._model_dims = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
        }
        
        self._feature_dim = self._model_dims.get(model_name, 512)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self) -> None:
        """Initialize the CLIP model."""
        try:
            self.model, self.preprocess = clip.load(
                self.model_name, device=self.device
            )
            self.model.eval()
            self.is_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")
    
    def extract_batch(self, frames: Union[np.ndarray, torch.Tensor], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Extract features in batches with frame sampling applied to the entire sequence.
        
        Args:
            frames: Input frames
            batch_size: Batch size for processing individual frames
            
        Returns:
            Feature array with same number of frames as input
        """
        if not self.is_initialized:
            self.initialize()
        
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # Ensure frames are in uint8 format
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        total_frames = len(frames)
        
        # Apply sampling at the sequence level
        if self.frame_sampling_rate > 1:
            sampled_indices = np.arange(0, total_frames, self.frame_sampling_rate)
            sampled_frames = frames[sampled_indices]
            self.logger.info(f"CLIP: Sampling {len(sampled_frames)} out of {total_frames} frames (rate: {self.frame_sampling_rate})")
            
            # Extract features from sampled frames in batches
            sampled_features = self._extract_features_from_frames(sampled_frames, batch_size)
            
            # Interpolate to match original frame count
            if len(sampled_features) > 1:
                features = self._interpolate_features(sampled_features, sampled_indices, total_frames)
            else:
                features = np.tile(sampled_features[0], (total_frames, 1))
        else:
            # Process all frames
            features = self._extract_features_from_frames(frames, batch_size)
        
        assert features.shape[0] == total_frames, \
            f"CLIP: Expected {total_frames} features, got {features.shape[0]}"
        
        return features
    
    def _extract_features_from_frames(self, frames: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Extract features from frames in batches (internal method).
        
        Args:
            frames: Input frames to process
            batch_size: Batch size for processing
            
        Returns:
            Feature array
        """
        features_list = []
        num_frames = len(frames)
        
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_features = []
                
                for frame in batch_frames:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Preprocess the image
                    image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                    
                    # Extract image features
                    image_features = self.model.encode_image(image_tensor)
                    
                    # Normalize features
                    image_features = torch.nn.functional.normalize(
                        image_features, p=2, dim=1
                    )
                    
                    batch_features.append(image_features.cpu().numpy())
                
                if batch_features:
                    features_list.append(np.concatenate(batch_features, axis=0))
        
        return np.concatenate(features_list, axis=0) if features_list else np.array([])
    
    def extract_features(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract CLIP features from frames (legacy method, use extract_batch instead).
        
        Args:
            frames: Input frames of shape (N, H, W, C) in RGB format
            
        Returns:
            Feature array of shape (N, 512)
        """
        # Delegate to extract_batch for consistency
        return self.extract_batch(frames)
    
    def _interpolate_features(self, sampled_features: np.ndarray, sampled_indices: np.ndarray, 
                             total_frames: int) -> np.ndarray:
        """
        Interpolate features to match the original frame count.
        
        Args:
            sampled_features: Features extracted from sampled frames (M, feature_dim)
            sampled_indices: Indices of sampled frames
            total_frames: Total number of original frames
            
        Returns:
            Interpolated features (total_frames, feature_dim)
        """
        if len(sampled_features) == 1:
            # Only one sample, replicate it
            return np.tile(sampled_features[0], (total_frames, 1))
        
        # Ensure we have exactly total_frames output
        target_indices = np.arange(total_frames)
        interpolated_features = np.zeros((total_frames, sampled_features.shape[1]))
        
        try:
            from scipy.interpolate import interp1d
            
            # Create interpolation function for each feature dimension
            for dim in range(sampled_features.shape[1]):
                if self.interpolation_method == 'linear':
                    f = interp1d(sampled_indices, sampled_features[:, dim], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
                elif self.interpolation_method == 'cubic':
                    if len(sampled_indices) >= 4:
                        f = interp1d(sampled_indices, sampled_features[:, dim], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                    else:
                        f = interp1d(sampled_indices, sampled_features[:, dim], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                else:  # nearest
                    f = interp1d(sampled_indices, sampled_features[:, dim], 
                               kind='nearest', bounds_error=False, fill_value='extrapolate')
                
                interpolated_features[:, dim] = f(target_indices)
                
        except ImportError:
            # Fallback: simple linear interpolation without scipy
            self.logger.warning("Scipy not available, using simple interpolation")
            for dim in range(sampled_features.shape[1]):
                interpolated_features[:, dim] = np.interp(
                    target_indices, sampled_indices, sampled_features[:, dim]
                )
        
        # Ensure we return exactly the right number of features
        assert interpolated_features.shape[0] == total_frames, \
            f"Expected {total_frames} features, got {interpolated_features.shape[0]}"
        
        return interpolated_features
    
    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the extracted features."""
        return self._feature_dim
    
    @property
    def name(self) -> str:
        """Return the name of the feature extractor."""
        return f"clip_{self.model_name.replace('/', '_')}"


# Register the extractor
FeatureExtractorRegistry.register("clip", CLIPFeatureExtractor)