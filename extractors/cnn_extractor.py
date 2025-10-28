"""
CNN Feature Extractor using ResNet-50.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from typing import Union
import logging

from .base import BaseFeatureExtractor, FeatureExtractorRegistry


class CNNFeatureExtractor(BaseFeatureExtractor):
    """
    CNN Feature Extractor using ResNet-50.
    
    Extracts 2048-dimensional features from the penultimate layer
    of a pretrained ResNet-50 model.
    Supports frame sampling to reduce computational cost for long videos.
    """
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True,
                 device: str = "auto", frame_sampling_rate: int = 1, 
                 interpolation_method: str = "linear"):
        """
        Initialize CNN feature extractor.
        
        Args:
            model_name: Name of the CNN model to use
            pretrained: Whether to use pretrained weights
            device: Device to run on
            frame_sampling_rate: Process every Nth frame (1 = all frames, 5 = every 5th frame)
            interpolation_method: Method to interpolate features ('linear', 'nearest', 'cubic')
        """
        super().__init__(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.transform = None
        self._feature_dim = 2048  # ResNet-50 penultimate layer size
        self.frame_sampling_rate = max(1, frame_sampling_rate)
        self.interpolation_method = interpolation_method
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self) -> None:
        """Initialize the CNN model and preprocessing."""
        # Load the model
        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=self.pretrained)
            self._feature_dim = 2048
        elif self.model_name == "resnet101":
            self.model = models.resnet101(pretrained=self.pretrained)
            self._feature_dim = 2048
        elif self.model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=self.pretrained)
            self._feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Remove the final classification layer
        if hasattr(self.model, 'fc'):
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif hasattr(self.model, 'classifier'):
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.is_initialized = True
    
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
            self.logger.info(f"CNN: Sampling {len(sampled_frames)} out of {total_frames} frames (rate: {self.frame_sampling_rate})")
            
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
            f"CNN: Expected {total_frames} features, got {features.shape[0]}"
        
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
                    # Preprocess frame
                    if self.transform:
                        frame_tensor = self.transform(frame)
                    else:
                        # Manual preprocessing if transform fails
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        frame_tensor = torch.nn.functional.interpolate(
                            frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                        ).squeeze(0)
                    
                    # Add batch dimension and move to device
                    frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                    
                    # Extract features
                    feature_vector = self.model(frame_tensor)
                    
                    # Global average pooling if needed
                    if len(feature_vector.shape) > 2:
                        feature_vector = torch.nn.functional.adaptive_avg_pool2d(
                            feature_vector, (1, 1)
                        )
                    
                    # Flatten and normalize
                    feature_vector = feature_vector.view(feature_vector.size(0), -1)
                    feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
                    
                    batch_features.append(feature_vector.cpu().numpy())
                
                if batch_features:
                    features_list.append(np.concatenate(batch_features, axis=0))
        
        return np.concatenate(features_list, axis=0) if features_list else np.array([])
    
    def extract_features(self, frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract CNN features from frames (legacy method, use extract_batch instead).
        
        Args:
            frames: Input frames of shape (N, H, W, C) in RGB format
            
        Returns:
            Feature array of shape (N, 2048)
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
        return f"cnn_{self.model_name}"


# Register the extractor
FeatureExtractorRegistry.register("cnn", CNNFeatureExtractor)