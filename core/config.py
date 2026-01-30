"""
Configuration management for video shot detection system.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    hsv_enabled: bool = True
    hsv_bins: int = 256
    hsv_smoothing: bool = True
    hsv_frame_sampling_rate: int = 1  # Process every Nth frame for HSV (1 = all frames)
    hsv_interpolation_method: str = "linear"
    
    cnn_enabled: bool = True
    cnn_model: str = "resnet50"
    cnn_pretrained: bool = True
    cnn_frame_sampling_rate: int = 6  # Process every 6th frame for CNN (expensive)
    cnn_interpolation_method: str = "linear"
    
    clip_enabled: bool = True
    clip_model: str = "ViT-B/32"
    clip_frame_sampling_rate: int = 15  # Process every 15th frame for CLIP (expensive)
    clip_interpolation_method: str = "linear"
    
    optical_flow_enabled: bool = True
    optical_flow_method: str = "farneback"  # farneback, tvl1
    optical_flow_bins: int = 40
    optical_flow_frame_sampling_rate: int = 6  # Process every 6th frame for optical flow
    optical_flow_interpolation_method: str = "linear"


@dataclass
class FusionConfig:
    """Configuration for fusion strategies."""
    strategy: str = "concatenation"  # concatenation, weighted, pca, transformer, hierarchical
    
    # Weighted fusion parameters
    weights_optimization: str = "genetic"  # genetic, grid_search, manual
    ga_population: int = 50
    ga_generations: int = 100
    manual_weights: Optional[Dict[str, float]] = None
    
    # PCA parameters
    pca_variance_retained: float = 0.95
    
    # Transformer fusion parameters
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_hidden_dim: int = 512
    transformer_dropout: float = 0.1
    
    # Hierarchical fusion parameters
    hierarchical_conv_kernel: int = 3
    hierarchical_attention_heads: int = 4


@dataclass
class DetectionConfig:
    """Configuration for detection methods."""
    method: str = "statistical"  # statistical, transformer, both
    
    # Statistical detection parameters
    smoothing_method: str = "gaussian"  # gaussian, savgol
    smoothing_window: int = 5
    z_score_window: int = 30
    
    # Multi-scale analysis
    short_scale: int = 15
    medium_scale: int = 45
    long_scale: int = 75
    
    # Dynamic thresholding
    threshold_method: str = "adaptive"  # fixed, adaptive, percentile
    fixed_threshold: float = 0.5
    adaptive_beta: float = 1.5
    
    # Transformer detection parameters
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_hidden_dim: int = 512
    transformer_ff_dim: int = 2048
    transformer_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Loss function configuration
    loss_type: str = "multi_task"  # "focal" or "bce" - USE BCE for now, it's simpler and more stable
    focal_alpha: float = 0.25  # For focal loss: balanced (if you switch back to focal)
    focal_gamma: float = 2.0   # For focal loss: focusing parameter
    # class_weights: Dict[str, float] = field(default_factory=lambda: {"negative": 1.0, "positive": 30.0})  # Balanced weight for 1:85 imbalance


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    frame_rate: int = 30
    target_size: tuple = (224, 224)
    sequence_length: int = 100
    overlap: int = 50
    min_shot_duration: int = 30
    temporal_tolerance: int = 5
    
    # Performance settings
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Parallel processing settings
    parallel_feature_extraction: bool = False
    feature_extraction_workers: int = 4  # Max workers for parallel feature extraction


@dataclass
class LoggingConfig:
    """Configuration for logging and progress display."""
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file_logging: bool = True
    console_logging: bool = True
    progress_bars: bool = True
    quiet_mode: bool = False  # Suppress all non-essential output
    log_to_file_only: bool = False  # Only log to file, no console output


@dataclass
class VideoShotConfig:
    """Main configuration class for video shot detection."""
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths and directories
    data_dir: str = "data"
    output_dir: str = "output"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # Experiment settings
    experiment_name: str = "shot_detection"
    seed: int = 42
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "VideoShotConfig":
        """Load configuration from YAML file."""
        config_dict = OmegaConf.load(config_path)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = OmegaConf.structured(self)
        OmegaConf.save(config_dict, config_path)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.output_dir,
            self.model_dir,
            self.log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Default configurations for different use cases
DEFAULT_RESEARCH_CONFIG = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        cnn_frame_sampling_rate=6,  # High quality for research
        clip_enabled=True,
        clip_frame_sampling_rate=15,  # High quality for research
        optical_flow_enabled=True,
        optical_flow_frame_sampling_rate=6  # Fine-grained motion analysis
    ),
    fusion=FusionConfig(strategy="hierarchical"),
    detection=DetectionConfig(method="both"),
    processing=ProcessingConfig(
        sequence_length=100,
        batch_size=16,
        num_workers=4
    )
)

DEFAULT_PRODUCTION_CONFIG = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        cnn_frame_sampling_rate=5,  # Balanced for production
        clip_enabled=False,  # Disabled for production speed
        optical_flow_enabled=True,
        optical_flow_frame_sampling_rate=3  # Moderate sampling
    ),
    fusion=FusionConfig(strategy="weighted"),
    detection=DetectionConfig(method="statistical"),
    processing=ProcessingConfig(
        sequence_length=50,
        batch_size=32,
        num_workers=8
    )
)

DEFAULT_EDGE_CONFIG = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=False,  # Disabled for edge devices
        clip_enabled=False,  # Disabled for edge devices
        optical_flow_enabled=True,
        optical_flow_frame_sampling_rate=5  # Aggressive sampling for edge
    ),
    fusion=FusionConfig(strategy="concatenation"),
    detection=DetectionConfig(method="statistical"),
    processing=ProcessingConfig(
        sequence_length=25,
        batch_size=64,
        num_workers=2
    )
)