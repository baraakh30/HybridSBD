"""
Main processing coordinator for video shot detection.
"""

# CRITICAL: Configure OpenCV environment variables BEFORE any OpenCV imports
import os
os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '16384')  # Increase read attempts for problematic videos
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # Reduce OpenCV warning verbosity
os.environ.setdefault('OPENCV_FFMPEG_DEMUXER_OPTIONS', 'analyzeduration;10000000;probesize;50000000')  # Better stream analysis
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'video_stream_id;0')  # Force video stream only

import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import VideoShotConfig
from extractors import FeatureExtractorRegistry
from fusion import FusionStrategyRegistry
from detection import DetectionMethodRegistry
from utils.video_utils import VideoProcessor
from utils.logger import setup_logger


class ProcessingCoordinator:
    """
    Main coordinator that orchestrates the entire video shot detection pipeline.
    
    Responsibilities:
    - Video input handling
    - Feature extraction coordination  
    - Fusion strategy management
    - Detection method routing
    - Post-processing and output formatting
    """
    
    def __init__(self, config: VideoShotConfig):
        """
        Initialize the processing coordinator.
        
        Args:
            config: Complete system configuration
        """
        self.config = config
        
        # Setup logger with configuration
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
        self.logger = setup_logger(
            "ProcessingCoordinator", 
            config.log_dir, 
            level=log_level,
            enable_logging=config.logging.enabled,
            file_logging=config.logging.file_logging,
            console_logging=config.logging.console_logging,
            quiet_mode=config.logging.quiet_mode
        )
        
        # Initialize components
        self.video_processor = None
        self.feature_extractors = {}
        self.fusion_strategy = None
        self.detection_methods = {}
        
        # Results storage
        self.last_results = None
        
        # Performance tracking
        self.performance_stats = {}
        
    def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing video shot detection system...")
        
        # Create necessary directories
        self.config.create_directories()
        
        # Initialize video processor
        self.video_processor = VideoProcessor(self.config.processing)
        
        # Initialize feature extractors
        self._initialize_feature_extractors()
        
        # Initialize fusion strategy
        self._initialize_fusion_strategy()
        
        # Initialize detection methods
        self._initialize_detection_methods()
        self.logger.info('Used device: %s', self.feature_extractors['clip']._get_device(self.config.processing.device))
        self.logger.info("System initialization complete")
    
    def _initialize_feature_extractors(self) -> None:
        """Initialize enabled feature extractors."""
        feature_config = self.config.features
        
        if feature_config.hsv_enabled:
            self.feature_extractors["hsv_histogram"] = FeatureExtractorRegistry.get(
                "hsv_histogram",
                bins=feature_config.hsv_bins,
                smoothing=feature_config.hsv_smoothing,
                device=self.config.processing.device,
                frame_sampling_rate=getattr(feature_config, 'hsv_frame_sampling_rate', 1),
                interpolation_method=getattr(feature_config, 'hsv_interpolation_method', 'linear')
            )
            self.logger.info("Initialized HSV histogram extractor")
        
        if feature_config.cnn_enabled:
            self.feature_extractors["cnn"] = FeatureExtractorRegistry.get(
                "cnn",
                model_name=feature_config.cnn_model,
                pretrained=feature_config.cnn_pretrained,
                device=self.config.processing.device,
                frame_sampling_rate=feature_config.cnn_frame_sampling_rate,
                interpolation_method=feature_config.cnn_interpolation_method
            )
            self.logger.info(f"Initialized CNN extractor ({feature_config.cnn_model}) with sampling rate {feature_config.cnn_frame_sampling_rate}")
        
        if feature_config.clip_enabled:
            self.feature_extractors["clip"] = FeatureExtractorRegistry.get(
                "clip",
                model_name=feature_config.clip_model,
                device=self.config.processing.device,
                frame_sampling_rate=feature_config.clip_frame_sampling_rate,
                interpolation_method=feature_config.clip_interpolation_method
            )
            self.logger.info(f"Initialized CLIP extractor ({feature_config.clip_model}) with sampling rate {feature_config.clip_frame_sampling_rate}")
        
        if feature_config.optical_flow_enabled:
            self.feature_extractors["optical_flow"] = FeatureExtractorRegistry.get(
                "optical_flow",
                method=feature_config.optical_flow_method,
                device=self.config.processing.device,
                frame_sampling_rate=feature_config.optical_flow_frame_sampling_rate,
                interpolation_method=feature_config.optical_flow_interpolation_method
            )
            self.logger.info(f"Initialized optical flow extractor ({feature_config.optical_flow_method}) with sampling rate {feature_config.optical_flow_frame_sampling_rate}")
    
    def _initialize_fusion_strategy(self) -> None:
        """Initialize the fusion strategy."""
        fusion_config = self.config.fusion
        
        # Get feature dimensions
        feature_dims = {}
        for name, extractor in self.feature_extractors.items():
            feature_dims[name] = extractor.feature_dim
        
        # Initialize fusion strategy
        if fusion_config.strategy == "concatenation":
            self.fusion_strategy = FusionStrategyRegistry.get(
                "concatenation",
                feature_dims=feature_dims,
                device=self.config.processing.device
            )
        
        elif fusion_config.strategy == "weighted":
            self.fusion_strategy = FusionStrategyRegistry.get(
                "weighted",
                feature_dims=feature_dims,
                optimization_method=fusion_config.weights_optimization,
                manual_weights=fusion_config.manual_weights,
                ga_population=fusion_config.ga_population,
                ga_generations=fusion_config.ga_generations,
                device=self.config.processing.device
            )
        
        elif fusion_config.strategy == "pca":
            self.fusion_strategy = FusionStrategyRegistry.get(
                "pca",
                feature_dims=feature_dims,
                variance_retained=fusion_config.pca_variance_retained,
                device=self.config.processing.device
            )
        
        elif fusion_config.strategy == "cross_transformer":
            self.fusion_strategy = FusionStrategyRegistry.get(
                "cross_transformer",
                feature_dims=feature_dims,
                hidden_dim=fusion_config.transformer_hidden_dim,
                num_heads=fusion_config.transformer_heads,
                num_layers=fusion_config.transformer_layers,
                dropout=fusion_config.transformer_dropout,
                device=self.config.processing.device
            )
        
        elif fusion_config.strategy == "hierarchical":
            self.fusion_strategy = FusionStrategyRegistry.get(
                "hierarchical",
                feature_dims=feature_dims,
                conv_kernel=fusion_config.hierarchical_conv_kernel,
                attention_heads=fusion_config.hierarchical_attention_heads,
                device=self.config.processing.device
            )
        
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_config.strategy}")
        
        self.logger.info(f"Initialized fusion strategy: {fusion_config.strategy}")
    
    def _initialize_detection_methods(self) -> None:
        """Initialize detection methods."""
        detection_config = self.config.detection
        
        if detection_config.method in ["statistical", "both"]:
            stat_config = {
                "smoothing_method": detection_config.smoothing_method,
                "smoothing_window": detection_config.smoothing_window,
                "z_score_window": detection_config.z_score_window,
                "short_scale": detection_config.short_scale,
                "medium_scale": detection_config.medium_scale,
                "long_scale": detection_config.long_scale,
                "threshold_method": detection_config.threshold_method,
                "fixed_threshold": detection_config.fixed_threshold,
                "adaptive_beta": detection_config.adaptive_beta,
                "min_shot_duration": self.config.processing.min_shot_duration,
                "temporal_tolerance": self.config.processing.temporal_tolerance,
            }
            
            self.detection_methods["statistical"] = DetectionMethodRegistry.get(
                "statistical", config=stat_config
            )
            self.logger.info("Initialized statistical detection method")
        
        if detection_config.method in ["transformer", "both"]:
            # Determine input dimension based on fusion strategy
            if hasattr(self.fusion_strategy, 'output_dim'):
                input_dim = self.fusion_strategy.output_dim
            else:
                input_dim = sum(extractor.feature_dim for extractor in self.feature_extractors.values())
            
            transformer_config = {
                "input_dim": input_dim,
                "hidden_dim": detection_config.transformer_hidden_dim,
                "num_layers": detection_config.transformer_layers,
                "num_heads": detection_config.transformer_heads,
                "ff_dim": detection_config.transformer_ff_dim,
                "dropout": detection_config.transformer_dropout,
                "sequence_length": self.config.processing.sequence_length,
                "confidence_threshold": 0.5,
                "device": self.config.processing.device,
                "min_shot_duration": self.config.processing.min_shot_duration,
            }
            
            self.detection_methods["transformer"] = DetectionMethodRegistry.get(
                "transformer", config=transformer_config
            )
            self.logger.info("Initialized transformer detection method")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a video file and detect shot boundaries.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path for output results
            
        Returns:
            Dictionary containing detection results and metadata
        """
        self.logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load and preprocess video
        frames = self.video_processor.load_video(video_path)
        self.logger.info(f"Loaded {len(frames)} frames from video")
        
        # Extract features (with optional parallel processing)
        use_parallel = self.config.processing.parallel_feature_extraction
        max_workers = self.config.processing.feature_extraction_workers if use_parallel else None
        
        features = self._extract_all_features(frames, parallel=use_parallel, max_workers=max_workers)
        
        # Apply fusion strategy
        if self.config.fusion.strategy == "weighted" and hasattr(self.fusion_strategy, 'optimize_weights'):
            # For weighted fusion, we might want to optimize weights if training data is available
            pass
        
        if self.config.fusion.strategy == "pca" and hasattr(self.fusion_strategy, 'fit'):
            # For PCA fusion, fit on the current data if not already fitted
            if not getattr(self.fusion_strategy, 'is_fitted', False):
                self.fusion_strategy.fit(features)
        
        # Fuse features and compute distances/representations
        if self.config.fusion.strategy == "weighted":
            # Weighted fusion returns distances directly
            distance_signal = self.fusion_strategy(features)
            fused_features = None
        else:
            # Other strategies return fused features
            fused_features = self.fusion_strategy.fuse_features(features)
            distance_signal = self.fusion_strategy.compute_distances(fused_features)
        
        # Detect boundaries using enabled methods
        detection_results = {}
        
        if "statistical" in self.detection_methods:
            stat_boundaries = self.detection_methods["statistical"](distance_signal)
            detection_results["statistical"] = stat_boundaries
            self.logger.info(f"Statistical method detected {len(stat_boundaries)} boundaries")
        
        if "transformer" in self.detection_methods:
            if fused_features is not None:
                transformer_boundaries = self.detection_methods["transformer"].detect_boundaries(fused_features)
                detection_results["transformer"] = transformer_boundaries
                self.logger.info(f"Transformer method detected {len(transformer_boundaries)} boundaries")
            else:
                self.logger.warning("Transformer detection requires fused features, skipping")
        
        # Combine results if using both methods
        if len(detection_results) > 1:
            combined_boundaries = self._combine_detection_results(detection_results)
            detection_results["combined"] = combined_boundaries
        
        # Prepare final results
        processing_time = time.time() - start_time
        
        results = {
            "video_path": video_path,
            "num_frames": len(frames),
            "frame_rate": self.video_processor.get_frame_rate(video_path),
            "detection_results": detection_results,
            "fusion_strategy": self.config.fusion.strategy,
            "detection_methods": list(self.detection_methods.keys()),
            "processing_time": processing_time,
            "performance_stats": self._compute_performance_stats(frames, features, processing_time)
        }
        
        # Save results if output path provided
        if output_path:
            self._save_results(results, output_path)
        
        self.last_results = results
        self.logger.info(f"Video processing complete in {processing_time:.2f} seconds")
        
        return results
    
    def _extract_feature_worker(self, name: str, extractor, frames: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Worker function to extract features from a single extractor.
        
        Args:
            name: Name of the feature extractor
            extractor: Feature extractor instance
            frames: Video frames array
            
        Returns:
            Tuple of (feature_name, features, extraction_time)
        """
        start_time = time.time()
        
        try:
            if name == "optical_flow":
                # Optical flow needs sequential frames
                feat = extractor.extract_batch(frames, batch_size=self.config.processing.batch_size)
            else:
                feat = extractor.extract_batch(frames, batch_size=self.config.processing.batch_size)
            
            extraction_time = time.time() - start_time
            self.logger.info(f"Extracted {name} features: {feat.shape} in {extraction_time:.2f}s")
            
            return name, feat, extraction_time
            
        except Exception as e:
            self.logger.error(f"Failed to extract {name} features: {e}")
            raise RuntimeError(f"Feature extraction failed for {name}: {e}")

    def _extract_all_features_parallel(self, frames: np.ndarray, max_workers: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features using all enabled extractors in parallel.
        
        Args:
            frames: Video frames array
            max_workers: Maximum number of parallel workers (None = number of extractors)
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if max_workers is None:
            max_workers = len(self.feature_extractors)
        
        features = {}
        extraction_times = {}
        
        # Use ThreadPoolExecutor to parallelize feature extraction
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all feature extraction tasks
            future_to_name = {
                executor.submit(self._extract_feature_worker, name, extractor, frames): name
                for name, extractor in self.feature_extractors.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                feature_name = future_to_name[future]
                try:
                    name, feat, extraction_time = future.result()
                    features[name] = feat
                    extraction_times[name] = extraction_time
                except Exception as e:
                    self.logger.error(f"Feature extraction failed for {feature_name}: {e}")
                    raise
        
        # Log summary
        total_time = max(extraction_times.values()) if extraction_times else 0
        self.logger.info(f"Parallel feature extraction completed in {total_time:.2f}s")
        
        return features

    def _extract_all_features(self, frames: np.ndarray, parallel: bool = False, max_workers: int = None) -> Dict[str, np.ndarray]:
        """
        Extract features using all enabled extractors.
        
        Args:
            frames: Video frames array
            parallel: Whether to use parallel feature extraction
            max_workers: Maximum number of parallel workers (only used if parallel=True)
            
        Returns:
            Dictionary mapping feature names to feature arrays
        """
        self.logger.info(f'using device: {self.config.processing.device}')
        if parallel and len(self.feature_extractors) > 1:
            return self._extract_all_features_parallel(frames, max_workers)
        
        # Sequential implementation (original)
        features = {}
        total_frames = len(frames)
        
        for name, extractor in self.feature_extractors.items():
            # self.logger.info(f"Extracting {name} features...")
            start_time = time.time()
            
            try:
                if name == "optical_flow":
                    # Optical flow needs sequential frames
                    feat = extractor.extract_batch(frames, batch_size=self.config.processing.batch_size)
                else:
                    feat = extractor.extract_batch(frames, batch_size=self.config.processing.batch_size)
                
                features[name] = feat
                extraction_time = time.time() - start_time
                
                # Validate feature dimensions
                if feat.shape[0] != total_frames:
                    self.logger.error(f"Feature dimension mismatch for {name}: expected {total_frames} frames, got {feat.shape[0]}")
                    raise ValueError(f"Feature extractor {name} returned {feat.shape[0]} features for {total_frames} frames")
                
                self.logger.info(f"Extracted {name} features: {feat.shape} in {extraction_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to extract {name} features: {e}")
                raise
        
        # Final validation - ensure all features have the same length
        feature_lengths = {name: len(feat) for name, feat in features.items()}
        if len(set(feature_lengths.values())) > 1:
            error_msg = f"Feature arrays have mismatched lengths: {', '.join([f'{name}: {length}' for name, length in feature_lengths.items()])}. All feature extractors must produce the same number of features per video."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return features
    
    def _combine_detection_results(self, results: Dict[str, List[int]]) -> List[int]:
        """Combine results from multiple detection methods."""
        # Simple combination strategy: union with temporal tolerance
        all_boundaries = []
        for method_boundaries in results.values():
            all_boundaries.extend(method_boundaries)
        
        if not all_boundaries:
            return []
        
        # Sort and remove duplicates within temporal tolerance
        all_boundaries = sorted(set(all_boundaries))
        tolerance = self.config.processing.temporal_tolerance
        
        combined = [all_boundaries[0]]
        for boundary in all_boundaries[1:]:
            if boundary - combined[-1] > tolerance:
                combined.append(boundary)
        
        return combined
    
    def _compute_performance_stats(self, frames: np.ndarray, features: Dict[str, np.ndarray], 
                                 processing_time: float) -> Dict[str, Any]:
        """Compute performance statistics."""
        stats = {
            "frames_per_second": len(frames) / processing_time,
            "feature_extraction_fps": {},
            "memory_usage": self._estimate_memory_usage(features),
            "fusion_strategy": self.config.fusion.strategy,
            "num_features": len(features),
            "total_feature_dim": sum(feat.shape[1] for feat in features.values()),
        }
        
        return stats
    
    def _estimate_memory_usage(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Estimate memory usage for features."""
        memory_usage = {}
        
        for name, feat in features.items():
            # Estimate memory in MB
            memory_mb = feat.nbytes / (1024 * 1024)
            memory_usage[name] = memory_mb
        
        memory_usage["total"] = sum(memory_usage.values())
        
        return memory_usage
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results.copy())
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_available_extractors(self) -> List[str]:
        """Get list of available feature extractors."""
        return FeatureExtractorRegistry.list_extractors()
    
    def get_available_fusion_strategies(self) -> List[str]:
        """Get list of available fusion strategies."""
        return FusionStrategyRegistry.list_strategies()
    
    def get_available_detection_methods(self) -> List[str]:
        """Get list of available detection methods."""
        return DetectionMethodRegistry.list_methods()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "extractors": {
                "available": self.get_available_extractors(),
                "enabled": list(self.feature_extractors.keys())
            },
            "fusion": {
                "available": self.get_available_fusion_strategies(),
                "current": self.config.fusion.strategy
            },
            "detection": {
                "available": self.get_available_detection_methods(),
                "enabled": list(self.detection_methods.keys())
            },
            "config": self.config.__dict__
        }