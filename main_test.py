"""
Main script to test different video shot detection methods and configurations.

This script provides a comprehensive testing framework for comparing:
- Different detection methods (statistical vs transformer)
- Various feature extraction combinations
- Multiple fusion strategies
- Different configuration parameters
- Batch processing of video directories
- Accuracy evaluation using ground truth files

Usage:
    # Test single video
    python main_test.py --video_path path/to/video.mp4 --method statistical
    
    # Test directory of videos with ground truth
    python main_test.py --video_dir path/to/videos --ground_truth_dir path/to/gt --method both
    
    # Compare methods on dataset
    python main_test.py --video_dir path/to/videos --ground_truth_dir path/to/gt --compare_all
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import warnings

# Add the parent directory to the Python path to import our modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import our modules
from core.config import (
    VideoShotConfig, 
    DEFAULT_RESEARCH_CONFIG, 
    DEFAULT_PRODUCTION_CONFIG, 
    DEFAULT_EDGE_CONFIG
)
from core.coordinator import ProcessingCoordinator
from utils.logger import setup_logger

warnings.filterwarnings("ignore")


class VideoShotDetectionTester:
    """
    Comprehensive testing framework for video shot detection methods.
    Supports single video testing and batch directory processing with accuracy evaluation.
    """
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize the tester.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = setup_logger("VideoShotTester", os.path.join(output_dir, "logs"))
        self.results = {}
        
        # Feature caching for efficiency
        self._feature_cache = {}
        self._frames_cache = {}
        self._video_info_cache = {}
        
    def load_ground_truth(self, gt_file: str, video_name: str = None) -> List[int]:
        """
        Load ground truth shot boundaries from file using the same format as training script.
        
        Args:
            gt_file: Path to ground truth file
            video_name: Name of the video (needed for JSON format with multiple videos)
            
        Returns:
            List of ground truth boundary frame indices
            
        Supported formats:
        - .txt: One boundary per line
        - .json: Training script format with video names as keys
                 Two sub-formats supported:
                 1) {"video_name": [boundary1, boundary2, ...]}
                 2) {"video_name": {"transitions": [[start, end], ...]}}
        - .npy: Numpy array of boundaries
        """
        if not os.path.exists(gt_file):
            self.logger.warning(f"Ground truth file not found: {gt_file}")
            return []
            
        try:
            if gt_file.endswith('.txt'):
                with open(gt_file, 'r') as f:
                    boundaries = [int(line.strip()) for line in f if line.strip().isdigit()]
            elif gt_file.endswith('.json'):
                with open(gt_file, 'r') as f:
                    data = json.load(f)
                
                # Handle training script format - JSON with video names as keys
                if video_name and video_name in data:
                    video_data = data[video_name]
                    
                    # Handle both formats used in training script
                    if isinstance(video_data, dict) and 'transitions' in video_data:
                        # New format: extract first frame of each transition as boundary
                        boundaries = [transition[0] for transition in video_data['transitions']]
                    elif isinstance(video_data, list):
                        # Old format: direct list of boundaries
                        boundaries = video_data
                    else:
                        self.logger.error(f"Unknown annotation format for {video_name} in {gt_file}")
                        return []
                else:
                    # Fallback: try to extract boundaries from first level
                    # Support simple format: {"boundaries": [frame1, frame2, ...]}
                    boundaries = data.get('boundaries', data.get('shots', []))
                    
                    # If still empty and video_name not found, log available videos
                    if not boundaries and video_name:
 
                        return []
                        
            elif gt_file.endswith('.npy'):
                boundaries = np.load(gt_file).tolist()
            else:
                self.logger.error(f"Unsupported ground truth format: {gt_file}")
                return []
                
            return sorted([int(b) for b in boundaries if isinstance(b, (int, float))])
            
        except Exception as e:
            self.logger.error(f"Error loading ground truth from {gt_file}: {e}")
            return []
    
    def _get_cache_key(self, video_path: str, feature_config: Dict[str, Any]) -> str:
        """
        Generate a cache key for features based on video path and feature configuration.
        
        Args:
            video_path: Path to the video file
            feature_config: Dictionary of feature extraction settings
            
        Returns:
            String cache key
        """
        import hashlib
        
        # Create a string representation of the feature config
        config_str = str(sorted(feature_config.items()))
        
        # Combine video path and config for unique key
        cache_content = f"{video_path}_{config_str}"
        
        # Generate hash for shorter key
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _extract_features_cached(self, video_path: str, config: VideoShotConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Extract features with caching support.
        
        Args:
            video_path: Path to the video file
            config: VideoShotConfig containing feature extraction settings
            
        Returns:
            Tuple of (frames, features, video_info)
        """
        # Create feature config dictionary for caching
        feature_config = {
            "hsv_enabled": config.features.hsv_enabled,
            "cnn_enabled": config.features.cnn_enabled,
            "clip_enabled": config.features.clip_enabled,
            "optical_flow_enabled": config.features.optical_flow_enabled,
            "cnn_model": config.features.cnn_model if config.features.cnn_enabled else None,
            "clip_model": config.features.clip_model if config.features.clip_enabled else None,
            "optical_flow_method": config.features.optical_flow_method if config.features.optical_flow_enabled else None,
            "batch_size": config.processing.batch_size,
            "frame_rate": config.processing.frame_rate
        }
        
        cache_key = self._get_cache_key(video_path, feature_config)
        
        # Check if we have cached results
        if cache_key in self._feature_cache:
            self.logger.info(f"Using cached features for {os.path.basename(video_path)}")
            return (
                self._frames_cache[cache_key],
                self._feature_cache[cache_key],
                self._video_info_cache[cache_key]
            )
        
        # Extract features fresh
        self.logger.info(f"Extracting features for {os.path.basename(video_path)} (will be cached)")
        
        # Initialize a minimal coordinator just for feature extraction
        from utils.video_utils import VideoProcessor
        from extractors import FeatureExtractorRegistry
        
        video_processor = VideoProcessor(config.processing)
        frames = video_processor.load_video(video_path)
        
        # Get video info
        video_info = {
            "num_frames": len(frames),
            "fps": video_processor.get_frame_rate(video_path),
            "duration": len(frames) / video_processor.get_frame_rate(video_path)
        }
        
        # Initialize feature extractors
        feature_extractors = {}
        
        if config.features.hsv_enabled:
            feature_extractors["hsv_histogram"] = FeatureExtractorRegistry.get(
                "hsv_histogram",
                device=config.processing.device
            )
        
        if config.features.cnn_enabled:
            feature_extractors["cnn"] = FeatureExtractorRegistry.get(
                "cnn",
                model_name=config.features.cnn_model,
                device=config.processing.device
            )
        
        if config.features.clip_enabled:
            feature_extractors["clip"] = FeatureExtractorRegistry.get(
                "clip",
                model_name=config.features.clip_model,
                device=config.processing.device
            )
        
        if config.features.optical_flow_enabled:
            feature_extractors["optical_flow"] = FeatureExtractorRegistry.get(
                "optical_flow",
                method=config.features.optical_flow_method,
                device=config.processing.device
            )
        
        # Extract features
        features = {}
        for name, extractor in feature_extractors.items():
            start_time = time.time()
            try:
                if name == "optical_flow":
                    feat = extractor.extract_batch(frames, batch_size=config.processing.batch_size)
                else:
                    feat = extractor.extract_batch(frames, batch_size=config.processing.batch_size)
                
                features[name] = feat
                extraction_time = time.time() - start_time
                self.logger.info(f"Extracted {name} features: {feat.shape} in {extraction_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to extract {name} features: {e}")
                raise
        
        # Cache the results
        self._frames_cache[cache_key] = frames
        self._feature_cache[cache_key] = features
        self._video_info_cache[cache_key] = video_info
        
        return frames, features, video_info
    
    def _run_detection_with_cached_features(self, video_path: str, config: VideoShotConfig,
                                          cached_frames: np.ndarray = None,
                                          cached_features: Dict[str, np.ndarray] = None) -> Tuple[List[int], Dict[str, Any]]:
        """
        Run shot detection with pre-extracted features.
        
        Args:
            video_path: Path to the video file
            config: Detection configuration
            cached_frames: Pre-loaded video frames
            cached_features: Pre-extracted features
            
        Returns:
            Tuple of (boundaries, metadata)
        """
        from fusion import FusionStrategyRegistry
        from detection import DetectionMethodRegistry
        
        start_time = time.time()
        
        # Use cached features if provided, otherwise extract them
        if cached_features is None or cached_frames is None:
            cached_frames, cached_features, video_info = self._extract_features_cached(video_path, config)
        else:
            video_info = {
                "num_frames": len(cached_frames),
                "fps": 30.0,  # Default, would need to get actual FPS
                "duration": len(cached_frames) / 30.0
            }
        
        # Initialize fusion strategy
        feature_dims = {}
        for name, features in cached_features.items():
            feature_dims[name] = features.shape[1]
        
        fusion_strategy = None
        if config.fusion.strategy == "hierarchical":
            fusion_strategy = FusionStrategyRegistry.get(
                "hierarchical",
                feature_dims=feature_dims,
                device=config.processing.device
            )
        elif config.fusion.strategy == "concatenation":
            fusion_strategy = FusionStrategyRegistry.get(
                "concatenation",
                feature_dims=feature_dims,
                device=config.processing.device
            )
        elif config.fusion.strategy == "weighted":
            fusion_strategy = FusionStrategyRegistry.get(
                "weighted",
                feature_dims=feature_dims,
                optimization_method=config.fusion.weights_optimization,
                manual_weights=config.fusion.manual_weights,
                device=config.processing.device
            )
        elif config.fusion.strategy == "pca":
            fusion_strategy = FusionStrategyRegistry.get(
                "pca",
                feature_dims=feature_dims,
                variance_retained=config.fusion.pca_variance_retained,
                device=config.processing.device
            )
        else:
            # Fallback to concatenation for unknown strategies
            self.logger.warning(f"Unknown fusion strategy '{config.fusion.strategy}', falling back to concatenation")
            fusion_strategy = FusionStrategyRegistry.get(
                "concatenation",
                feature_dims=feature_dims,
                device=config.processing.device
            )
        
        # Ensure fusion strategy was created successfully
        if fusion_strategy is None:
            self.logger.error(f"Failed to create fusion strategy '{config.fusion.strategy}'")
            raise RuntimeError(f"Failed to create fusion strategy '{config.fusion.strategy}'")
        
        # Apply fusion and get distance signal
        if config.fusion.strategy == "weighted":
            distance_signal = fusion_strategy(cached_features)
            fused_features = None
        else:
            if config.fusion.strategy == "pca" and hasattr(fusion_strategy, 'fit'):
                if not getattr(fusion_strategy, 'is_fitted', False):
                    fusion_strategy.fit(cached_features)
            
            fused_features = fusion_strategy.fuse_features(cached_features)
            distance_signal = fusion_strategy.compute_distances(fused_features)
        
        # Run detection
        boundaries = []
        
        if config.detection.method == "statistical":
            # Initialize statistical detector
            statistical_config = {
                "smoothing_method": config.detection.smoothing_method,
                "smoothing_window": config.detection.smoothing_window,
                "threshold_method": config.detection.threshold_method,
                "fixed_threshold": config.detection.fixed_threshold,
                "adaptive_beta": config.detection.adaptive_beta,
                "min_shot_duration": config.processing.min_shot_duration
            }
            detection_method = DetectionMethodRegistry.get("statistical", config=statistical_config)
            boundaries = detection_method(distance_signal)
            
        elif config.detection.method == "transformer":
            # Initialize transformer detector
            input_dim = fused_features.shape[1] if fused_features is not None else sum(feature_dims.values())
            
            transformer_config = {
                "input_dim": input_dim,
                "hidden_dim": config.detection.transformer_hidden_dim,
                "num_layers": config.detection.transformer_layers,
                "num_heads": config.detection.transformer_heads,
                "ff_dim": config.detection.transformer_ff_dim,
                "dropout": config.detection.transformer_dropout,
                "sequence_length": config.processing.sequence_length,
                "confidence_threshold": 0.5,
                "device": config.processing.device,
                "min_shot_duration": config.processing.min_shot_duration,
            }
            
            detection_method = DetectionMethodRegistry.get("transformer", config=transformer_config)
            
            if fused_features is not None:
                boundaries = detection_method.detect_boundaries(fused_features)
            else:
                self.logger.warning("Transformer detection requires fused features, using statistical fallback")
                # Fallback to statistical
                stat_detector = DetectionMethodRegistry.get(
                    "statistical",
                    smoothing_method="gaussian",
                    smoothing_window=5,
                    threshold_method="adaptive"
                )
                boundaries = stat_detector(distance_signal)
        
        processing_time = time.time() - start_time
        
        metadata = {
            "num_frames": video_info["num_frames"],
            "processing_time": processing_time,
            "method": config.detection.method,
            "fusion_strategy": config.fusion.strategy
        }
        
        return boundaries, metadata
    
    def clear_feature_cache(self) -> None:
        """Clear the feature cache to free memory."""
        self._feature_cache.clear()
        self._frames_cache.clear()
        self._video_info_cache.clear()
        self.logger.info("Feature cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        total_feature_memory = 0
        total_frames_memory = 0
        
        for features in self._feature_cache.values():
            for feat in features.values():
                total_feature_memory += feat.nbytes
        
        for frames in self._frames_cache.values():
            total_frames_memory += frames.nbytes
        
        return {
            "cached_videos": len(self._feature_cache),
            "feature_memory_mb": total_feature_memory / (1024 * 1024),
            "frames_memory_mb": total_frames_memory / (1024 * 1024),
            "total_memory_mb": (total_feature_memory + total_frames_memory) / (1024 * 1024)
        }
    
    def calculate_accuracy_metrics(self, predicted: List[int], ground_truth: List[int], 
                                 tolerance: int = 5) -> Dict[str, float]:
        """
        Calculate accuracy metrics for shot boundary detection.
        
        Args:
            predicted: List of predicted boundary frames
            ground_truth: List of ground truth boundary frames
            tolerance: Frame tolerance for matching boundaries
            
        Returns:
            Dictionary containing precision, recall, F1-score, and other metrics
        """
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}
            
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}
        
        # Convert to sets for easier manipulation
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        # Find matches within tolerance
        true_positives = 0
        matched_gt = set()
        matched_pred = set()
        
        for pred_boundary in pred_set:
            for gt_boundary in gt_set:
                if abs(pred_boundary - gt_boundary) <= tolerance:
                    if gt_boundary not in matched_gt:
                        true_positives += 1
                        matched_gt.add(gt_boundary)
                        matched_pred.add(pred_boundary)
                        break
        
        false_positives = len(pred_set) - true_positives
        false_negatives = len(gt_set) - true_positives
        
        # Calculate metrics
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Overall accuracy (true positives / total boundaries)
        total_boundaries = len(gt_set) + false_positives
        accuracy = true_positives / total_boundaries if total_boundaries > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "predicted_count": len(predicted),
            "ground_truth_count": len(ground_truth)
        }
    
    def find_video_files(self, video_dir: str, extensions: List[str] = None) -> List[str]:
        """
        Find all video files in a directory.
        
        Args:
            video_dir: Directory to search
            extensions: List of video file extensions to look for
            
        Returns:
            List of video file paths
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            
        video_files = []
        for ext in extensions:
            video_files.extend(Path(video_dir).glob(f"*{ext}"))
            video_files.extend(Path(video_dir).glob(f"*{ext.upper()}"))
            
        return [str(f) for f in sorted(video_files)]
    
    def test_video_directory(self, video_dir: str, ground_truth_dir: str = None,
                            ground_truth_file: str = None, methods: List[str] = None, 
                            tolerance: int = 5) -> Dict[str, Any]:
        """
        Test all videos in a directory and evaluate accuracy if ground truth is provided.
        
        Args:
            video_dir: Directory containing video files
            ground_truth_dir: Directory containing ground truth files (optional)
            ground_truth_file: Single JSON file with ground truth for all videos (optional)
            methods: List of detection methods to test
            tolerance: Frame tolerance for accuracy evaluation
            
        Returns:
            Dictionary containing results for all videos and methods
        """
        if methods is None:
            methods = ['statistical', 'transformer']
            
        video_files = self.find_video_files(video_dir)
        if not video_files:
            self.logger.error(f"No video files found in {video_dir}")
            return {}
            
        self.logger.info(f"Found {len(video_files)} video files")
        
        all_results = {}
        summary_stats = {method: [] for method in methods}
        
        # Initialize coordinators once for each method to reuse across videos
        coordinators = {}
        for method in methods:
            self.logger.info(f"Initializing {method} coordinator...")
            if method == "statistical":
                import copy
                config = copy.deepcopy(DEFAULT_PRODUCTION_CONFIG)
                config.detection.method = "statistical"
            elif method == "transformer":
                import copy
                config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
                config.detection.method = "transformer"
            else:
                self.logger.error(f"Unknown method: {method}")
                continue
                
            coordinator = ProcessingCoordinator(config)
            coordinator.initialize()
            coordinators[method] = coordinator
            self.logger.info(f"{method} coordinator initialized")
        
        for video_file in video_files:
            video_name = Path(video_file)
            
            # Load ground truth if available
            ground_truth = []
            if ground_truth_file:
                # Single JSON file containing all videos
                ground_truth = self.load_ground_truth(ground_truth_file, video_name)
                if ground_truth:
                    self.logger.info(f"Loaded ground truth: {len(ground_truth)} boundaries from {ground_truth_file}")
                else:
                    self.logger.warning(f"No ground truth found for {video_name} in {ground_truth_file}")
            elif ground_truth_dir:
                # Multiple files in directory
                gt_patterns = [
                    os.path.join(ground_truth_dir, f"{video_name}.txt"),
                    os.path.join(ground_truth_dir, f"{video_name}.json"),
                    os.path.join(ground_truth_dir, f"{video_name}.npy"),
                    os.path.join(ground_truth_dir, f"{video_name}_gt.txt"),
                    os.path.join(ground_truth_dir, f"{video_name}_boundaries.txt"),
                    # Also check for a single JSON file containing all videos (training script format)
                    os.path.join(ground_truth_dir, "ground_truth.json"),
                    os.path.join(ground_truth_dir, "annotations.json"),
                    os.path.join(ground_truth_dir, "boundaries.json")
                ]
                
                for gt_file in gt_patterns:
                    if os.path.exists(gt_file):
                        # For JSON files, pass video_name for training script format
                        if gt_file.endswith('.json'):
                            ground_truth = self.load_ground_truth(gt_file, video_name)
                        else:
                            ground_truth = self.load_ground_truth(gt_file)
                            
                        if ground_truth:  # Only break if we actually loaded some boundaries
                            self.logger.info(f"Loaded ground truth: {len(ground_truth)} boundaries from {gt_file}")
                            break
                
                if not ground_truth and ground_truth_dir:
                    self.logger.warning(f"No ground truth found for {video_name}")

            # Skip videos without ground truth when ground truth is expected
            if (ground_truth_dir or ground_truth_file) and not ground_truth:
                continue
            
            self.logger.info(f"Processing video: {video_name}")

            video_results = {}
            
            # Test each method
            for method in methods:
                if method not in coordinators:
                    self.logger.error(f"Coordinator not initialized for method: {method}")
                    continue
                    
                self.logger.info(f"Testing {method} method on {video_name}")
                
                try:
                    # Run detection using pre-initialized coordinator
                    start_time = time.time()
                    coordinator = coordinators[method]
                    results = coordinator.process_video(video_file)
                    
                    # Extract boundaries based on the method being used
                    detection_results = results.get("detection_results", {})
                    if method in detection_results:
                        boundaries = detection_results[method]
                    elif "combined" in detection_results:
                        boundaries = detection_results["combined"]
                    else:
                        # Fallback: take the first available method's results
                        boundaries = list(detection_results.values())[0] if detection_results else []
                    
                    processing_time = time.time() - start_time
                    
                    metadata = {
                        "method": method,
                        "config_type": "production" if method == "statistical" else "research",
                        "total_boundaries": len(boundaries)
                    }
                    
                    # Calculate accuracy if ground truth available
                    accuracy_metrics = {}
                    if ground_truth:
                        accuracy_metrics = self.calculate_accuracy_metrics(
                            boundaries, ground_truth, tolerance
                        )
                        summary_stats[method].append(accuracy_metrics)
                    
                    video_results[method] = {
                        "boundaries": boundaries,
                        "processing_time": processing_time,
                        "metadata": metadata,
                        "accuracy_metrics": accuracy_metrics,
                        "ground_truth": ground_truth
                    }
                    
                    self.logger.info(f"{method}: {len(boundaries)} boundaries, "
                                   f"{processing_time:.2f}s")
                    if accuracy_metrics:
                        self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                       f"R: {accuracy_metrics['recall']:.3f}, "
                                       f"F1: {accuracy_metrics['f1_score']:.3f}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {video_name} with {method}: {e}")
                    video_results[method] = {"error": str(e)}
            
            all_results[video_name] = video_results
            
            # Show cache info periodically and clear if requested
            if (video_files.index(video_file) + 1) % 10 == 0:  # Every 10 videos
                cache_info = self.get_cache_info()
                self.logger.info(f"Cache info: {cache_info['cached_videos']} videos cached, "
                               f"{cache_info['total_memory_mb']:.1f} MB used")
                
                # Optionally clear cache to save memory (can be configurable)
                if cache_info['total_memory_mb'] > 1000:  # Clear if over 1GB
                    self.logger.info("Clearing cache due to memory usage")
                    self.clear_feature_cache()
        
        # Cleanup coordinators
        for coordinator in coordinators.values():
            if hasattr(coordinator, 'cleanup'):
                coordinator.cleanup()
        
        # Calculate overall statistics
        overall_stats = {}
        for method in methods:
            if summary_stats[method]:
                metrics = summary_stats[method]
                overall_stats[method] = {
                    "mean_precision": np.mean([m['precision'] for m in metrics]),
                    "mean_recall": np.mean([m['recall'] for m in metrics]),
                    "mean_f1_score": np.mean([m['f1_score'] for m in metrics]),
                    "mean_accuracy": np.mean([m['accuracy'] for m in metrics]),
                    "std_precision": np.std([m['precision'] for m in metrics]),
                    "std_recall": np.std([m['recall'] for m in metrics]),
                    "std_f1_score": np.std([m['f1_score'] for m in metrics]),
                    "total_videos": len(metrics)
                }
        
        # Save results
        results_summary = {
            "video_results": all_results,
            "overall_statistics": overall_stats,
            "parameters": {
                "video_dir": video_dir,
                "ground_truth_dir": ground_truth_dir,
                "ground_truth_file": ground_truth_file,
                "methods": methods,
                "tolerance": tolerance,
                "total_videos": len(video_files)
            }
        }
        
        # Save to file
        results_file = os.path.join(self.output_dir, "batch_test_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results_summary, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Print summary
        self._print_batch_summary(overall_stats, len(video_files))
        
        return results_summary
    
    def _print_batch_summary(self, overall_stats: Dict[str, Any], total_videos: int):
        """Print a summary of batch test results."""
        print(f"\n{'='*60}")
        print(f"BATCH TEST SUMMARY ({total_videos} videos)")
        print(f"{'='*60}")
        
        for method, stats in overall_stats.items():
            print(f"\n{method.upper()} METHOD:")
            print(f"  Mean Precision: {stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}")
            print(f"  Mean Recall:    {stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}")
            print(f"  Mean F1-Score:  {stats['mean_f1_score']:.3f} ± {stats['std_f1_score']:.3f}")
            print(f"  Mean Accuracy:  {stats['mean_accuracy']:.3f}")
    
    def _run_single_video(self, video_path: str, method: str) -> Tuple[List[int], Dict[str, Any]]:
        """
        Run detection on a single video with specified method.
        
        Args:
            video_path: Path to video file
            method: Detection method ('statistical' or 'transformer')
            
        Returns:
            Tuple of (boundaries, metadata)
        """
        # Use appropriate configuration based on method
        import copy
        if method == "statistical":
            config = copy.deepcopy(DEFAULT_PRODUCTION_CONFIG)
            config.detection.method = "statistical"
        elif method == "transformer":
            config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
            config.detection.method = "transformer"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Use cached feature extraction for efficiency
        return self._run_detection_with_cached_features(video_path, config)
        
    def test_statistical_methods(self, video_path: str, 
                                configurations: List[Dict[str, Any]] = None,
                                ground_truth: List[int] = None,
                                tolerance: int = 5) -> Dict[str, Any]:
        """
        Test various statistical detection configurations.
        
        Args:
            video_path: Path to the video file
            configurations: List of configuration dictionaries to test
            ground_truth: Ground truth boundary frames for accuracy evaluation
            tolerance: Frame tolerance for accuracy evaluation
            
        Returns:
            Dictionary of results for each configuration
        """
        if configurations is None:
            configurations = self._get_default_statistical_configs()
        
        results = {}
        
        # Pre-extract features once for all statistical tests
        # Use research config as base for feature extraction
        import copy
        base_config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
        base_config.detection.method = "statistical"
        
        self.logger.info("Pre-extracting features for statistical method testing...")
        cached_frames, cached_features, video_info = self._extract_features_cached(video_path, base_config)
        
        for i, config_dict in enumerate(configurations):
            config_name = config_dict.get("name", f"statistical_config_{i}")
            self.logger.info(f"Testing statistical configuration: {config_name}")
            
            try:
                # Create configuration
                config = self._create_config_from_dict(config_dict)
                config.detection.method = "statistical"
                
                # Run detection with cached features
                start_time = time.time()
                boundaries, metadata = self._run_detection_with_cached_features(
                    video_path, config, cached_frames, cached_features
                )
                end_time = time.time()
                
                # Store results
                results[config_name] = {
                    "boundaries": boundaries,
                    "processing_time": end_time - start_time,
                    "metadata": metadata,
                    "config": config_dict
                }
                
                # Calculate accuracy if ground truth available
                if ground_truth:
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        boundaries, ground_truth, tolerance
                    )
                    results[config_name]["accuracy_metrics"] = accuracy_metrics
                    results[config_name]["ground_truth"] = ground_truth
                    
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                    self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                   f"R: {accuracy_metrics['recall']:.3f}, "
                                   f"F1: {accuracy_metrics['f1_score']:.3f}")
                else:
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error in configuration {config_name}: {str(e)}")
                results[config_name] = {"error": str(e)}
        
        return results
    
    def test_transformer_methods(self, video_path: str, 
                                model_paths: List[str] = None,
                                ground_truth: List[int] = None,
                                tolerance: int = 5) -> Dict[str, Any]:
        """
        Test transformer detection with different trained models.
        
        Args:
            video_path: Path to the video file
            model_paths: List of paths to trained transformer models
            ground_truth: Ground truth boundary frames for accuracy evaluation
            tolerance: Frame tolerance for accuracy evaluation
            
        Returns:
            Dictionary of results for each model
        """
        if model_paths is None:
            # Look for trained models in the models directory
            models_dir = "models"
            if os.path.exists(models_dir):
                model_paths = [
                    os.path.join(models_dir, f) for f in os.listdir(models_dir)
                    if f.endswith('.pth') or f.endswith('.pt')
                ]
            else:
                self.logger.warning("No models directory found, using default transformer config")
                model_paths = ["default"]
        
        results = {}
        
        # Pre-extract features once for all transformer tests
        import copy
        base_config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
        base_config.detection.method = "transformer"
        
        self.logger.info("Pre-extracting features for transformer method testing...")
        cached_frames, cached_features, video_info = self._extract_features_cached(video_path, base_config)
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path) if model_path != "default" else "default"
            self.logger.info(f"Testing transformer model: {model_name}")
            
            try:
                # Create configuration
                import copy
                config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
                config.detection.method = "transformer"
                
                if model_path != "default":
                    # Load specific model (would need to implement model loading)
                    pass
                
                # Run detection with cached features
                start_time = time.time()
                boundaries, metadata = self._run_detection_with_cached_features(
                    video_path, config, cached_frames, cached_features
                )
                end_time = time.time()
                
                # Store results
                results[model_name] = {
                    "boundaries": boundaries,
                    "processing_time": end_time - start_time,
                    "metadata": metadata,
                    "model_path": model_path
                }
                
                # Calculate accuracy if ground truth available
                if ground_truth:
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        boundaries, ground_truth, tolerance
                    )
                    results[model_name]["accuracy_metrics"] = accuracy_metrics
                    results[model_name]["ground_truth"] = ground_truth
                    
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                    self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                   f"R: {accuracy_metrics['recall']:.3f}, "
                                   f"F1: {accuracy_metrics['f1_score']:.3f}")
                else:
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error with model {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def test_feature_combinations(self, video_path: str,
                                  ground_truth: List[int] = None,
                                  tolerance: int = 5) -> Dict[str, Any]:
        """
        Test different feature extraction combinations.
        
        Args:
            video_path: Path to the video file
            ground_truth: Ground truth boundary frames for accuracy evaluation
            tolerance: Frame tolerance for accuracy evaluation
            
        Returns:
            Dictionary of results for each feature combination
        """
        feature_combinations = [
            {"name": "hsv_only", "hsv_histogram": True, "cnn": False, "clip": False, "optical_flow": False},
            {"name": "cnn_only", "hsv_histogram": False, "cnn": True, "clip": False, "optical_flow": False},
            {"name": "optical_flow_only", "hsv_histogram": False, "cnn": False, "clip": False, "optical_flow": True},
            {"name": "hsv_optical_flow", "hsv_histogram": True, "cnn": False, "clip": False, "optical_flow": True},
            {"name": "cnn_optical_flow", "hsv_histogram": False, "cnn": True, "clip": False, "optical_flow": True},
            {"name": "all_features", "hsv_histogram": True, "cnn": True, "clip": True, "optical_flow": True},
            {"name": "no_clip", "hsv_histogram": True, "cnn": True, "clip": False, "optical_flow": True},
        ]
        
        results = {}
        
        # For feature combinations, we need to extract features separately for each combination
        # since they use different feature extractors
        for combo in feature_combinations:
            combo_name = combo["name"]
            self.logger.info(f"Testing feature combination: {combo_name}")
            
            try:
                # Create configuration
                import copy
                config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
                config.features.hsv_enabled = combo["hsv_histogram"]
                config.features.cnn_enabled = combo["cnn"]
                config.features.clip_enabled = combo["clip"]
                config.features.optical_flow_enabled = combo["optical_flow"]
                
                # Extract features for this specific combination
                start_time = time.time()
                cached_frames, cached_features, video_info = self._extract_features_cached(video_path, config)
                
                # Run detection with the extracted features
                boundaries, metadata = self._run_detection_with_cached_features(
                    video_path, config, cached_frames, cached_features
                )
                end_time = time.time()
                
                # Store results
                results[combo_name] = {
                    "boundaries": boundaries,
                    "processing_time": end_time - start_time,
                    "metadata": metadata,
                    "features": combo
                }
                
                # Calculate accuracy if ground truth available
                if ground_truth:
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        boundaries, ground_truth, tolerance
                    )
                    results[combo_name]["accuracy_metrics"] = accuracy_metrics
                    results[combo_name]["ground_truth"] = ground_truth
                    
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                    self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                   f"R: {accuracy_metrics['recall']:.3f}, "
                                   f"F1: {accuracy_metrics['f1_score']:.3f}")
                else:
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error with combination {combo_name}: {str(e)}")
                results[combo_name] = {"error": str(e)}
        
        return results
    
    def test_fusion_strategies(self, video_path: str,
                              ground_truth: List[int] = None,
                              tolerance: int = 5) -> Dict[str, Any]:
        """
        Test different fusion strategies.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of results for each fusion strategy
        """
        fusion_strategies = [
            "concatenation",
            "weighted", 
            "pca",
            "cross_transformer",
            "hierarchical"
        ]
        
        results = {}
        
        # Pre-extract features once for all fusion strategy tests
        import copy
        base_config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
        
        self.logger.info("Pre-extracting features for fusion strategy testing...")
        cached_frames, cached_features, video_info = self._extract_features_cached(video_path, base_config)
        
        for strategy in fusion_strategies:
            self.logger.info(f"Testing fusion strategy: {strategy}")
            
            try:
                # Create configuration
                import copy
                config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
                config.fusion.strategy = strategy
                
                # Run detection with cached features
                start_time = time.time()
                boundaries, metadata = self._run_detection_with_cached_features(
                    video_path, config, cached_frames, cached_features
                )
                end_time = time.time()
                
                # Store results
                results[strategy] = {
                    "boundaries": boundaries,
                    "processing_time": end_time - start_time,
                    "metadata": metadata,
                    "strategy": strategy
                }
                
                # Calculate accuracy if ground truth available
                if ground_truth:
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        boundaries, ground_truth, tolerance
                    )
                    results[strategy]["accuracy_metrics"] = accuracy_metrics
                    results[strategy]["ground_truth"] = ground_truth
                    
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                    self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                   f"R: {accuracy_metrics['recall']:.3f}, "
                                   f"F1: {accuracy_metrics['f1_score']:.3f}")
                else:
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error with strategy {strategy}: {str(e)}")
                results[strategy] = {"error": str(e)}
        
        return results
    
    def compare_preset_configs(self, video_path: str,
                              ground_truth: List[int] = None,
                              tolerance: int = 5) -> Dict[str, Any]:
        """
        Compare the three preset configurations.
        
        Args:
            video_path: Path to the video file
            ground_truth: Ground truth boundary frames for accuracy evaluation
            tolerance: Frame tolerance for accuracy evaluation
            
        Returns:
            Dictionary of results for each preset configuration
        """
        presets = {
            "research": DEFAULT_RESEARCH_CONFIG,
            "production": DEFAULT_PRODUCTION_CONFIG,
            "edge": DEFAULT_EDGE_CONFIG
        }
        
        results = {}
        
        # For preset comparisons, each preset might use different feature combinations
        # so we need to extract features separately for each
        for preset_name, config in presets.items():
            self.logger.info(f"Testing preset configuration: {preset_name}")
            
            try:
                # Extract features for this specific preset configuration
                start_time = time.time()
                cached_frames, cached_features, video_info = self._extract_features_cached(video_path, config)
                
                # Run detection with the extracted features
                boundaries, metadata = self._run_detection_with_cached_features(
                    video_path, config, cached_frames, cached_features
                )
                end_time = time.time()
                
                # Store results
                results[preset_name] = {
                    "boundaries": boundaries,
                    "processing_time": end_time - start_time,
                    "metadata": metadata,
                    "preset": preset_name
                }
                
                # Calculate accuracy if ground truth available
                if ground_truth:
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        boundaries, ground_truth, tolerance
                    )
                    results[preset_name]["accuracy_metrics"] = accuracy_metrics
                    results[preset_name]["ground_truth"] = ground_truth
                    
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                    self.logger.info(f"Accuracy - P: {accuracy_metrics['precision']:.3f}, "
                                   f"R: {accuracy_metrics['recall']:.3f}, "
                                   f"F1: {accuracy_metrics['f1_score']:.3f}")
                else:
                    self.logger.info(f"Found {len(boundaries)} boundaries in {end_time - start_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error with preset {preset_name}: {str(e)}")
                results[preset_name] = {"error": str(e)}
        
        return results
    
    def _run_detection(self, video_path: str, config: VideoShotConfig) -> Tuple[List[int], Dict[str, Any]]:
        """
        Run shot detection with the given configuration.
        Note: This method is kept for backward compatibility but now uses cached features.
        
        Args:
            video_path: Path to the video file
            config: Detection configuration
            
        Returns:
            Tuple of (boundaries, metadata)
        """
        # Use the new cached feature extraction method
        return self._run_detection_with_cached_features(video_path, config)
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> VideoShotConfig:
        """Create a VideoShotConfig from a dictionary."""
        import copy
        config = copy.deepcopy(DEFAULT_RESEARCH_CONFIG)
        
        # Update detection parameters
        if "detection" in config_dict:
            for key, value in config_dict["detection"].items():
                if hasattr(config.detection, key):
                    setattr(config.detection, key, value)
        
        # Update fusion parameters
        if "fusion" in config_dict:
            for key, value in config_dict["fusion"].items():
                if hasattr(config.fusion, key):
                    setattr(config.fusion, key, value)
        
        # Update feature parameters
        if "features" in config_dict:
            for key, value in config_dict["features"].items():
                if hasattr(config.features, key):
                    setattr(config.features, key, value)
        
        return config
    
    def _get_default_statistical_configs(self) -> List[Dict[str, Any]]:
        """Get default statistical configurations for testing."""
        return [
            {
                "name": "default_statistical",
                "detection": {
                    "smoothing_method": "gaussian",
                    "smoothing_window": 5,
                    "threshold_method": "adaptive"
                }
            },
            {
                "name": "aggressive_detection",
                "detection": {
                    "smoothing_method": "gaussian",
                    "smoothing_window": 3,
                    "threshold_method": "adaptive",
                    "adaptive_beta": 1.2
                }
            },
            {
                "name": "conservative_detection",
                "detection": {
                    "smoothing_method": "savgol",
                    "smoothing_window": 7,
                    "threshold_method": "adaptive",
                    "adaptive_beta": 2.0
                }
            },
            {
                "name": "fixed_threshold",
                "detection": {
                    "smoothing_method": "gaussian",
                    "smoothing_window": 5,
                    "threshold_method": "fixed",
                    "fixed_threshold": 0.7
                }
            }
        ]
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to a JSON file."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
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
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def visualize_results(self, results: Dict[str, Any], video_path: str) -> None:
        """
        Create visualizations of the detection results.
        
        Args:
            results: Detection results dictionary
            video_path: Path to the original video
        """
        # Create a comparison plot
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 2 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        # Get video duration for x-axis
        from utils.video_utils import VideoProcessor
        processor = VideoProcessor(DEFAULT_RESEARCH_CONFIG.processing)
        video_info = processor.get_video_info(video_path)
        duration = video_info["duration"]
        
        for i, (method_name, result) in enumerate(results.items()):
            if "error" in result:
                axes[i].text(0.5, 0.5, f"Error: {result['error']}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{method_name} (Error)")
                continue
            
            boundaries = result["boundaries"]
            
            # Convert frame indices to time
            fps = video_info["fps"]
            boundary_times = [b / fps for b in boundaries]
            
            # Plot boundaries as vertical lines
            for bt in boundary_times:
                axes[i].axvline(x=bt, color='red', alpha=0.7)
            
            axes[i].set_xlim(0, duration)
            axes[i].set_ylim(-0.5, 0.5)
            axes[i].set_title(f"{method_name} ({len(boundaries)} boundaries, "
                            f"{result['processing_time']:.2f}s)")
            axes[i].set_xlabel("Time (seconds)")
            axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "detection_comparison.png"), dpi=300)
        plt.close()
        
        self.logger.info("Visualization saved to detection_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Test video shot detection methods")
    
    # Input options - either single video or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", help="Path to a single video file")
    input_group.add_argument("--video_dir", help="Directory containing video files")
    
    # Ground truth for accuracy evaluation
    parser.add_argument("--ground_truth_dir", help="Directory containing ground truth files")
    parser.add_argument("--ground_truth", help="Single JSON file containing ground truth for all videos")
    parser.add_argument("--tolerance", type=int, default=5, 
                       help="Frame tolerance for accuracy evaluation (default: 5)")
    
    # Detection methods
    parser.add_argument("--method", choices=["statistical", "transformer", "both"], 
                       default="both", help="Detection method(s) to test")
    
    # Testing options
    parser.add_argument("--test_features", action="store_true", 
                       help="Test different feature combinations (single video only)")
    parser.add_argument("--test_fusion", action="store_true", 
                       help="Test different fusion strategies (single video only)")
    parser.add_argument("--test_presets", action="store_true", 
                       help="Test preset configurations (single video only)")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all methods on dataset (directory mode)")
    
    # Output options
    parser.add_argument("--compare", action="store_true", 
                       help="Generate comparison visualizations")
    parser.add_argument("--output_dir", default="test_results", 
                       help="Directory to save results")
    parser.add_argument("--models_dir", help="Directory containing trained transformer models")
    
    # Performance options
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear feature cache between videos to save memory")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = VideoShotDetectionTester(args.output_dir)
    
    print("Video Shot Detection Testing Framework")
    print("=" * 50)
    print("Note: Feature caching is enabled for improved performance")
    print("Features are extracted once per video and reused across configurations")
    print("=" * 50)
    
    try:
        # Directory mode - batch processing
        if args.video_dir:
            if not os.path.exists(args.video_dir):
                print(f"Error: Video directory not found: {args.video_dir}")
                return
                
            print(f"Testing directory: {args.video_dir}")
            if args.ground_truth:
                print(f"Ground truth file: {args.ground_truth}")
                print(f"Frame tolerance: {args.tolerance}")
            elif args.ground_truth_dir:
                print(f"Ground truth directory: {args.ground_truth_dir}")
                print(f"Frame tolerance: {args.tolerance}")
            
            # Determine methods to test
            methods = []
            if args.method == "both":
                methods = ["statistical", "transformer"]
            else:
                methods = [args.method]
            
            print(f"Testing methods: {', '.join(methods)}")
            
            # Run batch testing
            results = tester.test_video_directory(
                video_dir=args.video_dir,
                ground_truth_dir=args.ground_truth_dir,
                ground_truth_file=args.ground_truth,
                methods=methods,
                tolerance=args.tolerance
            )
            
            # Generate comparison report if requested
            if args.compare_all and len(methods) > 1:
                print("\n=== Generating Method Comparison Report ===")
                tester._generate_method_comparison_report(results)
        
        # Single video mode
        else:
            if not os.path.exists(args.video_path):
                print(f"Error: Video file not found: {args.video_path}")
                return
            
            print(f"Testing video: {args.video_path}")
            
            # Load ground truth if provided
            ground_truth = []
            video_name = Path(args.video_path).stem
            
            if args.ground_truth:
                # Add debug logging
                ground_truth = tester.load_ground_truth(args.ground_truth, video_name)
                if ground_truth:
                    print(f"Loaded ground truth: {len(ground_truth)} boundaries from {args.ground_truth}")
                    print(f"Frame tolerance: {args.tolerance}")
                else:
                    # Try with .mp4 extension
                    video_name_with_ext = video_name + ".mp4"
                    ground_truth = tester.load_ground_truth(args.ground_truth, video_name_with_ext)
                    if ground_truth:
                        print(f"Loaded ground truth: {len(ground_truth)} boundaries from {args.ground_truth}")
                        print(f"Frame tolerance: {args.tolerance}")
                    else:
                        print(f"Still no ground truth found with extension")
                        
            elif args.ground_truth_dir:
                # Look for ground truth file for this video
                gt_patterns = [
                    os.path.join(args.ground_truth_dir, f"{video_name}.txt"),
                    os.path.join(args.ground_truth_dir, f"{video_name}.json"),
                    os.path.join(args.ground_truth_dir, f"{video_name}.npy"),
                    os.path.join(args.ground_truth_dir, f"{video_name}_gt.txt"),
                    os.path.join(args.ground_truth_dir, f"{video_name}_boundaries.txt"),
                    os.path.join(args.ground_truth_dir, "ground_truth.json"),
                    os.path.join(args.ground_truth_dir, "annotations.json"),
                    os.path.join(args.ground_truth_dir, "boundaries.json")
                ]
                
                for gt_file in gt_patterns:
                    if os.path.exists(gt_file):
                        if gt_file.endswith('.json'):
                            ground_truth = tester.load_ground_truth(gt_file, video_name)
                        else:
                            ground_truth = tester.load_ground_truth(gt_file)
                        
                        if ground_truth:
                            print(f"Loaded ground truth: {len(ground_truth)} boundaries from {gt_file}")
                            print(f"Frame tolerance: {args.tolerance}")
                            break
                
                if not ground_truth:
                    print(f"Warning: No ground truth found for {video_name} in {args.ground_truth_dir}")
            
            all_results = {}
            
            # Test statistical methods
            if args.method in ["statistical", "both"]:
                print("\n=== Testing Statistical Methods ===")
                statistical_results = tester.test_statistical_methods(
                    args.video_path, 
                    ground_truth=ground_truth,
                    tolerance=args.tolerance
                )
                all_results["statistical"] = statistical_results
                tester.save_results(statistical_results, "statistical_results.json")
            
            # Test transformer methods
            if args.method in ["transformer", "both"]:
                print("\n=== Testing Transformer Methods ===")
                model_paths = None
                if args.models_dir:
                    model_paths = [
                        os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir)
                        if f.endswith('.pth') or f.endswith('.pt')
                    ]
                transformer_results = tester.test_transformer_methods(
                    args.video_path, 
                    model_paths, 
                    ground_truth=ground_truth,
                    tolerance=args.tolerance
                )
                all_results["transformer"] = transformer_results
                tester.save_results(transformer_results, "transformer_results.json")
            
            # Test feature combinations
            if args.test_features:
                print("\n=== Testing Feature Combinations ===")
                feature_results = tester.test_feature_combinations(
                    args.video_path,
                    ground_truth=ground_truth,
                    tolerance=args.tolerance
                )
                all_results["features"] = feature_results
                tester.save_results(feature_results, "feature_results.json")
            
            # Test fusion strategies
            if args.test_fusion:
                print("\n=== Testing Fusion Strategies ===")
                fusion_results = tester.test_fusion_strategies(
                    args.video_path,
                    ground_truth=ground_truth,
                    tolerance=args.tolerance
                )
                all_results["fusion"] = fusion_results
                tester.save_results(fusion_results, "fusion_results.json")
            
            # Test preset configurations
            if args.test_presets:
                print("\n=== Testing Preset Configurations ===")
                preset_results = tester.compare_preset_configs(
                    args.video_path,
                    ground_truth=ground_truth,
                    tolerance=args.tolerance
                )
                all_results["presets"] = preset_results
                tester.save_results(preset_results, "preset_results.json")
            
            # Save combined results
            tester.save_results(all_results, "all_results.json")
            
            # Show cache info
            cache_info = tester.get_cache_info()
            print(f"\nCache info: {cache_info['cached_videos']} videos cached, "
                  f"{cache_info['total_memory_mb']:.1f} MB used")
            
            # Clear cache if requested
            if args.clear_cache:
                tester.clear_feature_cache()
                print("Feature cache cleared")
            
            # Generate visualizations
            if args.compare and all_results:
                print("\n=== Generating Visualizations ===")
                # Flatten results for comparison
                flat_results = {}
                for category, results in all_results.items():
                    for method, result in results.items():
                        flat_results[f"{category}_{method}"] = result
                
                tester.visualize_results(flat_results, args.video_path)
            
            # Print summary
            print(f"\n=== Single Video Testing Complete ===")
            print(f"Results saved to: {args.output_dir}")
            print("\nSummary:")
            for category, results in all_results.items():
                print(f"\n{category.upper()}:")
                for method, result in results.items():
                    if "error" in result:
                        print(f"  {method}: ERROR - {result['error']}")
                    else:
                        boundaries = len(result.get('boundaries', []))
                        time_taken = result.get('processing_time', 0)
                        summary_line = f"  {method}: {boundaries} boundaries, {time_taken:.2f}s"
                        
                        # Add accuracy metrics if available
                        if "accuracy_metrics" in result:
                            metrics = result["accuracy_metrics"]
                            summary_line += f" | P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}"
                        
                        print(summary_line)
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n=== Testing Complete ===")
    print(f"All results saved to: {args.output_dir}")


def _generate_method_comparison_report(self, results: Dict[str, Any]):
    """Generate a detailed comparison report between methods."""
    overall_stats = results.get("overall_statistics", {})
    
    if not overall_stats:
        print("No statistics available for comparison")
        return
    
    print(f"\n{'='*60}")
    print("METHOD COMPARISON REPORT")
    print(f"{'='*60}")
    
    methods = list(overall_stats.keys())
    if len(methods) < 2:
        print("Need at least 2 methods for comparison")
        return
    
    # Create comparison table
    metrics = ['mean_precision', 'mean_recall', 'mean_f1_score', 'mean_accuracy']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    print(f"\n{'Metric':<12} ", end="")
    for method in methods:
        print(f"{method.capitalize():<12} ", end="")
    print("Winner")
    print("-" * (12 + 12 * len(methods) + 10))
    
    for metric, name in zip(metrics, metric_names):
        print(f"{name:<12} ", end="")
        values = []
        for method in methods:
            value = overall_stats[method].get(metric, 0.0)
            values.append(value)
            print(f"{value:.3f}        ", end="")
        
        # Determine winner
        best_idx = np.argmax(values)
        winner = methods[best_idx]
        print(f"{winner.capitalize()}")
    
    # Statistical significance test (if you want to add it)
    print(f"\nDataset size: {results['parameters']['total_videos']} videos")
    print(f"Frame tolerance: {results['parameters']['tolerance']} frames")
    
    # Save detailed report
    report_file = os.path.join(self.output_dir, "method_comparison_report.txt")
    with open(report_file, 'w') as f:
        f.write("METHOD COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for method, stats in overall_stats.items():
            f.write(f"{method.upper()} METHOD:\n")
            f.write(f"  Mean Precision: {stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}\n")
            f.write(f"  Mean Recall:    {stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}\n")
            f.write(f"  Mean F1-Score:  {stats['mean_f1_score']:.3f} ± {stats['std_f1_score']:.3f}\n")
            f.write(f"  Mean Accuracy:  {stats['mean_accuracy']:.3f}\n")
            f.write(f"  Total Videos:   {stats['total_videos']}\n\n")
    
    print(f"Detailed report saved to: {report_file}")


# Add the method to the class
VideoShotDetectionTester._generate_method_comparison_report = _generate_method_comparison_report


if __name__ == "__main__":
    main()