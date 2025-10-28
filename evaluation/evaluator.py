"""
Comprehensive evaluation framework for video shot detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import time
import json
import os
from pathlib import Path

from core.config import VideoShotConfig
from core.coordinator import ProcessingCoordinator


class ShotDetectionEvaluator:
    """
    Comprehensive evaluation framework for comparing fusion strategies
    and detection methods across multiple datasets.
    """
    
    def __init__(self, config: VideoShotConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.coordinator = ProcessingCoordinator(config)
        self.results_history = []
        
    def evaluate_fusion_strategies(self, 
                                 video_datasets: Dict[str, List[str]],
                                 ground_truth: Dict[str, List[int]],
                                 fusion_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate all fusion strategies across multiple datasets.
        
        Args:
            video_datasets: Dictionary mapping dataset names to video file lists
            ground_truth: Dictionary mapping video files to ground truth boundaries
            fusion_strategies: List of strategies to evaluate (None for all)
            
        Returns:
            Comprehensive evaluation results
        """
        if fusion_strategies is None:
            fusion_strategies = ["concatenation", "weighted", "pca", "cross_transformer", "hierarchical"]
        
        evaluation_results = {
            "summary": {},
            "detailed_results": {},
            "performance_comparison": {},
            "dataset_analysis": {}
        }
        
        print("Starting comprehensive fusion strategy evaluation...")
        
        for strategy in fusion_strategies:
            print(f"\nEvaluating fusion strategy: {strategy}")
            
            # Update configuration for this strategy
            original_strategy = self.config.fusion.strategy
            self.config.fusion.strategy = strategy
            
            # Re-initialize coordinator with new strategy
            self.coordinator = ProcessingCoordinator(self.config)
            self.coordinator.initialize()
            
            strategy_results = self._evaluate_single_strategy(
                strategy, video_datasets, ground_truth
            )
            
            evaluation_results["detailed_results"][strategy] = strategy_results
            
            # Restore original strategy
            self.config.fusion.strategy = original_strategy
        
        # Compute summary statistics
        evaluation_results["summary"] = self._compute_summary_statistics(
            evaluation_results["detailed_results"]
        )
        
        # Performance comparison
        evaluation_results["performance_comparison"] = self._compare_strategies(
            evaluation_results["detailed_results"]
        )
        
        # Dataset-specific analysis
        evaluation_results["dataset_analysis"] = self._analyze_by_dataset(
            evaluation_results["detailed_results"], video_datasets
        )
        
        return evaluation_results
    
    def _evaluate_single_strategy(self, 
                                strategy: str,
                                video_datasets: Dict[str, List[str]],
                                ground_truth: Dict[str, List[int]]) -> Dict[str, Any]:
        """Evaluate a single fusion strategy."""
        strategy_results = {
            "metrics": {},
            "performance": {},
            "per_video_results": {},
            "dataset_breakdown": {}
        }
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_processing_times = []
        
        for dataset_name, video_files in video_datasets.items():
            print(f"  Processing dataset: {dataset_name}")
            
            dataset_metrics = {
                "precision": [],
                "recall": [], 
                "f1_score": [],
                "processing_time": [],
                "detected_boundaries": [],
                "ground_truth_boundaries": []
            }
            
            for video_file in video_files:
                if video_file not in ground_truth:
                    print(f"    Warning: No ground truth for {video_file}, skipping")
                    continue
                
                print(f"    Processing: {os.path.basename(video_file)}")
                
                try:
                    # Process video
                    results = self.coordinator.process_video(video_file)
                    
                    # Get detection results (prefer combined if available)
                    detection_results = results["detection_results"]
                    if "combined" in detection_results:
                        detected = detection_results["combined"]
                    elif "statistical" in detection_results:
                        detected = detection_results["statistical"]
                    elif "transformer" in detection_results:
                        detected = detection_results["transformer"]
                    else:
                        detected = []
                    
                    gt_boundaries = ground_truth[video_file]
                    
                    # Compute metrics
                    precision, recall, f1 = self._compute_metrics(
                        detected, gt_boundaries, tolerance=5
                    )
                    
                    # Store results
                    dataset_metrics["precision"].append(precision)
                    dataset_metrics["recall"].append(recall)
                    dataset_metrics["f1_score"].append(f1)
                    dataset_metrics["processing_time"].append(results["processing_time"])
                    dataset_metrics["detected_boundaries"].append(detected)
                    dataset_metrics["ground_truth_boundaries"].append(gt_boundaries)
                    
                    # Store per-video results
                    strategy_results["per_video_results"][video_file] = {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "processing_time": results["processing_time"],
                        "detected_boundaries": detected,
                        "ground_truth_boundaries": gt_boundaries,
                        "num_frames": results["num_frames"]
                    }
                    
                    # Add to overall statistics
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1_scores.append(f1)
                    all_processing_times.append(results["processing_time"])
                    
                except Exception as e:
                    print(f"    Error processing {video_file}: {e}")
                    continue
            
            # Compute dataset-level statistics
            if dataset_metrics["precision"]:
                strategy_results["dataset_breakdown"][dataset_name] = {
                    "mean_precision": np.mean(dataset_metrics["precision"]),
                    "mean_recall": np.mean(dataset_metrics["recall"]),
                    "mean_f1_score": np.mean(dataset_metrics["f1_score"]),
                    "std_precision": np.std(dataset_metrics["precision"]),
                    "std_recall": np.std(dataset_metrics["recall"]),
                    "std_f1_score": np.std(dataset_metrics["f1_score"]),
                    "mean_processing_time": np.mean(dataset_metrics["processing_time"]),
                    "num_videos": len(dataset_metrics["precision"])
                }
        
        # Compute overall strategy metrics
        if all_precisions:
            strategy_results["metrics"] = {
                "mean_precision": np.mean(all_precisions),
                "mean_recall": np.mean(all_recalls),
                "mean_f1_score": np.mean(all_f1_scores),
                "std_precision": np.std(all_precisions),
                "std_recall": np.std(all_recalls),
                "std_f1_score": np.std(all_f1_scores),
                "median_precision": np.median(all_precisions),
                "median_recall": np.median(all_recalls),
                "median_f1_score": np.median(all_f1_scores)
            }
            
            strategy_results["performance"] = {
                "mean_processing_time": np.mean(all_processing_times),
                "total_videos_processed": len(all_processing_times),
                "mean_fps": 1.0 / np.mean(all_processing_times) if all_processing_times else 0
            }
        
        return strategy_results
    
    def _compute_metrics(self, detected: List[int], ground_truth: List[int], 
                        tolerance: int = 5) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1-score for boundary detection.
        
        Args:
            detected: List of detected boundary frame indices
            ground_truth: List of ground truth boundary frame indices
            tolerance: Temporal tolerance in frames
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if len(detected) == 0 and len(ground_truth) == 0:
            return 1.0, 1.0, 1.0
        
        if len(detected) == 0:
            return 0.0, 0.0, 0.0
        
        if len(ground_truth) == 0:
            return 0.0, 1.0, 0.0
        
        # Count true positives
        tp = 0
        matched_gt = set()
        
        for det_boundary in detected:
            for i, gt_boundary in enumerate(ground_truth):
                if i not in matched_gt and abs(det_boundary - gt_boundary) <= tolerance:
                    tp += 1
                    matched_gt.add(i)
                    break
        
        # Compute metrics
        precision = tp / len(detected) if len(detected) > 0 else 0.0
        recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1_score
    
    def _compute_summary_statistics(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all strategies."""
        summary = {
            "best_strategy": {},
            "strategy_rankings": {},
            "overall_comparison": {}
        }
        
        strategies = list(detailed_results.keys())
        
        # Find best strategies for each metric
        metrics = ["mean_precision", "mean_recall", "mean_f1_score"]
        
        for metric in metrics:
            best_score = -1
            best_strategy = None
            
            scores = {}
            for strategy in strategies:
                if "metrics" in detailed_results[strategy]:
                    score = detailed_results[strategy]["metrics"].get(metric, 0)
                    scores[strategy] = score
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
            
            summary["best_strategy"][metric] = {
                "strategy": best_strategy,
                "score": best_score
            }
            
            # Rank strategies for this metric
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            summary["strategy_rankings"][metric] = ranked
        
        # Overall comparison table
        comparison_data = []
        for strategy in strategies:
            if "metrics" in detailed_results[strategy]:
                metrics_data = detailed_results[strategy]["metrics"]
                performance_data = detailed_results[strategy]["performance"]
                
                comparison_data.append({
                    "strategy": strategy,
                    "precision": metrics_data.get("mean_precision", 0),
                    "recall": metrics_data.get("mean_recall", 0),
                    "f1_score": metrics_data.get("mean_f1_score", 0),
                    "processing_time": performance_data.get("mean_processing_time", 0),
                    "num_videos": performance_data.get("total_videos_processed", 0)
                })
        
        summary["overall_comparison"] = comparison_data
        
        return summary
    
    def _compare_strategies(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed performance comparison between strategies."""
        comparison = {
            "accuracy_comparison": {},
            "efficiency_comparison": {},
            "statistical_significance": {}
        }
        
        strategies = list(detailed_results.keys())
        
        # Accuracy comparison
        accuracy_data = {}
        for strategy in strategies:
            if "metrics" in detailed_results[strategy]:
                accuracy_data[strategy] = detailed_results[strategy]["metrics"]
        
        comparison["accuracy_comparison"] = accuracy_data
        
        # Efficiency comparison
        efficiency_data = {}
        for strategy in strategies:
            if "performance" in detailed_results[strategy]:
                efficiency_data[strategy] = detailed_results[strategy]["performance"]
        
        comparison["efficiency_comparison"] = efficiency_data
        
        # TODO: Add statistical significance testing
        comparison["statistical_significance"] = {
            "note": "Statistical significance testing to be implemented"
        }
        
        return comparison
    
    def _analyze_by_dataset(self, detailed_results: Dict[str, Any], 
                          video_datasets: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze performance by dataset type."""
        dataset_analysis = {}
        
        for dataset_name in video_datasets.keys():
            dataset_analysis[dataset_name] = {}
            
            for strategy in detailed_results.keys():
                if "dataset_breakdown" in detailed_results[strategy]:
                    if dataset_name in detailed_results[strategy]["dataset_breakdown"]:
                        dataset_analysis[dataset_name][strategy] = \
                            detailed_results[strategy]["dataset_breakdown"][dataset_name]
        
        return dataset_analysis
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file."""
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects."""
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
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_path: str) -> None:
        """Generate a comprehensive evaluation report."""
        report = []
        
        report.append("# Video Shot Detection Evaluation Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary section
        report.append("## Executive Summary")
        
        if "summary" in results and "best_strategy" in results["summary"]:
            best_strategies = results["summary"]["best_strategy"]
            
            report.append("### Best Performing Strategies:")
            for metric, info in best_strategies.items():
                if info["strategy"]:
                    report.append(f"- **{metric}**: {info['strategy']} ({info['score']:.3f})")
        
        report.append("")
        
        # Overall comparison
        if "summary" in results and "overall_comparison" in results["summary"]:
            report.append("## Overall Performance Comparison")
            report.append("")
            report.append("| Strategy | Precision | Recall | F1-Score | Processing Time (s) | Videos |")
            report.append("|----------|-----------|--------|----------|-------------------|--------|")
            
            for strategy_data in results["summary"]["overall_comparison"]:
                report.append(
                    f"| {strategy_data['strategy']} | "
                    f"{strategy_data['precision']:.3f} | "
                    f"{strategy_data['recall']:.3f} | "
                    f"{strategy_data['f1_score']:.3f} | "
                    f"{strategy_data['processing_time']:.2f} | "
                    f"{strategy_data['num_videos']} |"
                )
            
            report.append("")
        
        # Dataset analysis
        if "dataset_analysis" in results:
            report.append("## Dataset-Specific Analysis")
            
            for dataset_name, dataset_results in results["dataset_analysis"].items():
                report.append(f"### {dataset_name}")
                report.append("")
                
                if dataset_results:
                    report.append("| Strategy | Precision | Recall | F1-Score |")
                    report.append("|----------|-----------|--------|----------|")
                    
                    for strategy, metrics in dataset_results.items():
                        report.append(
                            f"| {strategy} | "
                            f"{metrics['mean_precision']:.3f} | "
                            f"{metrics['mean_recall']:.3f} | "
                            f"{metrics['mean_f1_score']:.3f} |"
                        )
                    
                    report.append("")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Evaluation report saved to: {output_path}")


class DatasetManager:
    """Utility class for managing video datasets and ground truth."""
    
    @staticmethod
    def load_clipshots_dataset(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
        """Load ClipShots dataset."""
        # Placeholder implementation
        return {}, {}
    
    @staticmethod
    def load_trecvid_dataset(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
        """Load TRECVID dataset."""
        # Placeholder implementation
        return {}, {}
    
    @staticmethod
    def load_custom_dataset(video_dir: str, ground_truth_file: str) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
        """Load custom dataset from directory and ground truth file."""
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(video_dir).glob(f"*{ext}"))
        
        video_datasets = {"custom": [str(f) for f in video_files]}
        
        # Load ground truth
        ground_truth = {}
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                gt_data = json.load(f)
                ground_truth.update(gt_data)
        
        return video_datasets, ground_truth