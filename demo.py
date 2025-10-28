"""
Example script demonstrating the video shot detection system.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import VideoShotConfig, DEFAULT_RESEARCH_CONFIG
from core.coordinator import ProcessingCoordinator
from evaluation.evaluator import ShotDetectionEvaluator


def main():
    """Main demonstration function."""
    print("Video Shot Detection System Demo")
    print("=" * 50)
    
    # Example 1: Basic usage with default configuration
    print("\n1. Basic Usage Example")
    print("-" * 30)
    
    # Use default research configuration
    config = DEFAULT_RESEARCH_CONFIG
    config.experiment_name = "demo_experiment"
    
    # Initialize coordinator
    coordinator = ProcessingCoordinator(config)
    coordinator.initialize()
    
    # Show system information
    system_info = coordinator.get_system_info()
    print(f"Available extractors: {system_info['extractors']['available']}")
    print(f"Enabled extractors: {system_info['extractors']['enabled']}")
    print(f"Current fusion strategy: {system_info['fusion']['current']}")
    print(f"Enabled detection methods: {system_info['detection']['enabled']}")
    
    # Example 2: Custom configuration
    print("\n2. Custom Configuration Example")
    print("-" * 30)
    
    # Create custom configuration
    custom_config = VideoShotConfig()
    
    # Enable only HSV and CNN features for faster processing
    custom_config.features.hsv_enabled = True
    custom_config.features.cnn_enabled = True
    custom_config.features.clip_enabled = False
    custom_config.features.optical_flow_enabled = False
    
    # Use concatenation fusion for simplicity
    custom_config.fusion.strategy = "concatenation"
    
    # Use only statistical detection
    custom_config.detection.method = "statistical"
    
    # Faster processing settings
    custom_config.processing.sequence_length = 50
    custom_config.processing.batch_size = 32
    
    print(f"Custom config - Fusion: {custom_config.fusion.strategy}")
    print(f"Custom config - Detection: {custom_config.detection.method}")
    
    # Example 3: Processing a video (placeholder)
    print("\n3. Video Processing Example")
    print("-" * 30)
    
    # Note: This would require an actual video file
    video_path = "path/to/your/video.mp4"
    
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        
        # Process the video
        results = coordinator.process_video(
            video_path=video_path,
            output_path="output/demo_results.json"
        )
        
        print(f"Detected {len(results['detection_results'].get('statistical', []))} boundaries")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Performance: {results['performance_stats']['frames_per_second']:.1f} FPS")
    else:
        print(f"Video file not found: {video_path}")
        print("To test with a real video, update the video_path variable")
    
    # Example 4: Evaluation framework demo
    print("\n4. Evaluation Framework Example")
    print("-" * 30)
    
    # Create evaluator
    evaluator = ShotDetectionEvaluator(config)
    
    # Example dataset structure (placeholder)
    video_datasets = {
        "test_dataset": [
            "path/to/video1.mp4",
            "path/to/video2.mp4"
        ]
    }
    
    ground_truth = {
        "path/to/video1.mp4": [100, 250, 400],  # Frame indices of shot boundaries
        "path/to/video2.mp4": [150, 300, 450]
    }
    
    print("Example evaluation setup:")
    print(f"Datasets: {list(video_datasets.keys())}")
    print(f"Total videos: {sum(len(videos) for videos in video_datasets.values())}")
    
    # Note: Actual evaluation would require real video files and ground truth
    print("To run full evaluation, provide real video files and ground truth data")
    
    # Example 5: Different fusion strategies
    print("\n5. Fusion Strategy Comparison")
    print("-" * 30)
    
    strategies = ["concatenation", "weighted", "pca", "hierarchical"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        # Create config for this strategy
        strategy_config = VideoShotConfig()
        strategy_config.fusion.strategy = strategy
        
        # Show strategy-specific parameters
        if strategy == "weighted":
            print(f"  - Optimization: {strategy_config.fusion.weights_optimization}")
            print(f"  - GA population: {strategy_config.fusion.ga_population}")
        
        elif strategy == "pca":
            print(f"  - Variance retained: {strategy_config.fusion.pca_variance_retained}")
        
        elif strategy == "hierarchical":
            print(f"  - Conv kernel: {strategy_config.fusion.hierarchical_conv_kernel}")
            print(f"  - Attention heads: {strategy_config.fusion.hierarchical_attention_heads}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nTo use the system with real videos:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Update video paths in the examples above")
    print("3. Prepare ground truth data in JSON format")
    print("4. Run the evaluation framework")


if __name__ == "__main__":
    main()