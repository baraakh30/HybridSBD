# Video Shot Detection System

A comprehensive multimodal video shot boundary detection system that integrates multiple feature representations through sophisticated fusion strategies to achieve robust and accurate shot boundary detection.

## System Overview

The system follows a modular design that can be deployed in both statistical and deep learning modes, featuring:

- **4 Feature Extractors**: HSV histograms, CNN features (ResNet-50), CLIP embeddings, and optical flow
- **5 Fusion Strategies**: Concatenation, weighted sum, PCA, cross-transformer, and hierarchical fusion
- **2 Detection Methods**: Statistical (dynamic thresholding) and transformer-based approaches
- **Comprehensive Evaluation**: Framework for comparing strategies across multiple datasets

## Architecture

```
Video Input → Feature Extraction → Fusion → Detection → Results
    ↓              ↓                ↓         ↓         ↓
Frame         HSV, CNN,      Concatenation,  Statistical,  Boundaries,
Extraction    CLIP, Flow     Weighted, PCA,  Transformer   Segments,
                            Transformer,                   Keyframes
                            Hierarchical
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video_shot_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies:
```bash
# For CLIP (if not installed via requirements)
pip install git+https://github.com/openai/CLIP.git

```

## Quick Start

### Basic Usage

```python
from core.config import DEFAULT_RESEARCH_CONFIG
from core.coordinator import ProcessingCoordinator

# Initialize with default configuration
config = DEFAULT_RESEARCH_CONFIG
coordinator = ProcessingCoordinator(config)
coordinator.initialize()

# Process a video
results = coordinator.process_video("path/to/video.mp4")

# Get detected boundaries
boundaries = results["detection_results"]["statistical"]
print(f"Detected {len(boundaries)} shot boundaries")
```

### Custom Configuration

```python
from core.config import VideoShotConfig

# Create custom configuration
config = VideoShotConfig()

# Configure features
config.features.hsv_enabled = True
config.features.cnn_enabled = True
config.features.clip_enabled = False
config.features.optical_flow_enabled = True

# Configure fusion strategy
config.fusion.strategy = "hierarchical"

# Configure detection
config.detection.method = "both"  # Use both statistical and transformer

# Initialize and process
coordinator = ProcessingCoordinator(config)
coordinator.initialize()
results = coordinator.process_video("video.mp4")
```

## Feature Extractors

### 1. HSV Histogram Features (768-dim)
- Converts RGB frames to HSV color space
- Computes histograms for H, S, V channels (256 bins each)
- Applies Gaussian smoothing and normalization

### 2. CNN Features (2048-dim)
- Uses pretrained ResNet-50 model
- Extracts features from penultimate layer
- Applies L2 normalization

### 3. CLIP Embeddings (512-dim)
- Uses OpenAI's CLIP vision encoder
- Provides semantic representation
- Multiple model variants supported

### 4. Optical Flow Features (40-dim)
- Computes dense optical flow between frames
- Extracts Histogram of Optical Flow (HOF)
- 8 direction bins × 5 magnitude bins

## Fusion Strategies

### 1. Concatenation-Based Fusion
Simple baseline that concatenates all features and computes Euclidean distance.

### 2. Weighted Sum of Individual Distances
Computes individual distances per feature type and combines using optimized weights via genetic algorithm.

### 3. PCA-Based Dimensionality Reduction
Applies PCA to concatenated features, retaining 95-99% variance for denoising and efficiency.

### 4. Cross-Transformer Fusion
Uses cross-modal attention to learn adaptive fusion weights based on context.

### 5. Hierarchical Fusion (3-Level)
- **Level 1**: HSV + Optical Flow → 256-dim (1D convolution)
- **Level 2**: CNN + Level1 → 512-dim (cross-attention)
- **Level 3**: CLIP + Level2 → 512-dim (gated fusion)

## Detection Methods

### Statistical Detection
- Dynamic thresholding with multi-scale analysis
- Adaptive window management
- Peak detection with prominence scoring
- Content-aware filtering

### Transformer Detection
- 6-layer transformer encoder
- Positional encoding for temporal relationships
- Binary boundary classification
- Sequence-level training with balanced sampling

## Evaluation Framework

Comprehensive evaluation system for comparing fusion strategies:

```python
from evaluation.evaluator import ShotDetectionEvaluator

# Setup evaluation
evaluator = ShotDetectionEvaluator(config)

# Define datasets and ground truth
video_datasets = {
    "dataset1": ["video1.mp4", "video2.mp4"],
    "dataset2": ["video3.mp4", "video4.mp4"]
}

ground_truth = {
    "video1.mp4": [100, 250, 400],  # Boundary frame indices
    "video2.mp4": [150, 300, 450]
}

# Run evaluation
results = evaluator.evaluate_fusion_strategies(
    video_datasets, ground_truth
)

# Generate report
evaluator.generate_evaluation_report(results, "evaluation_report.md")
```

## Configuration Options

### Feature Configuration
```python
config.features.hsv_enabled = True
config.features.hsv_bins = 256
config.features.cnn_model = "resnet50"
config.features.clip_model = "ViT-B/32"
config.features.optical_flow_method = "farneback"
```

### Fusion Configuration
```python
config.fusion.strategy = "hierarchical"
config.fusion.pca_variance_retained = 0.95
config.fusion.transformer_heads = 4
config.fusion.weights_optimization = "genetic"
```

### Detection Configuration
```python
config.detection.method = "both"
config.detection.threshold_method = "adaptive"
config.detection.smoothing_method = "gaussian"
config.detection.transformer_layers = 6
```

## Performance Optimization

### For Real-time Processing
```python
# Use lightweight configuration
config = DEFAULT_EDGE_CONFIG
config.features.clip_enabled = False
config.fusion.strategy = "concatenation"
config.detection.method = "statistical"
```

### For High Accuracy
```python
# Use full research configuration
config = DEFAULT_RESEARCH_CONFIG
config.fusion.strategy = "hierarchical"
config.detection.method = "both"
```

## Output Formats

### JSON Results
```json
{
  "video_path": "video.mp4",
  "detection_results": {
    "statistical": [45, 120, 200],
    "transformer": [47, 118, 202],
    "combined": [46, 119, 201]
  },
  "fusion_strategy": "hierarchical",
  "processing_time": 15.2,
  "performance_stats": {
    "frames_per_second": 12.5,
    "memory_usage": {"total": 2048}
  }
}
```

### Video Segments
The system can automatically extract:
- Shot segments as separate video files
- Keyframes at boundary locations
- Temporal annotations

## Deployment Options

### Research Environment
- Full feature set with all extractors
- Both detection methods enabled
- Comprehensive evaluation capabilities

### Production Environment
- Optimized feature extraction
- Statistical detection for real-time performance
- Configurable quality/speed trade-offs

### Edge Deployment
- Lightweight features (HSV + Optical Flow)
- Statistical detection only
- Memory-optimized processing

## Datasets Supported

The evaluation framework supports multiple datasets:
- **ClipShots**: Modern online content
- **TRECVID**: Broadcast content
- **MSU**: Mixed content types
- **RAI**: Documentary footage
- **BBC Planet Earth**: Nature content
- **Custom datasets**: With JSON ground truth

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{video_shot_detection_2024,
  title={Multimodal Video Shot Detection System},
  author={Graduate Research Team},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/video-shot-detection}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI CLIP for semantic embeddings
- PyTorch team for the deep learning framework
- Research communities for dataset contributions

train_transformer.py --video_dir "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --extract_features --features_dir "training_features"


C:/Users/Baraa/anaconda3/envs/unsloth_env/python.exe d:/grad/video_shot_detection/main_test.py --video_path "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train\3505586409.mp4" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --method statistical --tolerance 10

train_transformer.py --video_dir "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --extract_features --features_dir "training_features" --disable_logging --train_all_strategies

train_transformer.py --video_dir "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --extract_features --features_dir "training_features"  --max_workers 10 --parallel_features

train_transformer.py --video_dir "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --extract_features --features_dir "training_features" --disable_logging --max_workers 10 --parallel_features




python interactive_visualizer.py --video "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\videos\train\4001498009.mp4" --results "test_results_single\test_results.json" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json"




"C:/Users/Abdalrahman Khanfar/Documents/grad/.venv/Scripts/python.exe" test_sequence_model.py --model_path "C:\Users\Abdalrahman Khanfar\Documents\grad\models\111.pth" --features_dir "training_features" --ground_truth "C:\Users\Abdalrahman Khanfar\Documents\ClipShots\annotations\train.json" --output_dir "test_results_single" --num_videos 300 --visualize --sequence_threshold 0.6 --boundary_selection max_diff