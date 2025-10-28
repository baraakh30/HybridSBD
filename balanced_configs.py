"""
Balanced configurations to reduce false positives while maintaining good recall.
"""

from core.config import VideoShotConfig, FeatureConfig, FusionConfig, DetectionConfig, ProcessingConfig

# The issue: Current best config has F1=0.698 but Precision=0.562 (44% false positives)
# Goal: Increase precision to 0.7+ while keeping recall above 0.8

# =====================================================
# BALANCED FIX #1: Higher Threshold + Stricter Filtering
# =====================================================
BALANCED_CONFIG_V1 = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        clip_enabled=True,
        optical_flow_enabled=True,
        cnn_model="resnet50",
        clip_model="ViT-B/32",
        optical_flow_method="farneback"
    ),
    fusion=FusionConfig(
        strategy="weighted",
        weights_optimization="genetic",
    ),
    detection=DetectionConfig(
        method="statistical",
        smoothing_method="gaussian",
        smoothing_window=9,      # INCREASED: More smoothing to reduce noise peaks
        z_score_window=50,       # INCREASED: Better normalization
        threshold_method="adaptive",
        adaptive_beta=2.1,       # INCREASED: Higher threshold to reduce false positives
        short_scale=20,
        medium_scale=50,
        long_scale=80
    ),
    processing=ProcessingConfig(
        min_shot_duration=25,    # INCREASED: Filter very short false detections
        temporal_tolerance=5,
        batch_size=32,
        device="cuda"
    )
)

# =====================================================
# BALANCED FIX #2: Savgol Smoothing + Higher Beta
# =====================================================
# Savgol filter is better at preserving real peaks while removing noise
BALANCED_CONFIG_V2 = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        clip_enabled=True,
        optical_flow_enabled=True,
        cnn_model="resnet50",
        clip_model="ViT-B/32",
        optical_flow_method="farneback"
    ),
    fusion=FusionConfig(
        strategy="weighted",
        weights_optimization="genetic",
    ),
    detection=DetectionConfig(
        method="statistical",
        smoothing_method="savgol",  # CHANGED: Better peak preservation
        smoothing_window=11,         # INCREASED: Strong noise reduction
        z_score_window=50,
        threshold_method="adaptive",
        adaptive_beta=2.0,           # INCREASED
        short_scale=20,
        medium_scale=50,
        long_scale=80
    ),
    processing=ProcessingConfig(
        min_shot_duration=25,
        temporal_tolerance=8,        # INCREASED: Merge close false positives
        batch_size=32,
        device="cuda"
    )
)

# =====================================================
# BALANCED FIX #3: Multi-scale + Percentile Threshold
# =====================================================
# Use percentile thresholding to focus on top peaks only
BALANCED_CONFIG_V3 = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        clip_enabled=True,
        optical_flow_enabled=True,
        cnn_model="resnet50",
        clip_model="ViT-B/32",
        optical_flow_method="farneback"
    ),
    fusion=FusionConfig(
        strategy="weighted",
        weights_optimization="genetic",
    ),
    detection=DetectionConfig(
        method="statistical",
        smoothing_method="gaussian",
        smoothing_window=9,
        z_score_window=50,
        threshold_method="percentile",  # CHANGED: Only detect top X% peaks
        fixed_threshold=95,              # Top 5% of signal
        short_scale=25,
        medium_scale=60,
        long_scale=100
    ),
    processing=ProcessingConfig(
        min_shot_duration=25,
        temporal_tolerance=8,
        batch_size=32,
        device="cuda"
    )
)

# =====================================================
# BALANCED FIX #4: Concatenation with High Threshold
# =====================================================
# Concatenation has better signal (2.78x), use it with high threshold
BALANCED_CONFIG_V4 = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=True,
        cnn_enabled=True,
        clip_enabled=True,
        optical_flow_enabled=True,
        cnn_model="resnet50",
        clip_model="ViT-B/32",
        optical_flow_method="farneback"
    ),
    fusion=FusionConfig(
        strategy="concatenation",  # CHANGED: Better base signal
    ),
    detection=DetectionConfig(
        method="statistical",
        smoothing_method="savgol",
        smoothing_window=11,
        z_score_window=50,
        threshold_method="adaptive",
        adaptive_beta=2.5,       # VERY HIGH: Reduce false positives aggressively
        short_scale=25,
        medium_scale=60,
        long_scale=100
    ),
    processing=ProcessingConfig(
        min_shot_duration=25,
        temporal_tolerance=10,   # Higher tolerance to merge close detections
        batch_size=32,
        device="cuda"
    )
)

# =====================================================
# BALANCED FIX #5: Feature Subset (Remove Noisy Features)
# =====================================================
# Maybe some features are adding more noise than signal
BALANCED_CONFIG_V5_NO_HSV = VideoShotConfig(
    features=FeatureConfig(
        hsv_enabled=False,       # DISABLED: Test without HSV
        cnn_enabled=True,
        clip_enabled=True,
        optical_flow_enabled=True,
        cnn_model="resnet50",
        clip_model="ViT-B/32",
        optical_flow_method="farneback"
    ),
    fusion=FusionConfig(
        strategy="weighted",
    ),
    detection=DetectionConfig(
        method="statistical",
        smoothing_method="gaussian",
        smoothing_window=9,
        z_score_window=50,
        threshold_method="adaptive",
        adaptive_beta=2.0,
        short_scale=20,
        medium_scale=50,
        long_scale=80
    ),
    processing=ProcessingConfig(
        min_shot_duration=25,
        temporal_tolerance=8,
        batch_size=32,
        device="cuda"
    )
)

# =====================================================
# TARGET METRICS
# =====================================================
TARGET_METRICS = {
    "precision": 0.70,  # At least 70% of detections should be correct
    "recall": 0.82,     # Catch at least 80% of boundaries (40/49)
    "f1_score": 0.76,   # Target F1 score
    "false_positives": "< 20",  # Maximum acceptable FP
    "false_negatives": "< 10",  # Maximum acceptable FN
}

if __name__ == "__main__":
    print("="*70)
    print("BALANCED CONFIGURATIONS TO REDUCE FALSE POSITIVES")
    print("="*70)
    
    print("\nCURRENT PROBLEM:")
    print("  Best config (Weighted, beta=1.8):")
    print("    ✓ F1: 0.698, Recall: 0.918 (45/49 boundaries)")
    print("    ✗ Precision: 0.562 (35 false positives out of 80 detections)")
    print("    → 44% of detections are WRONG!")
    
    print("\n" + "="*70)
    print("TARGET PERFORMANCE:")
    print("="*70)
    for metric, value in TARGET_METRICS.items():
        print(f"  {metric}: {value}")
    
    print("\n" + "="*70)
    print("BALANCED CONFIGURATIONS:")
    print("="*70)
    
    configs = {
        "V1_Higher_Threshold": {
            "desc": "smoothing_window=9, adaptive_beta=2.1, min_shot=25",
            "expected": "Precision ≈ 0.65-0.70, Recall ≈ 0.85"
        },
        "V2_Savgol_Filter": {
            "desc": "savgol smoothing, window=11, beta=2.0, temporal_tol=8",
            "expected": "Precision ≈ 0.68-0.72, Recall ≈ 0.82"
        },
        "V3_Percentile_Threshold": {
            "desc": "Top 5% peaks only (percentile=95)",
            "expected": "Precision ≈ 0.70-0.75, Recall ≈ 0.78"
        },
        "V4_Concat_High_Threshold": {
            "desc": "concatenation fusion, beta=2.5, temporal_tol=10",
            "expected": "Precision ≈ 0.72-0.78, Recall ≈ 0.75"
        },
        "V5_Without_HSV": {
            "desc": "Remove HSV features, beta=2.0",
            "expected": "Precision ≈ 0.65-0.70, Recall ≈ 0.83"
        }
    }
    
    for name, info in configs.items():
        print(f"\n{name}:")
        print(f"  Settings: {info['desc']}")
        print(f"  Expected: {info['expected']}")
    
    print("\n" + "="*70)
    print("KEY CHANGES TO REDUCE FALSE POSITIVES:")
    print("="*70)
    print("1. INCREASE adaptive_beta: 1.8 → 2.0-2.5")
    print("   → Higher threshold = fewer false positives")
    print("\n2. INCREASE smoothing_window: 7 → 9-11")
    print("   → Removes noise peaks that cause false detections")
    print("\n3. INCREASE min_shot_duration: 20 → 25")
    print("   → Filters out very short false detections")
    print("\n4. INCREASE temporal_tolerance: 5 → 8-10")
    print("   → Merges nearby false positives into single detection")
    print("\n5. TRY savgol filter instead of gaussian")
    print("   → Better preserves real peaks while removing noise")
    print("\n6. TRY percentile thresholding")
    print("   → Only detects strongest peaks (top 5%)")
    print("\n7. TRY concatenation fusion with high beta")
    print("   → Simpler, more predictable signal")
    print("\n8. TRY removing HSV features")
    print("   → HSV might be adding noise")
    
    print("\n" + "="*70)
    print("TEST COMMAND:")
    print("="*70)
    print("python test_balanced_configs.py")
