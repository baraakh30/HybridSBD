"""Debug script to understand test vs training evaluation difference"""

import numpy as np
import torch
import json
from test_transformer_models import load_model, calculate_metrics
from core.config import VideoShotConfig

# Load one video's features and GT
video_name = '1t9hqz2sqqc'
features = np.load(f'testing_features/{video_name}_pca.npy')
print(f'Features shape: {features.shape}')

# Load GT
with open(r'C:\Users\Baraa\Documents\ClipShots\annotations\test.json') as f:
    gt = json.load(f)
gt_data = gt.get(video_name + '.mp4', {})
boundaries = [t[1] for t in gt_data.get('transitions', [])]
print(f'GT boundaries: {len(boundaries)} - {boundaries}')
print(f'Frame count in GT: {gt_data.get("frame_num")}')

# Check GT boundaries distance
print(f'\n--- GT boundary spacing ---')
for i in range(1, len(boundaries)):
    diff = boundaries[i] - boundaries[i-1]
    if diff < 30:
        print(f'  CLOSE: boundary {boundaries[i-1]} -> {boundaries[i]} = {diff} frames apart (< 30)')

# Load model and detect
config = VideoShotConfig()
detection = load_model('models/transformer_detection_pca.pth', 'pca', config)
print(f'\nModel settings:')
print(f'  confidence_threshold: {detection.confidence_threshold}')
print(f'  use_nms: {detection.use_nms}')
print(f'  nms_distance: {detection.nms_distance}')

# Get raw probabilities to see what the model actually predicts
import torch
windows = []
window_starts = []

# Use same sliding window logic as detect_boundaries
stride = max(1, int(100 * 0.2))  # center_ratio=0.2 typically
for start_idx in range(0, len(features) - 100 + 1, stride):
    end_idx = start_idx + 100
    windows.append(features[start_idx:end_idx])
    window_starts.append(start_idx)

# Batch inference
all_probs = []
detection.model.eval()
with torch.no_grad():
    for window in windows:
        seq_tensor = torch.from_numpy(window).float().unsqueeze(0).to(detection.device)
        logits = detection.model(seq_tensor, mode='frame')
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs[0, :, 0])

# Aggregate
full_probs = np.zeros(len(features) - 1)
prob_counts = np.zeros(len(features) - 1)
for start_idx, probs in zip(window_starts, all_probs):
    center_start = int(len(probs) * 0.4)  # 0.5 - 0.1
    center_end = int(len(probs) * 0.6)   # 0.5 + 0.1
    for j in range(center_start, min(center_end, len(probs))):
        global_idx = start_idx + j
        if 0 <= global_idx < len(full_probs):
            full_probs[global_idx] += probs[j]
            prob_counts[global_idx] += 1
prob_counts[prob_counts == 0] = 1
full_probs = full_probs / prob_counts

# Check probabilities around GT boundaries using actual detection method
print(f'\n--- Model probabilities at GT boundaries (using actual detection) ---')

# Reload model to get fresh detection
from importlib import reload
import detection.transformer_detection as td
reload(td)
from test_transformer_models import load_model
detection = load_model('models/transformer_detection_pca.pth', 'pca', config)

# Run detection to get probs
detected = detection.detect_boundaries(features)
print(f'\nDetected boundaries after edge fix: {len(detected)} - {detected}')

# Check which GT boundaries are missed
missed_gt = [gt for gt in boundaries if not any(abs(d - gt) <= 3 for d in detected)]
print(f'\nMissed GT boundaries: {len(missed_gt)} - {missed_gt}')

detected = detection.detect_boundaries(features)
print(f'\nDetected boundaries: {len(detected)} - {detected}')

# Compare
print(f'\n--- Comparison ---')
print(f'GT: {sorted(boundaries)}')
print(f'Detected: {sorted(detected)}')

# Check metrics
metrics = calculate_metrics(detected, boundaries, tolerance=3)
print(f'\nMetrics (tolerance=3):')
print(f'  Precision: {metrics["precision"]:.4f}')
print(f'  Recall: {metrics["recall"]:.4f}')
print(f'  F1: {metrics["f1_score"]:.4f}')
print(f'  TP: {metrics["true_positives"]}, FP: {metrics["false_positives"]}, FN: {metrics["false_negatives"]}')

