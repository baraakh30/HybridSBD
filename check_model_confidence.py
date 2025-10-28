"""
Check model prediction confidence to diagnose if model is learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from detection.transformer_detection import TransformerDetection
from train_transformer import BoundaryDataset
import json

def analyze_model_predictions(model_path: str, features_dir: str, ground_truth_path: str, num_samples: int = 1000):
    """
    Analyze model prediction distribution to check if it's learning properly.
    
    A good model should have:
    - Boundaries predicted with high confidence (>0.7)
    - Non-boundaries predicted with low confidence (<0.3)
    
    A bad model has everything clustered around 0.4-0.6 (ambiguous).
    """
    
    # Load checkpoint to get the actual configuration
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        # Use the config from checkpoint
        model_config = checkpoint['config']
        print(f"Using config from checkpoint: input_dim={model_config.get('input_dim', 'unknown')}")
    else:
        # Try to infer input_dim from model weights
        input_projection_weight = checkpoint['model_state_dict']['input_projection.weight']
        input_dim = input_projection_weight.shape[1]
        print(f"Inferred input_dim from model weights: {input_dim}")
        
        model_config = {
            "input_dim": input_dim,
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "ff_dim": 2048,
            "dropout": 0.1,
            "sequence_length": 100,
            "device": "cuda"
        }
    
    detector = TransformerDetection(model_config)
    detector.load_model(model_path)
    detector.model.eval()
    
    # Determine fusion strategy from input_dim
    # Based on actual feature dimensions: CNN=2048, HSV=768, CLIP=512, Flow=40
    input_dim = model_config.get('input_dim', 512)
    
    # Map input dimensions to fusion strategies
    if input_dim == 512:
        fusion_strategy = "hierarchical"  # or pca, cross_transformer
    elif input_dim == 3368:
        fusion_strategy = "concatenation"  # or weighted
    else:
        # Default to hierarchical for 512-dim models
        fusion_strategy = "hierarchical"
        print(f"⚠️  Unknown input_dim={input_dim}, defaulting to hierarchical fusion")
    
    print(f"Using fusion strategy: {fusion_strategy} (input_dim={input_dim})")
    
    # Load dataset
    dataset = BoundaryDataset(
        features_dir=features_dir,
        ground_truth_file=ground_truth_path,
        fusion_strategy=fusion_strategy,
        sequence_length=100,
        lazy_loading=True,
        quiet=True
    )
    
    # Collect predictions
    boundary_probs = []
    non_boundary_probs = []
    
    print(f"Analyzing {min(num_samples, len(dataset))} samples...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sequence, targets = dataset[i]
            
            # Convert to tensor if needed
            if isinstance(sequence, np.ndarray):
                sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(detector.device)
            else:
                sequence_tensor = sequence.unsqueeze(0).to(detector.device)
            
            if isinstance(targets, np.ndarray):
                targets_tensor = torch.from_numpy(targets)
            else:
                targets_tensor = targets
            
            # Get predictions
            logits = detector.model(sequence_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Separate by target class
            targets_np = targets_tensor.cpu().numpy() if isinstance(targets_tensor, torch.Tensor) else targets_tensor
            for prob, target in zip(probs, targets_np):
                if target == 1:
                    boundary_probs.append(prob)
                else:
                    non_boundary_probs.append(prob)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{min(num_samples, len(dataset))} samples")
    
    # Calculate statistics
    boundary_probs = np.array(boundary_probs)
    non_boundary_probs = np.array(non_boundary_probs)
    
    print("\n" + "="*60)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Boundary frames (n={len(boundary_probs)}):")
    print(f"   Mean: {boundary_probs.mean():.3f}")
    print(f"   Median: {np.median(boundary_probs):.3f}")
    print(f"   Std: {boundary_probs.std():.3f}")
    print(f"   High confidence (>0.7): {(boundary_probs > 0.7).sum()} ({(boundary_probs > 0.7).mean()*100:.1f}%)")
    print(f"   Medium (0.3-0.7): {((boundary_probs >= 0.3) & (boundary_probs <= 0.7)).sum()} ({((boundary_probs >= 0.3) & (boundary_probs <= 0.7)).mean()*100:.1f}%)")
    print(f"   Low confidence (<0.3): {(boundary_probs < 0.3).sum()} ({(boundary_probs < 0.3).mean()*100:.1f}%)")
    
    print(f"\n📊 Non-boundary frames (n={len(non_boundary_probs)}):")
    print(f"   Mean: {non_boundary_probs.mean():.3f}")
    print(f"   Median: {np.median(non_boundary_probs):.3f}")
    print(f"   Std: {non_boundary_probs.std():.3f}")
    print(f"   High confidence (>0.7): {(non_boundary_probs > 0.7).sum()} ({(non_boundary_probs > 0.7).mean()*100:.1f}%)")
    print(f"   Medium (0.3-0.7): {((non_boundary_probs >= 0.3) & (non_boundary_probs <= 0.7)).sum()} ({((non_boundary_probs >= 0.3) & (non_boundary_probs <= 0.7)).mean()*100:.1f}%)")
    print(f"   Low confidence (<0.3): {(non_boundary_probs < 0.3).sum()} ({(non_boundary_probs < 0.3).mean()*100:.1f}%)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    boundary_mean = boundary_probs.mean()
    non_boundary_mean = non_boundary_probs.mean()
    separation = boundary_mean - non_boundary_mean
    
    if separation > 0.3:
        print("✅ GOOD: Model has clear separation between classes")
        print(f"   Boundaries avg {boundary_mean:.3f}, Non-boundaries avg {non_boundary_mean:.3f}")
        print(f"   Separation: {separation:.3f}")
    elif separation > 0.1:
        print("⚠️  MODERATE: Model shows some separation but could be better")
        print(f"   Boundaries avg {boundary_mean:.3f}, Non-boundaries avg {non_boundary_mean:.3f}")
        print(f"   Separation: {separation:.3f}")
        print("   → Try training longer or adjusting learning rate")
    else:
        print("❌ POOR: Model has weak separation (not learning well)")
        print(f"   Boundaries avg {boundary_mean:.3f}, Non-boundaries avg {non_boundary_mean:.3f}")
        print(f"   Separation: {separation:.3f}")
        print("   → Model is not learning discriminative features")
        print("   → Consider: Better features, different architecture, or more training")
    
    # Plot histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(boundary_probs, bins=50, alpha=0.7, label='Boundaries', color='red')
    plt.hist(non_boundary_probs, bins=50, alpha=0.7, label='Non-boundaries', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    plt.axvline(x=0.35, color='gray', linestyle='--', label='Threshold=0.35')
    
    plt.subplot(1, 2, 2)
    plt.hist(boundary_probs, bins=50, alpha=0.7, label=f'Boundaries (μ={boundary_mean:.3f})', color='red', density=True)
    plt.hist(non_boundary_probs, bins=50, alpha=0.7, label=f'Non-boundaries (μ={non_boundary_mean:.3f})', color='blue', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Normalized Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_confidence_analysis.png', dpi=150)
    print("\n💾 Saved plot to: model_confidence_analysis.png")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model prediction confidence")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    analyze_model_predictions(args.model_path, args.features_dir, args.ground_truth, args.num_samples)
