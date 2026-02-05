"""
Extract training history from checkpoint file and generate plots
"""

import torch
import json
import sys
from pathlib import Path

# Import the plotting function from train_hybrid_detector
sys.path.insert(0, str(Path(__file__).parent))
from train_hybrid_detector import plot_training_history, make_serializable


def extract_history_from_checkpoint(checkpoint_path, output_dir=None):
    """
    Extract training history from a checkpoint file
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_dir: Directory to save outputs (defaults to same as checkpoint)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if history exists
    if 'history' in checkpoint:
        history = checkpoint['history']
        print(f"✓ Found training history with {len(history['train'])} epochs")
    else:
        print("⚠ No training history found in checkpoint")
        print("This checkpoint was created before history tracking was added.")
        return None
    
    # Get output directory
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history as JSON
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"✓ History saved to: {history_path}")
    
    # Generate plots
    print("\nGenerating training plots...")
    plot_training_history(history, str(output_dir))
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING HISTORY SUMMARY")
    print("="*80)
    
    if history['val']:
        best_idx = max(range(len(history['val'])), 
                      key=lambda i: history['val'][i].get('overall_f1', 0))
        best_epoch = history['val'][best_idx]['epoch']
        best_f1 = history['val'][best_idx]['overall_f1']
        
        print(f"Total epochs: {len(history['train'])}")
        print(f"Best epoch: {best_epoch} (Overall F1: {best_f1:.4f})")
        print(f"  - Cut F1: {history['val'][best_idx]['cut_f1']:.4f}")
        print(f"  - Gradual F1: {history['val'][best_idx]['gradual_f1']:.4f}")
        
        if history['train']:
            last_epoch = history['train'][-1]['epoch']
            last_f1 = history['val'][-1]['overall_f1']
            print(f"Last epoch: {last_epoch} (Overall F1: {last_f1:.4f})")
    
    print("="*80)
    
    return history


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_checkpoint_history.py <checkpoint.pt> [output_dir]")
        print("\nExamples:")
        print("  python extract_checkpoint_history.py outputs/hybrid_detector/best_model.pt")
        print("  python extract_checkpoint_history.py outputs/hybrid_detector/hybrid_detector_epoch14.pt")
        return
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    extract_history_from_checkpoint(checkpoint_path, output_dir)


if __name__ == '__main__':
    main()
