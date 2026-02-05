"""
Extract training history using wandb API
This works for both synced and offline runs
"""

import sys
import json
from pathlib import Path
import pandas as pd

def extract_history_from_wandb_api(entity, project, run_id, output_dir=None):
    """
    Extract training history using wandb API
    
    Args:
        entity: wandb entity (username/org)
        project: wandb project name
        run_id: wandb run ID
        output_dir: Where to save outputs
    """
    import wandb
    
    print(f"Connecting to wandb...")
    print(f"  Entity: {entity}")
    print(f"  Project: {project}")
    print(f"  Run ID: {run_id}")
    
    api = wandb.Api()
    run_path = f"{entity}/{project}/runs/{run_id}"
    
    print(f"\nFetching run: {run_path}")
    run = api.run(run_path)
    
    print(f"✓ Found run: {run.name}")
    print(f"  State: {run.state}")
    print(f"  Created: {run.created_at}")
    
    # Get history
    print("\nFetching history...")
    history_df = run.history()
    
    print(f"✓ Retrieved {len(history_df)} data points")
    
    # Convert to our format
    history = {'train': [], 'val': []}
    
    for idx, row in history_df.iterrows():
        epoch = int(row.get('epoch', idx + 1))
        
        # Train record
        train_record = {
            'epoch': epoch,
            'loss': float(row.get('train/loss', 0)),
            'focal_loss': float(row.get('train/focal_loss', 0)),
            'type_loss': float(row.get('train/type_loss', 0)),
            'coherence_loss': float(row.get('train/coherence_loss', 0)),
            'boundary_f1': float(row.get('train/boundary_f1', 0)),
            'boundary_precision': float(row.get('train/boundary_precision', 0)),
            'boundary_recall': float(row.get('train/boundary_recall', 0)),
            'lr': float(row.get('lr', 0))
        }
        
        # Val record
        val_record = {
            'epoch': epoch,
            'loss': float(row.get('val/loss', 0)),
            'focal_loss': float(row.get('val/focal_loss', 0)),
            'type_loss': float(row.get('val/type_loss', 0)),
            'cut_f1': float(row.get('val/cut_f1', 0)),
            'cut_precision': float(row.get('val/cut_precision', 0)),
            'cut_recall': float(row.get('val/cut_recall', 0)),
            'cut_tp': int(row.get('val/cut_tp', 0)),
            'cut_fp': int(row.get('val/cut_fp', 0)),
            'cut_fn': int(row.get('val/cut_fn', 0)),
            'gradual_f1': float(row.get('val/gradual_f1', 0)),
            'gradual_precision': float(row.get('val/gradual_precision', 0)),
            'gradual_recall': float(row.get('val/gradual_recall', 0)),
            'gradual_tp': int(row.get('val/gradual_tp', 0)),
            'gradual_fp': int(row.get('val/gradual_fp', 0)),
            'gradual_fn': int(row.get('val/gradual_fn', 0)),
            'overall_f1': float(row.get('val/overall_f1', 0)),
            'overall_precision': float(row.get('val/overall_precision', 0)),
            'overall_recall': float(row.get('val/overall_recall', 0))
        }
        
        history['train'].append(train_record)
        history['val'].append(val_record)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path('outputs/hybrid_detector')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history JSON
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ History saved to: {history_path}")
    
    # Save raw CSV
    csv_path = output_dir / 'training_history.csv'
    history_df.to_csv(csv_path, index=False)
    print(f"✓ Raw CSV saved to: {csv_path}")
    
    # Generate plots
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from train_hybrid_detector import plot_training_history
        
        print("\nGenerating plots...")
        plot_training_history(history, str(output_dir))
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING HISTORY SUMMARY")
    print("="*80)
    
    if history['val']:
        import numpy as np
        best_idx = np.argmax([m['overall_f1'] for m in history['val']])
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
        print("Usage: python extract_wandb_api.py <run_path> [output_dir]")
        print("\nExamples:")
        print("  python extract_wandb_api.py baraakh-birzeit-university/hybrid-boundary-detector/gym3zbtl")
        print("  python extract_wandb_api.py baraakh-birzeit-university/hybrid-boundary-detector/gym3zbtl outputs/hybrid_detector")
        print("\nRun path format: entity/project/run_id")
        return
    
    # Parse run path
    run_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Split path
    parts = run_path.replace('/runs/', '/').split('/')
    
    if len(parts) == 3:
        entity, project, run_id = parts
    elif len(parts) == 2:
        # Assume default entity
        project, run_id = parts
        entity = "baraakh-birzeit-university"
    else:
        print(f"Error: Invalid run path format: {run_path}")
        print("Expected: entity/project/run_id")
        return
    
    # Extract history
    try:
        extract_history_from_wandb_api(entity, project, run_id, output_dir)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("TROUBLESHOOTING")
        print("="*80)
        print("If you see authentication errors, run:")
        print("  wandb login")
        print("\nIf the run is offline, you need to sync it first:")
        print("  wandb sync <run_directory>")
        print("="*80)


if __name__ == '__main__':
    main()
