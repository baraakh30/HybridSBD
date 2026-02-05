"""
Simple approach: Use wandb sync to export the offline run,
then you can download the CSV from wandb.ai
"""

import subprocess
import sys
from pathlib import Path

def sync_wandb_run(run_dir):
    """
    Sync the offline wandb run to wandb.ai
    This will upload the data so you can download CSV from the web interface
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return
    
    print(f"Syncing wandb run: {run_dir}")
    print("\nThis will upload the run to wandb.ai")
    print("You can then download the training history as CSV from the web interface")
    print("\nCommand:")
    print(f"  wandb sync {run_dir}")
    print("\nAfter syncing, go to wandb.ai and:")
    print("  1. Find your run")
    print("  2. Click on the 'Table' tab")
    print("  3. Click 'Export' to download as CSV")
    print("\n" + "="*80)
    
    # Try to run wandb sync
    try:
        result = subprocess.run(
            ['wandb', 'sync', str(run_dir)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Sync successful!")
            print("Visit https://wandb.ai to view your run and export the data")
        else:
            print(f"\n✗ Sync failed with code {result.returncode}")
            
    except FileNotFoundError:
        print("\nError: 'wandb' command not found")
        print("Make sure wandb is installed and in your PATH")
    except Exception as e:
        print(f"\nError running wandb sync: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sync_wandb_history.py <wandb_run_dir>")
        print("\nExample:")
        print(r'  python sync_wandb_history.py "D:\grad\wandb\offline-run-20260202_215400-gym3zbtl"')
        print("\nOr just run manually:")
        print(r'  wandb sync "D:\grad\wandb\offline-run-20260202_215400-gym3zbtl"')
        sys.exit(1)
    
    run_dir = sys.argv[1]
    sync_wandb_run(run_dir)
