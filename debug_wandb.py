"""
Debug script to inspect wandb file structure
"""

import sys
from pathlib import Path

def debug_wandb_file(wandb_file):
    """Debug what's in the wandb file"""
    print(f"Inspecting: {wandb_file}")
    
    from wandb.sdk.internal import datastore
    
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    
    record_types = {}
    sample_records = []
    
    for i, record in enumerate(ds.scan_data()):
        record_type = type(record).__name__
        record_types[record_type] = record_types.get(record_type, 0) + 1
        
        # Collect samples
        if len(sample_records) < 10:
            sample_records.append((i, record_type, record))
        
        # Stop after 1000 records for debugging
        if i > 1000:
            break
    
    ds.close()
    
    print(f"\nRecord types found:")
    for rtype, count in record_types.items():
        print(f"  {rtype}: {count}")
    
    print(f"\nSample records:")
    for idx, rtype, record in sample_records[:5]:
        print(f"\n[{idx}] {rtype}:")
        print(f"  Value: {record}")
        
        # If it has DESCRIPTOR, show fields
        if hasattr(record, 'DESCRIPTOR'):
            print(f"  Fields: {[f.name for f in record.DESCRIPTOR.fields]}")
            
            # Try to show some content
            for field in record.DESCRIPTOR.fields:
                try:
                    value = getattr(record, field.name)
                    if value:
                        print(f"    {field.name}: {str(value)[:200]}")
                except:
                    pass

if __name__ == '__main__':
    wandb_file = r"D:\grad\wandb\offline-run-20260202_215400-gym3zbtl\run-gym3zbtl.wandb"
    debug_wandb_file(wandb_file)
