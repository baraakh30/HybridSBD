"""SHOT validation-video checksum for training seeds 42/43/44: old rule (split follows the training seed) vs fixed split_seed=42."""
import hashlib, sys
from datasets import PredecodedDataset
cache = sys.argv[1]
def h(seed):
    ds = PredecodedDataset(cache, split="val", val_split_ratio=0.15, random_seed=seed)
    names = sorted({c["video_name"] for c in ds.mapping})
    return len(names), hashlib.sha256("\n".join(names).encode()).hexdigest()
print("OLD RULE (split seed = training seed):")
for s in (42, 43, 44): print(" ", s, *h(s))
print("FIXED (split_seed = 42 for every training seed):")
for s in (42, 43, 44): print(" ", s, *h(42))
