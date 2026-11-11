"""Full-video evaluation data for the 67 SHOT *validation* videos (Protocol B, threshold selection on validation).

The validation split (seed 42, 15 % of the 448 training videos) was previously available only as pre-cut training
clips. This script decodes those 67 videos completely, with the same decoder, resizing (48 x 64), window layout
(T = 128, stride 64, last window shifted back to end on the final frame, short videos padded) and index format as the
SHOT test cache (predecode_autoshot.py), so test_autoshot.py can evaluate them unchanged. Ground truth is built from the
SHOT label file (kuaishou_v2.txt) with the same transition -> scene conversion that is used for ClipShots
(transitions_to_scenes); transitions beyond the decoded length are dropped, as in training-data preparation.

Outputs (under --out): <video>/clip_XXXXXX.pt, index_mapping.json, val_gt_scenes.pickle, val_videos.json
"""
import argparse, hashlib, json, pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from datasets import PredecodedDataset
from frame_utils import decode_resized
from predecode_autoshot_train import parse_autoshot_annotations
from test_clipshots import transitions_to_scenes

SEQ, STRIDE, SIZE = 128, 64, (48, 64)
VDIRS = ["original_videos", "video_download", "ads_game_videos"]


def find_video(root, name):
    for d in VDIRS:
        p = Path(root) / d / name
        if p.exists():
            return str(p)
    return None


def work(a):
    name, path, transitions, out = a
    stem = Path(name).stem
    vdir = Path(out) / stem
    vdir.mkdir(parents=True, exist_ok=True)
    dec = VideoDecoder(path)
    total = int(len(dec))
    valid = [(s, e) for s, e in transitions if e < total]
    scenes = transitions_to_scenes(valid, total)
    clips = []
    idx = 0
    for start in range(0, total, STRIDE):
        if start + SEQ > total:
            start = max(0, total - SEQ)
        end = min(start + SEQ, total)
        frames = decode_resized(dec, list(range(start, end)), SIZE)
        if frames.shape[0] < SEQ:
            frames = torch.cat([frames, frames[-1:].repeat(SEQ - frames.shape[0], 1, 1, 1)], 0)
        f = vdir / f"clip_{idx:06d}.pt"
        torch.save({"frames": frames, "start_frame": start, "end_frame": end}, f)
        clips.append({"clip_file": str(f), "start_frame": start})
        idx += 1
        if end >= total:
            break
    return {"stem": stem, "total_frames": total, "n_transitions": len(transitions), "n_valid": len(valid),
            "scenes": scenes, "clips": clips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoshot_root", required=True)
    ap.add_argument("--train_cache", required=True, help="the SHOT training cache used for training (split source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    ds = PredecodedDataset(a.train_cache, split="val", val_split_ratio=0.15, random_seed=a.split_seed)
    names = sorted({c["video_name"] for c in ds.mapping})
    print("validation videos:", len(names), "sha256", hashlib.sha256("\n".join(names).encode()).hexdigest())
    ann = parse_autoshot_annotations(Path(a.autoshot_root) / "kuaishou_v2.txt")
    jobs, missing = [], []
    for n in names:
        key = n if n.endswith(".mp4") else n + ".mp4"
        p = find_video(a.autoshot_root, key)
        if p is None or key not in ann:
            missing.append(n); continue
        jobs.append((key, p, ann[key]["transitions"], a.out))
    if missing:
        raise SystemExit(f"missing videos/annotations: {missing}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=1)
    res.sort(key=lambda r: r["stem"])
    mapping = []
    for r in res:
        for c in r["clips"]:
            mapping.append({"index": len(mapping), "video_id": r["stem"], "clip_file": c["clip_file"], "start_frame": c["start_frame"]})
    json.dump({"dataset": "SHOT-validation-full", "num_videos": len(res), "num_clips": len(mapping), "mapping": mapping},
              open(Path(a.out) / "index_mapping.json", "w"))
    pickle.dump({r["stem"]: r["scenes"] for r in res}, open(Path(a.out) / "val_gt_scenes.pickle", "wb"))
    json.dump({"videos": [r["stem"] for r in res], "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
               "frames": {r["stem"]: r["total_frames"] for r in res},
               "transitions_total": sum(r["n_transitions"] for r in res), "transitions_used": sum(r["n_valid"] for r in res)},
              open(Path(a.out) / "val_videos.json", "w"), indent=1)
    print("videos", len(res), "clips", len(mapping), "transitions", sum(r["n_valid"] for r in res), "/", sum(r["n_transitions"] for r in res))


if __name__ == "__main__":
    main()
