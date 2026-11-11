"""Pre-decoded evaluation clips and ground-truth loaders for the SHOT, ClipShots and BBC test sets."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from hybridsbd.scoring import transitions_to_scenes


class PredecodedClipDataset(Dataset):
    """Clips of one test set, listed in ``<cache_root>/index_mapping.json`` (as written by ``data_prep/predecode_*``).

    Every item holds the clip's frames (float in [0, 1]), the id of the video it belongs to and the start frame of the
    clip inside that video, which is what the full-video reconstruction needs.

    Args:
        cache_root: directory with ``index_mapping.json`` and the clip files.
        id_key: key of the video identifier in the index entries (``video_id`` for SHOT and BBC, ``video_name`` for
            ClipShots).
        sort_clips: sort the entries by ``(video_id, start_frame)`` (BBC).
        name: dataset name used in the log line.
    """

    def __init__(self, cache_root: str | Path, id_key: str = "video_id", sort_clips: bool = False, name: str = "Dataset"):
        self.cache_root = Path(cache_root)
        with open(self.cache_root / "index_mapping.json", "r") as f:
            index = json.load(f)
        self.mapping = index["mapping"]
        self.id_key = id_key
        if sort_clips:
            self.mapping.sort(key=lambda e: (e[id_key], e["start_frame"]))
        print(f"{name}: {index['num_videos']} videos, {len(self.mapping)} clips")

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, idx: int) -> dict:
        entry = self.mapping[idx]
        clip = torch.load(entry["clip_file"], weights_only=False)
        return {
            "frames": clip["frames"].float() / 255.0,  # [T, 3, H, W]
            "video_id": entry[self.id_key],
            "start_frame": entry["start_frame"],
            "clip_index": idx,
        }


def collate_clips(batch: list) -> dict:
    """Stack a list of dataset items into one batch."""
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "video_id": [item["video_id"] for item in batch],
        "start_frame": torch.tensor([item["start_frame"] for item in batch]),
        "clip_index": torch.tensor([item["clip_index"] for item in batch]),
    }


# --------------------------------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------------------------------


def load_shot_gt(gt_path: str | Path) -> Dict[str, np.ndarray]:
    """SHOT ground truth: a pickle ``{video_id: N x 2 scenes}`` (``gt_scenes_dict_baseline_v2.pickle``)."""
    with open(gt_path, "rb") as f:
        return pickle.load(f)


def load_clipshots_gt(ann_path: str | Path, video_names: Iterable[str]) -> Dict[str, np.ndarray]:
    """ClipShots ground truth from ``annotations/test.json`` (``{video: {transitions: [[s, e], ...], frame_num: N}}``).

    Video names are matched with and without the ``.mp4`` extension; videos without an annotation are skipped.
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)

    ann_by_key = {}
    for key, value in ann.items():
        ann_by_key[key] = value
        ann_by_key[key.replace(".mp4", "")] = value
        ann_by_key[key + ".mp4"] = value

    gt: Dict[str, np.ndarray] = {}
    missing = 0
    for name in video_names:
        entry = ann_by_key.get(name)
        if entry is None:
            missing += 1
            continue
        gt[name] = transitions_to_scenes(entry["transitions"], entry["frame_num"])

    if missing:
        print(f"  warning: {missing} videos not found in annotations (will be skipped)")
    print(f"  {len(gt)} videos with GT scenes loaded")
    return gt


def read_bbc_shot_annotations(annotation_file: str | Path) -> np.ndarray:
    """Read one BBC annotation file (one ``start_frame end_frame`` pair per line) into an ``N x 2`` array."""
    shots = []
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                shots.append([int(parts[0]), int(parts[1])])
    shots.sort(key=lambda x: x[0])
    return np.array(shots, dtype=np.int32)


def _normalize_episode_name(name: str) -> str:
    """Strip the numeric prefix, lower-case and drop spaces, underscores and hyphens."""
    name = re.sub(r"^\d+[_\s]*", "", name)
    return re.sub(r"[\s_\-]+", "", name).lower()


def load_bbc_gt(bbc_root: str | Path, cache_root: str | Path) -> Dict[str, np.ndarray]:
    """Ground-truth scenes of the BBC episodes in the cache: ``{video_id: N x 2 scenes}``.

    Annotation files are matched to cache videos by *episode name* (numeric prefix ignored), because the cache was
    pre-decoded in file-system order, which differs from the numbering of the annotation files.
    """
    ann_dir = Path(bbc_root) / "annotations" / "shots"
    with open(Path(cache_root) / "index_mapping.json", "r") as f:
        index = json.load(f)

    video_ids = []
    for clip in index["mapping"]:
        if clip["video_id"] not in video_ids:
            video_ids.append(clip["video_id"])

    ann_by_norm = {_normalize_episode_name(p.stem): p for p in ann_dir.glob("*.txt")}

    gt: Dict[str, np.ndarray] = {}
    for vid in video_ids:
        norm = _normalize_episode_name(vid)
        if norm not in ann_by_norm:
            print(f"  warning: no annotation file for '{vid}' (normalised='{norm}')")
            print(f"  available: {sorted(ann_by_norm.keys())}")
            continue
        scenes = read_bbc_shot_annotations(ann_by_norm[norm])
        print(f"  {vid} -> {ann_by_norm[norm].name}  ({len(scenes)} shots)")
        if len(scenes) > 0:
            gt[vid] = scenes
    return gt
