"""Verify the predecoded caches before any training starts.

For every cache (ClipShots train/test, SHOT train/test, BBC):
  * every clip file listed in index_mapping.json exists and loads;
  * frames are uint8 [128, 3, 48, 64]; train caches also have targets/lengths [128] with labels in {0, 1, 2};
  * every video that SHOULD be in the cache (reconciled against the original annotations) is present;
    videos missing because the source file is unreadable must be allow-listed explicitly;
  * test caches (stride-64 sliding windows): per video, windows start at 0, are <= 64 apart and cover the whole video.
Writes a JSON report and exits non-zero on any failure, so the pipeline can gate on it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

SEQ, CH, H, W = 128, 3, 48, 64


def check_clip(job):
    path, has_labels = job
    import torch

    try:
        clip = torch.load(path, map_location="cpu", weights_only=False)
        frames = clip["frames"]
        if tuple(frames.shape) != (SEQ, CH, H, W) or frames.dtype != torch.uint8:
            return path, f"bad frames {tuple(frames.shape)} {frames.dtype}", {}
        stats = {"black": int(int(frames.max()) == 0)}
        if has_labels:
            t, l = clip["targets"], clip["lengths"]
            if tuple(t.shape) != (SEQ,) or tuple(l.shape) != (SEQ,):
                return path, f"bad labels {tuple(t.shape)} {tuple(l.shape)}", {}
            if int(t.min()) < 0 or int(t.max()) > 2:
                return path, f"label values outside 0..2: {int(t.min())}..{int(t.max())}", {}
            stats["cut_frames"] = int((t == 1).sum())
            stats["gradual_frames"] = int((t == 2).sum())
            stats["has_transition"] = int(bool((t > 0).any()))
        return path, None, stats
    except Exception as exc:  # noqa: BLE001
        return path, repr(exc)[:200], {}


def load_mapping(cache_dir: Path):
    mapping_file = cache_dir / "index_mapping.json"
    if not mapping_file.exists():
        return None
    return json.loads(mapping_file.read_text(encoding="utf-8"))["mapping"]


def expected_windows(frame_num, stride=64):
    last = max(0, int(frame_num) - SEQ)
    starts = list(range(0, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def video_key(entry):
    return entry.get("video_name") or entry.get("video_id")


def stem(name):
    return str(name)[:-4] if str(name).endswith(".mp4") else str(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_test", action="store_true", help="SHOT train excludes the 200 test videos (default policy)")
    parser.add_argument(
        "--allow_missing",
        action="append",
        default=[],
        metavar="CACHE:VIDEO",
        help="a video that may be absent from a cache (e.g. clipshots_train:5186782609.mp4)",
    )
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.data_root)
    allowed = defaultdict(set)
    for item in args.allow_missing:
        cache, video = item.split(":", 1)
        allowed[cache].add(video)

    paths = json.loads((root / "paths.json").read_text(encoding="utf-8"))
    clip_root = Path(paths["clipshots_root"])
    clip_train_ann = json.loads((clip_root / "annotations" / "train.json").read_text(encoding="utf-8"))
    clip_test_ann = json.loads((clip_root / "annotations" / "test.json").read_text(encoding="utf-8"))
    gt = pickle.load(open(paths["autoshot_gt_pickle"], "rb"))
    test_ids = {str(k) for k in gt}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    # SHOT training videos that SHOULD exist: labelled in kuaishou_v2.txt, minus test (if excluded), with a video file
    import os

    os.environ.setdefault("AUTOSHOT_ROOT", paths["autoshot_root"])
    import contextlib
    import io

    from data_prep import predecode_shot_train as shot_train

    with contextlib.redirect_stdout(io.StringIO()):  # the parser logs every malformed line
        labels = shot_train.parse_autoshot_annotations(Path(paths["autoshot_root"]) / "kuaishou_v2.txt")
    on_disk = set()
    for d in ("original_videos", "video_download", "ads_game_videos"):
        on_disk |= {p.name for p in (Path(paths["autoshot_root"]) / d).glob("*.mp4")}
    shot_expected = {v for v in labels if v in on_disk and not (args.exclude_test and stem(v) in test_ids)}
    from data_prep import predecode_bbc

    bbc_expected = {vid_id for _, vid_id, _ in predecode_bbc.VIDEO_MAP}

    caches = [
        # name, dir, has_labels, expected video set, stride-64 test windows?
        ("clipshots_train", root / "cache/clipshots/train", True, set(clip_train_ann), False),
        ("clipshots_test", root / "cache/clipshots/test", False, set(clip_test_ann), True),
        ("shot_train", root / "cache/autoshot_train/train", True, shot_expected, False),
        ("shot_test", root / "cache/autoshot_test", False, {k + ".mp4" for k in test_ids} | test_ids, True),
        ("bbc", root / "cache/bbc", False, bbc_expected, True),
    ]
    if (
        not (root / "cache/autoshot_train/train/index_mapping.json").exists()
        and (root / "cache/autoshot_train/index_mapping.json").exists()
    ):
        caches[2] = ("shot_train", root / "cache/autoshot_train", True, shot_expected, False)

    report, failures = {}, []
    pool = Pool(args.workers)
    for name, cache_dir, has_labels, expected, windows in caches:
        print(f"== {name}: {cache_dir}", flush=True)
        entries = load_mapping(cache_dir)
        if entries is None:
            failures.append(f"{name}: index_mapping.json missing in {cache_dir}")
            report[name] = {"status": "missing index"}
            continue
        by_video = defaultdict(list)
        for e in entries:
            by_video[video_key(e)].append(e)
        present = set(by_video)
        exp_norm = {stem(v) for v in expected}
        miss = sorted(v for v in exp_norm if v not in {stem(p) for p in present})
        miss_allowed = sorted(v for v in miss if v in {stem(a) for a in allowed[name]})
        miss_bad = [v for v in miss if v not in miss_allowed]
        extra = sorted(v for v in {stem(p) for p in present} if v not in exp_norm)

        results = list(pool.imap_unordered(check_clip, [(e["clip_file"], has_labels) for e in entries], chunksize=64))
        bad = [(p, msg) for p, msg, _ in results if msg]
        agg = Counter()
        for _, _, st in results:
            agg.update(st)

        cover_bad = []
        if windows:
            for vid, es in by_video.items():
                starts = sorted(e["start_frame"] for e in es)
                meta = cache_dir / stem(vid) / "metadata.json"
                total = None
                if meta.exists():
                    total = json.loads(meta.read_text(encoding="utf-8")).get("total_frames")
                if name == "clipshots_test":
                    ann = clip_test_ann.get(vid) or clip_test_ann.get(vid + ".mp4")
                    if ann is not None and starts != expected_windows(ann["frame_num"]):
                        cover_bad.append((vid, "windows != expected for annotated frame_num"))
                        continue
                if (
                    starts[0] != 0
                    or any(b - a > 64 for a, b in zip(starts, starts[1:]))
                    or (total is not None and starts[-1] + SEQ < total)
                ):
                    cover_bad.append((vid, f"gap/coverage: starts {starts[:3]}..{starts[-2:]} total {total}"))

        info = {
            "videos_expected": len(exp_norm),
            "videos_present": len(present),
            "clips": len(entries),
            "missing_videos": miss_bad,
            "missing_but_allowed": miss_allowed,
            "unexpected_videos": extra[:10],
            "unreadable_or_bad_clips": len(bad),
            "bad_examples": bad[:5],
            "all_black_clips": agg.get("black", 0),
            "coverage_problems": cover_bad[:5],
        }
        if has_labels:
            info.update(
                {
                    "clips_with_transition": agg.get("has_transition", 0),
                    "cut_frames": agg.get("cut_frames", 0),
                    "gradual_frames": agg.get("gradual_frames", 0),
                }
            )
        report[name] = info
        print(json.dumps(info, indent=1)[:1500], flush=True)
        if miss_bad:
            failures.append(f"{name}: {len(miss_bad)} expected videos missing, e.g. {miss_bad[:3]}")
        if bad:
            failures.append(f"{name}: {len(bad)} bad/unreadable clips, e.g. {bad[:2]}")
        if cover_bad:
            failures.append(f"{name}: {len(cover_bad)} videos with window coverage problems, e.g. {cover_bad[:2]}")
        if not entries:
            failures.append(f"{name}: empty cache")
    pool.close()

    out = {
        "ok": not failures,
        "failures": failures,
        "policy_exclude_shot_test": bool(args.exclude_test),
        "allowed_missing": {k: sorted(v) for k, v in allowed.items()},
        "caches": report,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nVERIFICATION", "PASSED" if not failures else "FAILED")
    for f in failures:
        print(" -", f)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
