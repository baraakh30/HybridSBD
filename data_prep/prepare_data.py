"""Download, validate, extract, and predecode the Protocol-A datasets.

The Google Drive folders are treated as canonical sources.  The script keeps
archives intact, extracts independent archives into one dataset root, and
joins only files that are clearly split archive members.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: lets the script import the repository packages

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from hybridsbd.utils import file_sha256

HERE = Path(__file__).resolve().parent  # data_prep/ (the predecode scripts live next to this file)


CLIPSHOTS_URL = "https://drive.google.com/drive/folders/1AAhTbNroSFsygHBXa88emCU7f50MxI8t"
AUTOSHOT_URL = "https://drive.google.com/drive/folders/1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ"


def safe_member(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise RuntimeError(f"Unsafe archive member: {name}")
    return target


def extract_archive(path: Path, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    try:
        is_tar = tarfile.is_tarfile(path)
    except Exception:  # truncated / non-archive files are not archives
        is_tar = False
    if is_tar:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                safe_member(root, member.name)
            archive.extractall(root)
        return "tar"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                safe_member(root, name)
            archive.extractall(root)
        return "zip"
    return None


def split_key(path: Path):
    name = path.name
    # .part01 / .001 / .aa style, and the ClipShots-a / ClipShots-b / ClipShots-c style
    match = re.match(r"^(.*?)(?:\.part[-_]?\d+|\.\d{3}|\.[a-z]{2}|-[a-z])$", name, re.I)
    return match.group(1) if match else None


def join_split_archives(archive_dir: Path):
    groups = {}
    for path in archive_dir.iterdir():
        if path.is_file():
            key = split_key(path)
            if key:
                groups.setdefault(key, []).append(path)
    joined = []
    consumed = set()
    for key, parts in groups.items():
        if len(parts) < 2:
            continue
        parts.sort(key=lambda p: p.name)
        consumed.update(parts)
        output = archive_dir / f"{key}.joined"
        expected = sum(p.stat().st_size for p in parts)
        if output.exists() and output.stat().st_size == expected:
            joined.append(output)  # complete result of an earlier run
            continue
        temp = archive_dir / f"{key}.joined.partial"
        with open(temp, "wb") as dst:  # write to a temp name, rename only when complete
            for part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        if temp.stat().st_size != expected:
            raise RuntimeError(f"joined archive size {temp.stat().st_size} != sum of parts {expected}")
        temp.replace(output)
        joined.append(output)
    return joined, consumed


def download_folder(url: str, archive_dir: Path):
    archive_dir.mkdir(parents=True, exist_ok=True)
    marker = archive_dir / ".download_complete"
    if marker.exists():
        return
    if shutil.which("gdown") is None:
        raise RuntimeError(
            "gdown is required for Google Drive folders. Install it with `pip install gdown` "
            "or place the downloaded archives in " + str(archive_dir)
        )
    # Flags differ between gdown versions: old releases cap folders at 50 files unless
    # --remaining-ok is given; gdown >= 6 dropped it and offers --continue/--retries instead.
    help_text = subprocess.run(["gdown", "--help"], capture_output=True, text=True).stdout
    cmd = ["gdown", "--folder", url, "-O", str(archive_dir)]
    if "--remaining-ok" in help_text:
        cmd.append("--remaining-ok")
    if "--continue" in help_text:
        cmd.append("--continue")
    if "--retries" in help_text:
        cmd += ["--retries", "5"]
    subprocess.run(cmd, check=True)
    marker.write_text(json.dumps({"url": url}, indent=2), encoding="utf-8")


def extract_all(archive_dir: Path, dataset_root: Path):
    joined, consumed = join_split_archives(archive_dir)
    # raw split parts are never extracted on their own, only the joined archive
    candidates = [p for p in archive_dir.iterdir() if p.is_file() and p not in consumed and not p.name.startswith(".")]
    candidates += joined
    manifest = []
    for archive in sorted(set(candidates)):
        kind = extract_archive(archive, dataset_root)
        if kind:
            manifest.append({"archive": str(archive), "type": kind, "sha256": file_sha256(archive)})
    for temp in joined:  # reconstructible from the kept parts; frees disk after extraction
        try:
            temp.unlink()
        except OSError:
            pass
    (dataset_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest:
        raise RuntimeError(f"No tar/zip archives found in {archive_dir}")


def find_clipshots_root(root: Path):
    matches = list(root.rglob("train.json"))
    for match in matches:
        if (match.parent / "test.json").exists():
            return match.parent.parent
    raise FileNotFoundError(f"Could not find ClipShots annotations under {root}")


def find_autoshot_root(root: Path):
    matches = list(root.rglob("kuaishou_v2.txt"))
    if not matches:
        raise FileNotFoundError(f"Could not find kuaishou_v2.txt under {root}")
    return matches[0].parent


KUAISHOU_URL = "https://raw.githubusercontent.com/wentaozhu/AutoShot/main/kuaishou_v2.txt"
SPLIT_DIR = re.compile(r"^(video_download|ads_game_videos)_(\d+)$")


def finalize_autoshot(auto_raw: Path, download_dir: Path = None):
    """AutoShot README steps: fetch the label file (it lives on GitHub, not in the Drive folder)
    and merge video_download_2..5 / ads_game_videos_2 into their base folders."""
    import urllib.request

    label = auto_raw / "kuaishou_v2.txt"
    if not label.exists():
        print(f"downloading {KUAISHOU_URL}")
        with urllib.request.urlopen(KUAISHOU_URL, timeout=60) as response:
            label.write_bytes(response.read())
    # plain (non-archive) files of the Drive folder: test ground truth, baseline outputs
    if download_dir is not None:
        for name in ("gt_scenes_dict_baseline_v2.pickle", "baseline_one_hot_pred_dict_baseline.pickle"):
            src, dst = download_dir / name, auto_raw / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"copied {name} into {auto_raw}")
    moved = 0
    for child in sorted(auto_raw.iterdir()):
        match = SPLIT_DIR.match(child.name)
        base = auto_raw / match.group(1) if match else None
        if not match or not child.is_dir() or not base.exists():
            continue
        for video in child.iterdir():
            target = base / video.name
            if target.exists():
                print(f"  duplicate kept in {base.name}: {video.name}")
                continue
            video.replace(target)
            moved += 1
        try:
            child.rmdir()
        except OSError:
            print(f"  {child.name} not empty after merge; left in place")
    if moved:
        print(f"merged {moved} videos from split folders")


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--clipshots_url", default=CLIPSHOTS_URL)
    parser.add_argument("--autoshot_url", default=AUTOSHOT_URL)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_decode", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--autoshot_exclude_test",
        action="store_true",
        help="exclude the 200 SHOT test videos (present in kuaishou_v2.txt) from AutoShot training data",
    )
    parser.add_argument(
        "--only",
        choices=["clipshots", "autoshot"],
        help="process just one dataset (lets AutoShot be decoded while ClipShots still downloads)",
    )
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    downloads = root / "downloads"
    do_clip = args.only in (None, "clipshots")
    do_auto = args.only in (None, "autoshot")
    clip_raw, auto_raw = root / "ClipShots", root / "AutoShot"
    clip_cache = root / "cache" / "clipshots"
    auto_cache = root / "cache" / "autoshot_train"
    paths_file = root / "paths.json"
    paths = json.loads(paths_file.read_text(encoding="utf-8")) if paths_file.exists() else {}

    if not args.skip_download:
        if do_clip:
            download_folder(args.clipshots_url, downloads / "clipshots")
        if do_auto:
            download_folder(args.autoshot_url, downloads / "autoshot")
    if do_clip and not (clip_raw / ".extracted").exists():
        extract_all(downloads / "clipshots", clip_raw)
        (clip_raw / ".extracted").write_text("ok", encoding="utf-8")
    if do_auto and not (auto_raw / ".extracted").exists():
        extract_all(downloads / "autoshot", auto_raw)
        (auto_raw / ".extracted").write_text("ok", encoding="utf-8")

    if do_clip:
        clip_root = find_clipshots_root(clip_raw)
        print(f"ClipShots root: {clip_root}")
        paths.update(
            {
                "clipshots_root": str(clip_root),
                "clipshots_test_annotations": str(clip_root / "annotations" / "test.json"),
                "clipshots_cache": str(clip_cache),
            }
        )
    if do_auto:
        finalize_autoshot(auto_raw, downloads / "autoshot")
        auto_root = find_autoshot_root(auto_raw)
        print(f"AutoShot root:  {auto_root}")
        gt_pickles = list(auto_raw.rglob("gt_scenes_dict_baseline_v2.pickle"))
        paths.update(
            {
                "autoshot_root": str(auto_root),
                "autoshot_gt_pickle": str(gt_pickles[0]) if gt_pickles else str(auto_root / "gt_scenes_dict_baseline_v2.pickle"),
                "autoshot_train_cache": str(auto_cache),
            }
        )
    paths_file.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    print(f"Discovered paths saved to {paths_file}")

    if not args.skip_decode:
        if do_clip:
            run(
                [
                    sys.executable,
                    str(HERE / "predecode_clipshots.py"),
                    "--split",
                    "train",
                    "--num_workers",
                    str(args.num_workers),
                    "--input_root",
                    paths["clipshots_root"],
                    "--output_root",
                    str(clip_cache),
                ]
            )
            run(
                [
                    sys.executable,
                    str(HERE / "predecode_clipshots.py"),
                    "--split",
                    "test",
                    "--num_workers",
                    str(args.num_workers),
                    "--input_root",
                    paths["clipshots_root"],
                    "--output_root",
                    str(clip_cache),
                ]
            )
        if do_auto:
            cmd = [
                sys.executable,
                str(HERE / "predecode_shot_train.py"),
                "--num_workers",
                str(args.num_workers),
                "--autoshot_root",
                paths["autoshot_root"],
                "--output_root",
                str(auto_cache / "train"),
            ]
            if args.autoshot_exclude_test:
                cmd += ["--exclude_pickle", paths["autoshot_gt_pickle"]]
            run(cmd)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
