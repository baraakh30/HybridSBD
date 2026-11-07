"""Download, validate, extract, and predecode the Protocol-A datasets.

The Google Drive folders are treated as canonical sources.  The script keeps
archives intact, extracts independent archives into one dataset root, and
joins only files that are clearly split archive members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


CLIPSHOTS_URL = "https://drive.google.com/drive/folders/1AAhTbNroSFsygHBXa88emCU7f50MxI8t"
AUTOSHOT_URL = "https://drive.google.com/drive/folders/1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise RuntimeError(f"Unsafe archive member: {name}")
    return target


def extract_archive(path: Path, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(path):
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
    match = re.match(r"^(.*?)(?:\.part[-_]?\d+|\.\d{3}|\.aa|\.ab|\.ac)$", name, re.I)
    return match.group(1) if match else None


def join_split_archives(archive_dir: Path):
    groups = {}
    for path in archive_dir.iterdir():
        if path.is_file():
            key = split_key(path)
            if key:
                groups.setdefault(key, []).append(path)
    joined = []
    for key, parts in groups.items():
        if len(parts) < 2:
            continue
        parts.sort(key=lambda p: p.name)
        output = archive_dir / f"{key}.joined"
        if output.exists():
            continue
        with open(output, "wb") as dst:
            for part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        joined.append(output)
    return joined


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
    cmd = ["gdown", "--folder", url, "-O", str(archive_dir), "--remaining-ok"]
    subprocess.run(cmd, check=True)
    marker.write_text(json.dumps({"url": url}, indent=2), encoding="utf-8")


def extract_all(archive_dir: Path, dataset_root: Path):
    candidates = [p for p in archive_dir.iterdir() if p.is_file()]
    candidates += join_split_archives(archive_dir)
    manifest = []
    for archive in sorted(set(candidates)):
        kind = extract_archive(archive, dataset_root)
        if kind:
            manifest.append({"archive": str(archive), "type": kind, "sha256": sha256(archive)})
    (dataset_root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
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
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    downloads = root / "downloads"
    clip_download = downloads / "clipshots"
    auto_download = downloads / "autoshot"
    clip_raw = root / "ClipShots"
    auto_raw = root / "AutoShot"
    if not args.skip_download:
        download_folder(args.clipshots_url, clip_download)
        download_folder(args.autoshot_url, auto_download)
    if not clip_raw.exists() or not (clip_raw / ".extracted").exists():
        extract_all(clip_download, clip_raw)
        (clip_raw / ".extracted").write_text("ok", encoding="utf-8")
    if not auto_raw.exists() or not (auto_raw / ".extracted").exists():
        extract_all(auto_download, auto_raw)
        (auto_raw / ".extracted").write_text("ok", encoding="utf-8")
    clip_root = find_clipshots_root(clip_raw)
    auto_root = find_autoshot_root(auto_raw)
    print(f"ClipShots root: {clip_root}")
    print(f"AutoShot root:  {auto_root}")
    if not args.skip_decode:
        clip_cache = root / "cache" / "clipshots"
        auto_cache = root / "cache" / "autoshot_train"
        run([sys.executable, "predecode_dataset.py", "--split", "train",
             "--num_workers", str(args.num_workers), "--input_root", str(clip_root),
             "--output_root", str(clip_cache)])
        run([sys.executable, "predecode_dataset.py", "--split", "test",
             "--num_workers", str(args.num_workers), "--input_root", str(clip_root),
             "--output_root", str(clip_cache)])
        run([sys.executable, "predecode_autoshot_train.py", "--num_workers", str(args.num_workers),
             "--autoshot_root", str(auto_root), "--output_root", str(auto_cache / "train")])
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
