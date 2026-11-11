"""Download the BBC Planet Earth shot-detection dataset from Zenodo and arrange it the way
predecode_bbc.py / test_bbc.py expect:

    <output>/bbc_01.mp4 ... bbc_11.mp4
    <output>/annotations/shots/<episode>.txt      (11 files)

Source: https://zenodo.org/records/14865504 (CC-BY-4.0). Every file is md5-verified against
the checksum Zenodo publishes; completed files are skipped, so the script is resumable.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RECORD_ID = "14865504"
API = "https://zenodo.org/api/records/{}"
EXPECTED_VIDEOS = [f"bbc_{i:02d}.mp4" for i in range(1, 12)]


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} after {exc!r}")
            time.sleep(5 * (attempt + 1))


def download(url, dest, tries=4):
    part = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} for {dest.name}: {exc!r}")
            time.sleep(10 * (attempt + 1))


def target_path(root, key):
    if key.endswith(".mp4"):
        return root / key
    if key.lower() == "readme.txt":
        return root / "readme.txt"
    if key.endswith(".txt"):
        return root / "annotations" / "shots" / key
    return root / "other" / key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="BBC_ROOT to create")
    parser.add_argument("--record", default=RECORD_ID)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)

    record = fetch_json(API.format(args.record))
    files = record.get("files", [])
    names = {f["key"] for f in files}
    missing = [v for v in EXPECTED_VIDEOS if v not in names]
    annotations = sorted(n for n in names if n.endswith(".txt") and n.lower() != "readme.txt")
    print(f"Record {args.record}: {record['metadata'].get('title')}  license={record['metadata'].get('license', {}).get('id')}")
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB; "
          f"{len(EXPECTED_VIDEOS) - len(missing)}/11 videos, {len(annotations)} annotation files")
    if missing or len(annotations) != 11:
        print(f"ERROR: unexpected record contents (missing videos: {missing}, annotations: {len(annotations)})", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        for f in sorted(files, key=lambda x: x["key"]):
            print(f"  {f['key']:<28} -> {target_path(root, f['key'])}  ({f['size'] / 1e6:.1f} MB)")
        return

    manifest = {"record": args.record, "files": {}}
    for f in sorted(files, key=lambda x: x["key"]):
        dest = target_path(root, f["key"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = f["checksum"].split(":", 1)[1]
        if dest.exists() and md5sum(dest) == expected:
            print(f"ok (already present) {f['key']}")
        else:
            url = f["links"]["self"] if "links" in f else \
                f"https://zenodo.org/api/records/{args.record}/files/{urllib.parse.quote(f['key'])}/content"
            print(f"downloading {f['key']} ({f['size'] / 1e6:.1f} MB)", flush=True)
            download(url, dest)
            actual = md5sum(dest)
            if actual != expected:
                print(f"ERROR: md5 mismatch for {f['key']}: {actual} != {expected}", file=sys.stderr)
                sys.exit(2)
        manifest["files"][f["key"]] = {"md5": expected, "size": f["size"], "path": str(dest)}
    (root / "bbc_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"BBC dataset ready at {root} (11 videos, {len(annotations)} annotation files, md5 verified)")


if __name__ == "__main__":
    main()
