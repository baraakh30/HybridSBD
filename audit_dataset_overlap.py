"""Audit video identifiers across all supplied train/val/test caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ids_from_cache(path):
    path = Path(path)
    mappings = list(path.rglob("index_mapping.json"))
    if not mappings:
        raise FileNotFoundError(f"No index_mapping.json under {path}")
    ids = set()
    for mapping in mappings:
        data = json.loads(mapping.read_text(encoding="utf-8"))
        for item in data.get("mapping", []):
            value = item.get("video_name") or item.get("video_id") or item.get("video")
            if value:
                ids.add(Path(str(value)).stem.lower())
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    parser.add_argument("--output", default="outputs/ablations/data_overlap.json")
    args = parser.parse_args()
    sets = {name: ids_from_cache(root) for name, root in args.dataset}
    overlaps = []
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            common = sorted(sets[left] & sets[right])
            overlaps.append({"left": left, "right": right, "count": len(common), "ids": common})
    result = {"sizes": {name: len(ids) for name, ids in sets.items()}, "overlaps": overlaps,
              "has_overlap": any(item["count"] for item in overlaps)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["has_overlap"]:
        raise SystemExit("Dataset overlap detected; inspect the JSON report")


if __name__ == "__main__":
    main()
