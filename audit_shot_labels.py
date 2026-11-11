"""Audit of the SHOT (AutoShot/Kuaishou) label file: which videos are labelled, label-file irregularities, and how the file's labels for
the 200 test videos relate to the final test annotation. Writes a JSON report.
   python audit_shot_labels.py --data_root /workspace/data --output outputs/ablations_v2/shot_label_audit.json"""
import argparse, collections, json, pickle, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
paths = json.load(open(Path(a.data_root) / "paths.json")); root = Path(paths["autoshot_root"])
gt = pickle.load(open(paths["autoshot_gt_pickle"], "rb"))
test = {str(k): [(int(x[1]), int(y[0])) for x, y in zip(v[:-1], v[1:])] for k, v in gt.items()}
lines = open(root / "kuaishou_v2.txt", encoding="utf-8", errors="replace").read().split("\n")

disk_by_dir = {d: {p.stem for p in (root / d).glob("*.mp4") if not p.name.startswith("._")} for d in ("original_videos", "video_download", "ads_game_videos")}
disk = set().union(*disk_by_dir.values())


def old_parse(line):  # strict "a,b" / "a b" pair only (what a strict parser keeps)
    c = line.strip().strip("[]"); parts = c.split(",") if "," in c else c.split()
    if len(parts) == 2 and parts[0] and parts[1]:
        try:
            x, y = int(parts[0].strip()), int(parts[1].strip()); return [(x, y)] if y >= x else []
        except ValueError:
            return []
    return []


entries, cur = [], None
for l in lines:
    t = l.strip()
    if ".mp4" in t:
        cur = {"vid": re.match(r"\d+", t).group(0), "strict": 0, "robust": [], "odd": 0, "header_shape": re.sub(r"\d", "9", t)[:30]}; entries.append(cur)
    elif t and cur is not None:
        nums = [int(x) for x in re.findall(r"\d+", t)]
        cur["strict"] += len(old_parse(t)); cur["robust"] += [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]; cur["odd"] += len(nums) % 2
by = collections.defaultdict(list)
for e in entries: by[e["vid"]].append(e)
labelled = set(by)
last = {v: es[-1]["robust"] for v, es in by.items()}
nf = ng = tol2 = 0
for k, g in test.items():
    gs = set(g); f = last[k]; nf += len(f); ng += len(g)
    tol2 += sum(any(abs(t[0] - x) <= 2 and abs(t[1] - y) <= 2 for x, y in gs) for t in f)
dup_nontest = [(v, es) for v, es in by.items() if len(es) > 1 and v not in test]
rep = {
    "files_in_three_video_folders": sum(len(v) for v in disk_by_dir.values()), "per_folder": {k: len(v) for k, v in disk_by_dir.items()},
    "distinct_videos_on_disk": len(disk), "labelled_videos": len(labelled), "labelled_and_on_disk": len(labelled & disk),
    "labelled_not_on_disk": len(labelled - disk), "on_disk_without_labels": len(disk - labelled),
    "test_videos": len(test), "test_videos_inside_label_file": len(set(test) & labelled),
    "train_videos_after_excluding_test": len((labelled & disk) - set(test)),
    "header_entries": len(entries), "videos_with_duplicate_entries": sum(1 for es in by.values() if len(es) > 1),
    "non_test_duplicate_videos": len(dup_nontest), "non_test_duplicates_identical": sum(1 for _, es in dup_nontest if es[0]["robust"] == es[-1]["robust"]),
    "header_shapes": dict(collections.Counter(e["header_shape"] for e in entries)),
    "transitions_recovered_by_tolerant_parse": sum(len(e["robust"]) for e in entries),
    "transitions_kept_by_strict_parse": sum(e["strict"] for e in entries),
    "lines_with_odd_number_of_integers": sum(e["odd"] for e in entries),
    "test_labels_in_file": nf, "test_labels_final_annotation": ng, "test_file_labels_within_2_frames_of_final": tol2,
}
rep["transitions_dropped_by_strict_parse"] = rep["transitions_recovered_by_tolerant_parse"] - rep["transitions_kept_by_strict_parse"]
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(a.output, "w"), indent=1); print(json.dumps(rep, indent=1))
