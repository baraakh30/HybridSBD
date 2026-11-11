"""One CSV: run, seed, protocol, test set, l_min, tau_c, tau_g, precision, recall, F1 (Protocol A with l_min=1 and Protocol B)."""
import csv, json, re
A = json.load(open("server_results/reeval_lmin1/reeval_lmin1.json"))["runs"]
B = json.load(open("server_results/protocolB/protocolB_results.json"))["runs"]
NAMES = {"shot": "SHOT", "clipshots": "ClipShots", "bbc": "BBC"}
rows = []
for run in A:
    m = re.search(r"_s(\d+)$", run); seed = int(m.group(1)) if m else 42
    for proto, get in (("A (l_min=1)", lambda s: A[run]["sets"][s]["lmin1"]), ("B (validation-selected)", lambda s: B[run]["sets"][s])):
        for s in ("shot", "clipshots", "bbc"):
            v = get(s)
            rows.append([run, seed, proto, NAMES[s], v["l_min"] if v.get("l_min") is not None else 1, v["tau_c"],
                         "" if v.get("tau_g") is None else v["tau_g"], round(v["p"], 4), round(v["r"], 4), round(v["f1"], 4)])
with open("results_all_runs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "seed", "protocol", "test set", "l_min", "tau_c", "tau_g", "precision", "recall", "F1"]); w.writerows(rows)
print(len(rows), "rows")
