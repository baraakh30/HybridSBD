"""Merge the speed benchmarks (ours.json + autoshot.json) into one table; adds GFLOPs (CPU, torch FlopCounterMode: conv/matmul only).
   python speed_summary.py --dir /workspace/outputs/speed_benchmark"""
import argparse, json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rows_of(path):
    d = json.load(open(path))
    lst = d.get("models") or d.get("results") or next(v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "label" in v[0])
    return d, lst


def gflops(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    D = Path(a.dir)
    _, ours = rows_of(D / "ours.json")
    auto = rows_of(D / "autoshot.json")[1] if (D / "autoshot.json").exists() else []
    flops = {}
    from load_ablation_model import load_detector
    for r in ours:
        try:
            m, _ = load_detector(r["checkpoint"], torch.device("cpu"))
            flops[r["label"]] = gflops(m, torch.rand(1, 128, 3, 48, 64))
        except Exception as e:  # noqa: BLE001
            print("flops failed", r["label"], repr(e)[:120])
    for r in auto:
        if r["label"].startswith("AutoShot"):
            try:
                import benchmark_baselines as b
                m = b.load_autoshot(torch.device("cpu"))
                flops[r["label"]] = gflops(m, b.make_dummy_input("autoshot", 1, 128, torch.device("cpu")))
            except Exception as e:  # noqa: BLE001
                print("flops failed autoshot", repr(e)[:120])
    bss = sorted({int(k) for r in ours + auto for k in r["batch_results"]})
    hdr = ["Model", "Params (M)", "FP32 MB", "GFLOPs/128-frame clip"] + [f"ms @B={b}" for b in bss] + [f"frames/s @B={b}" for b in bss] + [f"peak VRAM MB @B={b}" for b in bss]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out = []
    for r in ours + [x for x in auto if x["label"].startswith("AutoShot")]:
        br = {int(k): v for k, v in r["batch_results"].items()}
        g = flops.get(r["label"])
        cells = [r["label"], f"{r['total_params']/1e6:.2f}", f"{r['size_fp32_mb']}", f"{g:.2f}" if g else "n/a"]
        cells += [f"{br[b]['mean_ms']:.1f}" if b in br else "-" for b in bss]
        cells += [f"{br[b]['frames_per_sec']:.0f}" if b in br else "-" for b in bss]
        cells += [f"{br[b]['peak_memory_mb']:.0f}" if b in br and br[b]['peak_memory_mb'] is not None else "-" for b in bss]
        lines.append("| " + " | ".join(cells) + " |")
        out.append({"label": r["label"], "gflops": g, **{k: r[k] for k in ("total_params", "size_fp32_mb")}, "batch_results": br})
    (D / "speed_table.md").write_text("\n".join(lines) + "\n\nFP32, synthetic input, forward only, 10 warm-up + 80 timed runs, idle GPU. Ours: 48x64 frames; AutoShot: its native 27x48. GFLOPs count conv/matmul only.\n")
    (D / "speed_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
