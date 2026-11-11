"""Speed, memory and size benchmarks, as sub-commands.

``ours``       benchmark trained HybridSBD checkpoints (parameters, latency, throughput, peak GPU memory).
``baselines``  benchmark AutoShot (TransNetV2Supernet), TransNet V2 and PERSIST from their official repositories (paths
               via AUTOSHOT_DIR, TRANSNETV2_DIR and PERSIST_DIR; PERSIST also needs mmengine, mmcv-lite and mmaction2).
``gpu-busy``   wall-clock time per forward pass versus GPU kernel time: shows whether a model is limited by the host.
``summary``    merge the two result files of a directory into one table and add GFLOPs (CPU, conv/matmul only).

All timings use FP32, synthetic input and a forward pass only: 10 warm-up and 80 timed runs per batch size. Before each
measurement the peak-memory counter is reset and the CUDA cache is emptied.

Examples::

    python benchmarks/speed.py ours --checkpoints outputs/ablations/A0/best_model.pt --batch_sizes 1 16 32 --output out/ours.json
    python benchmarks/speed.py baselines --batch_sizes 1 16 32 --output out/baselines.json
    python benchmarks/speed.py summary --dir out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

ROOT_DIR = Path(__file__).resolve().parents[1]
AS_DIR = Path(os.environ.get("AUTOSHOT_DIR", ROOT_DIR / "data" / "autoshot"))  # AutoShot repository (supernet code, optional weights)
TNV2_DIR = Path(os.environ.get("TRANSNETV2_DIR", AS_DIR / "transnetv2" / "inference-pytorch"))  # soCzech/TransNetV2 inference-pytorch
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", ROOT_DIR / "data" / "PERSIST"))  # linty5/PERSIST
AS_CKPT = AS_DIR / "ckpt_0_200_0.pth"

OURS_FRAME = (48, 64)  # input size of our models (must match training / pre-decoding)
BASELINE_FRAME = (27, 48)  # input size hard-coded in the AutoShot and TransNetV2 forward passes
PERSIST_FRAME = (112, 112)  # PERSIST: frame size of its test pipeline (the model resizes to 27 x 48 inside its backbone)
DEFAULT_CHECKPOINTS = [f"outputs/ablations/{run}/best_model.pt" for run in ("A0", "A4", "C1", "C6")]


# --------------------------------------------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module):
    """``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _reset_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def measure_inference_time(
    model, make_input: Callable[[], torch.Tensor], device, batch_size: int, seq_len: int, num_runs: int = 80, warmup: int = 10
) -> dict:
    """Timing statistics (ms) and throughput of one batch size."""
    model.eval()
    _reset_cuda(device)
    x = make_input()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    t = np.array(times)
    return {
        "mean_ms": float(t.mean() * 1000),
        "std_ms": float(t.std() * 1000),
        "min_ms": float(t.min() * 1000),
        "max_ms": float(t.max() * 1000),
        "clips_per_sec": float(batch_size / t.mean()),
        "frames_per_sec": float(batch_size * seq_len / t.mean()),
    }


def measure_peak_memory_mb(model, make_input: Callable[[], torch.Tensor], device) -> Optional[float]:
    """Peak GPU memory (MB) of one forward pass; ``None`` on CPU."""
    if device.type != "cuda":
        return None
    model.eval()
    _reset_cuda(device)
    x = make_input()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024**2


def benchmark_model(
    label: str, model, make_input: Callable[[int], torch.Tensor], device, batch_sizes: List[int], seq_len: int, **meta
) -> dict:
    """Parameters, per-batch-size latency and memory, and the real-time factor (25 fps content) of one model."""
    total, _ = count_parameters(model)
    size_mb = total * 4 / 1024**2
    print(
        f"\n{'─' * 72}\n  {label}  |  "
        + "  ".join(f"{k}={v}" for k, v in meta.items() if k in ("epoch", "main_dim", "num_fusion_blocks"))
        + f"  params={total / 1e6:.2f}M\n{'─' * 72}"
    )
    result = {
        "label": label,
        **meta,
        "total_params": total,
        "size_fp32_mb": round(size_mb, 1),
        "size_fp16_mb": round(size_mb / 2, 1),
        "batch_results": {},
    }
    for bs in batch_sizes:
        print(f"    batch_size={bs:>2} ... ", end="", flush=True)
        timing = measure_inference_time(model, lambda: make_input(bs), device, bs, seq_len)
        mem = measure_peak_memory_mb(model, lambda: make_input(bs), device)
        print(
            f"{timing['mean_ms']:7.1f} ms  {timing['clips_per_sec']:6.0f} clips/s  {timing['frames_per_sec']:8.0f} frames/s  {(f'{round(mem):4d} MB' if mem is not None else 'CPU')}"
        )
        result["batch_results"][bs] = {**timing, "peak_memory_mb": mem}
    try:  # operations of one clip (B = 1); torch FlopCounterMode counts convolutions and matrix multiplies only
        result["gflops"] = gflops(model, make_input(1))
    except Exception as exc:  # noqa: BLE001
        print("    GFLOPs failed:", repr(exc)[:120])
        result["gflops"] = None
    mid = batch_sizes[len(batch_sizes) // 2]
    per_clip_ms = result["batch_results"][mid]["mean_ms"] / mid
    result["realtime_speedup_x"] = round(seq_len / 25 * 1000 / per_clip_ms, 1)
    return result


def print_comparison_table(results: list, batch_sizes: list, seq_len: int, show_architecture: bool) -> None:
    mid = batch_sizes[len(batch_sizes) // 2]
    width = 106
    print(
        f"\n{'=' * width}\n  COMPARISON TABLE  —  batch_size={mid}, seq_len={seq_len}, device={results[0].get('device', '?')}\n{'=' * width}"
    )
    arch = f"{'Dim':>4} {'Blk':>3}  " if show_architecture else ""
    print(
        f"  {'Model':<24} {arch}{'Params':>7}  {'FP32 MB':>7}  {'FP16 MB':>7}  {'Latency ms':>10}  {'Clips/s':>7}  {'Kframes/s':>9}  {'Mem MB':>7}  {'RT speedup':>10}\n"
        + "─" * width
    )
    for r in results:
        br = r["batch_results"].get(mid, {})
        mem = br.get("peak_memory_mb")
        arch_cells = f"{r['main_dim']:>4} {r['num_fusion_blocks']:>3}  " if show_architecture else ""
        print(
            f"  {r['label']:<24} {arch_cells}{r['total_params'] / 1e6:>6.2f}M  {r['size_fp32_mb']:>7.1f}  {r['size_fp16_mb']:>7.1f}  "
            f"{br.get('mean_ms', 0):>10.1f}  {br.get('clips_per_sec', 0):>7.0f}  {br.get('frames_per_sec', 0) / 1000:>9.1f}  "
            f"{(str(round(mem)) if mem is not None else 'CPU'):>7}  {r.get('realtime_speedup_x', 0):>9.1f}x"
        )
    print("=" * width)


def _device(args) -> torch.device:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 72}\n  Device      : {device}\n  Batch sizes : {args.batch_sizes}\n  Seq len     : {args.seq_len}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU         : {props.name}  ({props.total_memory / 1024**3:.1f} GB)")
    return device


def _save(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), **payload}, f, indent=2)
    print(f"\n  Results saved -> {path}\n")


# --------------------------------------------------------------------------------------------------------------
# ours
# --------------------------------------------------------------------------------------------------------------


def run_ours(args) -> None:
    from hybridsbd.checkpoint import load_detector
    from hybridsbd.configs.ablation_config import environment_snapshot
    from hybridsbd.logging_utils import tee_to_file

    tee_to_file(Path(args.output).with_suffix(".log"))
    device = _device(args)
    h, w = OURS_FRAME
    results = []
    for path in map(Path, args.checkpoints):
        if not path.exists():
            print(f"  SKIP {path} — file not found")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model, _ = load_detector(path, device)
        model.eval()
        r = benchmark_model(
            path.parent.name,
            model,
            lambda bs: torch.randn(bs, args.seq_len, 3, h, w, device=device),
            device,
            args.batch_sizes,
            args.seq_len,
            checkpoint=str(path),
            epoch=ckpt.get("epoch", "?"),
            main_dim=cfg.get("main_dim", 512),
            num_fusion_blocks=cfg.get("num_fusion_blocks", 3),
        )
        r["device"] = str(device)
        results.append(r)
        del model
        _reset_cuda(device)
    if not results:
        print("\nNo checkpoints found — nothing to report.")
        return
    print_comparison_table(results, args.batch_sizes, args.seq_len, show_architecture=True)
    _save(
        args.output,
        {
            "environment": environment_snapshot(),
            "protocol": "FP32, 10 warm-up + 80 timed, synthetic input, forward pass only",
            "device": str(device),
            "batch_sizes": args.batch_sizes,
            "seq_len": args.seq_len,
            "frame_size": [h, w],
            "models": results,
        },
    )


# --------------------------------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------------------------------


def load_autoshot(device: torch.device):
    """TransNetV2Supernet (AutoShot): pretrained weights if ``ckpt_0_200_0.pth`` is present, otherwise random weights."""
    sys.path.insert(0, str(AS_DIR))
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

    model = TransNetV2Supernet().to(device)
    if AS_CKPT.exists():
        state = torch.load(AS_CKPT, map_location=device, weights_only=False)["net"]
        own = model.state_dict()
        matched = {k: v for k, v in state.items() if k in own}
        own.update(matched)
        model.load_state_dict(own)
        print(f"    Loaded {len(matched)}/{len(own)} keys from {AS_CKPT.name}")
    else:
        print(f"    WARNING: {AS_CKPT} not found — benchmarking with random weights")
    return model.eval()


def load_transnetv2(device: torch.device):
    """Official TransNetV2 (PyTorch, ``inference-pytorch``) with random weights: latency and memory do not depend on them."""
    sys.path.insert(0, str(TNV2_DIR))
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2().to(device).eval()
    print("    TransNetV2 instantiated with random weights")
    return model


class PersistForward(torch.nn.Module):
    """The official PERSIST inference path without post-processing, as one forward pass.

    Input: uint8 ``[B, 3, T, H, W]``. The official data preprocessor (cast to float and normalise), the backbone (which
    resizes the frames to 27 x 48 itself) and the head are run exactly as ``Recognizer3D.predict`` runs them; the returned
    tensor is the head's boundary probability. Copying results to the CPU and the debug fields of ``predict`` are left out.
    """

    def __init__(self, recognizer: torch.nn.Module):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = self.recognizer.data_preprocessor.preprocess(list(x.unsqueeze(1)), None, False)  # [B, 1, 3, T, H, W]
        feats, kwargs = self.recognizer.extract_feat(batch, test_mode=True)
        return self.recognizer.cls_head.forward(feats, **kwargs)["probs"]


def load_persist(device: torch.device):
    """Official PERSIST model (``linty5/PERSIST``, MMAction2 fork) built from its evaluation config, random weights."""
    import warnings

    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules
    from mmengine.config import Config

    warnings.filterwarnings("ignore")
    register_all_modules()
    cfg = Config.fromfile(str(PERSIST_DIR / "configs" / "persist" / "persist_eval.py"))
    model = PersistForward(MODELS.build(cfg.model)).to(device).eval()
    print("    PERSIST built from configs/persist/persist_eval.py with random weights")
    return model


def make_baseline_input(model_name: str, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Synthetic clip in the model's native format.

    AutoShot: float ``[B, 3, T, 27, 48]`` in 0-255 (its first layer divides by 255). TransNetV2: uint8 ``[B, T, 27, 48, 3]``
    (its forward permutes and divides by 255). PERSIST: uint8 ``[B, 3, T, 112, 112]`` (its test pipeline).
    """
    h, w = BASELINE_FRAME
    if model_name == "autoshot":
        return torch.rand(batch_size, 3, seq_len, h, w, device=device) * 255
    if model_name == "persist":
        return torch.randint(0, 256, (batch_size, 3, seq_len, *PERSIST_FRAME), dtype=torch.uint8, device=device)
    return torch.randint(0, 256, (batch_size, seq_len, h, w, 3), dtype=torch.uint8, device=device)


INPUT_DESCRIPTION = {
    "autoshot": "{T} x 27 x 48, float32 RGB",
    "transnetv2": "{T} x 27 x 48, uint8 RGB",
    "persist": "{T} x 112 x 112, uint8 RGB (resized to 27 x 48 inside the model)",
}


def _git_commit(path: Path) -> Optional[str]:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def run_baselines(args) -> None:
    device = _device(args)
    results = []
    jobs = []  # (label, model name, loader, clip length)
    for name in args.models:
        if name == "autoshot":
            jobs.append(("AutoShot (TransNetV2Supernet)", name, load_autoshot, args.seq_len))
        elif name == "transnetv2":
            jobs.append(("TransNet V2 (PyTorch)", name, load_transnetv2, args.seq_len))
        else:
            jobs += [(f"PERSIST (T={t})", name, load_persist, t) for t in args.persist_seq_lens]
    for label, name, loader, seq_len in jobs:
        print(f"\n  Loading: {label}")
        try:
            model = loader(device)
            r = benchmark_model(
                label,
                model,
                lambda bs, n=name, t=seq_len: make_baseline_input(n, bs, t, device),
                device,
                args.batch_sizes,
                seq_len,
                input=INPUT_DESCRIPTION[name].format(T=seq_len),
            )
            r["device"], r["seq_len"] = str(device), seq_len
            results.append(r)
        except Exception as exc:  # a model that fails is reported, and must not stop the others
            import traceback

            print(f"  ERROR benchmarking {label}: {exc}")
            traceback.print_exc()
            results.append({"label": label, "input": INPUT_DESCRIPTION[name].format(T=seq_len), "seq_len": seq_len, "error": repr(exc)})
        finally:
            model = None
            _reset_cuda(device)
    ok = [r for r in results if "error" not in r]
    if ok:
        print_comparison_table(ok, args.batch_sizes, args.seq_len, show_architecture=False)
    _save(
        args.output,
        {
            "device": str(device),
            "batch_sizes": args.batch_sizes,
            "seq_len": args.seq_len,
            "repositories": {
                "AutoShot": _git_commit(AS_DIR),
                "TransNetV2": _git_commit(TNV2_DIR.parent if TNV2_DIR.name == "inference-pytorch" else TNV2_DIR),
                "PERSIST": _git_commit(PERSIST_DIR),
            },
            "note": "Random weights unless stated in the log. AutoShot: float32 [B,3,T,H,W] 0-255. TransNetV2: uint8 [B,T,H,W,3] (forward permutes and divides by 255). PERSIST: uint8 [B,3,T,112,112]; the official preprocessor, backbone (resizes to 27x48) and head are run, the post-processing of predict() is not.",
            "models": results,
        },
    )


# --------------------------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------------------------


def _rows_of(path: Path) -> list:
    d = json.load(open(path))
    return (
        d.get("models")
        or d.get("results")
        or next(v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "label" in v[0])
    )


def gflops(model, x: torch.Tensor) -> float:
    """GFLOPs of one forward pass (``torch`` FlopCounterMode: convolutions and matrix multiplies only)."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


def run_summary(args) -> None:
    from hybridsbd.checkpoint import load_detector

    d = Path(args.dir)
    ours = _rows_of(d / "ours.json")
    baselines = [r for r in (_rows_of(d / args.baselines) if (d / args.baselines).exists() else []) if "error" not in r]
    failed = [r for r in (_rows_of(d / args.baselines) if (d / args.baselines).exists() else []) if "error" in r]
    cpu = torch.device("cpu")
    for r in ours:  # files written before GFLOPs were stored with the timings
        if r.get("gflops") is None:
            try:
                model, _ = load_detector(r["checkpoint"], cpu)
                r["gflops"] = gflops(model, torch.rand(1, 128, 3, *OURS_FRAME))
            except Exception as exc:  # noqa: BLE001
                print("flops failed", r["label"], repr(exc)[:120])

    sizes = sorted({int(k) for r in ours + baselines for k in r["batch_results"]})
    header = (
        ["Model", "Input", "Params (M)", "FP32 MB", "GFLOPs/clip"]
        + [f"ms @B={b}" for b in sizes]
        + [f"frames/s @B={b}" for b in sizes]
        + [f"peak VRAM MB @B={b}" for b in sizes]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out = []
    for r in ours + baselines:
        br = {int(k): v for k, v in r["batch_results"].items()}
        g = r.get("gflops")
        cells = [
            r["label"],
            r.get("input", f"128 x {OURS_FRAME[0]} x {OURS_FRAME[1]}, float32 RGB"),
            f"{r['total_params'] / 1e6:.2f}",
            f"{r['size_fp32_mb']}",
            f"{g:.2f}" if g else "n/a",
        ]
        cells += [f"{br[b]['mean_ms']:.1f} ± {br[b]['std_ms']:.1f}" if b in br else "-" for b in sizes]
        cells += [f"{br[b]['frames_per_sec']:.0f}" if b in br else "-" for b in sizes]
        cells += [f"{br[b]['peak_memory_mb']:.0f}" if b in br and br[b]["peak_memory_mb"] is not None else "-" for b in sizes]
        lines.append("| " + " | ".join(cells) + " |")
        out.append({"label": r["label"], "gflops": g, **{k: r[k] for k in ("total_params", "size_fp32_mb")}, "batch_results": br})
    # speedup of HybridSBD over each other model at B = 16: latency of the other model / latency of HybridSBD
    ref = [r for r in ours if r["label"] in args.reference]
    if ref and baselines:
        lines += ["", "| Model | " + " | ".join(f"× vs {r['label']} (B=16)" for r in ref) + " |", "|---|" + "---|" * len(ref)]
        for r in baselines:
            ms = r["batch_results"].get("16", r["batch_results"].get(16))["mean_ms"]
            lines.append(
                f"| {r['label']} | "
                + " | ".join(f"{ms / (rr['batch_results'].get('16') or rr['batch_results'][16])['mean_ms']:.1f}" for rr in ref)
                + " |"
            )
    for r in failed:
        lines.append(f"\n{r['label']}: FAILED — {r['error']}")
    (d / "speed_table.md").write_text(
        "\n".join(lines)
        + "\n\nFP32, synthetic input, forward only, 10 warm-up + 80 timed runs, idle GPU. GFLOPs per clip (B = 1) count convolutions and matrix multiplies only (torch FlopCounterMode).\n"
    )
    (d / "speed_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(lines))


def run_gpu_busy(args) -> None:
    """Wall-clock time per forward pass versus the time the GPU actually spends in kernels (B = 16 by default).

    A model whose GPU-busy fraction is well below 100 % is limited by the host (CPU, Python, launches, synchronisation)
    and its latency depends on the machine's CPU, not only on the GPU.
    """
    from torch.profiler import ProfilerActivity, profile

    from hybridsbd.checkpoint import load_detector

    device = _device(args)
    cases = []
    for path in map(Path, args.checkpoints):
        model, _ = load_detector(path, device)
        cases.append((path.parent.name, model.eval(), lambda bs: torch.randn(bs, args.seq_len, 3, *OURS_FRAME, device=device)))
    for name in args.models:
        loader = {"autoshot": load_autoshot, "transnetv2": load_transnetv2, "persist": load_persist}[name]
        cases.append((name, loader(device), lambda bs, n=name: make_baseline_input(n, bs, args.seq_len, device)))
    bs = args.batch_size
    print(f"\n{'model':16s} {'wall ms':>9s} {'GPU kernel ms':>14s} {'GPU busy':>9s}")
    rows = []
    for label, model, make in cases:
        x = make(bs)
        with torch.no_grad():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            wall = 1000 * sum(times) / len(times)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(x)
                torch.cuda.synchronize()
            gpu = sum(e.self_device_time_total for e in prof.key_averages()) / 1000
        print(f"{label:16s} {wall:9.1f} {gpu:14.1f} {100 * gpu / wall:8.0f}%", flush=True)
        rows.append({"label": label, "batch_size": bs, "wall_ms": wall, "gpu_kernel_ms": gpu, "gpu_busy": gpu / wall})
        _reset_cuda(device)
    if args.output:
        _save(args.output, {"models": rows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, output):
        p.add_argument("--device", default="cuda")
        p.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
        p.add_argument("--seq_len", type=int, default=128)
        p.add_argument("--output", default=output)

    p = sub.add_parser("ours", help="benchmark trained checkpoints")
    p.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    common(p, "outputs/benchmark_results.json")
    p.set_defaults(func=run_ours)
    p = sub.add_parser("baselines", help="benchmark AutoShot, TransNet V2 and PERSIST")
    p.add_argument("--models", nargs="+", default=["autoshot", "transnetv2", "persist"], choices=["autoshot", "transnetv2", "persist"])
    p.add_argument("--persist_seq_lens", type=int, nargs="+", default=[128, 64], help="clip lengths for PERSIST (its native window is 64)")
    common(p, "outputs/benchmark_baselines.json")
    p.set_defaults(func=run_baselines)
    p = sub.add_parser("gpu-busy", help="wall-clock time versus GPU kernel time (is a model host-bound?)")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--models", nargs="*", default=[], choices=["autoshot", "transnetv2", "persist"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[16], help=argparse.SUPPRESS)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--output", default=None)
    p.set_defaults(func=run_gpu_busy)
    p = sub.add_parser("summary", help="merge ours.json and the baselines file into one table")
    p.add_argument("--dir", required=True)
    p.add_argument("--baselines", default="baselines.json", help="file name of the baselines result inside --dir")
    p.add_argument("--reference", nargs="+", default=["A0", "A4"], help="HybridSBD runs the speedups are computed against")
    p.set_defaults(func=run_summary)
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    ns.func(ns)
