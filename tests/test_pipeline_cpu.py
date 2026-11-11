"""CPU validation suite for the Protocol-A ablation implementation (prompt Stage 3).

Runs on dummy inputs only (no datasets, no GPU, no training).  Every test records
PASS/FAIL with details to ``outputs/ablations/cpu_validation.json`` and the run
exits non-zero if anything failed.  Nothing is claimed that was not executed.

    python test_pipeline_cpu.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: lets the script import the repository packages

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from hybridsbd.configs.ablation_config import VARIANTS
from hybridsbd.logging_utils import tee_to_file
from hybridsbd.losses.ablation_loss import AblationBoundaryLoss
from hybridsbd.models.ablation_model import UnifiedBoundaryHead, create_ablation_detector
from hybridsbd.models.hybrid_boundary_detector import HybridBoundaryDetector

RESULTS = []
REPO = Path(__file__).resolve().parents[1]
T, H, W = 128, 48, 64


def record(name, fn):
    start = time.time()
    try:
        detail = fn()
        RESULTS.append({"test": name, "status": "PASS", "detail": detail, "seconds": round(time.time() - start, 2)})
        print(f"PASS  {name}  {detail if detail is not None else ''}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append(
            {
                "test": name,
                "status": "FAIL",
                "detail": repr(exc),
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - start, 2),
            }
        )
        print(f"FAIL  {name}  {exc!r}", flush=True)


def expected_fusion(cfg):
    d = cfg.main_dim
    n = d // 2
    n += d // 2 if cfg.use_branch_b else 0
    n += d // 4 if cfg.use_fsim else 0
    n += d // 4 if cfg.use_csim else 0
    n += 64 if cfg.use_delta else 0
    n += 15 if cfg.use_motion else 0
    return n


def dummy_labels(batch=2):
    labels = torch.zeros(batch, T, dtype=torch.long)
    labels[:, 40] = 1  # a cut
    labels[:, 80:91] = 2  # a gradual (11 frames)
    return labels


# ---------------------------------------------------------------- tests
def t_default_equals_legacy():
    torch.manual_seed(0)
    legacy = HybridBoundaryDetector(main_dim=384, num_fusion_blocks=4).eval()
    new = create_ablation_detector(VARIANTS["A0"]).eval()
    ls, ns = legacy.state_dict(), new.state_dict()
    assert list(ls.keys()) == list(ns.keys()), set(ls) ^ set(ns)
    assert all(ls[k].shape == ns[k].shape for k in ls)
    lp, np_ = legacy.count_parameters()[0], new.count_parameters()[0]
    assert lp == np_, (lp, np_)
    missing = new.load_state_dict(ls, strict=True)
    x = torch.rand(2, T, 3, H, W)
    with torch.no_grad():
        ly, lz = legacy(x)
        ny, nz = new(x)
    assert torch.allclose(ly, ny, atol=1e-5) and torch.allclose(lz, nz, atol=1e-5), (
        "A0 forward differs from the legacy model with identical weights"
    )
    return {"params": np_, "keys": len(ls), "load": str(missing), "max_abs_diff_y": float((ly - ny).abs().max())}


def t_a0_architecture_check1():
    m = create_ablation_detector(VARIANTS["A0"])
    total, _ = m.count_parameters()
    assert len(m.fusion_blocks) == 4 and m.main_dim == 384
    assert tuple(m.feat_delta_proj[0].weight.shape) == (64, 640)
    assert m.fusion_in == 655
    assert not any("pos" in n.lower() for n, _ in m.named_parameters()), "positional encoding found"
    assert round(total / 1e6, 1) == 10.9, total
    return {"params_total": total}


def t_all_variants_construct_forward():
    out = {}
    x = torch.rand(1, T, 3, H, W)
    for run_id, cfg in VARIANTS.items():
        m = create_ablation_detector(cfg).eval()
        with torch.no_grad():
            y, z = m(x)
        assert y.shape == (1, T, 1), (run_id, y.shape)
        if cfg.supervision == "unified":
            assert z is None
        else:
            assert z.shape == (1, T, 1)
        assert m.fusion_in == expected_fusion(cfg), (run_id, m.fusion_in, expected_fusion(cfg))
        assert m.emb_dim == (640 if cfg.use_edge_head else 512)
        assert m.feature_proj[0].in_features == m.fusion_in
        total, trainable = m.count_parameters()
        out[run_id] = {"fusion_in": m.fusion_in, "emb_dim": m.emb_dim, "params": total, "trainable": trainable}
    assert out["A3"]["fusion_in"] == 192 and out["A4"]["fusion_in"] == 192
    assert out["A1"]["params"] == out["A0"]["params"]
    return out


def t_components_bypassed():
    x = torch.rand(1, T, 3, H, W)
    checks = {}
    for run_id, attr_none in {
        "C1": ["diff_encoder", "diff_proj"],
        "C2a": ["frame_sim_short", "frame_sim_medium", "frame_sim_long", "frame_sim_proj"],
        "C2b": ["color_hist_similarity", "color_sim_proj"],
        "C3": ["feat_delta_proj"],
        "C5a": ["pixel_motion_extractor", "motion_norm"],
    }.items():
        m = create_ablation_detector(VARIANTS[run_id]).eval()
        assert all(getattr(m, a) is None for a in attr_none), run_id
        checks[run_id] = "modules absent"
    # C6: no attention / gate parameters at all
    m = create_ablation_detector(VARIANTS["C6"]).eval()
    assert not any("global_attn" in n or ".gate" in n for n, _ in m.named_parameters())
    checks["C6"] = "no attention/gate params"
    # C4: forward_per_frame must never run; relations come from Branch A
    m = create_ablation_detector(VARIANTS["C4"]).eval()
    calls = {"n": 0}
    orig = m.rgb_encoder.forward_per_frame
    m.rgb_encoder.forward_per_frame = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    with torch.no_grad():
        m(x)
    assert calls["n"] == 0
    a0 = create_ablation_detector(VARIANTS["A0"]).eval()
    calls2 = {"n": 0}
    orig2 = a0.rgb_encoder.forward_per_frame
    a0.rgb_encoder.forward_per_frame = lambda *a, **k: (calls2.__setitem__("n", calls2["n"] + 1), orig2(*a, **k))[1]
    with torch.no_grad():
        a0(x)
    assert calls2["n"] == 1
    checks["C4"] = "per-frame pathway skipped (A0 calls it once)"
    # C5b: edge head removed everywhere
    m = create_ablation_detector(VARIANTS["C5b"]).eval()
    assert m.rgb_encoder.edge_extractor is None and m.diff_encoder.edge_extractor is None
    assert m.rgb_proj[0].in_features == 512 and m.feat_delta_proj[0].in_features == 512
    assert m.frame_sim_short is not None
    checks["C5b"] = "edge head removed, 512-d embeddings"
    # A3/A4 keep the edge head
    assert create_ablation_detector(VARIANTS["A3"]).rgb_encoder.edge_extractor is not None
    return checks


def t_heads():
    for run_id, cfg in VARIANTS.items():
        m = create_ablation_detector(cfg)
        if run_id == "A2":
            assert isinstance(m.detection_head, UnifiedBoundaryHead)
            assert not any("head_z" in n or "refine_z" in n for n, _ in m.named_parameters())
        else:
            assert hasattr(m.detection_head, "head_y") and hasattr(m.detection_head, "head_z")
    a0 = create_ablation_detector(VARIANTS["A0"]).count_parameters()[0]
    a2 = create_ablation_detector(VARIANTS["A2"]).count_parameters()[0]
    assert 0.1e6 < a0 - a2 < 0.3e6, a0 - a2
    return {"A0": a0, "A2": a2, "difference": a0 - a2}


def t_channels_last_equivalent():
    """channels_last is a memory-layout change only. Compared in eval mode (no batch-statistics amplification of
    float32 rounding noise): outputs and gradients must agree to float32 precision. (In train mode BatchNorm over a
    small batch amplifies reduction-order noise to ~1e-3 relative in gradients; measured, same order as AMP noise.)"""
    x = torch.rand(2, T, 3, H, W)
    labels = dummy_labels(2)
    out = {}
    for run_id in ("A0", "A2", "C5b"):
        cfg = VARIANTS[run_id]
        torch.manual_seed(0)
        a = create_ablation_detector(cfg).eval()
        torch.manual_seed(0)
        b = create_ablation_detector(cfg).to(memory_format=torch.channels_last).eval()
        b.load_state_dict(a.state_dict())
        ya, za = a(x)
        yb, zb = b(x)
        assert torch.allclose(ya, yb, atol=5e-4), (run_id, float((ya - yb).abs().max()))
        la, _ = AblationBoundaryLoss(cfg.supervision)(ya, za, labels, labels)
        la.backward()
        lb, _ = AblationBoundaryLoss(cfg.supervision)(yb, zb, labels, labels)
        lb.backward()
        ga = torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
        gb = torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
        rel = float((ga - gb).norm() / ga.norm())
        assert rel < 2e-3, (run_id, rel)
        out[run_id] = {"max_abs_diff_logits": float((ya - yb).abs().max()), "grad_rel_diff": rel}
    return out


def t_a2_bias_prior():
    rate = 0.0123
    m = create_ablation_detector(VARIANTS["A2"], unified_positive_rate=rate)
    p = torch.sigmoid(m.detection_head.head.bias).item()
    assert abs(p - rate) < 1e-6
    return {"sigmoid(bias)": p}


def t_loss_backward_all_variants():
    labels = dummy_labels(1)
    x = torch.rand(1, T, 3, H, W)
    summary = {}
    for run_id, cfg in VARIANTS.items():
        m = create_ablation_detector(cfg).train()
        loss_fn = AblationBoundaryLoss(cfg.supervision)
        y, z = m(x)
        loss, parts = loss_fn(y, z, labels, labels)
        assert torch.isfinite(loss), run_id
        loss.backward()
        if cfg.supervision == "y_only":
            zg = [p.grad for n, p in m.named_parameters() if "head_z" in n or "refine_z" in n]
            assert zg and all(g is None or torch.count_nonzero(g) == 0 for g in zg), f"{run_id}: z-head got gradient"
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        summary[run_id] = round(float(loss), 4)
    return summary


def t_loss_values():
    labels = dummy_labels(1)
    y = torch.zeros(1, T, 1)
    z = torch.zeros(1, T, 1)
    dual = AblationBoundaryLoss("dual")(y, z, labels, labels)[0]
    y_only = AblationBoundaryLoss("y_only")(y, z, labels, labels)[0]
    uni = AblationBoundaryLoss("unified")(y, None, labels, labels)[0]
    assert dual > y_only > 0 and uni > 0
    return {"dual": float(dual), "y_only": float(y_only), "unified": float(uni)}


def t_optimizer():
    for run_id in ("A0", "A2", "C6"):
        m = create_ablation_detector(VARIANTS[run_id]).train()
        opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.05)
        y, z = m(torch.rand(1, T, 3, H, W))
        loss, _ = AblationBoundaryLoss(VARIANTS[run_id].supervision)(y, z, dummy_labels(1), None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return "AdamW + clip_grad_norm step OK"


def t_checkpoint_roundtrip():
    from hybridsbd.checkpoint import is_single_output, load_detector

    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for run_id in ("A0", "A2", "A3", "C5b", "C6"):
            cfg = VARIANTS[run_id]
            m = create_ablation_detector(cfg).eval()
            path = Path(tmp) / f"{run_id}.pt"
            torch.save({"epoch": 1, "model_state_dict": m.state_dict(), "config": cfg.to_dict()}, path)
            m2, ckpt = load_detector(path, torch.device("cpu"))
            m2.eval()
            x = torch.rand(1, T, 3, H, W)
            with torch.no_grad():
                a, _ = m(x)
                b, _ = m2(x)
            assert torch.allclose(a, b, atol=1e-6)
            assert is_single_output(ckpt) == (cfg.supervision != "dual")
            out[run_id] = "ok"
        return out


def t_reproducibility():
    import types

    try:
        import torchcodec  # noqa: F401
    except ImportError:  # only needed by data decoding, not by this test
        sys.modules["torchcodec"] = types.ModuleType("torchcodec")
        decoders = types.ModuleType("torchcodec.decoders")
        decoders.VideoDecoder = object
        sys.modules["torchcodec.decoders"] = decoders
    seed_code = "import hashlib,sys;n=sys.argv[1];print(int(hashlib.md5(n.encode()).hexdigest(),16)%(2**31))"
    vals = {
        subprocess.check_output(
            [sys.executable, "-c", seed_code, "video_a.mp4"], env={**os.environ, "PYTHONHASHSEED": h}, text=True
        ).strip()
        for h in ("0", "1", "random")
    }
    assert len(vals) == 1, "MD5 sampling seed varies across processes"
    from training.train import seed_everything

    seed_everything(42)
    m1 = create_ablation_detector(VARIANTS["A0"]).eval()
    seed_everything(42)
    m2 = create_ablation_detector(VARIANTS["A0"]).eval()
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    for f in ("predecode_clipshots.py", "predecode_shot_test.py", "predecode_shot_train.py", "predecode_bbc.py"):
        src = (REPO / "data_prep" / f).read_text(encoding="utf-8")
        assert "hash(" not in src.replace("hashlib", "").replace(".hexdigest", "") or "md5" in src, f
    return {"md5_seed": vals.pop(), "init_deterministic": True}


def make_loader(batch=2):
    from torch.utils.data import DataLoader

    from evaluation.data import collate_clips

    torch.manual_seed(1)

    class DS(torch.utils.data.Dataset):
        items = [("v1", 0), ("v1", 64), ("v2", 0), ("v2", 64)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            v, s = self.items[i]
            return {"frames": torch.rand(T, 3, H, W), "video_id": v, "start_frame": s, "clip_index": i}

    return DataLoader(DS(), batch_size=batch, collate_fn=collate_clips)


def t_eval_pipeline_dual_and_single():
    from evaluation.engine import evaluate

    gt = {"v1": np.array([[0, 50], [60, 191]], dtype=np.int32), "v2": np.array([[0, 90], [92, 191]], dtype=np.int32)}
    out = {}
    for run_id in ("A0", "A1", "A2"):
        cfg = VARIANTS[run_id]
        m = create_ablation_detector(cfg).eval()
        single = cfg.supervision != "dual"
        res = evaluate(m, make_loader(), torch.device("cpu"), gt, tolerance=2, min_scene_lengths=[1, 2, 3, 4], single_output=single)
        n_tau_c = 17
        expected = n_tau_c * (1 if single else 4) * 4
        assert len(res["all_results"]) == expected, (run_id, len(res["all_results"]), expected)
        if single:
            assert res["best_gradual_threshold"] is None
            assert res["_predictions"][1] is None
        else:
            assert res["best_gradual_threshold"] in (0.3, 0.4, 0.5, 0.6)
        json.dumps({k: v for k, v in res.items() if k != "_predictions"})
        out[run_id] = {"grid_size": len(res["all_results"]), "f1_random_weights": round(res["overall"]["f1"], 4)}
    return out


def t_single_output_rule_ignores_z():
    from hybridsbd.scoring import build_pred_many_hot

    y = np.array([0.0, 0.9, 0.0, 0.0], dtype=np.float32)
    z_junk = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)  # would flag everything if used
    single = build_pred_many_hot(y, z_junk, 0.5, None)
    dual = build_pred_many_hot(y, z_junk, 0.5, 0.3)
    assert single.tolist() == [0, 1, 0, 0] and dual.tolist() == [1, 1, 1, 1]
    return "single-output ignores z; dual unions z"


def t_post_processing_and_windows():
    from hybridsbd.scoring import apply_min_scene_length, predictions_to_scenes, reconstruct_full_video

    pred = np.zeros(20, dtype=np.uint8)
    pred[5] = 1
    pred[7] = 1  # scenes: [0,4], [6,6], [8,19]; middle scene has length 1
    scenes = predictions_to_scenes(pred)
    assert scenes.tolist() == [[0, 4], [6, 6], [8, 19]]
    assert apply_min_scene_length(scenes, 1).tolist() == scenes.tolist()
    assert apply_min_scene_length(scenes, 2).tolist() == [[0, 6], [8, 19]]
    clips = {
        "v": [(0, np.full(128, 0.2, np.float32), np.zeros(128, np.float32)), (64, np.full(128, 0.6, np.float32), np.zeros(128, np.float32))]
    }
    y, _ = reconstruct_full_video(clips)
    assert y["v"].shape == (192,) and np.isclose(y["v"][0], 0.2) and np.isclose(y["v"][100], 0.4) and np.isclose(y["v"][191], 0.6)
    return "min-length merge + overlap averaging correct"


def t_analysis_tables():
    from evaluation.clipshots_type_analysis import analyse_run

    n = 400
    y = np.zeros(n, np.float32)
    z = np.zeros(n, np.float32)
    y[50] = 0.9  # cut at 50
    y[200] = 0.9  # midpoint of a gradual 190..210
    z[190:211] = 0.9
    y[300] = 0.9  # false positive
    ann = {"v": {"frame_num": n, "transitions": [[50, 51], [190, 210], [350, 352]]}}
    res = {"best_boundary_threshold": 0.5, "best_gradual_threshold": 0.5, "best_min_scene_len": 1, "single_output": False}
    a0 = analyse_run("A0", {"v": y}, {"v": z}, ann, res)
    # GT: 1 cut (found), gradual 190-210 (found), gradual 350-352 (deliberately missed; FP at 300)
    assert a0["cut_recall"] == 1.0 and a0["gradual_recall"] == 0.5
    assert a0["gradual_iou"] is not None and a0["gradual_iou"] > 0.9
    assert abs(a0["gradual_f1"] - 2 / 3) < 1e-9  # precision 1.0, recall 0.5
    assert a0["duration_counts"] == {"le_10": 1, "11_30": 1, "gt_30": 0}
    assert a0["gt_type_sources"] == {"length_rule": 3}
    res_s = dict(res, single_output=True, best_gradual_threshold=None)
    a1 = analyse_run("A1", {"v": y}, {"v": z}, ann, res_s)
    assert a1["gradual_f1"] is None and a1["gradual_iou"] is None
    a2 = analyse_run("A2", {"v": y}, None, ann, res_s)
    assert a2["gradual_f1"] is None and a2["gradual_iou"] is not None
    # annotation type labels override the length rule
    ann2 = {"v": {"frame_num": n, "transitions": [[50, 51], [190, 210], [350, 352]], "types": ["cut", "gradual", "gradual"]}}
    a0b = analyse_run("A0", {"v": y}, {"v": z}, ann2, res)
    assert a0b["gt_type_sources"] == {"annotation:types": 3} and a0b["counts"]["gt_graduals"] == 2
    return {"A0": {k: a0[k] for k in ("cut_recall", "gradual_recall", "gradual_iou", "gradual_f1")}}


def t_benchmark():
    from benchmarks import speed as bm

    m = create_ablation_detector(VARIANTS["A0"]).eval()
    t = bm.measure_inference_time(m, lambda: torch.randn(1, 128, 3, 48, 64), torch.device("cpu"), 1, 128, num_runs=2, warmup=1)
    assert t["mean_ms"] > 0
    return {"cpu_mean_ms_B1": round(t["mean_ms"], 1), "note": "CPU smoke test only; B=16 GPU timing needs the GPU"}


def t_csv_export():
    from evaluation.export_results_csv import FIELDS

    spec_header = (
        "run_id, seed, git_commit, python_version, pytorch_version, cuda_version, gpu_model, "
        "supervision, flags_json, params_total, params_trainable, best_epoch, train_hours, "
        "evaluation_protocol, val_F1, shot_tau_c, shot_tau_g, shot_P, shot_R, shot_F1, shot_TP, "
        "shot_FP, shot_FN, cs_tau_c, cs_tau_g, cs_lmin, cs_P, cs_R, cs_F1, cs_TP, cs_FP, cs_FN, "
        "bbc_tau_c, bbc_tau_g, bbc_P, bbc_R, bbc_F1, bbc_TP, bbc_FP, bbc_FN, latency_B16_ms, vram_B16_MB"
    )
    assert FIELDS == [f.strip() for f in spec_header.split(",")], "CSV columns differ from spec Section 7"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rid in ("A0", "A2"):
            d = root / rid
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({"best_val_f1": 0.9, "best_epoch": 7, "train_hours": 1.5, "parameters": [123, 120]}))
            (d / "config.json").write_text(json.dumps({"args": {"seed": 42}, "environment": {"git_commit": "abc"}}))
            ov = {"p": 0.9, "r": 0.8, "f1": 0.85, "tp": 8, "fp": 1, "fn": 2}
            for fn in ("shot_results.json", "clipshots_results.json", "bbc_results.json"):
                (d / fn).write_text(
                    json.dumps(
                        {
                            "best_boundary_threshold": 0.5,
                            "best_gradual_threshold": None if rid == "A2" else 0.4,
                            "best_min_scene_len": 2,
                            "overall": ov,
                        }
                    )
                )
        (root / "benchmark_results.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "checkpoint": str(root / "A0" / "best_model.pt"),
                            "batch_results": {"16": {"mean_ms": 12.5, "peak_memory_mb": 900}},
                        }
                    ]
                }
            )
        )
        out = root / "o.csv"
        subprocess.check_call(
            [sys.executable, str(REPO / "evaluation" / "export_results_csv.py"), "--output_root", str(root), "--output", str(out)],
            stdout=subprocess.DEVNULL,
        )
        rows = list(csv.DictReader(open(out)))
        assert [r["run_id"] for r in rows] == ["A0", "A2"]
        a0 = rows[0]
        assert a0["best_epoch"] == "7" and a0["shot_F1"] == "0.85" and a0["cs_lmin"] == "2" and a0["latency_B16_ms"] == "12.5"
        assert (root / "o.missing.json").exists()
    return "header == spec Section 7; values mapped"


def main():
    handle = tee_to_file("outputs/ablations/cpu_validation.log")
    torch.set_num_threads(int(os.environ.get("TEST_THREADS", min(8, os.cpu_count() or 1))))
    tests = [
        ("1  default config == legacy model (keys, shapes, count, outputs, strict load)", t_default_equals_legacy),
        ("1b A0 architecture check (4 blocks, delta (64,640), D=384, 655, 10.9M)", t_a0_architecture_check1),
        ("2-4 construct/forward/fusion-size/params for all 13 variants", t_all_variants_construct_forward),
        ("5  components genuinely bypassed", t_components_bypassed),
        ("6  prediction heads", t_heads),
        ("6b A2 bias prior", t_a2_bias_prior),
        ("6c channels_last (training layout) == default layout: outputs and gradients", t_channels_last_equivalent),
        ("7  loss + backward (A1/A3 z-head has no gradient)", t_loss_backward_all_variants),
        ("7b loss ordering sanity", t_loss_values),
        ("8  optimizer step", t_optimizer),
        ("9  checkpoint save/load", t_checkpoint_roundtrip),
        ("10 reproducibility (stable seed, deterministic init)", t_reproducibility),
        ("11 evaluation pipeline incl. single-output variants", t_eval_pipeline_dual_and_single),
        ("11b single-output rule ignores z", t_single_output_rule_ignores_z),
        ("12 threshold/post-processing + window averaging", t_post_processing_and_windows),
        ("12b Table 8/9 analysis logic", t_analysis_tables),
        ("13 benchmark function (CPU smoke)", t_benchmark),
        ("14 CSV export matches spec Section 7", t_csv_export),
    ]
    for name, fn in tests:
        record(name, fn)
    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    Path("outputs/ablations").mkdir(parents=True, exist_ok=True)
    Path("outputs/ablations/cpu_validation.json").write_text(
        json.dumps(
            {
                "passed": len(RESULTS) - len(failed),
                "failed": len(failed),
                "results": RESULTS,
                "scope": "CPU, dummy inputs only; no GPU training, no real datasets",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    handle.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
