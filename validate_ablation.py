"""CPU/GPU preflight validation for all 13 Protocol-A variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ablation_config import VARIANTS, environment_snapshot, write_json
from ablation_loss import AblationBoundaryLoss
from ablation_model import create_ablation_detector, UnifiedBoundaryHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ablations/preflight.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    x = torch.randn(1, args.seq_len, 3, 48, 64, device=device)
    results = {"environment": environment_snapshot(), "device": str(device), "variants": {}}

    for run_id, cfg in VARIANTS.items():
        model = create_ablation_detector(cfg).to(device)
        model.eval()
        with torch.no_grad():
            y, z = model(x)
        total, trainable = model.count_parameters()
        row = {
            "output_shape_y": list(y.shape), "output_shape_z": list(z.shape),
            "fusion_in": model.fusion_in, "embedding_dim": model.emb_dim,
            "parameters_total": total, "parameters_trainable": trainable,
            "head_type": type(model.detection_head).__name__,
        }
        if run_id == "A0":
            row["expected_fusion_in"] = 655
            row["delta_weight_shape"] = list(model.feat_delta_proj[0].weight.shape)
            row["four_fusion_blocks"] = len(model.fusion_blocks) == 4
            row["a0_parameter_count_rounds_to_millions"] = round(total / 1e6, 1)
            if model.fusion_in != 655 or len(model.fusion_blocks) != 4:
                raise AssertionError(f"A0 architecture mismatch: {row}")
        if run_id == "A2" and not isinstance(model.detection_head, UnifiedBoundaryHead):
            raise AssertionError("A2 does not use a single unified head")
        results["variants"][run_id] = row

        if run_id == "A1":
            labels = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
            labels[:, args.seq_len // 2] = 1
            loss_fn = AblationBoundaryLoss("y_only")
            model.train()
            yy, zz = model(x)
            loss, _ = loss_fn(yy, zz, labels, labels)
            loss.backward()
            z_grads = [p.grad for n, p in model.named_parameters() if "head_z" in n]
            if any(g is not None and torch.any(g != 0) for g in z_grads):
                raise AssertionError("A1 z-head received a gradient")
            row["a1_z_head_no_gradient"] = True

    write_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
