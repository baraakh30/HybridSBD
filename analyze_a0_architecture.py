"""A0 reproduction Check 1 (spec Section 1) on the TRAINED checkpoint: 4 fusion blocks, no positional encoding,
delta projection weight (64, 640), D = 384, exact integer parameter count (rounds to 10.9M). Writes a JSON report.
   python analyze_a0_architecture.py --checkpoint outputs/ablations_v2/A0/best_model.pt --output outputs/ablations_v2/a0_architecture_check.json"""
import argparse, json
import torch
from ablation_config import AblationConfig
from ablation_model import create_ablation_detector

ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--output", required=True); a = ap.parse_args()
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
m = create_ablation_detector(AblationConfig(**ck["config"]))
m.load_state_dict(sd)  # also proves the checkpoint matches the constructed model (no missing / unexpected keys)
params = sum(p.numel() for p in m.parameters())
rep = {
    "fusion_blocks": len({k.split(".")[1] for k in sd if k.startswith("fusion_blocks.")}),
    "positional_encoding_keys": [k for k in sd if "pos" in k.lower()],
    "delta_projection_weight_shape": list(sd["feat_delta_proj.0.weight"].shape),
    "main_dim": ck["config"]["main_dim"], "fusion_in": m.fusion_in, "embedding_dim": m.emb_dim,
    "parameters_total_exact": params, "parameters_round_to_M": round(params / 1e6, 1),
}
rep["checks"] = {"four_fusion_blocks": rep["fusion_blocks"] == 4, "no_positional_encoding": not rep["positional_encoding_keys"],
                 "delta_weight_is_64x640": rep["delta_projection_weight_shape"] == [64, 640], "D_is_384": rep["main_dim"] == 384,
                 "params_round_to_10.9M": rep["parameters_round_to_M"] == 10.9}
rep["all_pass"] = all(rep["checks"].values())
json.dump(rep, open(a.output, "w"), indent=2); print(json.dumps(rep, indent=1))
