"""Authoritative configurations for the HybridSBD Protocol-A ablations."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class AblationConfig:
    run_id: str
    supervision: str = "dual"  # dual | y_only | unified
    use_branch_b: bool = True
    use_branch_c: bool = True
    rel_source: str = "pf"  # pf | rgb
    use_fsim: bool = True
    use_csim: bool = True
    use_delta: bool = True
    use_motion: bool = True
    use_edge_head: bool = True
    use_attention: bool = True
    main_dim: int = 384
    num_fusion_blocks: int = 4

    def __post_init__(self):
        if self.supervision not in {"dual", "y_only", "unified"}:
            raise ValueError(f"Unsupported supervision mode: {self.supervision}")
        if self.rel_source not in {"pf", "rgb"}:
            raise ValueError(f"Unsupported relation source: {self.rel_source}")
        if not self.use_branch_c and self.rel_source != "rgb":
            raise ValueError("Branch C disabled requires rel_source='rgb'")
        if self.supervision == "unified" and self.run_id != "A2":
            raise ValueError("Only A2 uses unified supervision")

    def to_dict(self) -> Dict:
        return asdict(self)


VARIANTS = {
    "A0": AblationConfig("A0"),
    "A1": AblationConfig("A1", supervision="y_only"),
    "A2": AblationConfig("A2", supervision="unified"),
    "A3": AblationConfig(
        "A3", supervision="y_only", use_branch_b=False, use_branch_c=False,
        rel_source="rgb", use_fsim=False, use_csim=False, use_delta=False,
        use_motion=False,
    ),
    "A4": AblationConfig(
        "A4", use_branch_b=False, use_branch_c=False, rel_source="rgb",
        use_fsim=False, use_csim=False, use_delta=False, use_motion=False,
    ),
    "C1": AblationConfig("C1", use_branch_b=False),
    "C2a": AblationConfig("C2a", use_fsim=False),
    "C2b": AblationConfig("C2b", use_csim=False),
    "C3": AblationConfig("C3", use_delta=False),
    "C4": AblationConfig("C4", use_branch_c=False, rel_source="rgb"),
    "C5a": AblationConfig("C5a", use_motion=False),
    "C5b": AblationConfig("C5b", use_edge_head=False),
    "C6": AblationConfig("C6", use_attention=False),
}

# Build-up (add-one-in) ablation, NOT part of the 13-run Protocol-A table. Base = A4 (backbone only: RGB encoder + fusion
# blocks + dual head; no diff branch, no fsim/csim/delta/motion, per-frame pathway unused). Each U_* adds ONE component to the
# backbone; S_* are cumulative stacks (S3 + diff branch == A0). B_* remove a part of the backbone itself.
_BASE = dict(supervision="dual", use_branch_b=False, use_branch_c=False, rel_source="rgb",
             use_fsim=False, use_csim=False, use_delta=False, use_motion=False)


def _b(run_id, **kw):
    return AblationConfig(run_id, **{**_BASE, **kw})


BUILDUP = {
    "U_fsim": _b("U_fsim", use_fsim=True),
    "U_csim": _b("U_csim", use_csim=True),
    "U_delta": _b("U_delta", use_delta=True),
    "U_motion": _b("U_motion", use_motion=True),
    "U_diff": _b("U_diff", use_branch_b=True),
    "S1_rel": _b("S1_rel", use_fsim=True, use_csim=True, use_delta=True),
    "S2_rel_pf": _b("S2_rel_pf", use_fsim=True, use_csim=True, use_delta=True, use_branch_c=True, rel_source="pf"),
    "S3_rel_pf_motion": _b("S3_rel_pf_motion", use_fsim=True, use_csim=True, use_delta=True, use_branch_c=True,
                           rel_source="pf", use_motion=True),
    # Minimal ladder BELOW the backbone: M0 = RGB encoder -> MLP -> dual head only (no edge head, no fusion blocks);
    # M1 = M0 + edge head; B_noatt = M1 + 4 conv-only fusion blocks; A4 = B_noatt + self-attention/gate.
    "M0_min": _b("M0_min", use_edge_head=False, num_fusion_blocks=0, use_attention=False),
    "M1_edge": _b("M1_edge", use_edge_head=True, num_fusion_blocks=0, use_attention=False),
    "B_noatt": _b("B_noatt", use_attention=False),
    "B_noedge": _b("B_noedge", use_edge_head=False),
}


def get_variant(run_id: str) -> AblationConfig:
    if run_id in BUILDUP:
        return BUILDUP[run_id]
    try:
        return VARIANTS[run_id]
    except KeyError as exc:
        raise ValueError(f"Unknown run_id {run_id!r}; choose from {sorted(VARIANTS)}") from exc


def environment_snapshot() -> Dict:
    """Return reproducibility metadata without requiring optional packages."""
    result = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        result.update({
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        result["torch_error"] = repr(exc)
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception as exc:
        result["git_error"] = repr(exc)
    return result


def write_json(path, value) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
