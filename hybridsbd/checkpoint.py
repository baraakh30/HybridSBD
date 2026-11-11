"""Load either a legacy checkpoint or a Protocol-A ablation checkpoint."""

from hybridsbd.models.hybrid_boundary_detector import create_hybrid_detector


def is_single_output(ckpt):
    """True when the variant's inference rule uses only p_y > tau_c (A1, A2, A3)."""
    return ckpt.get("config", {}).get("supervision", "dual") != "dual"


def load_detector(checkpoint, device):
    import torch

    from hybridsbd.configs.ablation_config import AblationConfig
    from hybridsbd.models.ablation_model import create_ablation_detector

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    if "run_id" in config:
        model = create_ablation_detector(AblationConfig(**config))
    else:
        model = create_hybrid_detector(
            main_dim=config.get("main_dim", 512),
            num_fusion_blocks=config.get("num_fusion_blocks", 3),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), ckpt
