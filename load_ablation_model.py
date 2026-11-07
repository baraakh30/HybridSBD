"""Load either a legacy checkpoint or a Protocol-A ablation checkpoint."""

from detection.hybrid_boundary_detector import create_hybrid_detector


def load_detector(checkpoint, device):
    import torch
    from ablation_config import AblationConfig
    from ablation_model import create_ablation_detector

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
