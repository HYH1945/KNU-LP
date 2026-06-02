"""Denoise pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

from pipeline.types import UploadedImage
from pipeline.utils import image_to_url
from pipeline.utils import opencv_to_url

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent
if str(BE_ROOT) not in sys.path:
    sys.path.insert(0, str(BE_ROOT))


def _load_config() -> dict:
    """Load denoiser configuration from settings.yaml."""

    import yaml

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["denoiser"]


def run_denoise(images: list[UploadedImage]) -> list[str]:
    """Run DnCNN denoising when weights are present, otherwise pass images through."""

    try:
        import torch

        from pipeline.denoiser.inference import build_and_load_model
        from pipeline.denoiser.inference import denoise_single

        cfg = _load_config()
        model = build_and_load_model(cfg)
        try:
            results = [denoise_single(img, model, cfg) for img in images]
        finally:
            del model
            if cfg["model"]["device"] != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return [opencv_to_url(arr) for arr in results]
    except Exception as exc:
        print(f"[Denoiser] fallback to input images: {exc}")
        return image_to_url(images)
