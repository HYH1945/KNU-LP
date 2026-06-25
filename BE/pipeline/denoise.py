"""Denoise pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from pipeline.types import UploadedImage
from pipeline.utils import image_to_url
from pipeline.utils import opencv_to_url

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent
if str(BE_ROOT) not in sys.path:
    sys.path.insert(0, str(BE_ROOT))

_LAST_WARNING: str | None = None


def _load_config() -> dict:
    """Load denoiser configuration from settings.yaml."""

    import yaml

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["denoiser"]


@lru_cache(maxsize=1)
def _get_model():
    """Load the DnCNN model once per backend process."""

    from pipeline.denoiser.inference import build_and_load_model

    return build_and_load_model(_load_config())


def get_last_warning() -> str | None:
    """Return the latest fallback warning emitted by this stage."""

    return _LAST_WARNING


def run_denoise(images: list[UploadedImage]) -> list[str]:
    """Run DnCNN denoising when weights are present, otherwise pass images through."""

    global _LAST_WARNING
    _LAST_WARNING = None

    try:
        from pipeline.denoiser.inference import denoise_single

        cfg = _load_config()
        model = _get_model()
        results = [denoise_single(img, model, cfg) for img in images]
        _log_pixel_changes(images, results)
        return [opencv_to_url(arr) for arr in results]
    except Exception as exc:
        _LAST_WARNING = f"Denoiser fallback to input images: {exc}"
        print(f"[Denoiser] {_LAST_WARNING}")
        return image_to_url(images)


def _log_pixel_changes(images, results) -> None:
    """Print lightweight denoising diagnostics without affecting inference."""

    try:
        import cv2
        import numpy as np

        for idx, (orig, denoised) in enumerate(zip(images, results)):
            orig_arr = cv2.imdecode(np.frombuffer(orig.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if orig_arr is None or denoised is None or orig_arr.shape != denoised.shape:
                continue
            diff = np.abs(orig_arr.astype(np.float32) - denoised.astype(np.float32))
            print(f"[Denoiser] Sample {idx} -> Mean Pixel Change: {np.mean(diff):.4f}")
    except Exception as exc:
        print(f"[Denoiser] pixel-change log skipped: {exc}")
