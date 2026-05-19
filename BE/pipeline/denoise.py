"""Denoise pipeline."""

from __future__ import annotations
from pathlib import Path
import sys

import torch
import yaml

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent
if str(BE_ROOT) not in sys.path:
    sys.path.insert(0, str(BE_ROOT))

from pipeline.types import UploadedImage
from pipeline.utils import opencv_to_url
from pipeline.denoiser.inference import build_and_load_model, denoise_single


def _load_config() -> dict:
    """Input: 없음. Output: dict. Purpose: settings.yaml의 denoiser 섹션을 로드해 반환."""

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["denoiser"]


def run_denoise(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: 요청당 모델 1회 로드 → 순차 추론 → data URL 반환 → 모델 명시적 해제."""

    cfg = _load_config()
    model = build_and_load_model(cfg)
    try:
        results = [denoise_single(img, model, cfg) for img in images]
    finally:
        del model
        if cfg["model"]["device"] != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [opencv_to_url(arr) for arr in results]
