"""GP-LPR OCR model adapter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib
import re
import sys

import cv2
import numpy as np

from pipeline.types import UploadedImage
from pipeline.utils import decode_image

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent.parent
VENDOR_ROOT = BE_ROOT / "vendors" / "gplpr"


class GPLPRLabelConverter:
    """Decode GP-LPR class indices into license plate text."""

    def __init__(self, alphabet: str):
        self.alphabet = "-" + str(alphabet)

    def decode_list(self, indices) -> list[str]:
        texts: list[str] = []
        for row in indices:
            chars: list[str] = []
            for value in row:
                index = int(value.item())
                if index == 0:
                    continue
                if 0 <= index < len(self.alphabet):
                    chars.append(self.alphabet[index])
            texts.append("".join(chars))
        return texts


def run_single_ocr(image: UploadedImage) -> str:
    """Return one GP-LPR prediction for a single image."""

    predictions = run_batch_ocr([image])
    return predictions[0] if predictions else "UNKNOWN"


def run_batch_ocr(images: list[UploadedImage]) -> list[str]:
    """Run GP-LPR OCR on a batch of cropped plate images."""

    if not images:
        return []

    try:
        import torch

        cfg = _load_config()
        model, converter, in_channels = _get_model()
        tensor = _images_to_tensor(images, cfg, in_channels)
        device = next(model.parameters()).device

        with torch.no_grad():
            model_out = model(tensor.to(device))
            if isinstance(model_out, (tuple, list)):
                logits = model_out[1] if len(model_out) >= 2 else model_out[0]
            else:
                logits = model_out
            if not (torch.is_tensor(logits) and logits.dim() == 3):
                raise RuntimeError(f"unexpected GP-LPR output shape: {type(logits)}")

            pred_indices = logits.argmax(dim=2).detach().cpu()
            predictions = converter.decode_list(pred_indices)

        return [_normalize_prediction(text) for text in predictions]
    except Exception as exc:
        print(f"[OCR] fallback to UNKNOWN: {exc}")
        return ["UNKNOWN" for _ in images]


def run_ocr(images: list[UploadedImage]) -> str:
    """Run OCR on the first image in the selected pipeline output."""

    if not images:
        return "UNKNOWN"
    return run_single_ocr(images[0])


@lru_cache(maxsize=1)
def _get_model():
    import torch

    cfg = _load_config()
    model_cfg = cfg["model"]
    checkpoint_path = _resolve_path(cfg["paths"]["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GP-LPR checkpoint not found: {checkpoint_path}")

    gplpr_models = _load_gplpr_models_module()
    device_name = model_cfg.get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model_spec = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(model_spec, dict):
        raise RuntimeError("GP-LPR checkpoint format mismatch: missing model spec")

    args = model_spec.get("args", {})
    in_channels = int(args.get("nc", 3))
    alphabet = str(model_cfg.get("alphabet") or args.get("alphabet") or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    converter = GPLPRLabelConverter(alphabet)

    model = gplpr_models.make(model_spec, load_model=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f"[OCR] GP-LPR loaded: {checkpoint_path} | device={device}")
    return model, converter, in_channels


def _images_to_tensor(images: list[UploadedImage], cfg: dict, in_channels: int):
    import torch

    width = int(cfg["model"].get("img_width", 96))
    height = int(cfg["model"].get("img_height", 32))
    arrays = [_uploaded_to_chw_float(image, width, height, in_channels) for image in images]
    return torch.from_numpy(np.stack(arrays, axis=0)).float()


def _uploaded_to_chw_float(image: UploadedImage, width: int, height: int, in_channels: int) -> np.ndarray:
    frame = decode_image(image, cv2.IMREAD_COLOR)
    if frame is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    if in_channels == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return (gray.astype(np.float32) / 255.0)[None, :, :]

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[:in_channels]


def _load_gplpr_models_module():
    vendor_root = str(VENDOR_ROOT.resolve())
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    existing = sys.modules.get("models")
    if existing is not None:
        module_file = Path(getattr(existing, "__file__", "")).resolve()
        if str(module_file).startswith(vendor_root):
            return existing
        del sys.modules["models"]
        for name in [key for key in sys.modules if key.startswith("models.")]:
            del sys.modules[name]

    return importlib.import_module("models")


def _normalize_prediction(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    return normalized or "UNKNOWN"


def _load_config() -> dict:
    import yaml

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["ocr"]


def _resolve_path(path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (BE_ROOT / value).resolve()
