"""RGDiffSR super-resolution adapter."""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from functools import lru_cache
from pathlib import Path
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage
from pipeline.utils import decode_image
from pipeline.utils import image_to_url
from pipeline.utils import opencv_to_uploaded

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent.parent
VENDORS_ROOT = BE_ROOT / "vendors"
_LAST_WARNING: str | None = None


def get_last_warning() -> str | None:
    """Return the latest fallback warning emitted by this stage."""

    return _LAST_WARNING


def run_sr(
    images: list[UploadedImage],
    placeholder_hr: UploadedImage | None,
    options: PipelineOptions,
) -> list[str]:
    """Run the 5-frame RGDiffSR model and return one SR image URL."""

    global _LAST_WARNING
    _LAST_WARNING = None

    if not images:
        return []

    try:
        sr_image = _run_rgdiffsr(images, placeholder_hr, options)
        return image_to_url([sr_image])
    except Exception as exc:
        _LAST_WARNING = f"SR fallback to bicubic first crop: {exc}"
        print(f"[SR] {_LAST_WARNING}")
        return image_to_url([_bicubic_first(images, options)])


def _run_rgdiffsr(
    images: list[UploadedImage],
    placeholder_hr: UploadedImage | None,
    options: PipelineOptions,
) -> UploadedImage:
    import torch

    cfg = _load_config()
    model_cfg = cfg["model"]
    model = _get_model()
    device = next(model.parameters()).device
    seed = int(model_cfg.get("seed", 23))

    _set_seed(seed)
    lr_tensor = _images_to_lr_tensor(images, cfg).to(device)
    hr_tensor = _placeholder_to_hr_tensor(placeholder_hr or images[0], cfg, options).to(device)
    batch = {
        "image": hr_tensor,
        "LR_image": lr_tensor,
        "label": [""],
        "id": [0],
    }

    with torch.no_grad():
        result = model.recognize_sample(
            batch,
            N=1,
            split="predict",
            inpaint=False,
            ddim_steps=int(model_cfg.get("ddim_steps", 200)),
            ddim_eta=float(model_cfg.get("ddim_eta", 1.0)),
        )

    samples = result["samples"]
    image = samples[0, :3].detach().float().cpu()
    if image.min().item() < -1e-6:
        image = (image + 1.0) * 0.5
    image = image.clamp(0.0, 1.0)
    rgb = (image.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    output_width, output_height = _get_output_size(model_cfg)
    resized = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    return opencv_to_uploaded(resized)


@lru_cache(maxsize=1)
def _get_model():
    import torch
    from omegaconf import OmegaConf

    cfg = _load_config()
    paths = cfg["paths"]
    repo_root = _resolve_path(paths["repo"])
    checkpoint_path = _resolve_path(paths["checkpoint"])
    vqgan_path = _resolve_path(paths["vqgan"])
    parseq_path = _resolve_path(paths["parseq"])
    parseq_repo = _resolve_repo_path(repo_root, paths["parseq_repo"])
    config_path = _resolve_repo_path(repo_root, paths["config"])

    _require_file(checkpoint_path, "SR checkpoint")
    _require_file(vqgan_path, "VQGAN checkpoint")
    _require_file(parseq_path, "PARSeq checkpoint")
    _require_file(config_path, "SR config")
    if not repo_root.is_dir():
        raise FileNotFoundError(f"RGDiffSR repo not found: {repo_root}")
    if not parseq_repo.is_dir():
        raise FileNotFoundError(f"PARSeq repo not found: {parseq_repo}")

    # sys.modules에서 taming 관련 모듈을 캐시 아웃하여, 올바른 경로에서 다시 로드되도록 합니다.
    for key in list(sys.modules.keys()):
        if key in {"ldm", "strhub", "taming", "text_super_resolution"} or key.startswith(
            ("ldm.", "strhub.", "taming.", "text_super_resolution.")
        ):
            sys.modules.pop(key, None)

    _set_sys_path(BE_ROOT, 0)
    _set_sys_path(VENDORS_ROOT, 1)
    _set_sys_path(repo_root, 2)

    with _temporary_cwd(repo_root):
        from ldm.util import instantiate_from_config

        config = OmegaConf.load(str(config_path))
        model_params = config.model.params
        model_params.ckpt_path = str(checkpoint_path)
        model_params.first_stage_config.params.ckpt_path = str(vqgan_path)
        model_params.cond_stage_config.params.checkpoint_path = str(parseq_path)
        model_params.cond_stage_config.params.parseq_repo_path = str(parseq_repo)

        model = instantiate_from_config(config.model)

    device_name = cfg["model"].get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[SR] RGDiffSR loaded: {checkpoint_path} | device={device}")
    return model


def _images_to_lr_tensor(images: list[UploadedImage], cfg: dict):
    import torch

    model_cfg = cfg["model"]
    frames = int(model_cfg.get("frames", 5))
    width = int(model_cfg.get("input_width", 64))
    height = int(model_cfg.get("input_height", 16))
    padded = images[:frames]
    while len(padded) < frames:
        padded.append(padded[-1])

    arrays = [_uploaded_to_rgb_float(image, width, height) for image in padded]
    stacked = np.stack(arrays, axis=0)[None, ...]  # B x T x H x W x C
    return torch.from_numpy(stacked).float()


def _placeholder_to_hr_tensor(image: UploadedImage, cfg: dict, options: PipelineOptions):
    import torch

    model_cfg = cfg["model"]
    width, height = _get_internal_size(model_cfg)
    if width <= 0:
        width = max(options.hr_width, 128)
    if height <= 0:
        height = max(options.hr_height, 32)

    rgb = _uploaded_to_rgb_float(image, width, height)
    return torch.from_numpy(rgb[None, ...]).float()


def _uploaded_to_rgb_float(image: UploadedImage, width: int, height: int) -> np.ndarray:
    try:
        resized = (
            Image.open(BytesIO(image.data))
            .convert("RGB")
            .resize((width, height), Image.BICUBIC)
        )
        rgb = np.asarray(resized, dtype=np.float32)
    except Exception:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
    return rgb / 255.0


def _bicubic_first(images: list[UploadedImage], options: PipelineOptions) -> UploadedImage:
    cfg = _load_config()
    model_cfg = cfg["model"]
    output_width, output_height = _get_output_size(model_cfg)
    frame = decode_image(images[0], cv2.IMREAD_COLOR)
    if frame is None:
        return images[0]
    resized = cv2.resize(
        frame,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )
    return opencv_to_uploaded(resized)


def _get_internal_size(model_cfg: dict) -> tuple[int, int]:
    width = int(model_cfg.get("sr_internal_width", 128))
    height = int(model_cfg.get("sr_internal_height", 32))
    return width, height


def _get_output_size(model_cfg: dict) -> tuple[int, int]:
    fallback_width, fallback_height = _get_internal_size(model_cfg)
    width = int(model_cfg.get("output_width", fallback_width))
    height = int(model_cfg.get("output_height", fallback_height))
    return width, height


def _load_config() -> dict:
    import yaml

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["superresolution"]


def _resolve_path(path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (BE_ROOT / value).resolve()


def _resolve_repo_path(repo_root: Path, path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (repo_root / value).resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _set_sys_path(path: Path, index: int) -> None:
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(index, path_text)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
