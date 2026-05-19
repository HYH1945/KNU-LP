"""
DnCNN Inference
- UploadedImage(bytes) 입력을 받아 grayscale로 디코드한 뒤
  학습된 DnCNN으로 디노이즈하여 ndarray로 반환
- 모델 로드는 build_and_load_model로 분리 (caller가 1회 로드 후 주입)
"""

from __future__ import annotations
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent.parent
if str(BE_ROOT) not in sys.path:
    sys.path.insert(0, str(BE_ROOT))

from pipeline.types import UploadedImage
from pipeline.denoiser.model import DnCNN


def _resolve_weights_path(weights: str) -> str:
    """
    Input:
        weights (str) : settings.yaml에 기재된 가중치 경로 (상대/절대)
    Output:
        absolute_path (str) : BE 루트 기준 절대 경로
    """
    weights_path = Path(weights)
    if weights_path.is_absolute():
        return str(weights_path)
    return str(BE_ROOT / weights_path)


# ---------------------------------------------------------------------- #
def build_and_load_model(cfg: dict) -> nn.Module:
    """
    Input:
        cfg (dict) : settings.yaml의 denoiser 섹션
    Output:
        model (nn.Module) : checkpoint 로드 + device 이동 + eval 완료된 DnCNN
    """
    model_cfg = cfg["model"]
    device = model_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Denoiser] CUDA 없음 → CPU 전환")
        device = "cpu"

    model = DnCNN(
        in_channels=model_cfg["in_channels"],
        depth=model_cfg["depth"],
        features=model_cfg["features"],
    )

    weights_path = Path(_resolve_weights_path(cfg["paths"]["weights"]))
    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일 없음: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[Denoiser] 로드 완료 | device={device}")
    return model


# ---------------------------------------------------------------------- #
def denoise_single(image: UploadedImage, model: nn.Module, cfg: dict) -> np.ndarray:
    """
    Input:
        image (UploadedImage) : 단일 입력 이미지 (bytes + content_type)
        model (nn.Module)     : build_and_load_model 결과
        cfg   (dict)          : settings.yaml의 denoiser 섹션
    Output:
        denoised (np.ndarray) : 디노이즈된 grayscale 이미지 (H, W), uint8,
                                입력 해상도로 복원
    """
    # bytes → grayscale ndarray
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError("빈 이미지 입력")
    frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError("이미지 디코드 실패")

    # original_shape = frame.shape  # (H, W)

    # 전처리: resize → normalize → tensor
    # model_h, model_w = cfg["model"]["image_size"]
    # resized = cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
    # normalized = resized.astype(np.float32) / 255.0
    normalized = frame.astype(np.float32) / 255.0

    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        output = model(tensor)

    # 후처리: tensor → ndarray → uint8 → 원본 해상도 복원
    output_np = output.squeeze().detach().cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # orig_h, orig_w = original_shape
    # output_np = cv2.resize(output_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    return output_np


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트:
    - settings.yaml 로드 → 모델 빌드 → 더미 grayscale 이미지 1장 추론
    - best_model.pt 가 있어야 통과
    """
    import yaml

    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["denoiser"]

    # 더미 grayscale 이미지 생성 (YOLO crop 사이즈 335x170)
    dummy_array = np.random.randint(0, 255, (170, 335), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", dummy_array)
    assert success, "더미 이미지 인코드 실패"
    dummy_image = UploadedImage(data=encoded.tobytes(), content_type="image/png")

    model = build_and_load_model(cfg)
    try:
        result = denoise_single(dummy_image, model, cfg)
    finally:
        del model
        if cfg["model"]["device"] != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert result.shape == (170, 335), f"출력 shape 오류: {result.shape}"
    assert result.dtype == np.uint8, f"출력 dtype 오류: {result.dtype}"
    print(f"✅ denoise_single 단독 테스트 통과 | shape={result.shape}, dtype={result.dtype}")
