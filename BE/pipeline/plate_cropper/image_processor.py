# 이미지 하이라이팅 및 크롭 기능을 담당하는 모듈

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

MODULE_ROOT = Path(__file__).resolve().parent
BE_ROOT = MODULE_ROOT.parent.parent
if str(BE_ROOT) not in sys.path:
    sys.path.insert(0, str(BE_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pipeline.types import UploadedImage
from pipeline.utils import opencv_to_url
from pipeline.utils import image_to_url

from modules.plate_segmenter import PlateSegmenter
from modules.perspective_aligner import PerspectiveAligner


def _load_config() -> dict:
    config_path = BE_ROOT / "configs" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_weights_path(weights: str) -> str:
    weights_path = Path(weights)
    if weights_path.is_absolute():
        return str(weights_path)
    return str(BE_ROOT / weights_path)


@lru_cache(maxsize=1)
def _get_dummy_image() -> UploadedImage:
    cfg = _load_config()["yolo"]
    dummy_path = BE_ROOT / cfg["paths"]["dummy"]
    return UploadedImage(data=dummy_path.read_bytes(), content_type="image/jpeg")


@lru_cache(maxsize=1)
def _get_models() -> tuple[PlateSegmenter, PerspectiveAligner]:
    cfg = _load_config()["yolo"]

    segmenter = PlateSegmenter(
        weights=_resolve_weights_path(cfg["paths"]["weights"]),
        device=cfg["model"]["device"],
        conf_thr=cfg["model"]["conf_thr"],
        iou_thr=cfg["model"]["iou_thr"],
        imgsz=cfg["model"]["imgsz"],
    )
    segmenter.warmup()

    aligner = PerspectiveAligner(
        output_width=cfg["plate"]["output_width"],
        output_height=cfg["plate"]["output_height"],
    )
    return segmenter, aligner


def _decode_uploaded_image(image: UploadedImage) -> np.ndarray | None:
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        mask = det.mask_full
        color_mask = np.zeros_like(annotated)
        color_mask[mask > 0] = (0, 0, 255)

        alpha = 0.5
        mask_indices = mask > 0
        annotated[mask_indices] = cv2.addWeighted(
            annotated[mask_indices], 1 - alpha,
            color_mask[mask_indices], alpha, 0
        )
    return annotated


def _select_best_detection(detections: list):
    if not detections:
        return None
    return max(detections, key=lambda det: det.conf)


def image_highlighter(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return highlighted image data"""

    segmenter, _ = _get_models()
    highlighted_urls: list[str] = []
    dummy_url = image_to_url([_get_dummy_image()])[0]

    for image in images:
        frame = _decode_uploaded_image(image)
        if frame is None:
            highlighted_urls.append(dummy_url)
            continue

        detections = segmenter.detect(frame)
        best_detection = _select_best_detection(detections)
        if best_detection is None:
            highlighted_urls.append(dummy_url)
            continue

        highlighted = _draw_detections(frame, [best_detection])
        highlighted_urls.append(opencv_to_url(highlighted))

    return highlighted_urls


def image_cropper(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return cropped image data"""

    segmenter, aligner = _get_models()
    cropped_urls: list[str] = []
    dummy_url = image_to_url([_get_dummy_image()])[0]

    for image in images:
        frame = _decode_uploaded_image(image)
        if frame is None:
            cropped_urls.append(dummy_url)
            continue

        detections = segmenter.detect(frame)
        best_detection = _select_best_detection(detections)
        if best_detection is None:
            cropped_urls.append(dummy_url)
            continue

        align_result = aligner.align(frame, best_detection)
        if not align_result.success or align_result.warped is None:
            cropped_urls.append(dummy_url)
            continue
        cropped_urls.append(opencv_to_url(align_result.warped))

    return cropped_urls


if __name__ == "__main__":
    dummy_images = [_get_dummy_image()]
    highlighted_data = image_highlighter(dummy_images)
    cropped_data = image_cropper(dummy_images)

    print("Highlighted Data:", highlighted_data)
    print("Cropped Data:", cropped_data)
