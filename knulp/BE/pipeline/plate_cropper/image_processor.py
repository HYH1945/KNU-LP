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
from pipeline.utils import image_to_url
from pipeline.utils import opencv_to_uploaded
from pipeline.utils import opencv_to_url

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


def _bbox_size(detection) -> tuple[int, int]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return max(0, int(x2 - x1)), max(0, int(y2 - y1))


def _bbox_area(detection) -> int:
    width, height = _bbox_size(detection)
    return width * height


def _crop_bbox(frame: np.ndarray, detection) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _natural_aligned_crop(frame: np.ndarray, detection, aligner: PerspectiveAligner) -> np.ndarray | None:
    corners, _ = aligner._extract_corners(detection.mask_full)
    if corners is None:
        return _crop_bbox(frame, detection)

    top_w = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    out_w = max(1, int(round(max(top_w, bottom_w))))
    out_h = max(1, int(round(max(left_h, right_h))))

    dst_pts = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
    return cv2.warpPerspective(frame, matrix, (out_w, out_h))


def process_images(images: list[UploadedImage]) -> list[dict]:
    """Run YOLO once per image and return crop/highlight metadata together."""

    segmenter, aligner = _get_models()
    dummy_image = _get_dummy_image()
    dummy_url = image_to_url([dummy_image])[0]
    results: list[dict] = []

    for index, image in enumerate(images):
        frame = _decode_uploaded_image(image)
        fallback = {
            "crop_url": dummy_url,
            "highlighted_url": dummy_url,
            "detected": False,
            "detection_area": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "crop_width": 0,
            "crop_height": 0,
            "confidence": 0.0,
            "source_index": index,
        }
        if frame is None:
            results.append(fallback)
            continue

        detections = segmenter.detect(frame)
        best_detection = _select_best_detection(detections)
        if best_detection is None:
            results.append(fallback)
            continue

        highlighted = _draw_detections(frame, [best_detection])
        highlighted_url = opencv_to_url(highlighted)
        crop_frame = _natural_aligned_crop(frame, best_detection, aligner)
        if crop_frame is None or crop_frame.size == 0:
            fallback["highlighted_url"] = highlighted_url
            fallback["detection_area"] = _bbox_area(best_detection)
            bbox_width, bbox_height = _bbox_size(best_detection)
            fallback["bbox_width"] = bbox_width
            fallback["bbox_height"] = bbox_height
            fallback["confidence"] = float(best_detection.conf)
            results.append(fallback)
            continue

        crop = opencv_to_uploaded(crop_frame)
        crop_height, crop_width = crop_frame.shape[:2]
        bbox_width, bbox_height = _bbox_size(best_detection)
        results.append(
            {
                "crop_url": image_to_url([crop])[0],
                "highlighted_url": highlighted_url,
                "detected": True,
                "detection_area": bbox_width * bbox_height,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "confidence": float(best_detection.conf),
                "source_index": index,
            }
        )

    return results


def image_highlighter(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return highlighted image data"""

    return [result["highlighted_url"] for result in process_images(images)]


def image_cropper(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return cropped image data"""

    return [result["crop_url"] for result in process_images(images)]


if __name__ == "__main__":
    dummy_images = [_get_dummy_image()]
    highlighted_data = image_highlighter(dummy_images)
    cropped_data = image_cropper(dummy_images)

    print("Highlighted Data:", highlighted_data)
    print("Cropped Data:", cropped_data)
