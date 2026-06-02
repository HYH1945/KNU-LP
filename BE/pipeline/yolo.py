"""YOLO plate crop/highlight adapter."""

from __future__ import annotations

from pipeline.types import UploadedImage
from pipeline.types import YoloCandidate
from pipeline.utils import image_dimensions
from pipeline.utils import url_to_image


def run_yolo(images: list[UploadedImage]) -> list[YoloCandidate]:
    """Run YOLO cropper when available, otherwise keep the pipeline usable."""

    try:
        from pipeline.plate_cropper.image_processor import process_images

        return [
            _candidate_from_result(images[index], index, result)
            for index, result in enumerate(process_images(images))
        ]
    except Exception as exc:
        print(f"[YOLO] fallback to original images: {exc}")
        return [_fallback_candidate(image, index) for index, image in enumerate(images)]


def _candidate_from_result(
    source: UploadedImage,
    index: int,
    result: dict,
) -> YoloCandidate:
    crop = _single_url_to_image(result.get("crop_url")) or source
    highlighted = _single_url_to_image(result.get("highlighted_url")) or source
    width, height = image_dimensions(crop)
    return YoloCandidate(
        source=source,
        crop=crop,
        highlighted=highlighted,
        detected=bool(result.get("detected")),
        detection_area=int(result.get("detection_area") or 0),
        bbox_width=int(result.get("bbox_width") or 0),
        bbox_height=int(result.get("bbox_height") or 0),
        crop_width=int(result.get("crop_width") or width),
        crop_height=int(result.get("crop_height") or height),
        confidence=float(result.get("confidence") or 0.0),
        source_index=int(result.get("source_index", index)),
    )


def _single_url_to_image(url: str | None) -> UploadedImage | None:
    if not url:
        return None
    images = url_to_image([url])
    return images[0] if images else None


def _fallback_candidate(image: UploadedImage, index: int) -> YoloCandidate:
    width, height = image_dimensions(image)
    return YoloCandidate(
        source=image,
        crop=image,
        highlighted=image,
        detected=False,
        detection_area=0,
        bbox_width=0,
        bbox_height=0,
        crop_width=width,
        crop_height=height,
        confidence=0.0,
        source_index=index,
    )
