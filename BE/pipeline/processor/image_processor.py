"""Image selection and HR-based routing logic."""

from __future__ import annotations

import cv2

from pipeline.types import PipelineOptions
from pipeline.types import SelectedCandidate
from pipeline.types import UploadedImage
from pipeline.types import YoloCandidate
from pipeline.utils import decode_image
from pipeline.utils import image_dimensions
from pipeline.utils import opencv_to_uploaded


def select_top_candidates(
    yolo_candidates: list[YoloCandidate],
    output_slots: int,
) -> list[SelectedCandidate]:
    """Keep the best YOLO crops by original plate bbox area."""

    if not yolo_candidates:
        raise ValueError("at least one input image is required")

    candidates: list[SelectedCandidate] = []
    for yolo_candidate in yolo_candidates:
        width, height = image_dimensions(yolo_candidate.crop)
        ranking_area = yolo_candidate.detection_area if yolo_candidate.detected else 0
        candidates.append(
            SelectedCandidate(
                source=yolo_candidate.source,
                crop=yolo_candidate.crop,
                highlighted=yolo_candidate.highlighted,
                width=yolo_candidate.crop_width or width,
                height=yolo_candidate.crop_height or height,
                area=ranking_area,
                detected=yolo_candidate.detected,
                detection_area=yolo_candidate.detection_area,
                bbox_width=yolo_candidate.bbox_width,
                bbox_height=yolo_candidate.bbox_height,
                confidence=yolo_candidate.confidence,
                source_index=yolo_candidate.source_index,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.detected,
            item.area,
            item.confidence,
            -item.source_index,
        ),
        reverse=True,
    )
    selected = candidates[:output_slots]
    while len(selected) < output_slots:
        selected.append(selected[-1])
    return selected


def get_high_resolution_count(candidates: list[SelectedCandidate], options: PipelineOptions) -> int:
    """Count detected plate bboxes whose area is greater than the HR threshold."""

    threshold = int(options.hr_width * options.hr_height * 0.8)
    return sum(
        candidate.detected and candidate.detection_area > threshold
        for candidate in candidates
    )


def should_use_sr(candidates: list[SelectedCandidate], options: PipelineOptions) -> bool:
    """Decide whether the selected crops should enter the SR module."""

    sr_mode = options.sr_mode.lower()
    if sr_mode == "always":
        return True
    if sr_mode == "skip":
        return False

    return get_high_resolution_count(candidates, options) < 3


def build_placeholder_hr(image: UploadedImage, options: PipelineOptions) -> UploadedImage:
    """Create a bicubic HR placeholder for SR modules that require one."""

    frame = decode_image(image, cv2.IMREAD_COLOR)
    if frame is None:
        return image

    resized = cv2.resize(
        frame,
        (options.hr_width, options.hr_height),
        interpolation=cv2.INTER_CUBIC,
    )
    return opencv_to_uploaded(resized)
