"""Image selection and HR-based routing logic."""

from __future__ import annotations

import cv2

from pipeline.types import PipelineOptions
from pipeline.types import SelectedCandidate
from pipeline.types import UploadedImage
from pipeline.utils import decode_image
from pipeline.utils import image_dimensions
from pipeline.utils import opencv_to_uploaded
from pipeline.utils import pad_uploaded_images
from pipeline.utils import url_to_image


def select_top_candidates(
    source_images: list[UploadedImage],
    crop_urls: list[str],
    highlighted_urls: list[str],
    output_slots: int,
) -> list[SelectedCandidate]:
    """Pair YOLO outputs with source frames and keep the highest-resolution crops."""

    if not source_images:
        raise ValueError("at least one input image is required")

    crops = url_to_image(crop_urls)
    highlighted = url_to_image(highlighted_urls)
    crops = pad_uploaded_images(crops or source_images, len(source_images))
    highlighted = pad_uploaded_images(highlighted or source_images, len(source_images))

    candidates: list[SelectedCandidate] = []
    for index, source in enumerate(source_images):
        crop = crops[min(index, len(crops) - 1)]
        selected = highlighted[min(index, len(highlighted) - 1)]
        width, height = image_dimensions(crop)
        candidates.append(
            SelectedCandidate(
                source=source,
                crop=crop,
                highlighted=selected,
                width=width,
                height=height,
                area=width * height,
                source_index=index,
            )
        )

    candidates.sort(key=lambda item: item.area, reverse=True)
    selected = candidates[:output_slots]
    while len(selected) < output_slots:
        selected.append(selected[-1])
    return selected


def get_high_resolution_count(candidates: list[SelectedCandidate], options: PipelineOptions) -> int:
    """Count crops whose area is greater than the user-defined HR threshold."""

    threshold = int(options.hr_width * options.hr_height * 0.8)
    return sum(candidate.area > threshold for candidate in candidates)


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
