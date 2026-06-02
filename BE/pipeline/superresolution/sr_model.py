"""Super-resolution model adapter.

The current implementation is a lightweight stub. It preserves the final
model interface by accepting denoised crops, an HR placeholder, and runtime
options, then returning five browser-safe image URLs.
"""

from __future__ import annotations

import cv2

from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage
from pipeline.utils import decode_image
from pipeline.utils import image_to_url
from pipeline.utils import opencv_to_uploaded


def run_sr(
    images: list[UploadedImage],
    placeholder_hr: UploadedImage | None,
    options: PipelineOptions,
) -> list[str]:
    """Return bicubic HR placeholders until the real SR model is integrated."""

    _ = placeholder_hr
    restored: list[UploadedImage] = []
    for image in images:
        frame = decode_image(image, cv2.IMREAD_COLOR)
        if frame is None:
            restored.append(image)
            continue

        resized = cv2.resize(
            frame,
            (options.hr_width, options.hr_height),
            interpolation=cv2.INTER_CUBIC,
        )
        restored.append(opencv_to_uploaded(resized))

    return image_to_url(restored)
