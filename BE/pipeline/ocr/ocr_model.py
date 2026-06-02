"""OCR model adapter.

This module keeps the final OCR interface stable while the real model is
being prepared by the OCR task owner.
"""

from pipeline.types import UploadedImage


def run_single_ocr(image: UploadedImage) -> str:
    """Return one OCR prediction for a single image."""

    _ = image
    return "00가 0000"


def run_ocr(images: list[UploadedImage]) -> str:
    """Run OCR on the first image in the selected pipeline output."""

    if not images:
        return "00가 0000"
    return run_single_ocr(images[0])
