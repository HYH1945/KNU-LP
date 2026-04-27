"""OCR pipeline stub."""

from pipeline.types import UploadedImage


def run_ocr(images: list[UploadedImage]) -> str:
    """Input: list[UploadedImage] images. Output: str. Purpose: return a fixed dummy OCR string while preserving the stable pipeline signature for later model replacement."""

    _ = images
    return "00가 0000"
