"""Denoise pipeline stub."""

from pipeline.types import UploadedImage
from pipeline.utils import image_to_url


def run_denoise(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return dummy denoised image payloads using shared data URL conversion."""

    return image_to_url(images)
