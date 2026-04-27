"""Super-resolution pipeline stub."""

from pipeline.types import UploadedImage
from pipeline.utils import to_data_urls


def run_sr(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: return dummy super-resolution payloads using shared data URL conversion."""

    return to_data_urls(images)
