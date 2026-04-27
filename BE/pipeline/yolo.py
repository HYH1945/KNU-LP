"""YOLO pipeline stub."""

from pipeline.types import UploadedImage
from pipeline.utils import to_data_urls


def run_yolo(images: list[UploadedImage]) -> tuple[list[str], list[str]]:
    """Input: list[UploadedImage] images. Output: tuple[list[str], list[str]]. Purpose: return dummy YOLO crop and selected-region payloads using shared data URL conversion."""

    data_urls = to_data_urls(images)
    return data_urls, data_urls
