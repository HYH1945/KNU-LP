"""Shared pipeline helpers."""

from base64 import b64encode

from pipeline.types import UploadedImage


def to_data_urls(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: convert uploaded image bytes into browser-safe data URLs while preserving each original MIME type."""

    return [
        f"data:{image.content_type};base64,{b64encode(image.data).decode('ascii')}"
        for image in images
    ]
