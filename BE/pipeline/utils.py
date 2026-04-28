"""Shared pipeline helpers."""

from base64 import b64encode
from base64 import b64decode

import cv2
import numpy as np

from pipeline.types import UploadedImage


def image_to_url(images: list[UploadedImage]) -> list[str]:
    """Input: list[UploadedImage] images. Output: list[str]. Purpose: convert uploaded image bytes into browser-safe data URLs while preserving each original MIME type."""

    return [
        f"data:{image.content_type};base64,{b64encode(image.data).decode('ascii')}"
        for image in images
    ]

def opencv_to_url(image: np.ndarray, content_type: str = "image/png") -> str:
    """Input: np.ndarray image, str content_type. Output: str. Purpose: encode an OpenCV image into a browser-safe data URL without writing it to disk."""

    extension_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }
    extension = extension_map.get(content_type, ".png")
    encoded, buffer = cv2.imencode(extension, image)
    if not encoded:
        raise ValueError(f"failed to encode image as {content_type}")

    uploaded = UploadedImage(data=buffer.tobytes(), content_type=content_type)
    return image_to_url([uploaded])[0]

def url_to_image(urls: list[str]) -> list[UploadedImage]:
    """Input: list[str] urls. Output: list[UploadedImage]. Purpose: convert browser-safe data URLs back into UploadedImage objects, extracting the original MIME type and binary data."""

    images = []
    for url in urls:
        if url.startswith("data:") and ";base64," in url:
            header, b64_data = url.split(";base64,")
            content_type = header[len("data:"):]
            data = b64decode(b64_data)
            images.append(UploadedImage(content_type=content_type, data=data))
    return images
