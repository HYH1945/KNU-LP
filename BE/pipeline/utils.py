"""Shared pipeline helpers."""

from base64 import b64decode
from base64 import b64encode

import cv2
import numpy as np

from pipeline.types import UploadedImage


def image_to_url(images: list[UploadedImage]) -> list[str]:
    """Convert uploaded image bytes into browser-safe data URLs."""

    return [
        f"data:{image.content_type};base64,{b64encode(image.data).decode('ascii')}"
        for image in images
    ]


def opencv_to_url(image: np.ndarray, content_type: str = "image/png") -> str:
    """Encode an OpenCV image into a browser-safe data URL."""

    return image_to_url([opencv_to_uploaded(image, content_type)])[0]


def opencv_to_uploaded(image: np.ndarray, content_type: str = "image/png") -> UploadedImage:
    """Encode an OpenCV image into an UploadedImage without writing it to disk."""

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

    return UploadedImage(data=buffer.tobytes(), content_type=content_type)


def url_to_image(urls: list[str]) -> list[UploadedImage]:
    """Convert browser-safe data URLs back into UploadedImage objects."""

    images = []
    for url in urls:
        if url.startswith("data:") and ";base64," in url:
            header, b64_data = url.split(";base64,")
            content_type = header[len("data:"):]
            data = b64decode(b64_data)
            images.append(UploadedImage(content_type=content_type, data=data))
    return images


def decode_image(image: UploadedImage, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an UploadedImage into an OpenCV ndarray."""

    buffer = np.frombuffer(image.data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def image_dimensions(image: UploadedImage) -> tuple[int, int]:
    """Return image width and height. Undecodable images report 0x0."""

    frame = decode_image(image, cv2.IMREAD_UNCHANGED)
    if frame is None:
        return 0, 0
    height, width = frame.shape[:2]
    return width, height


def pad_uploaded_images(images: list[UploadedImage], target_count: int) -> list[UploadedImage]:
    """Pad a non-empty image list by repeating the last image."""

    if not images:
        raise ValueError("at least one image is required")

    padded = images[:target_count]
    while len(padded) < target_count:
        padded.append(padded[-1])
    return padded
