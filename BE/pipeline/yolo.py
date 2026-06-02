"""YOLO plate crop/highlight adapter."""

from pipeline.types import UploadedImage
from pipeline.utils import image_to_url


def run_yolo(images: list[UploadedImage]) -> tuple[list[str], list[str]]:
    """Run YOLO cropper when available, otherwise keep the pipeline usable."""

    try:
        from pipeline.plate_cropper.image_processor import image_cropper
        from pipeline.plate_cropper.image_processor import image_highlighter

        cropped_images = image_cropper(images)
        highlighted_images = image_highlighter(images)
        return cropped_images, highlighted_images
    except Exception as exc:
        print(f"[YOLO] fallback to original images: {exc}")
        fallback = image_to_url(images)
        return fallback, fallback
