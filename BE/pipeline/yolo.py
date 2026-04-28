"""YOLO pipeline stub."""

from pipeline.types import UploadedImage
from pipeline.utils import image_to_url

from pipeline.plate_cropper.image_processor import image_cropper
from pipeline.plate_cropper.image_processor import image_highlighter

def run_yolo(images: list[UploadedImage]) -> tuple[list[str], list[str]]:
    """Input: list[UploadedImage] images. Output: tuple[list[str], list[str]]. Purpose: return dummy YOLO crop and selected-region payloads using shared data URL conversion."""

    cropped_image = image_cropper(images)
    highlighted_image = image_highlighter(images)
    return cropped_image, highlighted_image
