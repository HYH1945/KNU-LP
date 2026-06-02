"""Shared pipeline types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadedImage:
    """Pipeline image payload."""

    data: bytes
    content_type: str


@dataclass(frozen=True)
class PipelineOptions:
    """Runtime options received from the web UI."""

    input_mode: str = "image"
    hr_width: int = 335
    hr_height: int = 170
    video_start: float = 0.0
    video_end: float | None = None
    max_video_frames: int = 60
    output_slots: int = 5
    sr_mode: str = "auto"
    denoise_enabled: bool = True


@dataclass(frozen=True)
class YoloCandidate:
    """YOLO crop output and detection metadata for one source frame."""

    source: UploadedImage
    crop: UploadedImage
    highlighted: UploadedImage
    detected: bool
    detection_area: int
    bbox_width: int
    bbox_height: int
    crop_width: int
    crop_height: int
    confidence: float
    source_index: int


@dataclass(frozen=True)
class SelectedCandidate:
    """Selected source frame, crop, and metadata kept in the same order."""

    source: UploadedImage
    crop: UploadedImage
    highlighted: UploadedImage
    width: int
    height: int
    area: int
    detected: bool
    detection_area: int
    bbox_width: int
    bbox_height: int
    confidence: float
    source_index: int
