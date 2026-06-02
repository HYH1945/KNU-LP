"""Response schemas for the backend API."""

from typing import Annotated

from pydantic import BaseModel
from pydantic import Field

ImageList = Annotated[list[str], Field(min_length=5, max_length=5)]
NonEmptyText = Annotated[str, Field(min_length=1)]


class AnalyzeResponse(BaseModel):
    """Stable frontend response contract for the staged pipeline."""

    selected_inputs: ImageList
    yolo_crops: ImageList
    yolo_selected: ImageList
    denoised: ImageList
    sr: ImageList
    ocr_text: NonEmptyText
    pipeline_route: str
    sr_applied: bool
    high_resolution_count: int
    hr_width: int
    hr_height: int
