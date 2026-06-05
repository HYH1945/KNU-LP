"""Response schemas for the backend API."""

from typing import Annotated
from typing import List

from pydantic import BaseModel
from pydantic import Field

ImageList = Annotated[List[str], Field(min_length=5, max_length=5)]
SrImageList = Annotated[List[str], Field(min_length=1, max_length=5)]
NonEmptyText = Annotated[str, Field(min_length=1)]


class PlateBbox(BaseModel):
    """Original-frame plate bounding box metadata for one selected candidate."""

    detected: bool
    width: int
    height: int
    area: int
    confidence: float
    source_index: int


BboxList = Annotated[List[PlateBbox], Field(min_length=5, max_length=5)]


class AnalyzeResponse(BaseModel):
    """Stable frontend response contract for the staged pipeline."""

    input_preview: ImageList
    input_omitted_count: int
    selected_inputs: ImageList
    selected_source_indices: List[int]
    selected_plate_bboxes: BboxList
    yolo_crops: ImageList
    yolo_selected: ImageList
    denoised: ImageList
    sr: SrImageList
    ocr_text: NonEmptyText
    pipeline_route: str
    sr_applied: bool
    high_resolution_count: int
    hr_width: int
    hr_height: int
