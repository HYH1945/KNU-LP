"""Response schemas for the backend skeleton."""

from typing import Annotated

from pydantic import BaseModel, Field

ImageList = Annotated[list[str], Field(min_length=5, max_length=5)]
NonEmptyText = Annotated[str, Field(min_length=1)]


class AnalyzeResponse(BaseModel):
    """Input: API pipeline output fields. Output: validated response model. Purpose: enforce the frontend response contract."""

    yolo_crops: ImageList
    yolo_selected: ImageList
    denoised: ImageList
    sr: ImageList
    ocr_text: NonEmptyText
