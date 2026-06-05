"""API routes for the analysis backend."""

from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile

from pipeline.orchestrator import run_analyze_pipeline
from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage
from schemas import AnalyzeResponse

router = APIRouter()


def _is_image_file(file: UploadFile) -> bool:
    return bool(file.content_type and file.content_type.startswith("image/"))


def _is_video_file(file: UploadFile) -> bool:
    return bool(file.content_type and file.content_type.startswith("video/"))


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_media(
    files: Optional[List[UploadFile]] = File(default=None),
    input_mode: str = Form(default="image"),
    hr_width: int = Form(default=96),
    hr_height: int = Form(default=32),
    video_start: float = Form(default=0.0),
    video_end: Optional[float] = Form(default=None),
    sr_mode: str = Form(default="auto"),
    denoise_enabled: bool = Form(default=True),
) -> AnalyzeResponse:
    """Analyze uploaded images or a video through the staged pipeline."""

    input_mode = input_mode.lower()
    sr_mode = sr_mode.lower()

    if input_mode not in {"image", "video"}:
        raise HTTPException(status_code=400, detail="input_mode must be image or video")
    if sr_mode not in {"auto", "always", "skip"}:
        raise HTTPException(status_code=400, detail="sr_mode must be auto, always, or skip")
    if hr_width <= 0 or hr_height <= 0:
        raise HTTPException(status_code=400, detail="HR width and height must be positive")
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    if input_mode == "video":
        if len(files) != 1 or not _is_video_file(files[0]):
            raise HTTPException(status_code=400, detail="video mode requires exactly one video file")
    elif any(not _is_image_file(file) for file in files):
        raise HTTPException(status_code=400, detail="image mode accepts image files only")

    uploaded_files = [
        UploadedImage(
            data=await file.read(),
            content_type=file.content_type or "application/octet-stream",
        )
        for file in files
    ]
    options = PipelineOptions(
        input_mode=input_mode,
        hr_width=hr_width,
        hr_height=hr_height,
        video_start=video_start,
        video_end=video_end,
        sr_mode=sr_mode,
        denoise_enabled=denoise_enabled,
    )

    try:
        result = run_analyze_pipeline(uploaded_files, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc

    return AnalyzeResponse(**result)
