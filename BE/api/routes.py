"""API routes for the backend skeleton."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from pipeline.denoise import run_denoise
from pipeline.ocr import run_ocr
from pipeline.sr import run_sr
from pipeline.types import UploadedImage
from pipeline.yolo import run_yolo
from schemas import AnalyzeResponse

router = APIRouter()


def _is_image_file(file: UploadFile) -> bool:
    """Input: UploadFile file. Output: bool. Purpose: validate that an uploaded file declares an image MIME type."""

    return bool(file.content_type and file.content_type.startswith("image/"))


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_images(files: list[UploadFile] | None = File(default=None)) -> AnalyzeResponse:
    """Input: optional list[UploadFile]. Output: AnalyzeResponse. Purpose: validate uploads, assemble pipeline stubs, and return the frontend contract payload."""

    if not files or len(files) != 5:
        raise HTTPException(status_code=400, detail="exactly 5 files required")

    if any(not _is_image_file(file) for file in files):
        raise HTTPException(status_code=400, detail="all files must be images")

    images = [
        UploadedImage(
            data=await file.read(),
            content_type=file.content_type or "application/octet-stream",
        )
        for file in files
    ]

    yolo_crops, yolo_selected = run_yolo(images)
    denoised = run_denoise(images)
    sr = run_sr(images)
    ocr_text = run_ocr(images)

    return AnalyzeResponse(
        yolo_crops=yolo_crops,
        yolo_selected=yolo_selected,
        denoised=denoised,
        sr=sr,
        ocr_text=ocr_text,
    )
