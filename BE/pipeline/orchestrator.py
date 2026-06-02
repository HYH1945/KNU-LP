"""End-to-end analysis pipeline orchestration."""

from __future__ import annotations

from pipeline.denoise import run_denoise
from pipeline.ocr.ensemble import run_ensemble_ocr
from pipeline.ocr.ocr_model import run_ocr
from pipeline.processor.image_processor import build_placeholder_hr
from pipeline.processor.image_processor import get_high_resolution_count
from pipeline.processor.image_processor import select_top_candidates
from pipeline.processor.image_processor import should_use_sr
from pipeline.processor.video_processor import extract_video_frames
from pipeline.superresolution.sr_model import run_sr
from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage
from pipeline.utils import image_to_url
from pipeline.utils import url_to_image
from pipeline.yolo import run_yolo


def run_analyze_pipeline(uploaded_files: list[UploadedImage], options: PipelineOptions) -> dict:
    """Run preprocess -> denoise -> SR/OCR route -> OCR."""

    source_images = _prepare_sources(uploaded_files, options)
    yolo_crops, yolo_selected = run_yolo(source_images)
    candidates = select_top_candidates(
        source_images=source_images,
        crop_urls=yolo_crops,
        highlighted_urls=yolo_selected,
        output_slots=options.output_slots,
    )

    crop_images = [candidate.crop for candidate in candidates]
    if options.denoise_enabled:
        denoised_urls = run_denoise(crop_images)
        denoised_images = url_to_image(denoised_urls) or crop_images
    else:
        denoised_images = crop_images
        denoised_urls = image_to_url(denoised_images)

    if len(denoised_images) != options.output_slots:
        denoised_images = crop_images
        denoised_urls = image_to_url(denoised_images)

    sr_applied = should_use_sr(candidates, options)
    if sr_applied:
        placeholder_hr = build_placeholder_hr(denoised_images[0], options)
        sr_urls = run_sr(denoised_images, placeholder_hr, options)
        ocr_text = run_ocr(url_to_image(sr_urls) or denoised_images)
        pipeline_route = "sr_then_ocr"
    else:
        sr_urls = image_to_url(denoised_images)
        ocr_text = run_ensemble_ocr(denoised_images)
        pipeline_route = "ocr_ensemble"

    return {
        "selected_inputs": image_to_url([candidate.source for candidate in candidates]),
        "yolo_crops": image_to_url(crop_images),
        "yolo_selected": image_to_url([candidate.highlighted for candidate in candidates]),
        "denoised": denoised_urls,
        "sr": sr_urls,
        "ocr_text": ocr_text,
        "pipeline_route": pipeline_route,
        "sr_applied": sr_applied,
        "high_resolution_count": get_high_resolution_count(candidates, options),
        "hr_width": options.hr_width,
        "hr_height": options.hr_height,
    }


def _prepare_sources(uploaded_files: list[UploadedImage], options: PipelineOptions) -> list[UploadedImage]:
    """Normalize image/video input into source frames."""

    if not uploaded_files:
        raise ValueError("at least one file is required")

    if options.input_mode == "video":
        if len(uploaded_files) != 1:
            raise ValueError("video mode accepts exactly one video file")
        return extract_video_frames(uploaded_files[0], options)

    return uploaded_files
