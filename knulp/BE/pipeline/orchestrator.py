"""End-to-end analysis pipeline orchestration."""

from __future__ import annotations

from pipeline.denoise import run_denoise
from pipeline.ocr.ensemble import run_ensemble_ocr
from pipeline.ocr.ocr_model import run_ocr
from pipeline.processor.image_processor import get_high_resolution_count
from pipeline.processor.image_processor import select_top_candidates
from pipeline.processor.image_processor import should_use_sr
from pipeline.processor.video_processor import extract_video_frames
from pipeline.superresolution.sr_model import run_sr
from pipeline.types import PipelineOptions
from pipeline.types import SelectedCandidate
from pipeline.types import UploadedImage
from pipeline.utils import image_to_url
from pipeline.utils import image_dimensions
from pipeline.utils import pad_uploaded_images
from pipeline.utils import url_to_image
from pipeline.yolo import run_yolo


def run_analyze_pipeline(uploaded_files: list[UploadedImage], options: PipelineOptions) -> dict:
    """Run preprocess -> denoise -> SR/OCR route -> OCR."""

    source_images = _prepare_sources(uploaded_files, options)
    if options.input_mode == "crop" or _looks_like_pre_cropped(source_images, options):
        candidates = _prepare_crop_candidates(source_images, options)
    else:
        yolo_candidates = run_yolo(source_images)
        
        # [검증 파이프라인 수동 옵션] is_validation == True 인 경우, 16x(32~48) 크기 필터를 타이트하게 강제 적용
        # (세로 12~20 & 가로 28~54 & 비율 2배 이상 만족하는 것만 검출로 남김)
        if options.is_validation:
            filtered_candidates = []
            for yc in yolo_candidates:
                w = yc.bbox_width
                h = yc.bbox_height
                # 가로가 세로 대비 2배 이상이면서, 픽셀수(면적)는 1000 이하인 경우만 통과
                if yc.detected and (w >= h * 2.0) and (w * h <= 1000):
                    filtered_candidates.append(yc)
                else:
                    from pipeline.types import YoloCandidate
                    filtered_candidates.append(
                        YoloCandidate(
                            source=yc.source, crop=yc.crop, highlighted=yc.highlighted,
                            detected=False, detection_area=0, bbox_width=0, bbox_height=0,
                            crop_width=yc.crop_width, crop_height=yc.crop_height,
                            confidence=0.0, source_index=yc.source_index
                        )
                    )
            yolo_candidates = filtered_candidates

        candidates = select_top_candidates(
            yolo_candidates=yolo_candidates,
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
        sr_urls = run_sr(denoised_images, denoised_images[0], options)
        ocr_text = run_ocr(url_to_image(sr_urls) or denoised_images)
        pipeline_route = "sr_then_ocr"
    else:
        sr_urls = image_to_url(denoised_images)
        ocr_text = run_ensemble_ocr(denoised_images)
        pipeline_route = "ocr_ensemble"

    from pipeline.ocr.ocr_model import run_batch_ocr
    if sr_applied:
        sr_images = url_to_image(sr_urls) or denoised_images
        yolo_ocr_preds = run_batch_ocr(sr_images)
    else:
        yolo_ocr_preds = run_batch_ocr(denoised_images)

    return {
        "input_preview": image_to_url(_build_input_preview(source_images, options)),
        "input_omitted_count": max(0, len(source_images) - options.output_slots),
        "selected_inputs": image_to_url([candidate.source for candidate in candidates]),
        "selected_source_indices": [candidate.source_index for candidate in candidates],
        "selected_plate_bboxes": [_candidate_bbox(candidate) for candidate in candidates],
        "yolo_crops": image_to_url(crop_images),
        "yolo_selected": image_to_url([candidate.highlighted for candidate in candidates]),
        "denoised": denoised_urls,
        "sr": sr_urls,
        "ocr_text": ocr_text,
        "yolo_ocr_preds": yolo_ocr_preds,
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


def _prepare_crop_candidates(
    crop_images: list[UploadedImage],
    options: PipelineOptions,
) -> list[SelectedCandidate]:
    """Treat already-cropped plate images as selected candidates."""

    selected = pad_uploaded_images(crop_images[: options.output_slots], options.output_slots)
    candidates: list[SelectedCandidate] = []
    for index, crop in enumerate(selected):
        width, height = image_dimensions(crop)
        area = max(0, width) * max(0, height)
        candidates.append(
            SelectedCandidate(
                source=crop,
                crop=crop,
                highlighted=crop,
                width=width,
                height=height,
                area=area,
                detected=True,
                detection_area=area,
                bbox_width=width,
                bbox_height=height,
                confidence=1.0,
                source_index=index,
            )
        )
    return candidates


def _looks_like_pre_cropped(
    source_images: list[UploadedImage],
    options: PipelineOptions,
) -> bool:
    """Auto-skip YOLO when image inputs are already no larger than HR plate size."""

    if options.input_mode != "image" or not source_images:
        return False

    dimensions = [image_dimensions(image) for image in source_images]
    return all(
        0 < width <= options.hr_width and 0 < height <= options.hr_height
        for width, height in dimensions
    )


def _build_input_preview(source_images: list[UploadedImage], options: PipelineOptions) -> list[UploadedImage]:
    """Return the first source frames for frontend preview slots."""

    preview = source_images[: options.output_slots]
    return pad_uploaded_images(preview, options.output_slots)


def _candidate_bbox(candidate) -> dict:
    """Expose original-frame YOLO bbox metadata for frontend detail views."""

    return {
        "detected": candidate.detected,
        "width": candidate.bbox_width,
        "height": candidate.bbox_height,
        "area": candidate.detection_area,
        "confidence": candidate.confidence,
        "source_index": candidate.source_index,
    }
