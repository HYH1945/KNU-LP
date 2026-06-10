"""Video frame extraction helpers."""

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile

import cv2

from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage
from pipeline.utils import opencv_to_uploaded


def extract_video_frames(video: UploadedImage, options: PipelineOptions) -> list[UploadedImage]:
    """Extract frames from the selected time range and return PNG images."""

    suffix = _suffix_from_content_type(video.content_type)
    temp_path = ""
    capture = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video.data)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("failed to open uploaded video")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = max(int(options.video_start * fps), 0)
        if options.video_end is None:
            end_frame = total_frames - 1 if total_frames > 0 else start_frame
        else:
            end_frame = int(options.video_end * fps)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)
        if end_frame < start_frame:
            raise ValueError("video_end must be greater than video_start")

        frame_count = max(end_frame - start_frame + 1, 1)
        stride = max(frame_count // options.max_video_frames, 1)

        frames: list[UploadedImage] = []
        frame_index = start_frame
        while frame_index <= end_frame and len(frames) < options.max_video_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                break
            frames.append(opencv_to_uploaded(frame))
            frame_index += stride

        if not frames:
            raise ValueError("no frames were extracted from the uploaded video")
        return frames
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _suffix_from_content_type(content_type: str) -> str:
    suffix_map = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/ogg": ".ogg",
    }
    return suffix_map.get(content_type, ".mp4")
