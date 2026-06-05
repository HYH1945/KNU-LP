"""Backward-compatible SR wrapper."""

from pipeline.superresolution.sr_model import run_sr as _run_sr
from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage


def run_sr(
    images: list[UploadedImage],
    placeholder_hr: UploadedImage | None = None,
    options: PipelineOptions | None = None,
) -> list[str]:
    """Run the current SR adapter with defaults for older callers."""

    return _run_sr(images, placeholder_hr, options or PipelineOptions())
