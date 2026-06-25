"""OCR module package."""

from pipeline.ocr.ensemble import run_ensemble_ocr
from pipeline.ocr.ocr_model import run_ocr

__all__ = ["run_ocr", "run_ensemble_ocr"]
