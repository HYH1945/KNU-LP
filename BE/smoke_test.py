import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline.orchestrator import run_analyze_pipeline
from pipeline.types import PipelineOptions
from pipeline.types import UploadedImage


def _resolve_video_path() -> Path | None:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()

    env_path = os.environ.get("KNU_LP_SMOKE_VIDEO")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return None


def run_test() -> bool:
    print("Starting smoke test for pipeline...")

    video_path = _resolve_video_path()
    if video_path is None:
        print("Usage: python smoke_test.py <video_path>")
        print("Or set KNU_LP_SMOKE_VIDEO to a local mp4 path.")
        return False

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return False

    print(f"Loading video from {video_path}...")
    uploaded_video = UploadedImage(
        data=video_path.read_bytes(),
        content_type="video/mp4",
    )

    options = PipelineOptions(
        input_mode="video",
        hr_width=96,
        hr_height=32,
        video_start=0.0,
        video_end=2.0,
        sr_mode="auto",
        denoise_enabled=True,
    )

    try:
        print("Running pipeline...")
        result = run_analyze_pipeline([uploaded_video], options)
        print("Pipeline ran successfully!")
        print("-" * 40)
        print(f"Pipeline Route: {result.get('pipeline_route')}")
        print(f"OCR Result: {result.get('ocr_text')}")
        print(f"Batch OCR Predictions: {result.get('yolo_ocr_preds')}")
        print(f"SR Applied: {result.get('sr_applied')}")
        print(f"Denoised Images Count: {len(result.get('denoised', []))}")
        print("-" * 40)
        print("Smoke test PASSED.")
        return True
    except Exception:
        print("Smoke test FAILED with exception:")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
