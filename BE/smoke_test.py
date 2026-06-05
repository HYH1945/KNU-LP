import sys
import os

# BE 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import run_analyze_pipeline
from pipeline.types import PipelineOptions, UploadedImage

def run_test():
    print("Starting smoke test for pipeline...")
    
    # 테스트용 비디오 파일 경로
    video_path = r"c:\Experimental\knu-lp\vidieo\시연용 영상\lowvideo.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
        
    print(f"Loading video from {video_path}...")
    with open(video_path, 'rb') as f:
        video_data = f.read()
        
    uploaded_files = [
        UploadedImage(
            data=video_data,
            content_type="video/mp4"
        )
    ]
    
    # 2초 동안의 영상을 분석하는 옵션으로 실행
    options = PipelineOptions(
        input_mode="video",
        hr_width=96,
        hr_height=32,
        video_start=0.0,
        video_end=2.0,
        sr_mode="auto",
        denoise_enabled=True
    )
    
    try:
        print("Running pipeline...")
        result = run_analyze_pipeline(uploaded_files, options)
        print("Pipeline ran successfully!")
        print("-" * 40)
        print(f"Pipeline Route: {result.get('pipeline_route')}")
        print(f"OCR Result: {result.get('ocr_text')}")
        print(f"SR Applied: {result.get('sr_applied')}")
        print(f"Denoised Images Count: {len(result.get('denoised', []))}")
        print("-" * 40)
        print("Smoke test PASSED.")
        return True
    except Exception as e:
        print("Smoke test FAILED with exception:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
