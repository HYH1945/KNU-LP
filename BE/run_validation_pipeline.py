import os
import sys
import re
import shutil
from collections import defaultdict, Counter

# VLM(AI)이 선별해 준 정예 등수들만 정밀 검증하려면 여기에 등수 리스트를 입력하세요.
# 예: TARGET_RANKS = [1, 5, 12, 19]
# 비어 있으면 (즉, []) 폴더에 있는 모든 등수를 전체 처리합니다.
TARGET_RANKS = []

# BE 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import run_analyze_pipeline
from pipeline.types import PipelineOptions, UploadedImage
from pipeline.utils import url_to_image

def get_gt_labels(crops_dir):
    """
    폴더의 파일명들로부터 각 rank별 다수결 GT 번호판 텍스트를 유추합니다.
    """
    files = os.listdir(crops_dir)
    pattern = re.compile(r"rank(\d+)_cand\d+_sec(?:\d+p\d+|\d+)_(.+?)_(HR|LR)_sharp")
    backup_pattern = re.compile(r"fullvideo_720_rank(\d+)_sec(?:\d+p\d+|\d+)_(.+?)_(HR|LR)_sharp")
    
    rank_ocr_texts = defaultdict(list)
    for f in files:
        match = pattern.match(f)
        if not match:
            match = backup_pattern.match(f)
            
        if match:
            rank = int(match.group(1))
            ocr_text = match.group(2)
            rank_ocr_texts[rank].append(ocr_text)
            
    gt_labels = {}
    for rank, texts in rank_ocr_texts.items():
        if texts:
            most_common = Counter(texts).most_common(1)[0][0]
            gt_labels[rank] = most_common
    return gt_labels

def save_pipeline_images(url_list, output_dir, file_prefix):
    """
    파이프라인 결과 URL 리스트를 디코딩하여 이미지 파일로 저장합니다.
    """
    if not url_list:
        return
    images = url_to_image(url_list)
    if not images:
        return
    for idx, img in enumerate(images):
        save_path = os.path.join(output_dir, f"{file_prefix}_{idx}.jpg")
        try:
            with open(save_path, "wb") as f:
                f.write(img.data)
        except Exception as e:
            print(f"Failed to save image {save_path}: {e}")

def main():
    BE_ROOT = os.path.dirname(os.path.abspath(__file__))
    outputs_parent = os.path.join(BE_ROOT, "test_outputs")
    
    # 가장 최근에 생성된 demo_crops_XXXXXX 분석 폴더 자동 감지
    if not os.path.exists(outputs_parent):
        print(f"Parent outputs directory not found: {outputs_parent}")
        return
        
    candidates = [
        os.path.join(outputs_parent, d)
        for d in os.listdir(outputs_parent)
        if (d.startswith("demo_crops_") and os.path.isdir(os.path.join(outputs_parent, d)))
    ]
    
    if candidates:
        crops_dir = max(candidates, key=os.path.getmtime)
    else:
        crops_dir = os.path.join(outputs_parent, "demo_crops_o")
        
    if not os.path.exists(crops_dir):
        print(f"Target crops directory not found: {crops_dir}")
        return
        
    # 결과가 기록될 고유 폴더 내의 pipeline_results에 저장하여 실험 데이터를 깔끔하게 병합
    result_root_dir = os.path.join(crops_dir, "pipeline_results")
    os.makedirs(result_root_dir, exist_ok=True)
    
    print(f"\n[Validation Pipeline] Target Crops Directory: {crops_dir}")
    print(f"[Validation Pipeline] Outputs will be saved to: {result_root_dir}")
    
    # 1. 랭크별 GT 라벨 딕셔너리 생성
    gt_labels = get_gt_labels(crops_dir)
    if not gt_labels:
        print("Error: Could not find any GT labels from the target directory.")
        return
        
    print(f"Extracted {len(gt_labels)} Ground Truth labels.")
    
    # 2. 잘라진 비디오 조각들 목록 수집
    files = os.listdir(crops_dir)
    video_pattern = re.compile(r"rank(\d+)_fullvideo_(\d+)_sec.+?\.mp4")
    
    videos_to_process = []
    for f in files:
        match = video_pattern.match(f)
        if match:
            rank = int(match.group(1))
            resolution = int(match.group(2))
            video_path = os.path.join(crops_dir, f)
            videos_to_process.append({
                "rank": rank,
                "resolution": resolution,
                "path": video_path,
                "filename": f
            })
            
    if not videos_to_process:
        print("Error: No cropped mp4 video segments found in target folder.")
        print("Please run trim_ranked_videos.py first to generate the video segments.")
        return
        
    if TARGET_RANKS:
        print(f"Active Filter: Processing only specified Ranks {TARGET_RANKS}")
    else:
        print("Processing all ranks detected in the folder.")
        
    print(f"Found {len(videos_to_process)} video clips to process through the pipeline.")
    
    # 종합 성공 여부를 기록할 구조 정의
    # 구조: { rank_id: { "GT": 'xxxx', "HR": 'xxxx', "LR_results": { 480: { "ocr": 'xxxx', "sr_applied": True }, 320: ... } } }
    validation_records = defaultdict(lambda: {"GT": "", "HR": "", "LR_results": {}})
    
    # 3. 비디오별 파이프라인 작동
    for item in sorted(videos_to_process, key=lambda x: (x["rank"], x["resolution"])):
        rank = item["rank"]
        resolution = item["resolution"]
        vpath = item["path"]
        
        # TARGET_RANKS 필터링 수행
        if TARGET_RANKS and rank not in TARGET_RANKS:
            continue
            
        # GT 라벨 조회
        gt_label = gt_labels.get(rank, f"Rank{rank}_UNKNOWN")
        validation_records[rank]["GT"] = gt_label
        
        # 폴더 세분화 분류: {GT_label}/HR, {GT_label}/LR_480, {GT_label}/LR_320 등
        if resolution == 720:
            sub_folder = "HR"
        else:
            sub_folder = f"LR_{resolution}"
            
        target_dir = os.path.join(result_root_dir, gt_label, sub_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\nProcessing Rank {rank} ({resolution}p) -> GT: {gt_label} [{sub_folder}]")
        
        # [요청사항 반영] 나중에 웹 시연 시 찾기 쉽도록 원본 비디오 조각 파일을 해당 서브폴더 안에 복사해 넣어줌
        try:
            shutil.copy(vpath, os.path.join(target_dir, os.path.basename(vpath)))
            print(f"  -> Copied original video clip to target folder.")
        except Exception as e:
            print(f"  -> Warning: Failed to copy video clip: {e}")
        
        # 비디오 바이너리 로드
        try:
            with open(vpath, "rb") as vf:
                vdata = vf.read()
        except Exception as e:
            print(f"Failed to read video file {vpath}: {e}")
            continue
            
        uploaded_video = UploadedImage(data=vdata, content_type="video/mp4")
        
        # 실제 웹 시연과 100% 동일하게 'auto'로 설정하여 
        # 파이프라인 내부 자동 분기 알고리즘(should_use_sr)을 검증합니다.
        sr_mode = "auto"
        
        options = PipelineOptions(
            input_mode="video",
            hr_width=96,
            hr_height=32,
            video_start=0.0,
            video_end=None,
            sr_mode="auto",  # 원래 스펙대로 auto 모드로 자동 판단
            denoise_enabled=False,
            is_validation=True  # 검증 전용 16x(32~48) 픽셀 필터 작동
        )
        
        # 파이프라인 실행
        try:
            print("  -> Running run_analyze_pipeline...")
            result = run_analyze_pipeline([uploaded_video], options)
            print("  -> Execution successful.")
            
            ocr_text = result.get("ocr_text", "UNKNOWN")
            sr_applied = bool(result.get("sr_applied", False))
            yolo_ocr_preds = result.get("yolo_ocr_preds", [])
            
            # [수정 사항] 5개의 크롭 이미지 중 하나라도 매칭을 성공했는지 판단 (정답 기준은 gt_label)
            is_any_match = any(pred == gt_label for pred in yolo_ocr_preds)
            matched_ocr = gt_label if is_any_match else ocr_text
            
            # 종합 성공 판별 기록을 위해 보관
            if resolution == 720:
                validation_records[rank]["HR"] = matched_ocr
            else:
                validation_records[rank]["LR_results"][resolution] = {
                    "ocr": matched_ocr,
                    "sr_applied": sr_applied,
                    "all_preds": yolo_ocr_preds
                }
            
            # OCR 텍스트 결과 저장
            ocr_out_path = os.path.join(target_dir, "ocr_result.txt")
            with open(ocr_out_path, "w", encoding="utf-8") as f:
                f.write(f"GT Label: {gt_label}\n")
                f.write(f"Pipeline Result OCR (Ensemble): {ocr_text}\n")
                f.write(f"Pipeline Batch OCR Predictions: {yolo_ocr_preds}\n")
                f.write(f"Pipeline Route: {result.get('pipeline_route')}\n")
                f.write(f"SR Applied: {sr_applied}\n")
                f.write(f"Any Candidate Matched: {is_any_match}\n")
            print(f"  -> Saved OCR Result: '{ocr_text}' (Batch matched: {is_any_match})")
            
            # 이미지 파일들 복원하여 폴더에 저장
            # 1) YOLO 크롭 이미지들
            save_pipeline_images(result.get("yolo_crops"), target_dir, "yolo_crop")
            # 2) 디노이즈 처리된 이미지들
            save_pipeline_images(result.get("denoised"), target_dir, "denoised")
            # 3) 최종 결과 이미지 (LR의 경우 SR 복원 이미지, HR의 경우 원본 Denoise 크롭 이미지)
            save_pipeline_images(result.get("sr"), target_dir, "final_output")
            
            print(f"  -> Saved pipeline output images in {target_dir}")
            
        except Exception as e:
            print(f"  -> Error executing pipeline for {item['filename']}: {e}")
            
    # [요청사항 반영] 최종 종합 성공 검증 보고서 작성
    # 조건: SR이 적용(sr_applied == True)되었으며, 그 결과 OCR이 고화질(HR)로 구동한 GT OCR 결과와 100% 동일하게 일치한 경우 성공 표기
    summary_path = os.path.join(result_root_dir, "validation_summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("===================================================\n")
        sf.write("       KNU Plate SR Validation Matching Report     \n")
        sf.write("===================================================\n")
        sf.write("본 리포트는 [SR이 작동하여 성공적으로 복원된 결과]와 [HR 원본 결과]가\n")
        sf.write("완벽하게 일치(HR == SR)하는 검증 성공 클립들만 표시합니다.\n")
        sf.write("*참고: 파이프라인에서 SR이 적용되지 않았거나 매칭이 실패한 케이스는 패스되었습니다.\n\n")
        
        success_count = 0
        
        for rank, data in sorted(validation_records.items()):
            gt = data["GT"]
            hr_ocr = data["HR"]
            
            # 만약 스크립트에 HR 비디오가 누락되어 hr_ocr가 수집되지 않은 경우를 대비해 GT 라벨을 백업 매칭값으로 사용
            base_reference_ocr = hr_ocr if hr_ocr else gt
            
            for res, lr_data in sorted(data["LR_results"].items()):
                lr_ocr = lr_data["ocr"]
                sr_applied = lr_data["sr_applied"]
                
                # 조건식: SR을 적용했고, 최종 복원 OCR 텍스트가 HR 기준 텍스트와 100% 동일한 경우 성공!
                if sr_applied and lr_ocr == base_reference_ocr:
                    success_count += 1
                    ok_line = f"Rank {rank} (GT: {base_reference_ocr}) -> HR=SR({res}) [OK]\n"
                    sf.write(ok_line)
                    print(f"[SUCCESS] {ok_line.strip()}")
                    
        sf.write(f"\n===================================================\n")
        sf.write(f"Total Successful SR Demonstrations Found: {success_count}\n")
        sf.write("===================================================\n")
        
    print(f"\nAll pipelines processed. Check results in: {result_root_dir}")
    print(f"Validation summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
