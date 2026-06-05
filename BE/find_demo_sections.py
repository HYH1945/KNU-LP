import os
import sys
import cv2
import numpy as np
import re
from datetime import datetime

# BE 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.yolo import run_yolo
from pipeline.types import UploadedImage
from pipeline.ocr.ocr_model import run_ocr
from pipeline.ocr.ensemble import vote_texts
from pipeline.utils import decode_image

CACHE_INTERVAL_SEC = 0.1   # [1단계] 프레임 사전 추론 간격 (0.1초마다 1프레임씩 촘촘히 스캔해둡니다.)
WINDOW_DURATION = 1.0      # [2단계] 비디오 슬라이스 윈도우 크기 (1.0초)
SLIDE_INTERVAL_SEC = 1.0   # [2단계] 슬라이딩 윈도우 이동 간격 (1.0초 단위로 쪼개기)
OUTPUT_SLOTS = 5           # 0.5초 내에서 최종 앙상블에 사용하는 프레임 수 (5장)

def calculate_blur(crop_img_uploaded):
    img = decode_image(crop_img_uploaded, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def select_top_candidates_local(candidates, output_slots=5):
    detected_candidates = [c for c in candidates if c["detected"]]
    if not detected_candidates:
        return []
    detected_candidates.sort(
        key=lambda x: (x["bbox_area"], x["confidence"], -x["sec"]),
        reverse=True
    )
    return detected_candidates[:output_slots]

def trim_video(video_path, start_sec, end_sec, output_path):
    """
    OpenCV를 사용하여 원본 비디오에서 지정된 구간(start_sec ~ end_sec)을 잘라내어 .mp4 비디오 파일로 저장합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Trim] Error: Failed to open source video {video_path}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    curr_frame = start_frame
    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        curr_frame += 1
        
    cap.release()
    out.release()
    return True

def analyze_video_cached(video_path, hr_threshold):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return []

    print(f"\n[Step 1] Pre-Caching Video Frames: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f}s")

    cached_results = []
    time_steps = np.arange(0.0, duration_sec, CACHE_INTERVAL_SEC)

    from tqdm import tqdm
    for sec in tqdm(time_steps, desc="Inferencing Frames", unit="frame"):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        _, encoded = cv2.imencode('.jpg', frame)
        uploaded_img = UploadedImage(data=encoded.tobytes(), content_type="image/jpeg")

        candidates = run_yolo([uploaded_img])
        if not candidates:
            continue

        candidate = candidates[0]
        detected = candidate.detected
        
        if detected:
            w = candidate.bbox_width
            h = candidate.bbox_height
            # 사용자의 요청: 가로가 세로 대비 2배 이상이면서 픽셀수는 1000 이하인 번호판
            if not (w >= h * 2.0 and w * h <= 1000):
                detected = False
                
        if detected:
            blur_val = calculate_blur(candidate.crop)
            bbox_area = w * h
            is_hr = bbox_area >= hr_threshold
            ocr_text = run_ocr([candidate.crop])
            
            cached_results.append({
                "sec": float(sec),
                "detected": True,
                "confidence": candidate.confidence,
                "bbox_size": (candidate.bbox_width, candidate.bbox_height),
                "bbox_area": bbox_area,
                "is_hr": is_hr,
                "blur_val": blur_val,
                "ocr_text": ocr_text,
                "crop_data": candidate.crop.data
            })
        else:
            cached_results.append({
                "sec": float(sec),
                "detected": False,
                "confidence": 0.0,
                "bbox_size": (0, 0),
                "bbox_area": 0,
                "is_hr": False,
                "blur_val": 0.0,
                "ocr_text": "NONE",
                "crop_data": None
            })

    cap.release()
    return cached_results

def find_perfect_3s_segments(cached_results, hr_threshold):
    """
    2단계: 캐싱된 결과를 활용해 0.5초 윈도우 슬라이딩을 수행합니다.
    YOLO 번호판 검출 및 유효한 OCR 결과가 나온 모든 고유 차량 세그먼트를 1등부터 끝까지 전부 추출합니다.
    """
    if not cached_results:
        return []

    print(f"\n[Step 2] Sliding {WINDOW_DURATION}s windows to search for candidate segments...")
    duration_sec = cached_results[-1]["sec"]
    
    # 윈도우 시작점들을 SLIDE_INTERVAL_SEC 단위로 슬라이딩
    window_starts = np.arange(0.0, max(0.0, duration_sec - WINDOW_DURATION), SLIDE_INTERVAL_SEC)
    perfect_segments = []
    backup_segments = []

    for start in window_starts:
        end = start + WINDOW_DURATION
        window_frames = [r for r in cached_results if start <= r["sec"] <= end]
        
        # 0.5초 구간 안에서 YOLO 검출이 된 것들 중 상위 5개 후보 선정
        top_candidates = select_top_candidates_local(window_frames, OUTPUT_SLOTS)
        
        if not top_candidates:
            continue
            
        pred_texts = [c["ocr_text"] for c in top_candidates]
        final_ensemble_ocr = vote_texts(pred_texts)
        
        if final_ensemble_ocr == "UNKNOWN" or final_ensemble_ocr == "NONE":
            continue
            
        match_count = sum(1 for text in pred_texts if text == final_ensemble_ocr)
        
        avg_blur = np.mean([x["blur_val"] for x in top_candidates])
        avg_conf = np.mean([x["confidence"] for x in top_candidates])
        hr_ratio = sum([1 for x in top_candidates if x["is_hr"]]) / len(top_candidates)
        
        # 점수 산정
        score = (avg_blur / 50.0) + (avg_conf * 10.0) + (hr_ratio * 5.0) + (match_count * 2.0)
        
        seg_data = {
            "start": start,
            "end": end,
            "ensemble_ocr": final_ensemble_ocr,
            "avg_blur": avg_blur,
            "hr_ratio": hr_ratio,
            "avg_conf": avg_conf,
            "score": score,
            "match_count": match_count,
            "all_frames": top_candidates
        }
        
        if match_count == len(top_candidates): # 100% 일치
            perfect_segments.append(seg_data)
        elif match_count >= (len(top_candidates) * 0.6): # 60% 이상 일치 (백업용)
            backup_segments.append(seg_data)
            
    # 중복 제거 및 대표 구간 선정
    # 모든 발견된 고유 번호판 중에서 각 번호판별로 가장 점수가 높은 베스트 0.5초 구간을 1등부터 끝까지 전부 모읍니다.
    from collections import defaultdict
    all_grouped = defaultdict(list)
    
    # 100% 일치 세그먼트 그룹화
    for seg in perfect_segments:
        all_grouped[seg["ensemble_ocr"]].append(seg)
        
    # 일부 오차 세그먼트 그룹화 (아직 완벽 그룹에 없는 고유 번호판들만 추가하여 데이터 다양성 확보)
    for seg in backup_segments:
        if seg["ensemble_ocr"] not in all_grouped:
            all_grouped[seg["ensemble_ocr"]].append(seg)
            
    # 각 고유 번호판별로 가장 점수가 우수한 대표 구간 1개씩 선정
    final_segs = []
    for ocr_txt, segs in all_grouped.items():
        final_segs.append(max(segs, key=lambda x: x["score"]))
        
    # 만약 수집된 고유 번호판 세그먼트 수가 적다면, 시간대가 다른 중복 세그먼트들도 점수순으로 차례대로 덧붙여 확보합니다.
    # 이를 통해 최대한 영상 속 모든 주행 순간의 비디오/이미지를 긁어모읍니다.
    all_candidates = perfect_segments + backup_segments
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    for seg in all_candidates:
        # 시간대가 완전히 겹치지 않는(독립적인) 차순위 세그먼트들을 추가
        if not any(abs(x["start"] - seg["start"]) < 1.0 for x in final_segs):
            final_segs.append(seg)
                
    final_segs.sort(key=lambda x: x["score"], reverse=True)
    return final_segs

def main():
    hr_width = 96
    hr_height = 32
    try:
        import yaml
        with open("configs/settings.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            hr_width = cfg.get("yolo", {}).get("plate", {}).get("output_width", 96)
            hr_height = cfg.get("yolo", {}).get("plate", {}).get("output_height", 32)
    except Exception:
        pass
    
    hr_threshold = hr_width * hr_height * 0.8
    print(f"HR Plate Area Threshold: {hr_threshold:.1f} ({hr_width}x{hr_height} * 0.8)")

    # 현재 날짜 및 시간을 이용한 고유 타임스탬프 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BE_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BE_ROOT)
    
    video_root = os.path.join(PROJECT_ROOT, "vidieo")
    output_crop_dir = os.path.join(BE_ROOT, "test_outputs", f"demo_crops_{timestamp}")
    os.makedirs(output_crop_dir, exist_ok=True)
    print(f"Output cropped images/videos will be saved to: {output_crop_dir}")

    target_video = os.path.join(video_root, "fullvideo_720.mp4")
    if not os.path.exists(target_video):
        found = False
        for root, dirs, files in os.walk(video_root):
            for f in files:
                if f.startswith("fullvideo_720") and f.endswith(".mp4"):
                    target_video = os.path.join(root, f)
                    found = True
                    break
            if found:
                break
                
        if not found:
            print(f"Error: Target video 'fullvideo_720.mp4' not found.")
            return

    # 리포트 파일도 날짜 시간명을 붙여 고유하게 폴더 내부에 생성
    report_path = os.path.join(output_crop_dir, f"video_analysis_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("=== KNU Plate Detection Perfect 2.0s Demo Segments Report ===\n")
        rf.write("■ 실험 목적: 고화질(HR) 영상을 분석해 오인식 없이 100% 일치하게 추출된 구간을 확보하고 이를 GT(Ground Truth)로 정의합니다.\n")
        rf.write("■ 검증 방법: 본 리포트의 구간(Start ~ End)을 참고하여 저화질(LR) 비디오의 동일 구간을 잘라 파이프라인에 입력한 뒤,\n")
        rf.write("             SR 복원 후의 OCR 결과가 본 리포트의 GT(Ensemble OCR Output)와 일치하는지 판별하여 SR의 성공 여부를 검증하십시오.\n\n")
        rf.write(f"Target Video: {os.path.basename(target_video)}\n\n")
        
        # 1단계: 캐싱 및 추론
        cached_results = analyze_video_cached(target_video, hr_threshold)
        if not cached_results:
            print("No frames processed.")
            return
            
        # 2단계: 0.5초 윈도우 슬라이딩 및 필터링 없이 전체 구간 목록 추출
        perfect_segs = find_perfect_3s_segments(cached_results, hr_threshold)
        
        rf.write(f"Found {len(perfect_segs)} unique 2.0s segments in total:\n\n")
        print(f"\n--- Found {len(perfect_segs)} Unique 2.0s Segments (Saving all, 1 to End) ---")
        
        for idx, seg in enumerate(perfect_segs):
            if seg['hr_ratio'] >= 0.7:
                res_type = "HR (High Resolution)"
            elif seg['hr_ratio'] <= 0.3:
                res_type = "LR (Low Resolution - SR 필수 대상)"
            else:
                res_type = "Mixed"
                
            log_str = (
                f"Rank {idx+1}: {seg['start']:.1f}s ~ {seg['end']:.1f}s (Duration: 2.0s) - Type: {res_type}\n"
                f"  - Ensemble OCR Output: '{seg['ensemble_ocr']}' (Match Rate: {seg['match_count']}/5 candidates)\n"
                f"  - Avg Sharpness (Blur Value): {seg['avg_blur']:.1f}\n"
                f"  - HR Plates Ratio: {seg['hr_ratio']*100:.1f}%\n"
                f"  - Recommendation Score: {seg['score']:.2f}\n"
            )
            print(log_str)
            rf.write(log_str + "\n")
            
            sanitized_ocr = sanitize_filename(seg['ensemble_ocr'])
            
            # 2.0초 분량 비디오 크롭 저장
            start_str = f"{seg['start']:.1f}".replace('.', 'p')
            end_str = f"{seg['end']:.1f}".replace('.', 'p')
            video_filename = f"rank{idx+1}_video_sec{start_str}_{end_str}_{sanitized_ocr}.mp4"
            video_output_path = os.path.join(output_crop_dir, video_filename)
            
            try:
                trim_video(target_video, seg["start"], seg["end"], video_output_path)
            except Exception as e:
                print(f"Failed to trim and save video segment: {e}")
            
            # 구간 내의 후보 프레임의 크롭 이미지 전체 저장
            for c_idx, c_frame in enumerate(seg["all_frames"]):
                res_suffix = "HR" if c_frame["is_hr"] else "LR"
                sec_str = f"{c_frame['sec']:.1f}".replace('.', 'p')
                
                crop_filename = f"rank{idx+1}_cand{c_idx+1}_sec{sec_str}_{sanitized_ocr}_{res_suffix}_sharp{int(c_frame['blur_val'])}.jpg"
                crop_filepath = os.path.join(output_crop_dir, crop_filename)
                
                try:
                    if c_frame["crop_data"] is not None:
                        with open(crop_filepath, "wb") as cf:
                            cf.write(c_frame["crop_data"])
                except Exception as e:
                    print(f"Failed to save candidate crop: {e}")
                
            rf.write("  Candidate Frames Details:\n")
            for c_frame in seg["all_frames"]:
                rf.write(
                    f"    {c_frame['sec']:.1f}s: BBox={c_frame['bbox_size']}, Sharpness={c_frame['blur_val']:.1f}, "
                    f"Conf={c_frame['confidence']:.2f}, OCR='{c_frame['ocr_text']}'\n"
                )
            rf.write("\n")
            
    print(f"\nAnalysis complete. Detailed report saved to: {os.path.abspath(report_path)}")
    print(f"Check representative cropped preview images/videos in: {output_crop_dir}")

if __name__ == "__main__":
    main()
