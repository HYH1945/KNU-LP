import os
import re
import cv2
from collections import defaultdict

# VLM(AI)이 선별해 준 정예 등수들만 정밀 트리밍하려면 여기에 등수 리스트를 입력하세요.
# 예: TARGET_RANKS = [1, 5, 12, 19]
# 비어 있으면 (즉, []) 폴더에 있는 모든 등수를 전체 처리합니다.
TARGET_RANKS = []

def trim_video(video_path, start_sec, end_sec, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open {video_path}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_frame = max(0, int(start_sec * fps))
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

def main():
    BE_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BE_ROOT)
    outputs_parent = os.path.join(BE_ROOT, "test_outputs")
    video_root = os.path.join(PROJECT_ROOT, "vidieo")
    
    # 720p 비디오 소스 경로만 설정 (720p 기준으로만 구동)
    video_sources = {
        "720": os.path.join(video_root, "fullvideo_720.mp4")
    }
    
    # 가장 최근에 분석 완료된 demo_crops_XXXXXX 폴더를 자동으로 감지합니다.
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
        print(f"Crops directory not found: {crops_dir}")
        return
        
    print(f"\n[Video Trimmer] Target Crops Directory: {crops_dir}")
    files = os.listdir(crops_dir)
    
    # rank별로 sec 리스트를 파싱하여 모음 (파일명에 sec_str이 floatp형식이거나 정수형인 경우 모두 파싱)
    # 패턴: rank(\d+)_cand\d+_sec(\d+p\d+|\d+)_
    pattern = re.compile(r"rank(\d+)_cand\d+_sec(\d+p\d+|\d+)_")
    backup_pattern = re.compile(r"fullvideo_720_rank(\d+)_sec(\d+p\d+|\d+)_")
    
    rank_secs = defaultdict(list)
    
    for f in files:
        match = pattern.match(f)
        if not match:
            match = backup_pattern.match(f)
            
        if match:
            rank = int(match.group(1))
            sec_str = match.group(2).replace('p', '.')
            sec = float(sec_str)
            rank_secs[rank].append(sec)
            
    if not rank_secs:
        print("No ranked candidate files matched the pattern in the folder.")
        return
        
    # 필터링 적용 메시지
    if TARGET_RANKS:
        print(f"Active Filter: Processing only specified Ranks {TARGET_RANKS}")
    else:
        print("Processing all ranks detected in the folder.")
        
    print(f"Found {len(rank_secs)} unique ranks to slice.")
    
    for rank, secs in sorted(rank_secs.items()):
        # TARGET_RANKS 필터링 수행
        if TARGET_RANKS and rank not in TARGET_RANKS:
            continue
            
        min_sec = min(secs)
        max_sec = max(secs)
        
        # 0.5초 크롭 타임 윈도우 셋팅 (안전하게 감지 구간 시작과 끝을 감싸도록 설정)
        start_sec = max(0.0, float(min_sec))
        end_sec = float(max_sec) + 0.1  # 0.5초의 마지막 부분까지 온전히 포함하도록 미세 보정
        
        # 만약 구간이 너무 짧거나 0초인 경우 최소 1.0초 분량을 보장
        if end_sec - start_sec < 1.0:
            end_sec = start_sec + 1.0
            
        print(f"Rank {rank}: Slicing range {start_sec:.1f}s ~ {end_sec:.1f}s (Duration: {end_sec - start_sec:.2f}s)")
        
        start_str = f"{start_sec:.1f}".replace('.', 'p')
        end_str = f"{end_sec:.1f}".replace('.', 'p')
        
        # 각 해상도 비디오 소스가 존재하면 슬라이스 수행
        for res, v_src in video_sources.items():
            if os.path.exists(v_src):
                out_name = f"rank{rank}_fullvideo_{res}_sec{start_str}_{end_str}.mp4"
                out_path = os.path.join(crops_dir, out_name)
                print(f"  -> Trimming fullvideo_{res}...")
                trim_video(v_src, start_sec, end_sec, out_path)
            else:
                print(f"  -> Warning: Source video '{os.path.basename(v_src)}' not found in '{video_root}'. Skipping.")

    print("\nTrimming complete for all available resolutions!")

if __name__ == "__main__":
    main()
