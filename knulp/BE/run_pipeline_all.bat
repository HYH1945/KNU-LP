@echo off
chcp 65001
echo =========================================================
echo  [KNU-LP] 1초 간격 720p 극소형 번호판 SR 검증 파이프라인
echo =========================================================

echo [1/3] 720p 비디오에서 16x32 이하의 작은 번호판 구간들을 1.0초 단위로 촘촘히 검출하는 중...
python find_demo_sections.py
if %errorlevel% neq 0 (
    echo [에러] find_demo_sections.py 수행 실패!
    pause
    exit /b %errorlevel%
)

echo [2/3] 검출된 구간들에 대해 720p 원본 비디오 조각(.mp4)들을 컷팅하는 중...
python trim_ranked_videos.py
if %errorlevel% neq 0 (
    echo [에러] trim_ranked_videos.py 수행 실패!
    pause
    exit /b %errorlevel%
)

echo [3/3] 잘라진 720p 영상 조각들을 입력으로 넣어 강제로 디퓨전(SR) 및 OCR 검증 수행하는 중...
python run_validation_pipeline.py
if %errorlevel% neq 0 (
    echo [에러] run_validation_pipeline.py 수행 실패!
    pause
    exit /b %errorlevel%
)

echo =========================================================
echo  파이프라인의 모든 검증 프로세스가 성공적으로 완료되었습니다!
echo  결과 요약 리포트는 최신 demo_crops_[TIMESTAMP]/pipeline_results/validation_summary_report.txt 에 저장됩니다.
echo =========================================================
pause
