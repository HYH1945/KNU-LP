@echo off
chcp 65001
echo =========================================================
echo  [KNU-LP] SR validation pipeline
echo =========================================================

echo [1/3] Finding candidate demo sections...
python find_demo_sections.py
if %errorlevel% neq 0 (
    echo [Error] find_demo_sections.py failed.
    pause
    exit /b %errorlevel%
)

echo [2/3] Trimming ranked video clips...
python trim_ranked_videos.py
if %errorlevel% neq 0 (
    echo [Error] trim_ranked_videos.py failed.
    pause
    exit /b %errorlevel%
)

echo [3/3] Running validation pipeline...
python run_validation_pipeline.py
if %errorlevel% neq 0 (
    echo [Error] run_validation_pipeline.py failed.
    pause
    exit /b %errorlevel%
)

echo =========================================================
echo  Validation pipeline finished.
echo =========================================================
pause
