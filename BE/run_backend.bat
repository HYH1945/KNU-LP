@echo off
chcp 65001
echo [Backend] Activating conda environment and starting FastAPI...
call conda activate rgdiffsr
if %errorlevel% neq 0 (
    echo [Warning] Failed to activate conda environment: rgdiffsr
    echo If you use a different environment name, activate it manually before running this script.
)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
