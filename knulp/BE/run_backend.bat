@echo off
chcp 65001
echo [Backend] Conda 가상환경(rgdiffsr) 활성화 및 FastAPI 실행 중...
call conda activate rgdiffsr
if %errorlevel% neq 0 (
    echo [경고] conda activate rgdiffsr 실패. conda가 path에 없거나 가상환경명이 다를 수 있습니다.
    echo 가상환경이 켜지지 않았다면 conda activate rgdiffsr를 수동으로 입력해주세요.
)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
