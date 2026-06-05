@echo off
chcp 65001
echo ==================================================
echo  BE (FastAPI) 및 FE (Vite React) 서버 일괄 실행
echo ==================================================

echo 백엔드 실행 창을 띄웁니다...
start "Backend FastAPI Server" cmd /k "cd /d %~dp0BE && run_backend.bat"

echo 프론트엔드 실행 창을 띄웁니다...
start "Frontend Vite Server" cmd /k "cd /d %~dp0FE && run_frontend.bat"

echo 두 서버가 별도 창으로 실행되었습니다. 이 창은 닫으셔도 됩니다.
timeout /t 5
