@echo off
title Motorcycle DSS - Backend API (FastAPI)
cd /d "%~dp0"

echo ========================================================
echo   KHOI DONG BACKEND REST API (FASTAPI + UVICORN)
echo   Dia chi API:     http://127.0.0.1:8000
echo   Swagger UI Docs: http://127.0.0.1:8000/docs
echo   Health Check:    http://127.0.0.1:8000/health
echo ========================================================
echo.

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

pause
