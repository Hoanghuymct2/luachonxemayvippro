@echo off
title Motorcycle DSS - Web UI (Streamlit)
cd /d "%~dp0"

echo ========================================================
echo   KHOI DONG STREAMLIT WEB UI
echo   Dia chi Web UI:  http://localhost:8501
echo   Backend API ket noi: http://127.0.0.1:8000
echo ========================================================
echo.

python -m streamlit run web/app.py

pause
