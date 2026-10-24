@echo off
title Motorcycle DSS - He Ho Tro Ra Quyet Dinh
cd /d "%~dp0"

echo ========================================================
echo   HE HO TRO RA QUYET DINH GOI Y XE MAY (DSS MULTI-TIER)
echo   FastAPI Backend + Weighted KNN + Gemini LLM + Streamlit
echo ========================================================
echo.
echo [1/2] Dang khoi dong Backend API tai http://127.0.0.1:8000 ...
start "Motorcycle DSS Backend API" python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
timeout /t 2 /nobreak >nul

echo [2/2] Dang khoi dong Giao dien Web Streamlit tai http://localhost:8501 ...
echo.
echo - Web UI:        http://localhost:8501
echo - Backend API:   http://127.0.0.1:8000
echo - Swagger Docs:  http://127.0.0.1:8000/docs
echo.

python -m streamlit run web/app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANH BAO] Co loi khi khoi dong. Dang kiem tra dependencies...
    pip install -r requirements.txt
    python -m streamlit run web/app.py
)

pause
