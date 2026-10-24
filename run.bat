@echo off
title Motorcycle DSS - He Ho Tro Ra Quyet Dinh
cd /d "%~dp0"

echo ========================================================
echo   HE HO TRO RA QUYET DINH GOI Y XE MAY (KNN + GEMINI)
echo   Giao dien Streamlit truc quan & Bao mat
echo ========================================================
echo.
echo Dang khoi dong may chu va tu dong mo trinh duyet...
echo Dia chi Web: http://localhost:8501
echo.

:: Tu dong mo trinh duyet web cua Windows sau 2 giay
start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:8501"

python -m streamlit run app.py --server.headless=false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANH BAO] Co loi khi khoi dong. Dang kiem tra dependencies...
    pip install -r requirements.txt
    start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:8501"
    python -m streamlit run app.py --server.headless=false
)

pause
