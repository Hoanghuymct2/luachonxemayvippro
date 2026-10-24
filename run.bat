@echo off
title Motorcycle DSS - He Ho Tro Ra Quyet Dinh
cd /d "%~dp0"

echo ========================================================
echo   HE HO TRO RA QUYET DINH GOI Y XE MAY (KNN + GEMINI)
echo   Giao dien Streamlit truc quan & Bao mat
echo ========================================================
echo.
echo Dang khoi dong may chu va mo trinh duyet tai http://localhost:8501 ...
echo.

python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANH BAO] Co loi khi khoi dong. Dang kiem tra dependencies...
    pip install -r requirements.txt
    python -m streamlit run app.py
)

pause
