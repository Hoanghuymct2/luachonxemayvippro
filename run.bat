@echo off
title Motorcycle DSS - He Ho Tro Ra Quyet Dinh
cd /d "%~dp0"

echo ========================================================
echo   HE HO TRO RA QUYET DINH GOI Y XE MAY (KNN + GEMINI)
echo ========================================================
echo.
echo Dang khoi dong ung dung Streamlit tren trinh duyet...
echo Dia chi web: http://localhost:8501
echo.

python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANH BAO] Phat hien thieu thu vien hoac moi truong chua san sang.
    echo Dang tien hanh cai dat bo sung tu requirements.txt...
    echo.
    pip install -r requirements.txt
    echo.
    echo Dang thu khoi dong lai...
    python -m streamlit run app.py
)

pause

