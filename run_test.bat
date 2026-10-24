@echo off
title Motorcycle DSS - Kiem Thu Tu Dong (Pytest)
cd /d "%~dp0"

echo ========================================================
echo   CHAY TOAN BO 39 TEST CASES KIEM THU TU DONG (PYTEST)
echo   (Bao gom Unit tests, Math verification va API tests)
echo ========================================================
echo.

python -m pytest -v

echo.
echo ========================================================
pause
