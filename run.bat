@echo off
title SmartSearch AI - Hybrid Search Engine
echo ========================================================
echo               SMARTSEARCH AI LAUNCHER
echo ========================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [INFO] Installing required dependencies...
    call .\venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .\venv\Scripts\activate.bat
)

echo.
echo [INFO] Starting SmartSearch AI on http://localhost:8000 ...
echo [INFO] Press Ctrl+C in this terminal to stop the server.
echo.

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
