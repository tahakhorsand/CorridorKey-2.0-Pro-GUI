@echo off
cd /d "%~dp0"
title CorridorKey 2.0 GUI

set "PATH=%~dp0backend\ffmpeg;%~dp0backend\python;%PATH%"
set "PYTHON=%~dp0backend\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    if exist "%~dp0backend\python\python.exe" (
        set "PYTHON=%~dp0backend\python\python.exe"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON=python"
        ) else (
            echo [ERROR] Python environment not found at: %PYTHON%
            echo Please make sure Python is installed or create the virtual environment:
            echo   python -m venv backend\venv
            echo   backend\venv\Scripts\pip install -r backend\requirements.txt
            pause
            exit /b 1
        )
    )
)

"%PYTHON%" backend\main.py %*
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
