@echo off
cd /d "%~dp0"
title CorridorKey 2.0 GUI

set "PATH=%~dp0backend\ffmpeg;%~dp0backend\python;%PATH%"
set "PYTHON=%~dp0backend\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Python environment not found at: %PYTHON%
    echo Please make sure the backend\venv folder is present.
    pause
    exit /b 1
)

"%PYTHON%" backend\main.py %*
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
