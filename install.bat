@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title CorridorKey 2.0 Pro GUI - Installer

echo =======================================================
echo     CorridorKey 2.0 Pro GUI - Portable Environment Setup
echo =======================================================
echo.

set "BACKEND_DIR=%~dp0backend"
set "PYTHON_DIR=%BACKEND_DIR%\python"
set "VENV_DIR=%BACKEND_DIR%\venv"

:: 1. Ensure backend directory exists
if not exist "%BACKEND_DIR%" (
    mkdir "%BACKEND_DIR%"
)

:: 2. Check if embedded portable Python already exists
if not exist "%PYTHON_DIR%\python.exe" (
    echo [*] Setting up portable Python 3.12...
    echo [*] Downloading CPython 3.12 embedded runtime...
    mkdir "%PYTHON_DIR%" 2>nul
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip', '%BACKEND_DIR%\python_embed.zip')"
    if not exist "%BACKEND_DIR%\python_embed.zip" (
        echo [ERROR] Failed to download portable Python. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [*] Extracting Python...
    powershell -Command "Expand-Archive -Path '%BACKEND_DIR%\python_embed.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    del "%BACKEND_DIR%\python_embed.zip" 2>nul

    :: Enable site-packages in embedded python
    powershell -Command "$f = '%PYTHON_DIR%\python312._pth'; if (Test-Path $f) { (Get-Content $f) -replace '^#import site', 'import site' | Set-Content $f }"

    :: Download get-pip.py
    echo [*] Installing pip...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%BACKEND_DIR%\get-pip.py')"
    "%PYTHON_DIR%\python.exe" "%BACKEND_DIR%\get-pip.py" --no-warn-script-location
    del "%BACKEND_DIR%\get-pip.py" 2>nul
)

:: 3. Setup virtual environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [*] Initializing isolated virtual environment...
    "%PYTHON_DIR%\Scripts\pip.exe" install --no-warn-script-location virtualenv
    "%PYTHON_DIR%\Scripts\virtualenv.exe" "%VENV_DIR%"
)

:: 4. Install PyTorch with CUDA 12.4
echo [*] Installing PyTorch with CUDA 12.4 acceleration...
"%VENV_DIR%\Scripts\pip.exe" install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: 5. Install dependencies from requirements.txt
if exist "%BACKEND_DIR%\requirements.txt" (
    echo [*] Installing application dependencies...
    "%VENV_DIR%\Scripts\pip.exe" install --no-warn-script-location -r "%BACKEND_DIR%\requirements.txt"
)

echo.
echo =======================================================
echo [SUCCESS] CorridorKey 2.0 Pro GUI environment is ready!
echo Models and FFmpeg can now be downloaded directly in-app.
echo You can now launch the application using Run_GUI.bat
echo =======================================================
echo.
pause
