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
    if errorlevel 1 (
        echo [ERROR] Failed to download portable Python. Please check your internet connection.
        pause
        exit /b 1
    )
    if not exist "%BACKEND_DIR%\python_embed.zip" (
        echo [ERROR] Downloaded portable Python archive is missing.
        pause
        exit /b 1
    )
    echo [*] Extracting Python...
    powershell -Command "Expand-Archive -Path '%BACKEND_DIR%\python_embed.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to extract portable Python archive.
        del "%BACKEND_DIR%\python_embed.zip" 2>nul
        pause
        exit /b 1
    )
    del "%BACKEND_DIR%\python_embed.zip" 2>nul

    :: Enable site-packages in embedded python
    powershell -Command "$f = '%PYTHON_DIR%\python312._pth'; if (Test-Path $f) { (Get-Content $f) -replace '^#import site', 'import site' | Set-Content $f }"
    if errorlevel 1 (
        echo [ERROR] Failed to configure python312._pth.
        pause
        exit /b 1
    )

    :: Download get-pip.py
    echo [*] Installing pip...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%BACKEND_DIR%\get-pip.py')"
    if errorlevel 1 (
        echo [ERROR] Failed to download get-pip.py. Please check your internet connection.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%BACKEND_DIR%\get-pip.py" --no-warn-script-location
    if errorlevel 1 (
        echo [ERROR] Failed to install pip into portable Python.
        del "%BACKEND_DIR%\get-pip.py" 2>nul
        pause
        exit /b 1
    )
    del "%BACKEND_DIR%\get-pip.py" 2>nul
)

:: 3. Setup virtual environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [*] Initializing isolated virtual environment...
    "%PYTHON_DIR%\Scripts\pip.exe" install --no-warn-script-location virtualenv
    if errorlevel 1 (
        echo [ERROR] Failed to install virtualenv.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\Scripts\virtualenv.exe" "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment in "%VENV_DIR%".
        pause
        exit /b 1
    )
)

:: 4. Install PyTorch with CUDA 12.4
echo [*] Installing PyTorch with CUDA 12.4 acceleration...
"%VENV_DIR%\Scripts\pip.exe" install --no-warn-script-location torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch with CUDA 12.4.
    pause
    exit /b 1
)

:: 5. Install dependencies from requirements.txt
if exist "%BACKEND_DIR%\requirements.txt" (
    echo [*] Installing application dependencies...
    "%VENV_DIR%\Scripts\pip.exe" install --no-warn-script-location --upgrade-strategy only-if-needed -r "%BACKEND_DIR%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements from requirements.txt.
        pause
        exit /b 1
    )
)

:: 6. Ensure bundled FFmpeg / FFprobe (full build) is installed
echo [*] Checking FFmpeg / FFprobe installation...
set "PYTHONPATH=%BACKEND_DIR%"
"%VENV_DIR%\Scripts\python.exe" -c "from backend.ffmpeg_tools import validate_ffmpeg_install, repair_ffmpeg_install; r = validate_ffmpeg_install(require_probe=True); print('[OK] FFmpeg ready:', r.ffmpeg_version.token if r.ffmpeg_version else r.ffmpeg_path) if r.ok else (print('[*] Downloading FFmpeg full build (gyan.dev)...'), repair_ffmpeg_install(lambda ph, c, t: print(f'    {ph}: {c//1048576}/{t//1048576} MB' if t else f'    {ph}', end='\r')), print('\n[OK] FFmpeg installed!'))"
if errorlevel 1 (
    echo [WARNING] Automatic FFmpeg download failed ^(you can retry later via Preferences ^> Repair FFmpeg^).
)

echo.
echo =======================================================
echo [SUCCESS] CorridorKey 2.0 Pro GUI environment is ready!
echo FFmpeg is configured. Models ^(including SAM 3.1^) can be
echo downloaded on first launch via Setup Wizard / Download Manager.
echo You can now launch the application using Run_GUI.bat
echo =======================================================
echo.
pause
