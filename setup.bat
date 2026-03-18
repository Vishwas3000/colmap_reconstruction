@echo off
echo ============================================
echo   COLMAP Point Cloud Extraction - Setup
echo ============================================
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

REM Activate and install dependencies
echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create folders
echo [3/3] Creating project folders...
if not exist "images" mkdir images
if not exist "workspace" mkdir workspace

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Install COLMAP from https://github.com/colmap/colmap/releases
echo      (download the Windows binary and add to PATH)
echo   2. Place your photos in the "images" folder
echo   3. Activate venv:  venv\Scripts\activate.bat
echo   4. Run:  python colmap_pipeline.py --image_dir images --workspace workspace
echo.
pause
