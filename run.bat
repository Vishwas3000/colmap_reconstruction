@echo off
echo ============================================
echo   COLMAP Point Cloud Extraction - Run
echo ============================================
echo.

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

REM Check images folder
if not exist "images" (
    echo [ERROR] "images" folder not found. Create it and add your photos.
    pause
    exit /b 1
)

REM Count images
set count=0
for %%f in (images\*.jpg images\*.jpeg images\*.png images\*.tif images\*.tiff images\*.bmp) do set /a count+=1
if %count% equ 0 (
    echo [ERROR] No images found in "images" folder.
    echo         Supported formats: jpg, jpeg, png, tif, tiff, bmp
    pause
    exit /b 1
)
echo Found %count% images.
echo.

REM Run pipeline
python colmap_pipeline.py --image_dir images --workspace workspace --visualize
pause
