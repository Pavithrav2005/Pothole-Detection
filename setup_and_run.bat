@echo off
echo Pothole Detection using YOLOv8 - Setup and Run
echo =============================================

REM Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Checking and installing required packages...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Error installing required packages.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo Choose an option:
echo 1. Run main pothole detection program
echo 2. Run quick test on a sample image
echo 3. Exit

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    python pothole_detection.py
) else if "%choice%"=="2" (
    echo Running quick test on a sample image...
    set "sample_img=Dataset\valid\images\552_jpg.rf.b38e499d872c1ed47b181022b8b1ee9d.jpg"
    if exist "%sample_img%" (
        python quick_test.py "%sample_img%"
    ) else (
        echo Sample image not found.
        echo Please specify an image path:
        set /p img_path="Image path: "
        python quick_test.py "%img_path%"
    )
) else if "%choice%"=="3" (
    exit /b 0
) else (
    echo Invalid choice.
)

pause 