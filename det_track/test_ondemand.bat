@echo off
REM Quick test of on-demand crop extraction
REM 
REM Prerequisites:
REM 1. Download canonical_persons.npz from Google Drive:
REM    Location: /content/drive/MyDrive/pipelineoutputs/kohli_nets/canonical_persons.npz
REM    Save to: D:\trials\unifiedpipeline\newrepo\det_track\test_data\
REM
REM 2. Video should be at: D:\trials\unifiedpipeline\newrepo\demo_data\videos\kohli_nets.mp4
REM    (We already have this from Stage 0 testing)

echo ================================================================================
echo Testing On-Demand Crop Extraction
echo ================================================================================
echo.

cd /d %~dp0

REM Check if canonical_persons.npz exists
if not exist "test_data\canonical_persons.npz" (
    echo ERROR: canonical_persons.npz not found!
    echo.
    echo Please download it from Google Drive:
    echo   Location: /content/drive/MyDrive/pipelineoutputs/kohli_nets/canonical_persons.npz
    echo   Save to:  test_data\canonical_persons.npz
    echo.
    pause
    exit /b 1
)

REM Check if video exists
if not exist "..\demo_data\videos\kohli_nets.mp4" (
    echo ERROR: kohli_nets.mp4 not found!
    echo   Expected: ..\demo_data\videos\kohli_nets.mp4
    echo.
    pause
    exit /b 1
)

echo All files found! Starting test...
echo.

REM Run the test
python test_ondemand_extraction.py ^
    --video ..\demo_data\videos\kohli_nets.mp4 ^
    --data test_data\canonical_persons.npz ^
    --output test_output\ ^
    --crops-per-person 50

echo.
echo ================================================================================
echo Test complete!
echo.
echo Check outputs:
echo   - WebP animations: test_output\
echo   - Timing results:  test_output\timing_results.json
echo ================================================================================
pause
