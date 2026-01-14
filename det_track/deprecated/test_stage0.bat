@echo off
REM Quick test script for Stage 0 (Video Normalization)
REM Tests the normalization on kohli_nets.mp4

echo ================================================================================
echo Testing Stage 0: Video Normalization
echo ================================================================================
echo.

cd /d %~dp0

python stage0_normalize_video.py --config configs/pipeline_config.yaml

echo.
echo ================================================================================
echo Test complete! Check outputs\kohli_nets\canonical_video.mp4
echo ================================================================================
pause
