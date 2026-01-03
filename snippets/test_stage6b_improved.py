#!/usr/bin/env python3
"""
Quick test script for improved Stage 6b crop selection
Run this in Colab to test the intelligent crop selection
"""

import subprocess
import sys

# Change to det_track directory
import os
os.chdir('/content/unifiedposepipeline/det_track')

# Run Stage 6b (stage 9) with --force to re-run
result = subprocess.run(
    [sys.executable, 'run_pipeline.py', 
     '--config', 'configs/pipeline_config.yaml',
     '--stages', '9',
     '--force'],
    capture_output=False
)

sys.exit(result.returncode)
