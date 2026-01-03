#!/usr/bin/env python3
"""Quick test script to run Stage 11 MP4 generation"""

import os
import sys
import time
from pathlib import Path

# Add det_track to path
sys.path.insert(0, '/content/unifiedposepipeline/det_track')

os.chdir('/content/unifiedposepipeline/det_track')

# Run stage 11
from stage9_generate_person_gifs import main

start_time = time.time()
exit_code = main.__wrapped__() if hasattr(main, '__wrapped__') else main()
end_time = time.time()

print(f"\n{'='*70}")
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"{'='*70}\n")

sys.exit(exit_code)
