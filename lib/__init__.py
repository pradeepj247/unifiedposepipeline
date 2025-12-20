"""
Unified Pose Estimation Pipeline - Main Package
Combines ViTPose+HybrIK and RTMLib into a single interface
"""

__version__ = "1.0.0"
__author__ = "Unified Pose Team"

# Import main components
from .rtmlib import *
from .vitpose import *

__all__ = [
    'RTMLibPipeline',
    'ViTPosePipeline',
    'UnifiedPoseEstimator',
]
