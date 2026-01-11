"""
Centralized logging module for the Unified Pipeline.

Provides standardized console output across all pipeline stages with support
for verbose and silent modes. Used for both development and production.
"""

import time
from pathlib import Path


class PipelineLogger:
    """Unified logger for all pipeline stages."""
    
    def __init__(self, stage_name, verbose=False):
        """
        Initialize logger for a stage.
        
        Args:
            stage_name (str): Name of the stage (e.g., "Stage 1: YOLO Detection")
            verbose (bool): Enable verbose output with detailed stats
        """
        self.stage_name = stage_name
        self.verbose = verbose
        self._stage_start_time = time.time()
    
    def header(self):
        """Print stage header."""
        print(f"\n>>> {self.stage_name}")
        print(f"{'-' * 70}")
    
    def step(self, message):
        """
        Print normal-mode step information.
        
        Args:
            message (str): Information to display
        """
        print(f"   {message}")
    
    def info(self, message):
        """
        Always displayed info (essential information).
        
        Args:
            message (str): Information to display
        """
        print(f"   [+] {message}")
    
    def verbose_info(self, message):
        """
        Debug details (only shown in verbose mode).
        
        Args:
            message (str): Verbose information to display
        """
        if self.verbose:
            print(f"   (V) {message}")
    
    def timing(self, label, duration):
        """
        Sub-step timing information (verbose only).
        
        Args:
            label (str): Name of the operation
            duration (float): Duration in seconds
        """
        if self.verbose:
            print(f"   [T] {label}: {duration:.2f}s")
    
    def stat(self, label, value, format_str=None):
        """
        Statistics/count information (verbose only).
        
        Args:
            label (str): Name of the statistic
            value: The value to display
            format_str (str, optional): Format string (e.g., ".1f" for floats)
        """
        if self.verbose:
            if format_str:
                formatted = f"{value:{format_str}}"
            else:
                formatted = str(value)
            print(f"   [S] {label}: {formatted}")
    
    def file_size(self, filename, size_mb):
        """
        File size information (verbose only).
        
        Args:
            filename (str): Name of the file
            size_mb (float): Size in megabytes
        """
        if self.verbose:
            print(f"   [F] {filename}: {size_mb:.2f} MB")
    
    def warning(self, message):
        """
        Warning message (always displayed).
        
        Args:
            message (str): Warning message
        """
        print(f"   [!] WARNING: {message}")
    
    def error(self, message):
        """
        Error message (always displayed).
        
        Args:
            message (str): Error message
        """
        print(f"   [X] ERROR: {message}")
    
    def success(self):
        """Print completion message with elapsed time."""
        elapsed = time.time() - self._stage_start_time
        print(f"\n[OK] {self.stage_name} completed in {elapsed:.2f}s")
