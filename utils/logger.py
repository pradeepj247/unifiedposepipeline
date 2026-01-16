"""
Centralized logging module for the Unified Pipeline.

Provides standardized console output across all pipeline stages with support
for verbose and silent modes. Uses consistent emoji legend for readability.

EMOJI LEGEND (Referenced in logs):
 1. ✅ Success          7. ⏱️ Timing         13. 👤 Person/ReID
 2. 🚀 Start/Step       8. 💡 Tip/Note       14. ⚡ Speed/Fast
 3. 🔍 Found/Present    9. 📁 File/Saved     15. 💥 Error/Crash
 4. ❌ Missing/Fail    10. 🧾 Summary        16. ❓ Question
 5. ⬇️  In-progress    11. ✔️ Completed     17. 📌 Important
 6. ⚠️  Warning        12. 🛠️ Execute       18. 🔄 Retry
                       20. 📊 Stats/Metrics

On Windows consoles without UTF-8 support, falls back to ASCII brackets.
"""

import time
import sys
import os
from pathlib import Path


class PipelineLogger:
    """Unified logger for all pipeline stages with emoji legend."""
    
    # Detect if we can use emojis (Windows cmd.exe typically can't)
    _CAN_USE_EMOJI = (
        sys.platform != 'win32' or 
        os.environ.get('TERM') == 'xterm' or
        'WT_SESSION' in os.environ or  # Windows Terminal
        'ConEmuANSI' in os.environ      # ConEmu
    )
    
    # Emoji mappings with ASCII fallbacks
    EMOJI = {
        'success': ('✅', '[OK]'),           # 1: Success
        'start': ('🚀', '[>>]'),             # 2: Start/Step
        'found': ('🔍', '[+]'),              # 3: Found/Present
        'fail': ('❌', '[X]'),               # 4: Missing/Fail
        'progress': ('⬇️', '[*]'),           # 5: In-progress
        'warning': ('⚠️', '[!]'),            # 6: Warning
        'timing': ('⏱️', '[T]'),             # 7: Timing
        'note': ('💡', '[i]'),               # 8: Tip/Note
        'file': ('📁', '[F]'),               # 9: File/Saved
        'summary': ('🧾', '[S]'),            # 10: Summary
        'completed': ('✔️', '[V]'),          # 11: Completed
        'execute': ('🛠️', '[>]'),            # 12: Execute
        'person': ('👤', '[P]'),             # 13: Person/ReID
        'speed': ('⚡', '[~]'),              # 14: Speed/Fast
        'error': ('💥', '[!]'),              # 15: Error/Crash
        'question': ('❓', '[?]'),           # 16: Question
        'important': ('📌', '[*]'),          # 17: Important
        'retry': ('🔄', '[<]'),              # 18: Retry
        'stats': ('📊', '[#]'),              # 20: Stats/Metrics
    }
    
    @classmethod
    def _get_emoji(cls, key):
        """Get emoji or ASCII fallback based on platform."""
        if key not in cls.EMOJI:
            return '?'
        emoji, fallback = cls.EMOJI[key]
        return emoji if cls._CAN_USE_EMOJI else fallback
    
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
        """Print stage header with separator box (consistent with Stage 1 & 2)."""
        print(f"\n{'='*70}")
        print(f"📍 {self.stage_name.upper()}")
        print(f"{'='*70}\n")
    
    def step(self, message):
        """
        Print normal-mode step information.
        
        Args:
            message (str): Information to display
        """
        print(f"   {message}")
    
    def info(self, message):
        """
        Always displayed info (essential information). Uses success emoji #1.
        
        Args:
            message (str): Information to display
        """
        success_emoji = self._get_emoji('success')
        print(f"   {success_emoji} {message}")
    
    def found(self, message):
        """
        Found/present information. Uses found emoji #3.
        
        Args:
            message (str): Message about what was found
        """
        if self.verbose:
            found_emoji = self._get_emoji('found')
            print(f"   {found_emoji} {message}")
    
    def verbose_info(self, message):
        """
        Debug details (only shown in verbose mode). Uses note emoji #8.
        
        Args:
            message (str): Verbose information to display
        """
        if self.verbose:
            note_emoji = self._get_emoji('note')
            print(f"   {note_emoji} {message}")
    
    def timing(self, label, duration):
        """
        Sub-step timing information (verbose only). Uses timing emoji #7.
        
        Args:
            label (str): Name of the operation
            duration (float): Duration in seconds
        """
        if self.verbose:
            timing_emoji = self._get_emoji('timing')
            print(f"   {timing_emoji} {label}: {duration:.2f}s")
    
    def stat(self, label, value, format_str=None):
        """
        Statistics/count information (verbose only). Uses stats emoji #20.
        
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
            stats_emoji = self._get_emoji('stats')
            print(f"   {stats_emoji} {label}: {formatted}")
    
    def file_size(self, filename, size_mb):
        """
        File size information (verbose only). Uses file emoji #9.
        
        Args:
            filename (str): Name of the file
            size_mb (float): Size in megabytes
        """
        if self.verbose:
            file_emoji = self._get_emoji('file')
            print(f"   {file_emoji} {filename}: {size_mb:.2f} MB")
    
    def warning(self, message):
        """
        Warning message (always displayed). Uses warning emoji #6.
        
        Args:
            message (str): Warning message
        """
        warning_emoji = self._get_emoji('warning')
        print(f"   {warning_emoji} WARNING: {message}")
    
    def error(self, message):
        """
        Error message (always displayed). Uses error emoji #15.
        
        Args:
            message (str): Error message
        """
        error_emoji = self._get_emoji('error')
        print(f"   {error_emoji} ERROR: {message}")
    
    def success(self):
        """Print completion message with elapsed time. Uses success emoji #1."""
        success_emoji = self._get_emoji('success')
        elapsed = time.time() - self._stage_start_time
        print(f"\n   {success_emoji} {self.stage_name} completed in {elapsed:.2f}s")
