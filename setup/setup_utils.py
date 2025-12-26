"""
Shared Utility Functions for Setup Scripts

This module provides common helper functions used across all setup scripts.
"""

import os
import sys
import subprocess


# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"


def is_colab_environment():
    """
    Check if running in Google Colab environment.
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def print_header(title, color=None):
    """
    Print a formatted header for section titles.
    
    Args:
        title (str): The title to display
        color (str, optional): ANSI color code for the title
    """
    color_code = color or ""
    reset_code = COLOR_RESET if color else ""
    print("\n" + "=" * 70)
    print(f"{color_code}  {title}{reset_code}")
    print("=" * 70 + "\n")


def print_step(step_num, description, indent=False):
    """
    Print a formatted step header.
    
    Args:
        step_num (str or int): Step number (can be like "1.0", "1.1", etc.)
        description (str): Step description
        indent (bool): Whether to indent the step (for sub-steps)
    """
    indent_str = "  " if indent else ""
    print(f"\n{indent_str}{'─' * 66}")
    print(f"{indent_str}STEP {step_num}: {description}")
    print(f"{indent_str}{'─' * 66}\n")


def run_command(cmd, shell=True, check=True):
    """
    Execute a shell command and handle errors.
    
    Args:
        cmd (str): Command to execute
        shell (bool): Whether to use shell execution
        check (bool): Whether to raise exception on non-zero exit
        
    Returns:
        subprocess.CompletedProcess: Result of command execution
        
    Raises:
        subprocess.CalledProcessError: If command fails and check=True
    """
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=check, 
                          capture_output=False, text=True)
    return result


def create_directory(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"✓ Created directory: {path}")
    else:
        print(f"✓ Directory exists: {path}")


def check_file_exists(filepath):
    """
    Check if a file exists.
    
    Args:
        filepath (str): Path to file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    exists = os.path.isfile(filepath)
    if exists:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ Found: {filepath} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Missing: {filepath}")
    return exists


def check_import(module_name, package_name=None):
    """
    Check if a Python module can be imported.
    
    Args:
        module_name (str): Name of module to import
        package_name (str, optional): Display name if different from module_name
        
    Returns:
        bool: True if import successful, False otherwise
    """
    display_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False


def print_success(message, color=None):
    """
    Print a success message.
    
    Args:
        message (str): Success message to display
        color (str, optional): ANSI color code for the message
    """
    color_code = color or COLOR_GREEN
    print(f"\n{'=' * 70}")
    print(f"{color_code}✓ SUCCESS: {message}{COLOR_RESET}")
    print(f"{'=' * 70}\n")


def print_error(message):
    """
    Print an error message.
    
    Args:
        message (str): Error message to display
    """
    print(f"\n{'=' * 70}")
    print(f"✗ ERROR: {message}")
    print(f"{'=' * 70}\n")


def print_warning(message):
    """
    Print a warning message.
    
    Args:
        message (str): Warning message to display
    """
    print(f"\n{'─' * 70}")
    print(f"⚠ WARNING: {message}")
    print(f"{'─' * 70}\n")
