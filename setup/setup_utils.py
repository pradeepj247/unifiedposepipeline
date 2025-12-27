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


def run_command(cmd, shell=True, check=True, message=None, indent=False):
    """
    Execute a shell command and handle errors.
    
    Args:
        cmd (str): Command to execute
        shell (bool): Whether to use shell execution
        check (bool): Whether to raise exception on non-zero exit
        message (str, optional): Custom message to display instead of full command
        indent (bool): Whether to indent the message (for sub-steps)
        
    Returns:
        subprocess.CompletedProcess: Result of command execution
        
    Raises:
        subprocess.CalledProcessError: If command fails and check=True
    """
    indent_str = "  " if indent else ""
    if message:
        print(f"{indent_str}{message}")
    else:
        print(f"{indent_str}Running: {cmd}")
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
        print(f"  ✓ Created directory: {path}")
    else:
        print(f"  ✓ Directory exists: {path}")


def check_file_exists(filepath, quiet=False):
    """
    Check if a file exists.
    
    Args:
        filepath (str): Path to file
        quiet (bool): Suppress stdout logging when True
        
    Returns:
        bool: True if file exists, False otherwise
    """
    exists = os.path.isfile(filepath)
    if quiet:
        return exists
    if exists:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ Found: {filepath} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ Missing: {filepath}")
    return exists


def check_import(module_name, package_name=None, silent=False):
    """
    Check if a Python module can be imported.
    
    Args:
        module_name (str): Name of module to import
        package_name (str, optional): Display name if different from module_name
        silent (bool): If True, suppress verbose output from the import
        
    Returns:
        bool: True if import successful, False otherwise
    """
    display_name = package_name or module_name
    try:
        if silent:
            # Suppress stdout/stderr during import
            import sys
            import os
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                __import__(module_name)
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            __import__(module_name)
        print(f"  ✓ {display_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {display_name}: {e}")
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
