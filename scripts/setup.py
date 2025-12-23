"""
Unified Pose Estimation Pipeline Setup
Combines ViTPose+HybrIK and RTMLib into a single framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]
else:
    requirements = []

setup(
    name="unified-pose-estimation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Unified pose estimation pipeline combining ViTPose+HybrIK and RTMLib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/unified-pose-estimation",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "jupyter>=1.0",
            "notebook>=6.4",
        ],
        "gpu": [
            "onnxruntime-gpu",
        ],
        "accelerate": [
            "openvino",
            "tensorrt",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-pose=lib.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
