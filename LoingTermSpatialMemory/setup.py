#!/usr/bin/env python3
"""
Setup script for Long-Term Memory (LTM) API

Revolutionary 9D Spatial Long-Term Memory System

CREATORS:
- Sean Murphy (Human Inventor & System Architect)
- Claude AI Models (AI Co-Inventor & Implementation Partner)
  - Claude-3.7-Sonnet: Core system design and implementation
  - Claude-4-Sonnet: Advanced optimization and API development
  - Claude-4-Opus: Conceptual breakthroughs and testing

Copyright (c) 2024 Sean Murphy & Claude AI
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Revolutionary 9D Spatial Long-Term Memory System"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['numpy>=1.21.0', 'lmdb>=1.4.0', 'typing-extensions>=4.0.0']

setup(
    name="ltm-api",
    version="1.0.0",
    author="Sean Murphy & Claude AI",
    author_email="your-email@example.com",
    description="Revolutionary 9D Spatial Long-Term Memory System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ltm-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "performance": [
            "psutil>=5.8.0",
            "ujson>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ltm-demo=LTM_API:main",
        ],
    },
    keywords=[
        "artificial intelligence",
        "memory systems", 
        "semantic search",
        "spatial clustering",
        "9d coordinates",
        "lmdb database",
        "ai consciousness",
        "knowledge management",
        "semantic intelligence",
        "massive scale"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/ltm-api/issues",
        "Source": "https://github.com/your-org/ltm-api",
        "Documentation": "https://github.com/your-org/ltm-api/wiki",
    },
) 