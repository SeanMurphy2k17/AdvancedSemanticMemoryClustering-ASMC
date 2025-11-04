#!/usr/bin/env python3
"""
Setup script for Semantic STM API
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="semantic-stm-api",
    version="1.0.0",
    author="Sean Murphy & Claude AI",
    author_email="contact@semantic-stm-api.org",
    description="Revolutionary 9D Spatial Semantic Memory System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/semantic-stm-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
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
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stm-demo=example_usage:main",
        ],
    },
    keywords=[
        "artificial intelligence",
        "semantic memory",
        "spatial coordinates", 
        "conversation memory",
        "contextual search",
        "9d coordinates",
        "semantic clustering",
        "ai memory",
        "consciousness",
        "semantic search"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/semantic-stm-api/issues",
        "Source": "https://github.com/your-org/semantic-stm-api",
        "Documentation": "https://semantic-stm-api.readthedocs.io/",
    },
) 