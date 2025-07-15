"""
Setup script for EpiSim Python Interface
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="episim-python",
    version="0.1.0",
    author="EpiSim Development Team",
    author_email="miguel.ponce@bsc.es",
    description="Python interface for EpiSim.jl epidemic simulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Epi-Sim/EpiSim.jl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "xarray>=0.16.0",
        "netcdf4>=1.5.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "episim-python=episim_python.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Epi-Sim/EpiSim.jl/issues",
        "Source": "https://github.com/Epi-Sim/EpiSim.jl",
        "Documentation": "https://github.com/Epi-Sim/EpiSim.jl/blob/main/README.md",
    },
)
