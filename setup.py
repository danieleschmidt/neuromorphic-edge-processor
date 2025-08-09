from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuromorphic_edge_processor",
    version="0.1.0",
    description="Brain-inspired ultra-low power computing at the edge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    url="https://github.com/danieleschmidt/neuromorphic-edge-processor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "experiment": [
            "wandb>=0.15.0",
            "tensorboard>=2.10.0",
            "psutil>=5.8.0",
        ],
        "benchmark": [
            "psutil>=5.8.0",
            "scikit-learn>=1.0.0",
        ],
        "all": [
            "wandb>=0.15.0",
            "tensorboard>=2.10.0", 
            "psutil>=5.8.0",
            "scikit-learn>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    keywords="neuromorphic computing, spiking neural networks, edge AI, event-driven processing",
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/neuromorphic-edge-processor/issues",
        "Source": "https://github.com/danieleschmidt/neuromorphic-edge-processor",
        "Documentation": "https://neuromorphic-edge-processor.readthedocs.io/",
    },
    entry_points={
        "console_scripts": [
            "neuromorphic-benchmark=benchmarks.cli:main",
        ],
    },
)
