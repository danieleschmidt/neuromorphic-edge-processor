from setuptools import setup, find_packages

setup(
    name="neuromorphic-edge-processor",
    version="1.0.0",
    description="Brain-inspired event-driven processor architecture simulator",
    author="Daniel Schmidt",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy>=1.21"],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
