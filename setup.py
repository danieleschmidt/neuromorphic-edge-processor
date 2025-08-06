from setuptools import setup, find_packages

setup(
    name="neuromorphic_edge_processor",
    version="0.1.0",
    description="Brain-inspired ultra-low power computing at the edge",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be added based on research needs
    ],
    python_requires=">=3.8",
)
