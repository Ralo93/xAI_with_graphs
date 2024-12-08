from setuptools import setup, find_packages
from pathlib import Path


def read_requirements(filename: str) -> list:
    """Read requirements from file"""
    return [
        line.strip()
        for line in Path(filename).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# Read requirements
base_reqs = read_requirements("requirements/base.txt")
prod_reqs = read_requirements("requirements/prod.txt")
dev_reqs = read_requirements("requirements/dev.txt")

setup(
    name="graph-attention-networks",
    version="0.1.0",
    description="Graph Attention Networks for Node Classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=base_reqs,
    extras_require={
        "dev": dev_reqs,
        "prod": prod_reqs,
    },
    entry_points={
        "console_scripts": [
            "gat-train=src.training.train:main",
            "gat-serve=src.api.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
