# setup.py
from setuptools import setup, find_packages
from pathlib import Path


def read_requirements(filename: str) -> list:
    """Read requirements from file"""
    try:
        return [
            line.strip()
            for line in Path(filename).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(("#", "-r"))
        ]
    except Exception as e:
        print(f"Warning: Could not read requirements file {filename}: {e}")
        return []


setup(
    name="gat-explainer",
    version="0.1.0",
    description="Explainable Graph Attention Networks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=read_requirements("requirements/base.txt"),
    extras_require={
        "dev": read_requirements("requirements/dev.txt"),
        "prod": read_requirements("requirements/prod.txt"),
    },
    entry_points={
        "console_scripts": [
            "gat-train=src.training.train:main",
            "gat-serve=src.api.main:main",
        ],
    },
)
