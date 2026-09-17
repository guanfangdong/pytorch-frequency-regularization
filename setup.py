"""Package metadata for the frequency-regularization library."""

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent


setup(
    name="frereg",
    version="0.1.0",
    description="Frequency Regularization Python Package",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/guanfangdong/pytorch-frequency-regularization",
    author="Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup",
    packages=find_packages(include=["frereg", "frereg.*"]),
    license="Apache License 2.0",
    install_requires=[
        "numpy",
        "imageio",
        "torch",
        "matplotlib",
    ],
    python_requires=">=3.8, <3.12",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
