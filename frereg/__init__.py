"""Frequency regularization layers for PyTorch models."""

from ._dct import idct, idct_2d, idct_4d
from ._freqreg_eva import (
    Conv2d,
    Conv2d_FR4d,
    ConvTranspose2d,
    ConvTranspose2d_FR4d,
    Linear,
    Linear_FR2d,
    ZIdx2Int,
    cleanParams,
    countParams,
    getZIdx_2d,
    getZIdx_4d,
)

__version__ = "0.1.0"

__all__ = [
    "Conv2d",
    "Conv2d_FR4d",
    "ConvTranspose2d",
    "ConvTranspose2d_FR4d",
    "Linear",
    "Linear_FR2d",
    "ZIdx2Int",
    "cleanParams",
    "countParams",
    "getZIdx_2d",
    "getZIdx_4d",
    "idct",
    "idct_2d",
    "idct_4d",
]
