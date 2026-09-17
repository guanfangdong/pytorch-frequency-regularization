# Frequency Regularization

This repository contains the reference implementation and an inference demo
for the paper **Frequency Regularization: Reducing Information Redundancy in
Convolutional Neural Networks**.

Frequency regularization stores convolution kernels in the frequency domain
and applies an inverse discrete cosine transform (DCT) before the usual
PyTorch operation. In the released UNet experiment, the paper reports a
reduction from 31M model parameters to 759 non-zero parameters while keeping
a Dice score above 97% on the [Carvana Image Masking Challenge
dataset](https://www.kaggle.com/c/carvana-image-masking-challenge).

## Repository layout

```text
frereg/                         Frequency-regularized PyTorch layers
run/
  detect_UNetFR.py              Pretrained UNet inference demo
  dice_score.py                 Dice metric and loss helpers
  unet/                         UNet architecture used by the demo
  unet_fr.tar.xz                Compressed sparse checkpoint
  testimgs/                     Demo input images and masks
tests/                          Regression and shape tests
```

The checkpoint archive expands to `run/unet_fr.pt`. That extracted file is a
generated artifact and is ignored by Git.

## Installation

The package can be installed directly from this repository:

```bash
python -m pip install .
```

For editable development installation:

```bash
python -m pip install -e .
```

The package metadata supports Python 3.8 through 3.11.

The package requires PyTorch, NumPy, ImageIO, and Matplotlib. A CUDA-enabled
PyTorch installation is optional; the demo automatically uses CUDA when it is
available and otherwise runs on the CPU.

## Run the demo

From the repository root, run:

```bash
bash run/run.sh
```

The script locates its own directory, extracts the bundled checkpoint, loads
the sample images, prints the number of non-zero weights and biases, reports
the average Dice score, and displays the predictions. It can also be started
from inside the `run/` directory with:

```bash
bash run.sh
```

## Use the layers

`Conv2d` and `ConvTranspose2d` have the same call style as their PyTorch
counterparts while keeping their trainable kernels in the frequency domain:

```python
import torch
import frereg

layer = frereg.Conv2d(3, 16, kernel_size=3, padding=1)
output = layer(torch.randn(1, 3, 32, 32))
```

The original public helper names, pruning metadata, and UNet state-dictionary
layout are retained so that existing training code and the released checkpoint
continue to work.

## Tests

With PyTorch and pytest installed:

```bash
pytest -q
```

## Citation

```bibtex
@ARTICLE{10266314,
  author={Zhao, Chenqiu and Dong, Guanfang and Zhang, Shupei and Tan, Zijie and Basu, Anup},
  journal={IEEE Access},
  title={Frequency Regularization: Reducing Information Redundancy in Convolutional Neural Networks},
  year={2023},
  volume={11},
  pages={106793-106802},
  doi={10.1109/ACCESS.2023.3320642}
}
```

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10266314)
