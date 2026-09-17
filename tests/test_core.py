"""Small regression tests for the public library and demo model."""

import io
import tarfile
from pathlib import Path

import torch

import frereg
from run.detect_UNetFR import cleanUNet
from run.unet import UNet


def test_dct_shapes():
    values = torch.randn(2, 3, 4, 5)

    assert frereg.idct(values).shape == values.shape
    assert frereg.idct_2d(values).shape == values.shape
    assert frereg.idct_4d(values).shape == values.shape


def test_frequency_regularized_layers_match_expected_shapes():
    convolution = frereg.Conv2d(3, 4, kernel_size=3, padding=1)
    output = convolution(torch.randn(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)

    transpose = frereg.ConvTranspose2d(4, 3, kernel_size=2, stride=2)
    output = transpose(torch.randn(2, 4, 4, 4))
    assert output.shape == (2, 3, 8, 8)

    linear = frereg.Linear(8, 5)
    assert linear(torch.randn(2, 8)).shape == (2, 5)


def test_clean_params_applies_zero_mask():
    layer = frereg.Conv2d(1, 1, kernel_size=1)
    with torch.no_grad():
        layer.IDROP.zero_()

    frereg.cleanParams(layer)

    assert torch.count_nonzero(layer.weight) == 0


def test_unet_state_dict_layout_and_output_shape():
    model = UNet(n_channels=3, n_classes=2, bilinear=False)

    assert len(model.state_dict()) == 73
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 2, 32, 32)


def test_bundled_checkpoint_loads():
    project_root = Path(__file__).resolve().parents[1]
    with tarfile.open(project_root / "run" / "unet_fr.tar.xz", "r:xz") as archive:
        checkpoint_bytes = archive.extractfile("unet_fr.pt").read()

    sparse_state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    cleanUNet(model)
    model.load_state_dict(
        {name: value.to_dense() for name, value in sparse_state.items()}
    )
