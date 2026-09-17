"""Frequency-domain neural-network layers.

The layers in this module store trainable weights in the frequency domain and
apply an inverse DCT immediately before the corresponding PyTorch operation.
The public class and attribute names intentionally remain compatible with the
original implementation and with the released UNet checkpoint.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ._dct import idct_2d, idct_4d

_CONV_PROTECTED_PARAMETERS = 127
_LINEAR_PROTECTED_PARAMETERS = 255
_TRANSPOSE_PROTECTED_PARAMETERS = 31
_EPSILON = 1e-22


def getZIdx_4d(data):
    """Build the four-dimensional frequency traversal matrix."""

    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0
    maxval3 = data.shape[2] + 1.0
    maxval4 = data.shape[3] + 1.0

    matdim1 = torch.tensor(range(data.shape[0]))
    matdim1 = matdim1 + torch.tensor(range(data.shape[0])) / maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[2], data.shape[3], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1]))
    matdim2 = matdim2 + torch.tensor(range(data.shape[1])) / (maxval1 * maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[2], data.shape[3], data.shape[1])

    matdim3 = torch.tensor(range(data.shape[2]))
    matdim3 = matdim3 + torch.tensor(range(data.shape[2])) / (
        maxval1 * maxval2 * maxval3
    )
    matdim3 = matdim3.expand(data.shape[0], data.shape[1], data.shape[3], data.shape[2])

    matdim4 = torch.tensor(range(data.shape[3]))
    matdim4 = matdim4 + torch.tensor(range(data.shape[3])) / (
        maxval1 * maxval2 * maxval3 * maxval4
    )
    matdim4 = matdim4.expand(data.shape[0], data.shape[1], data.shape[2], data.shape[3])

    return (
        matdim1.permute(3, 0, 1, 2)
        + matdim2.permute(0, 3, 1, 2)
        + matdim3.permute(0, 1, 3, 2)
        + matdim4
    )


def getZIdx_2d(data):
    """Build the two-dimensional equivalent of :func:`getZIdx_4d`."""

    maxval1 = data.shape[0] + 1.0
    maxval2 = data.shape[1] + 1.0

    matdim1 = torch.tensor(range(data.shape[0]))
    matdim1 = matdim1 + torch.tensor(range(data.shape[0])) / maxval1
    matdim1 = matdim1.expand(data.shape[1], data.shape[0])

    matdim2 = torch.tensor(range(data.shape[1]))
    matdim2 = matdim2 + torch.tensor(range(data.shape[1])) / (maxval1 * maxval2)
    matdim2 = matdim2.expand(data.shape[0], data.shape[1])

    return matdim1.permute(1, 0) + matdim2


def ZIdx2Int(inputmat):
    """Convert a traversal matrix into integer rank indices."""

    mat = inputmat.view(-1)
    idxpos = torch.tensor(range(mat.numel()))

    packmat = torch.cat((mat.unsqueeze(-1), idxpos.unsqueeze(-1)), dim=-1)
    packmat = packmat[packmat[:, 0].sort()[1]]

    x = torch.cat((packmat, idxpos.unsqueeze(-1)), dim=-1)[:, 1:]
    intZidx = (x[x[:, 0].sort()[1]])[:, 1].squeeze()

    return intZidx.reshape(inputmat.shape)


def _reset_mask(layer):
    """Restore a layer's drop mask and keep all coefficients enabled."""

    layer.IDROP.fill_(1.0)
    layer.keeprate = 1.0


class Conv2d_FR4d(nn.Module):
    """2-D convolution with a four-dimensional frequency-domain kernel."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        minrate=0.01,
        droprate=0.01,
        dynamicdrop=False,
        groups=1,
        dilation=1,
    ):
        super().__init__()

        weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(weight_shape).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None

        self.register_buffer("ZMAT", ZIdx2Int(getZIdx_4d(torch.empty(weight_shape))))
        self.register_buffer("IDROP", torch.zeros(weight_shape) + 1.0)

        self.minrate = minrate
        self.minnum = max(
            round(self.IDROP.numel() * minrate),
            _CONV_PROTECTED_PARAMETERS,
        )
        self.keeprate = 1.0
        self.droprate = droprate
        self.dynamicdrop = dynamicdrop
        self.threval = self.IDROP.numel()
        self.weightrate = 0
        self.weightnum = -1

    def setDroprate(self, droprate):
        """Set the coefficient drop rate used by external training code."""

        self.droprate = droprate

    def setminnum(self, minnum, protectnum=_CONV_PROTECTED_PARAMETERS):
        """Set the minimum number of retained coefficients."""

        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum / (self.IDROP.numel() * 1.0)
        _reset_mask(self)

    def setminrate(self, minrate, protectnum=_CONV_PROTECTED_PARAMETERS):
        """Set the minimum retained coefficient rate."""

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel() * minrate), protectnum)
        _reset_mask(self)

    def resetparams(self):
        """Reset the drop mask and recompute the minimum coefficient count."""

        self.minnum = max(
            round(self.weight.numel() * self.minrate // self.groups),
            _CONV_PROTECTED_PARAMETERS,
        )
        _reset_mask(self)
        self.threval = self.IDROP.numel()

    def forward(self, data):
        weight = idct_4d(self.weight)
        return F.conv2d(
            data,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class Linear_FR2d(nn.Module):
    """Linear layer with a two-dimensional frequency-domain weight matrix."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        minrate=0.01,
        droprate=0.01,
        dynamicdrop=False,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0, 0.1)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        else:
            self.bias = None

        weight_shape = (out_features, in_features)
        self.register_buffer(
            "ZMAT",
            ZIdx2Int(getZIdx_2d(torch.empty(weight_shape)).squeeze()),
        )
        self.register_buffer("IDROP", torch.zeros(weight_shape).squeeze() + 1.0)

        self.minrate = minrate
        self.minnum = max(
            round(self.IDROP.numel() * minrate),
            _LINEAR_PROTECTED_PARAMETERS,
        )
        self.keeprate = 1.0
        self.droprate = droprate
        self.dynamicdrop = dynamicdrop
        self.threval = self.IDROP.numel()
        self.weightrate = 0
        self.weightnum = -1

    def setDroprate(self, droprate):
        """Set the coefficient drop rate used by external training code."""

        self.droprate = droprate

    def setminnum(self, minnum, protectnum=_LINEAR_PROTECTED_PARAMETERS):
        """Set the minimum number of retained coefficients."""

        self.minnum = max(minnum, protectnum)
        self.minrate = self.minnum / (self.IDROP.numel() * 1.0)
        _reset_mask(self)
        self.threval = self.IDROP.numel()

    def setminrate(self, minrate, protectnum=_LINEAR_PROTECTED_PARAMETERS):
        """Set the minimum retained coefficient rate."""

        self.minrate = minrate
        self.minnum = max(round(self.IDROP.numel() * minrate), protectnum)
        _reset_mask(self)
        self.threval = self.IDROP.numel()

    def resetparams(self):
        """Reset the drop mask and recompute the minimum coefficient count."""

        self.minnum = max(
            round(self.weight.numel() * self.minrate),
            _LINEAR_PROTECTED_PARAMETERS,
        )
        _reset_mask(self)
        self.threval = self.IDROP.numel()

    def forward(self, data):
        weight = idct_2d(self.weight)
        return F.linear(data, weight, self.bias)


class Conv2d(Conv2d_FR4d):
    """Public alias for :class:`Conv2d_FR4d`."""


class Linear(Linear_FR2d):
    """Public alias for :class:`Linear_FR2d`."""


def countParams(model):
    """Return retained, minimum, maximum, and average mask statistics."""

    sumnum = 0
    maxnum = -1
    minnum = 10000000000000000
    totalnum = 0

    for layer in model.modules():
        drop_mask = getattr(layer, "IDROP", None)
        if drop_mask is None:
            continue

        num = drop_mask.sum()
        totalnum += drop_mask.numel()

        maxnum = max(maxnum, num)
        minnum = min(minnum, num)
        sumnum += num

    rate = sumnum * 1.0 / totalnum
    return sumnum, minnum, maxnum, rate


def cleanParams(model):
    """Zero weights whose drop masks mark them as removed."""

    with torch.no_grad():
        for layer in model.modules():
            drop_mask = getattr(layer, "IDROP", None)
            weight = getattr(layer, "weight", None)
            if drop_mask is None or weight is None:
                continue

            idx = drop_mask.abs() < _EPSILON
            weight[idx] = 0


class ConvTranspose2d_FR4d(nn.Module):
    """Transposed convolution with a four-dimensional frequency kernel."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        minrate=0.1,
        droprate=0.001,
        dropspeed=-1,
        dynamicdrop=False,
        groups=1,
        directdrop=False,
    ):
        super().__init__()

        weight_shape = (
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(weight_shape).normal_(0, 0.1))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).normal_(0, 0.1))
        else:
            self.bias = None

        self.groups = groups
        self.register_buffer("ZMAT", ZIdx2Int(getZIdx_4d(torch.empty(weight_shape))))
        self.register_buffer("IMAT", torch.zeros(weight_shape) + 1.0)
        self.register_buffer("IDROP", torch.zeros(weight_shape) + 1.0)

        self.minrate = minrate
        self.minnum = max(
            round(
                out_channels
                * in_channels
                * kernel_size
                * kernel_size
                * minrate
                // groups
            ),
            _TRANSPOSE_PROTECTED_PARAMETERS,
        )
        self.dynamicdrop = dynamicdrop
        self.dropcnt = self.ZMAT.numel()
        self.dropspeed = dropspeed if dropspeed > 0 else droprate * self.ZMAT.numel()
        self.weightrate = 0
        self.weightnum = -1

        if directdrop:
            self.dropcnt = self.minnum + 1

    def reset(self):
        """Reset the mask and restart the dropping counter."""

        self.minnum = max(
            round(self.weight.numel() * self.minrate // self.groups),
            16,
        )
        self.IDROP.fill_(1.0)
        self.dropcnt = self.minnum + 10

    def forward(self, data):
        weight = idct_4d(self.weight)
        return F.conv_transpose2d(
            data,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


class ConvTranspose2d(ConvTranspose2d_FR4d):
    """Public alias for :class:`ConvTranspose2d_FR4d`."""


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
    "idct_2d",
    "idct_4d",
]
