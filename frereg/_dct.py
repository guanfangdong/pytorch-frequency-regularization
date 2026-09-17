"""Inverse discrete cosine transform utilities used by :mod:`frereg`.

The implementation follows the original project closely. In particular, the
transform is implemented with PyTorch operations so gradients can flow back
to the learnable frequency-domain weights.
"""

import numpy as np
import torch

try:
    # PyTorch 1.x and later expose the complex FFT API through torch.fft.
    import torch.fft

    def _dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    def _dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def _idct_irfft_impl(values):
        return torch.fft.irfft(
            torch.view_as_complex(values),
            n=values.shape[1],
            dim=1,
        )

except ImportError:  # pragma: no cover - retained for older PyTorch releases

    def _dct1_rfft_impl(x):
        return torch.rfft(x, 1)

    def _dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def _idct_irfft_impl(values):
        return torch.irfft(values, 1, onesided=False)


# Keep the original helper names available for callers that imported them
# directly from this private module.
dct1_rfft_impl = _dct1_rfft_impl
dct_fft_impl = _dct_fft_impl
idct_irfft_impl = _idct_irfft_impl


def idct(X, norm=None):
    """Apply an inverse DCT to the last dimension of ``X``.

    Parameters
    ----------
    X:
        Input tensor. All dimensions except the last one are treated as
        independent signals.
    norm:
        Use ``"ortho"`` for the orthonormal normalization used by the model.
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def idct_2d(X, norm="ortho"):
    """Apply the inverse DCT over the final two dimensions of ``X``."""

    transformed = idct(X, norm=norm)
    transformed = idct(transformed.transpose(-1, -2), norm=norm)
    return transformed.transpose(-1, -2)


def idct_4d(X, norm="ortho"):
    """Apply the inverse DCT over all four dimensions of ``X``."""

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    x4 = idct(x3.transpose(-1, -4), norm=norm)
    return x4.transpose(-1, -4).transpose(-1, -3).transpose(-1, -2)


__all__ = ["idct", "idct_2d", "idct_4d"]
