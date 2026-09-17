"""Dice coefficient and Dice loss helpers for segmentation masks."""

import torch
from torch import Tensor


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    """Compute the Dice coefficient for binary masks."""

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    """Average Dice coefficient over all classes."""

    return dice_coeff(
        input.flatten(0, 1),
        target.flatten(0, 1),
        reduce_batch_first,
        epsilon,
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """Return one minus the Dice coefficient."""

    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
