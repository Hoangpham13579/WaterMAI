import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):

    assert input.size() == target.size()
    input = input.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (input * target).sum()
    cardinality = input.sum() + target.sum()

    dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
    return dice

def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    assert input.size() == target.size()

    # Flatten the tensors to [N * H * W, C]
    input = input.contiguous().view(-1, input.shape[1])
    target = target.contiguous().view(-1, target.shape[1])

    intersection = (input * target).sum(dim=0)
    cardinality = input.sum(dim=0) + target.sum(dim=0)

    dice_per_class = (2.0 * intersection + epsilon) / (cardinality + epsilon)

    # Average over classes
    dice = dice_per_class.mean()
    return dice

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    if multiclass:
        return 1 - multiclass_dice_coeff(input, target)
    else:
        return 1 - dice_coeff(input, target)
