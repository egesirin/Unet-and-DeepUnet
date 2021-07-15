import torch


def dice_loss(prediction, target):
    epsilon = 1e-3
    prediction_flat = prediction.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (prediction_flat * target_flat).sum()
    prediction_sum = torch.sum(prediction_flat)
    target_sum = torch.sum(target_flat)
    return (2. * intersection + epsilon) / (prediction_sum + target_sum + epsilon)
