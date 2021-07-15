import torch
import torch.nn.functional as F


def interpolation(target, n):
    targets = []
    m = target.shape[-1]
    for i in range(n-1):
        j = n - i - 1
        k = int(m / 2 ** j)
        itarget = F.interpolate(target, size=k, mode='trilinear', align_corners=False)
        itarget = (itarget > 0.5).to(torch.float32)
        targets.append(itarget)
    targets.append(target)
    return targets
