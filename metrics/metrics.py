"""
Evaluation metrics for super-resolved MRI images: PSNR and SSIM.

Both metrics are computed per-image in a batch and returned as a numpy array,
so callers can aggregate (mean/std) over an entire test set.
"""

import numpy as np
import torch


def PSNR(SR_Images: torch.Tensor, HR_Images: torch.Tensor):
    """Peak Signal-to-Noise Ratio between super-resolved and ground-truth HR images, in dB."""

    SR_Images = (SR_Images + 1) / 2
    HR_Images = (HR_Images + 1) / 2
    C, H, W = SR_Images[0].shape
    total_pixel_per_image = C * H * W
    psnr = 10 * torch.log10(
        (1 / (((HR_Images - SR_Images) ** 2).sum(dim=(1, 2, 3)) / total_pixel_per_image))
    )

    return np.array(psnr)


def SSIM(SR_Images: torch.Tensor, HR_Images: torch.Tensor, K1: float, K2: float):
    """Structural Similarity Index between super-resolved and ground-truth HR images."""

    B, C, H, W = SR_Images.shape
    Ux = SR_Images.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
    Uy = HR_Images.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)

    Xi_X = SR_Images - (SR_Images.mean(dim=(1, 2, 3), keepdim=True))
    Yi_Y = HR_Images - (HR_Images.mean(dim=(1, 2, 3), keepdim=True))

    covar_x_y = ((Xi_X * Yi_Y).sum(dim=(1, 2, 3)) / (C * H * W)).reshape(B, 1)

    var_x = (Xi_X ** 2).mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
    var_y = (Yi_Y ** 2).mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)

    term1 = ((Ux * Uy * 2) + K1) / ((Ux ** 2) + (Uy ** 2) + K1)
    term2 = (covar_x_y + K2) / (var_x + var_y + K2)

    ssim = (term1 * term2).reshape(B)

    return np.array(ssim)
