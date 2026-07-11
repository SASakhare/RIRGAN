"""
Loss functions used by RIRGAN:

  Discriminator (RaD):
    - RaDLossFun: relativistic average discriminator loss.

  Generator (RIR-G):
    - RIRGAdversarialLoss: relativistic average adversarial loss.
    - RIRPixelLoss: L1 pixel-wise reconstruction loss.
    - RIRPerceptualLoss: VGG19 feature-space perceptual loss.
    - RIRTotalVariation: total-variation regularization for smoothness.
"""

import torch
from torch.nn import functional as F


def RaDLossFun(HR_Output: torch.Tensor, SR_Output: torch.Tensor, device: torch.device = 'cpu'):
    """Relativistic average discriminator loss (wants HR to look 'more real' than SR, on average)."""

    avg_sr_output = SR_Output.mean()
    avg_hr_output = HR_Output.mean()

    relative_real_fake = HR_Output - avg_sr_output
    relative_fake_real = SR_Output - avg_hr_output

    # RaD wants to increase relative_real_fake (real image looks more real than fake, on average)
    loss_real = F.binary_cross_entropy_with_logits(
        input=relative_real_fake,
        target=torch.ones_like(HR_Output).to(device)
    )

    # RaD wants to decrease relative_fake_real (fake image looks less real than real, on average)
    loss_fake = F.binary_cross_entropy_with_logits(
        input=relative_fake_real,
        target=torch.zeros_like(SR_Output).to(device)
    )

    return (loss_real + loss_fake)


def RIRGAdversarialLoss(HR_Output: torch.Tensor, SR_Output: torch.Tensor, device: torch.device = 'cpu'):
    """Relativistic average adversarial loss for the generator (opposite objective to RaD)."""

    avg_hr_output = HR_Output.mean()
    avg_sr_output = SR_Output.mean()

    relative_real_fake = HR_Output - avg_sr_output
    relative_fake_real = SR_Output - avg_hr_output

    # generator wants to decrease relative_real_fake (fool the discriminator)
    loss_real = F.binary_cross_entropy_with_logits(
        input=relative_real_fake,
        target=torch.zeros_like(HR_Output).to(device)
    )

    # generator wants to increase relative_fake_real (fake looks real relative to HR)
    loss_fake = F.binary_cross_entropy_with_logits(
        input=relative_fake_real,
        target=torch.ones_like(SR_Output).to(device)
    )

    return (loss_real + loss_fake)


def RIRPixelLoss(HR_Images: torch.Tensor, SR_Images: torch.Tensor):
    """Mean absolute (L1) pixel-wise error between super-resolved and ground-truth HR images."""

    (B, C, H, W) = HR_Images.shape
    total_pixel = B * C * H * W

    loss = torch.abs(SR_Images - HR_Images).sum() / total_pixel

    return loss


def vgg_normalize(x, device: torch.device = 'cpu'):
    """Normalize an image batch with ImageNet mean/std, as expected by torchvision VGG19."""

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    return (x - imagenet_mean) / imagenet_std


def RIRPerceptualLoss(SR_Images: torch.Tensor, HR_Images: torch.Tensor, vgg19_feature_extractor: torch.nn.Module,
                       device: torch.device = 'cpu'):
    """
    Perceptual (feature-space) loss: MSE between VGG19 conv features of the
    SR and HR images. Single-channel MRI images are repeated to 3 channels
    to match VGG19's expected input.
    """

    sr_higher_feature_dim_space = vgg19_feature_extractor(
        vgg_normalize(((SR_Images + 1) / 2).repeat(1, 3, 1, 1), device=device))
    hr_higher_feature_dim_space = vgg19_feature_extractor(
        vgg_normalize(((HR_Images + 1) / 2).repeat(1, 3, 1, 1), device=device))

    perceptual_loss = ((sr_higher_feature_dim_space - hr_higher_feature_dim_space) ** 2).mean()

    return perceptual_loss


def RIRTotalVariation(Images: torch.Tensor, fun, device: torch.device = 'cpu'):
    """Total variation loss (encourages spatial smoothness), using torchmetrics.image.TotalVariation."""

    return fun(Images.to(device))
