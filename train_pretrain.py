"""
Stage 1 - Generator pretraining.

Trains the RIR-G generator alone (no discriminator) using a combination of
pixel-wise L1 loss and VGG19 perceptual loss. This gives the generator a
strong initialization before adversarial (GAN) fine-tuning in train_gan.py.

Usage:
    python train_pretrain.py
"""

import json

import torch
import torchvision

from configs.config import GENERATOR_CONFIG, DATA_CONFIG, PRETRAIN_CONFIG
from data.dataset import build_dataloaders
from losses.losses import RIRPixelLoss, RIRPerceptualLoss
from models.generator import RIRG
from utils.visualization import ON_OFF_Grad, save_MRI_images


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    # ---- Data ----
    train_loader, val_loader, _ = build_dataloaders(
        train_path=DATA_CONFIG["train_dataset_path"],
        val_path=DATA_CONFIG["val_dataset_path"],
        test_path=DATA_CONFIG["test_dataset_path"],
        train_val_size=DATA_CONFIG["train_val_size"],
        test_size=DATA_CONFIG["test_size"],
        batch_size=DATA_CONFIG["batch_size"],
    )

    # ---- Models ----
    gen = RIRG(**GENERATOR_CONFIG).to(device)

    vgg19 = torchvision.models.vgg19(torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    vgg19_feature_extractor = vgg19.features[:35].to(device)
    ON_OFF_Grad(vgg19_feature_extractor, 'OFF')

    # ---- Optimizer / scheduler ----
    optimizer_gen = torch.optim.Adam(
        gen.parameters(),
        lr=PRETRAIN_CONFIG["lr"],
        betas=PRETRAIN_CONFIG["betas"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer_gen,
        step_size=PRETRAIN_CONFIG["scheduler_step_size"],
        gamma=PRETRAIN_CONFIG["scheduler_gamma"]
    )

    perceptual_weight = PRETRAIN_CONFIG["perceptual_loss_weight"]
    EPOCHS = PRETRAIN_CONFIG["epochs"]

    total_losses_train, total_losses_val = [], []
    pixel_losses_train, pixel_losses_val = [], []
    perceptual_losses_train, perceptual_losses_val = [], []

    for epoch in range(1, EPOCHS + 1):

        # ---------------- Training ----------------
        gen.train()
        running_total, running_pixel, running_perceptual = 0.0, 0.0, 0.0

        for LR_Images, HR_Images in train_loader:
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)

            optimizer_gen.zero_grad()

            SR_Images = gen(LR_Images)

            loss_pixel = RIRPixelLoss(HR_Images=HR_Images, SR_Images=SR_Images)
            loss_perceptual = RIRPerceptualLoss(
                SR_Images=SR_Images, HR_Images=HR_Images,
                vgg19_feature_extractor=vgg19_feature_extractor, device=device
            )

            total_loss = loss_pixel + loss_perceptual * perceptual_weight

            total_loss.backward()
            optimizer_gen.step()

            running_total += total_loss.item()
            running_pixel += loss_pixel.item()
            running_perceptual += loss_perceptual.item()

        total_losses_train.append(running_total / len(train_loader))
        pixel_losses_train.append(running_pixel / len(train_loader))
        perceptual_losses_train.append(running_perceptual / len(train_loader))

        # ---------------- Validation ----------------
        gen.eval()
        running_total, running_pixel, running_perceptual = 0.0, 0.0, 0.0

        with torch.no_grad():
            for LR_Images, HR_Images in val_loader:
                LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)

                SR_Images = gen(LR_Images)

                loss_pixel = RIRPixelLoss(HR_Images=HR_Images, SR_Images=SR_Images)
                loss_perceptual = RIRPerceptualLoss(
                    SR_Images=SR_Images, HR_Images=HR_Images,
                    vgg19_feature_extractor=vgg19_feature_extractor, device=device
                )
                total_loss = loss_pixel + loss_perceptual * perceptual_weight

                running_total += total_loss.item()
                running_pixel += loss_pixel.item()
                running_perceptual += loss_perceptual.item()

        total_losses_val.append(running_total / len(val_loader))
        pixel_losses_val.append(running_pixel / len(val_loader))
        perceptual_losses_val.append(running_perceptual / len(val_loader))

        # ---------------- Sample image ----------------
        with torch.no_grad():
            LR_Images, HR_Images = next(iter(val_loader))
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)
            SR_Images = gen(LR_Images)

            save_MRI_images(
                LR_Images[0][0].cpu(), SR_Images[0][0].cpu(), HR_Images[0][0].cpu(),
                title1='LR', title2='SR', title3='HR',
                image_name=f'{epoch}', fold='outputs/pretrain_samples'
            )

        scheduler.step()

        print(f'Epoch : {epoch}/{EPOCHS} | '
              f'Train Total Loss : {total_losses_train[-1]:.5f} , '
              f'Val Total Loss : {total_losses_val[-1]:.5f}')

        if epoch % 100 == 0:
            torch.save(gen.state_dict(), f'checkpoints/gen_pretrain_{epoch}.pth')

    # ---- Save loss history ----
    history = {
        'train': {
            'total_loss': total_losses_train,
            'pixel_loss': pixel_losses_train,
            'perceptual_loss': perceptual_losses_train,
        },
        'val': {
            'total_loss': total_losses_val,
            'pixel_loss': pixel_losses_val,
            'perceptual_loss': perceptual_losses_val,
        }
    }
    with open('outputs/pretrain_history.json', 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == '__main__':
    main()
