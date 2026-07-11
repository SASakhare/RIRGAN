"""
Stage 2 - Adversarial (GAN) fine-tuning.

Loads the pretrained RIR-G generator (from train_pretrain.py) and jointly
trains it against the RaD relativistic-average discriminator, using a
weighted combination of pixel, perceptual, adversarial, and total-variation
losses for the generator, and the relativistic loss for the discriminator.

Usage:
    python train_gan.py --pretrained_gen checkpoints/gen_pretrain_200.pth
"""

import argparse
import json

import torch
import torchvision
from torchmetrics.image import TotalVariation

from configs.config import GENERATOR_CONFIG, DISCRIMINATOR_CONFIG, DATA_CONFIG, GAN_CONFIG
from data.dataset import build_dataloaders
from losses.losses import (
    RaDLossFun, RIRGAdversarialLoss, RIRPixelLoss, RIRPerceptualLoss, RIRTotalVariation
)
from models.discriminator import Rad
from models.generator import RIRG
from utils.visualization import ON_OFF_Grad, save_MRI_images


def main(pretrained_gen_path: str):
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
    gen = RIRG(**GENERATOR_CONFIG)
    gen.load_state_dict(torch.load(pretrained_gen_path, map_location='cpu'))
    gen = gen.to(device)

    dis = Rad(DISCRIMINATOR_CONFIG).to(device)

    vgg19 = torchvision.models.vgg19(torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    vgg19_feature_extractor = vgg19.features[:35].to(device)
    ON_OFF_Grad(vgg19_feature_extractor, 'OFF')

    total_variation_fn = TotalVariation().to(device)

    # ---- Optimizers ----
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=GAN_CONFIG["gen_lr"], betas=GAN_CONFIG["gen_betas"])
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=GAN_CONFIG["dis_lr"], betas=GAN_CONFIG["dis_betas"])

    LAMBDA, GAMMA, BETA, ETA = GAN_CONFIG["LAMBDA"], GAN_CONFIG["GAMMA"], GAN_CONFIG["BETA"], GAN_CONFIG["ETA"]
    EPOCHS = GAN_CONFIG["epochs"]

    history = {"gen": {"total": [], "pixel": [], "perceptual": [], "adversarial": [], "tv": []},
               "dis": {"loss": []}}

    for epoch in range(1, EPOCHS + 1):

        # ============= Discriminator step =============
        gen.eval()
        dis.train()
        ON_OFF_Grad(gen, "OFF")
        ON_OFF_Grad(dis, "ON")

        running_loss_dis = 0.0
        for LR_Images, HR_Images in train_loader:
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)

            optimizer_dis.zero_grad()

            with torch.no_grad():
                SR_Images = gen(LR_Images)

            sr_output = dis(SR_Images)
            hr_output = dis(HR_Images)

            loss_dis = RaDLossFun(HR_Output=hr_output, SR_Output=sr_output, device=device)
            loss_dis.backward()
            optimizer_dis.step()

            running_loss_dis += loss_dis.item()

        history["dis"]["loss"].append(running_loss_dis / len(train_loader))

        # ============= Generator step =============
        gen.train()
        dis.eval()
        ON_OFF_Grad(gen, "ON")
        ON_OFF_Grad(dis, "OFF")

        running = {"total": 0.0, "pixel": 0.0, "perceptual": 0.0, "adversarial": 0.0, "tv": 0.0}

        for LR_Images, HR_Images in train_loader:
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)

            optimizer_gen.zero_grad()

            SR_Images = gen(LR_Images)

            sr_output = dis(SR_Images)
            hr_output = dis(HR_Images)

            loss_pixel = RIRPixelLoss(HR_Images=HR_Images, SR_Images=SR_Images)
            loss_perceptual = RIRPerceptualLoss(
                SR_Images=SR_Images, HR_Images=HR_Images,
                vgg19_feature_extractor=vgg19_feature_extractor, device=device
            )
            loss_adversarial = RIRGAdversarialLoss(HR_Output=hr_output, SR_Output=sr_output, device=device)
            loss_tv = RIRTotalVariation(SR_Images, total_variation_fn, device)

            total_loss = (loss_pixel * LAMBDA) + (loss_perceptual * GAMMA) + \
                         (loss_adversarial * ETA) + (loss_tv * BETA)

            total_loss.backward()
            optimizer_gen.step()

            running["total"] += total_loss.item()
            running["pixel"] += loss_pixel.item()
            running["perceptual"] += loss_perceptual.item()
            running["adversarial"] += loss_adversarial.item()
            running["tv"] += loss_tv.item()

        for k in running:
            history["gen"][k].append(running[k] / len(train_loader))

        # ============= Sample image + logging =============
        with torch.no_grad():
            LR_Images, HR_Images = next(iter(val_loader))
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)
            SR_Images = gen(LR_Images)

            save_MRI_images(
                LR_Images[0][0].cpu(), SR_Images[0][0].cpu(), HR_Images[0][0].cpu(),
                title1='LR', title2='SR', title3='HR',
                image_name=f'{epoch}', fold='outputs/gan_samples'
            )

        print(f'Epoch : {epoch}/{EPOCHS} | '
              f'Gen Total Loss : {history["gen"]["total"][-1]:.5f} | '
              f'Dis Loss : {history["dis"]["loss"][-1]:.5f}')

        if epoch % 25 == 0:
            torch.save(gen.state_dict(), f'checkpoints/gen_gan_{epoch}.pth')

    with open('outputs/gan_history.json', 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_gen', type=str, required=True,
                         help='Path to a generator checkpoint produced by train_pretrain.py')
    args = parser.parse_args()
    main(args.pretrained_gen)
