"""
Evaluate a trained RIR-G generator on the test set using PSNR and SSIM.

Usage:
    python evaluate.py --checkpoint checkpoints/gen_gan_200.pth
"""

import argparse

import numpy as np
import torch

from configs.config import GENERATOR_CONFIG, DATA_CONFIG
from data.dataset import build_dataloaders
from metrics.metrics import PSNR, SSIM
from models.generator import RIRG


def main(checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = build_dataloaders(
        train_path=DATA_CONFIG["train_dataset_path"],
        val_path=DATA_CONFIG["val_dataset_path"],
        test_path=DATA_CONFIG["test_dataset_path"],
        train_val_size=DATA_CONFIG["train_val_size"],
        test_size=DATA_CONFIG["test_size"],
        batch_size=DATA_CONFIG["batch_size"],
    )

    gen = RIRG(**GENERATOR_CONFIG)
    gen.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    gen = gen.to(device)
    gen.eval()

    psnr_scores, ssim_scores = [], []

    with torch.no_grad():
        for LR_Images, HR_Images in test_loader:
            LR_Images, HR_Images = LR_Images.to(device), HR_Images.to(device)
            SR_Images = gen(LR_Images)

            psnr_scores.append(PSNR(SR_Images.cpu(), HR_Images.cpu()))
            ssim_scores.append(SSIM(SR_Images.cpu(), HR_Images.cpu(), K1=0.01, K2=0.2))

    psnr_scores = np.concatenate(psnr_scores)
    ssim_scores = np.concatenate(ssim_scores)

    print(f'PSNR : mean={psnr_scores.mean():.4f} dB , std={psnr_scores.std():.4f}')
    print(f'SSIM : mean={ssim_scores.mean():.5f} , std={ssim_scores.std():.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.checkpoint)
