"""
Dataset and data-utility functions for RIRGAN, built on the BraTS MRI dataset.

Each MRI volume is stored as a per-slice .h5 file with an 'image' dataset of
shape (H, W, 4) - one channel per MRI modality (we use channel index 1: the
Enhanced Tumor / T1ce-derived channel). A fixed-size HR patch is cropped from
the slice, Gaussian noise is added to emulate scanner noise, and a bicubic
downsample produces the paired LR input.
"""

import os
import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def AddGaussianNoise(image: np.array, mean: int = 0, std: int = 1):
    """Add Gaussian noise to an image normalized in [-1, 1]."""
    # note: input image is in [-1, 1] so we denormalize first.
    image = (image + 1) / 2
    noise = np.random.normal(mean, std, image.shape)

    image = image + noise

    # re-normalize back to [-1, 1]
    image = (image * 2 - 1)

    return image


def upscale_image(image: np.array, H: int, W: int):
    """Bicubic-resize an image to (H, W)."""
    image = np.array(image)

    image = cv2.resize(
        image,
        (H, W),
        interpolation=cv2.INTER_CUBIC
    )

    return image


class RIRGAN_Dataset(Dataset):
    """
    Loads BraTS-style .h5 MRI slices, crops a fixed-size HR patch (Enhanced
    Tumor channel), adds Gaussian noise, and bicubic-downsamples 4x to
    produce the paired LR/HR training pair used for super-resolution.
    """

    def __init__(self, dataset_path: str, size: int, add_noise: bool, mean: int = 0, std: int = 0.0005):
        super().__init__()

        self.all_files = list()
        self.size = size
        self.add_noise = add_noise
        self.mean = mean
        self.std = std

        for vol in os.listdir(dataset_path):
            vol_fold_path = f'{dataset_path}/{vol}'
            for file in os.listdir(vol_fold_path):
                file_path = f'{vol_fold_path}/{file}'
                self.all_files.append(file_path)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        file_path = self.all_files[index]

        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]

        offset_x = 20
        offset_y = 0

        H, W, _ = image.shape

        center_x = W // 2
        center_y = H // 2

        image = image[:, :, 1]  # index=1 -> Enhanced Tumor (Te) channel

        # crop the HR patch around the (offset) center
        size = self.size // 2
        img_hr = image[
            center_y - size + offset_y: center_y + size + offset_y,
            center_x - size + offset_x: center_x + size + offset_x
        ]

        hr_image_noise = AddGaussianNoise(img_hr.copy(), mean=self.mean, std=self.std)

        lr_image = cv2.resize(
            hr_image_noise,
            (self.size // 4, self.size // 4),
            interpolation=cv2.INTER_CUBIC
        )

        lr_image = torch.tensor(lr_image).float().reshape((1, self.size // 4, self.size // 4))
        hr_image = torch.tensor(img_hr).float().reshape((1, self.size, self.size))

        return lr_image, hr_image


def build_dataloaders(train_path: str, val_path: str, test_path: str,
                       train_val_size: int, test_size: int, batch_size: int):
    """Convenience helper that builds train/val/test datasets + dataloaders."""

    train_dataset = RIRGAN_Dataset(train_path, train_val_size, add_noise=True)
    val_dataset = RIRGAN_Dataset(val_path, train_val_size, add_noise=True)
    test_dataset = RIRGAN_Dataset(test_path, test_size, add_noise=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
