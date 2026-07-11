"""
Visualization and misc helper utilities: MRI image plotting/saving, per-channel
inspection, dataset copy helper, and a switch to freeze/unfreeze module gradients
(used to alternate generator/discriminator training in the GAN stage).
"""

import os

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

from data.dataset import upscale_image


def show_MRI_IMAGES_C(image: np.array):
    """Display all 4 channels of a raw BraTS slice (T1, T1ce, T2, FLAIR-style)."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    fig.align_titles()
    for c in range(4):
        axes[c].imshow((image[:, :, c] + 1) / 2)
        axes[c].set_title(f"Image (240x240x4) channel : {c}")
        axes[c].axis('off')

    plt.tight_layout()
    plt.show()
    return


def save_dataset(dataset_name: str, fold_list: list, src: str, dest: str):
    """Copy a subset of per-volume .h5 slice folders from `src` into a new `dest/dataset_name` tree."""

    dataset_path = f'{dest}/{dataset_name}'
    os.makedirs(dataset_path, exist_ok=True)

    for fold in fold_list:
        src_folder_path = f'{src}/{fold}'
        dest_folder_path = f'{dataset_path}/{fold}'
        os.makedirs(dest_folder_path, exist_ok=True)

        for file in os.listdir(src_folder_path):
            src_file_path = f'{src_folder_path}/{file}'
            dest_file_path = f'{dest_folder_path}/{file}'

            with h5py.File(src_file_path, 'r') as f:
                image = f['image'][:]

            with h5py.File(dest_file_path, 'w') as f:
                f.create_dataset('image', data=image)

    return True


def show_MRI_images(image1: np.array, image2: np.array, title1: str, title2: str):
    """Side-by-side plot of two images (e.g. LR upscaled-for-viewing vs HR)."""

    plt.figure()
    image1, image2 = np.array(image1), np.array(image2)
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes = axes.flatten()

    up_size = image1.shape[0] * 4
    axes[0].imshow(upscale_image(image1, up_size, up_size), cmap='gray')
    axes[0].set_title(f'{title1}')
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(f'{title2}')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def ON_OFF_Grad(model: torch.nn.Module, switch: str):
    """Enable/disable gradient tracking for all parameters of `model` ('ON' or 'OFF')."""

    for layer in model.parameters():
        if switch.upper() == 'ON':
            layer.requires_grad = True
        else:
            layer.requires_grad = False


def save_MRI_images(image1: np.array, image2: np.array, image3: np.array,
                     title1: str, title2: str, title3: str, image_name: str, fold: str):
    """Save a 3-panel comparison figure (typically LR / SR / HR) to `fold/image_name.png`."""

    os.makedirs(fold, exist_ok=True)

    plt.figure()
    image1, image2, image3 = np.array(image1), np.array(image2), np.array(image3)
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    axes = axes.flatten()
    up_size = image1.shape[0] * 4
    axes[0].imshow(upscale_image(image1, up_size, up_size), cmap='gray')
    axes[0].set_title(f'{title1}')
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(f'{title2}')
    axes[1].axis('off')

    axes[2].imshow(image3, cmap='gray')
    axes[2].set_title(f'{title3}')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f"./{fold}/{image_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    return
