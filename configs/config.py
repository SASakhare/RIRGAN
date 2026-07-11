"""
Central configuration for RIRGAN: generator hyper-parameters, discriminator
architecture spec, and loss-weighting / optimizer settings used during
pretraining and adversarial (GAN) training.
"""

# ----------------------------------------------------------------------------
# Generator (RIR-G) hyper-parameters
# ----------------------------------------------------------------------------
GENERATOR_CONFIG = {
    "in_channel": 1,
    "out_channel": 1,
    "in_RIR_block_channel": 64,
    "kernel_size_in_out_layer": 9,
    "kerne_size_RIR_block": 3,
    "no_of_RIR_Block": 8,
    "no_of_ERes_block": 5,
    "sub_PixelLayer_channel": 256,
    "alpha": 0.2,
    "scale": 2,  # each SubPixelLayer doubles resolution -> 2 layers => 4x total upscale
}

# ----------------------------------------------------------------------------
# Discriminator (RaD) architecture spec
# ----------------------------------------------------------------------------
DISCRIMINATOR_CONFIG = {
    'input_layer': {
        'input_channel': 1,
        'k': 3,
        'num_of_channels': 64,
        'stride': 1
    },
    'Rad_Repeat_block1': {
        'input_channel': 64,
        'k': 3,
        'num_of_channels': 64,
        'stride': 2
    },
    'Rad_Repeat_block2': {
        'input_channel': 64,
        'k': 3,
        'num_of_channels': 128,
        'stride': 1
    },
    'Rad_Repeat_block3': {
        'input_channel': 128,
        'k': 3,
        'num_of_channels': 128,
        'stride': 2
    },
    'Rad_Repeat_block4': {
        'input_channel': 128,
        'k': 3,
        'num_of_channels': 256,
        'stride': 1
    },
    'Rad_Repeat_block5': {
        'input_channel': 256,
        'k': 3,
        'num_of_channels': 256,
        'stride': 2
    },
    'Rad_Repeat_block6': {
        'input_channel': 256,
        'k': 3,
        'num_of_channels': 512,
        'stride': 1
    },
    'Rad_Repeat_block7': {
        'input_channel': 512,
        'k': 3,
        'num_of_channels': 512,
        'stride': 2
    },
    'RadOutput': {
        'input_features_1': 4608,
        'output_features_1': 1024,
        'output_features_2': 1,
    },
}

# ----------------------------------------------------------------------------
# Dataset / dataloader settings
# ----------------------------------------------------------------------------
DATA_CONFIG = {
    "train_dataset_path": "./data/train",
    "val_dataset_path": "./data/val",
    "test_dataset_path": "./data/test",
    "train_val_size": 96,   # HR crop size for train/val (LR = size // 4)
    "test_size": 128,       # HR crop size for test
    "batch_size": 16,
    "noise_mean": 0,
    "noise_std": 0.0005,
}

# ----------------------------------------------------------------------------
# Stage 1: Generator-only pretraining settings
# ----------------------------------------------------------------------------
PRETRAIN_CONFIG = {
    "epochs": 200,
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "scheduler_step_size": 20,
    "scheduler_gamma": 0.5,
    "perceptual_loss_weight": 0.01,  # total_loss = pixel_loss + 0.01 * perceptual_loss
}

# ----------------------------------------------------------------------------
# Stage 2: Adversarial (GAN) training settings
# ----------------------------------------------------------------------------
GAN_CONFIG = {
    "epochs": 200,
    "gen_lr": 1e-4,
    "gen_betas": (0.5, 0.999),
    "dis_lr": 1e-7,
    "dis_betas": (0.5, 0.99),
    # Loss weighting: total_gen_loss = LAMBDA*pixel + GAMMA*perceptual + ETA*adversarial + BETA*total_variation
    "LAMBDA": 1.0,
    "GAMMA": 0.01,
    "BETA": 1e-8,
    "ETA": 0.001,
}
