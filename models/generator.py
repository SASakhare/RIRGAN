"""
RIR-G : Residual-in-Residual Generator for MRI Super-Resolution.

Architecture (paper-inspired, RIRGAN):
    Input Layer -> [RIR Blocks (each made of Enhanced Residual Blocks)] ->
    Conv + Global Skip Connection -> Sub-Pixel (PixelShuffle) upscaling x2 twice ->
    Output Layer (Tanh) -> Super-Resolved image
"""

import torch
from torch import nn


class EResidualBlock(nn.Module):
    """Enhanced Residual Block: Conv -> PReLU -> Conv, with a local skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1
        )

        self.PRelu = nn.PReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1
        )

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.PRelu(output)
        output = self.conv2(output)
        output = output + inputs  # local residual connection
        return output


class RIRBlock(nn.Module):
    """Residual-in-Residual Block: a stack of EResidualBlocks with a scaled skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 num_of_ERes_block: int, alpha: int):
        super().__init__()
        self.alpha = alpha
        self.RIR_Block = nn.Sequential(
            *[
                EResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size
                )
                for _ in range(num_of_ERes_block)
            ]
        )

    def forward(self, inputs):
        output = self.RIR_Block(inputs)
        output = inputs + (self.alpha) * output
        return output


class SubPixelLayer(nn.Module):
    """Conv -> PixelShuffle -> PReLU, used to progressively upscale feature maps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, scale: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1
        )

        self.pixelShuffler = nn.PixelShuffle(upscale_factor=scale)
        self.PRelu = nn.PReLU()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.pixelShuffler(output)
        output = self.PRelu(output)
        return output


class RIRGInputLayer(nn.Module):
    """Initial feature extraction layer: Conv -> PReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1
        )

        self.PRelu = nn.PReLU()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.PRelu(output)
        return output


class RIRGOutputLayer(nn.Module):
    """Final reconstruction layer: Conv -> Tanh, maps features back to image space [-1, 1]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1
        )

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.tanh(output)
        return output


class RIRG(nn.Module):
    """
    Full RIR-based Generator.

    Pipeline:
        LR image -> Input Layer -> N x RIR Blocks -> Conv -> + (Global Skip Connection)
                  -> 2x Sub-Pixel upscaling stages -> Output Layer -> SR image
    """

    def __init__(self, in_channel: int, out_channel: int, in_RIR_block_channel: int,
                 kernel_size_in_out_layer: int, kerne_size_RIR_block: int,
                 no_of_RIR_Block: int, no_of_ERes_block: int,
                 sub_PixelLayer_channel: int, alpha: int, scale: int):
        super().__init__()

        self.inputLayer = RIRGInputLayer(
            in_channels=in_channel,
            out_channels=in_RIR_block_channel,
            kernel_size=kernel_size_in_out_layer
        )

        self.FeatureExtractionLayer = nn.Sequential(
            *[
                RIRBlock(
                    in_channels=in_RIR_block_channel,
                    out_channels=in_RIR_block_channel,
                    kernel_size=kerne_size_RIR_block,
                    num_of_ERes_block=no_of_ERes_block,
                    alpha=alpha
                )
                for _ in range(no_of_RIR_Block)
            ]
        )

        self.conv = nn.Conv2d(
            in_channels=in_RIR_block_channel,
            out_channels=in_RIR_block_channel,
            kernel_size=kerne_size_RIR_block,
            padding='same',
            stride=1
        )

        self.SubPixelLayer1 = SubPixelLayer(
            in_channels=in_RIR_block_channel,
            out_channels=sub_PixelLayer_channel,
            kernel_size=kerne_size_RIR_block,
            scale=scale
        )
        self.SubPixelLayer2 = SubPixelLayer(
            in_channels=in_RIR_block_channel,
            out_channels=sub_PixelLayer_channel,
            kernel_size=kerne_size_RIR_block,
            scale=2
        )

        self.OutputLayer = RIRGOutputLayer(
            in_channels=in_RIR_block_channel,
            out_channels=out_channel,
            kernel_size=kernel_size_in_out_layer
        )

    def forward(self, inputs):
        # input layer
        out_inputLayer = self.inputLayer(inputs)

        # feature extraction layer (stack of RIR blocks)
        out_featureExtra_layer = self.FeatureExtractionLayer(out_inputLayer)

        # conv layer
        out_conv_layer = self.conv(out_featureExtra_layer)

        # Global Residual/Skip Connection (GSC)
        out_GSC = out_conv_layer + out_inputLayer

        # Sub-Pixel (enlarging) layers -> total 4x upscale
        out_pixel_layer1 = self.SubPixelLayer1(out_GSC)
        out_pixel_layer2 = self.SubPixelLayer2(out_pixel_layer1)

        # Output / reconstruction layer
        output = self.OutputLayer(out_pixel_layer2)

        return output
