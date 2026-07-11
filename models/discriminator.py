"""
RaD : Relativistic average Discriminator for RIRGAN.

Architecture:
    Input Layer -> 7 x Repeat Blocks (Conv -> BatchNorm -> LeakyReLU, alternating stride) ->
    Flatten -> Dense(1024) -> LeakyReLU -> Dense(1) -> real/fake relativistic score
"""

from torch import nn


class RadInputLayer(nn.Module):
    """Initial feature layer: Conv -> LeakyReLU (no batch norm, as in SRGAN-style discriminators)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same'
        )

        self.LRelu = nn.LeakyReLU()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.LRelu(output)
        return output


class RadRepeatBlock(nn.Module):
    """Repeated conv block: Conv -> BatchNorm -> LeakyReLU. Stride 2 blocks progressively downsample."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

        self.BNLayer = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.LRelu = nn.LeakyReLU()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.BNLayer(output)
        output = self.LRelu(output)
        return output


class RadOutputLayer(nn.Module):
    """Classifier head: Flatten features -> Dense -> LeakyReLU -> Dense -> single logit."""

    def __init__(self, in_features_1: int, out_features_1: int, out_features_2: int, bias: bool = False):
        super().__init__()

        self.Dense1 = nn.Linear(
            in_features=in_features_1,
            out_features=out_features_1,
            bias=bias,
        )

        self.LRelu = nn.LeakyReLU()

        self.Dense2 = nn.Linear(
            in_features=out_features_1,
            out_features=out_features_2,
            bias=bias
        )

    def forward(self, inputs):
        output = self.Dense1(inputs)
        output = self.LRelu(output)
        output = self.Dense2(output)
        return output


class Rad(nn.Module):
    """
    Full Relativistic average Discriminator (RaD).

    `params` is a config dict describing the input layer, 7 repeat blocks, and the
    output head. See configs/config.py for the exact structure used in this project.
    """

    def __init__(self, params: dict):
        super().__init__()

        self.InputLayer = RadInputLayer(
            in_channels=params['input_layer']['input_channel'],
            out_channels=params['input_layer']['num_of_channels'],
            kernel_size=params['input_layer']['k']
        )

        self.InnerLayer = nn.Sequential(
            *[
                RadRepeatBlock(
                    in_channels=param['input_channel'],
                    out_channels=param['num_of_channels'],
                    kernel_size=param['k'],
                    stride=param['stride']
                )
                for ind, (name, param) in enumerate(params.items()) if ind in range(1, 8)
            ]
        )

        self.FlattenLayer = nn.Flatten()

        self.OutPutLayer = RadOutputLayer(
            in_features_1=params['RadOutput']['input_features_1'],
            out_features_1=params['RadOutput']['output_features_1'],
            out_features_2=params['RadOutput']['output_features_2']
        )

    def forward(self, inputs):
        output = self.InputLayer(inputs)
        output = self.InnerLayer(output)
        output = self.FlattenLayer(output)
        output = self.OutPutLayer(output)
        return output
