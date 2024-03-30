import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels, negative_slope: float=0.02):
        super().__init__()

        subnet = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels)
        ]

        self.subnet = nn.Sequential(*subnet)

    def forward(self, x):
        return self.subnet(x)


class HandsGenerator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int]):
        super().__init__()

        self.img_shape = img_shape

        if self.img_shape[0] == self.img_shape[1]:
            pass
        else:
            raise ValueError("Please provide the size of the image to generate with te same dimentions. "
                             "Example: img_shape = (256, 256).")
        
        layers = []

        initial_layer = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=self._calculate_num_conv_layers(),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        layers.append(initial_layer)

        for _ in range(self._calculate_num_conv_layers()):
            layers.append(
                GeneratorBlock(
                    in_channels=self._calculate_num_conv_layers(),
                    out_channels=self._calculate_num_conv_layers()
                )
            )

        output_layer = nn.ConvTranspose2d(
            in_channels=self._calculate_num_conv_layers(),
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        layers.append(output_layer)

        self.generator = nn.Sequential(*layers)

    def _calculate_num_conv_layers(self):

        power_of_two = 0
        divider = self.img_shape[0]
        while divider > 1:
            divider //= 2
            power_of_two += 1
        return power_of_two

    def forward(self, x):
        return self.generator(x)