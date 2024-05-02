import torch.nn as nn
from typing import Tuple


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float=0.1):
        super().__init__()

        subnet = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.BatchNorm2d(num_features=out_channels)
        ]

        self.subnet = nn.Sequential(*subnet)

    def forward(self, x):
        return self.subnet(x)


class HandsDiscriminator(nn.Module):
    def __init__(
            self,
            img_shape: Tuple[int, int],
            latent_dim: int
    ):
        super().__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim
        channels = self.latent_dim

        layers = []

        initial_layer = DiscriminatorBlock(
            in_channels=3,
            out_channels=channels
        )
        layers.append(initial_layer)

        for middle_layers in range(self._calculate_num_conv_layers()-2):
            layers.append(
                DiscriminatorBlock(
                    in_channels=channels,
                    out_channels=channels+self.latent_dim
                )
            )
            channels += self.latent_dim

        output_layer = DiscriminatorBlock(
            in_channels=channels,
            out_channels=1,
        )
        layers.append(output_layer)

        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def _calculate_num_conv_layers(self):

        power_of_two = 0
        divider = self.img_shape[0]
        while divider > 1:
            divider //= 2
            power_of_two += 1
        return power_of_two-2
    
    def forward(self, x):
        return self.discriminator(x)