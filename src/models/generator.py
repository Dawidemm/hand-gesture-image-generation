import torch.nn as nn
from typing import Tuple


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        self.subnet = nn.Sequential(*subnet)

    def forward(self, x):
        return self.subnet(x)


class HandsGenerator(nn.Module):
    def __init__(
            self, 
            img_shape: Tuple[int, int], 
            latent_dim: int
    ):
        super().__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        if self.img_shape[0] == self.img_shape[1]:
            pass
        else:
            raise ValueError("Please provide the size of the image to generate with te same dimentions. "
                             "Example: img_shape = (256, 256).")
        
        layers = []

        initial_block = GeneratorBlock(
            in_channels=self.latent_dim,
            out_channels=4*self.latent_dim
        )
        layers.append(initial_block)

        self.latent_dim = 4*self.latent_dim
        
        for block in range(self._calculate_num_conv_layers()-2):
            layers.append(
                GeneratorBlock(
                    in_channels=self.latent_dim,
                    out_channels=int(self.latent_dim/2)
                )
            )
            self.latent_dim = int(self.latent_dim/2)

        output_layer = nn.ConvTranspose2d(
            in_channels=self.latent_dim,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        layers.append(output_layer)

        layers.append(nn.Tanh())

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