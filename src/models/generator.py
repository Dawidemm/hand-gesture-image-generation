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
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]

        self.subnet = nn.Sequential(*subnet)

    def forward(self, x):
        return self.subnet(x)


class HandsGenerator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int], latent_dim: int=32):
        super().__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        if self.img_shape[0] == self.img_shape[1]:
            pass
        else:
            raise ValueError("Please provide the size of the image to generate with te same dimentions. "
                             "Example: img_shape = (256, 256).")
        
        layers = []

        initial_layer = nn.ConvTranspose2d(
            in_channels=self.latent_dim,
            out_channels=2*self.latent_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        layers.append(initial_layer)

        for i in range(self._calculate_num_conv_layers()):
            layers.append(
                GeneratorBlock(
                    in_channels=2*self.latent_dim,
                    out_channels=2*self.latent_dim
                )
            )

        output_layer = nn.ConvTranspose2d(
            in_channels=2*self.latent_dim,
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
    
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img