import os
import torch
import lightning
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt

from src.models.generator import HandsGenerator
from src.models.discriminator import HandsDiscriminator

from typing import Tuple


class HandsGAN(lightning.LightningModule):
    def __init__(
            self,
            input_size: Tuple[int, int],
            latent_dim: int,
            learning_rate: float=0.0005
    ):
        super().__init__()

        self.save_hyperparameters('input_size', 'latent_dim', 'learning_rate')

        self.automatic_optimization = False

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self.generator = HandsGenerator(
            img_shape=self.input_size,
            latent_dim=self.latent_dim
        )

        self.discriminator = HandsDiscriminator(
            img_shape=self.input_size,
            latent_dim=self.latent_dim
        )

        self.epoch = -1

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):

        X, _ = batch

        opt_generator, opt_discriminator = self.optimizers()

        z = torch.randn(X.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(X)

        self.toggle_optimizer(opt_generator)
        self.generated_imgs = self(z)

        valid = torch.ones(X.size(0), 1, 4, 4)
        valid = valid.type_as(X)

        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_generator.step()
        opt_generator.zero_grad()
        self.untoggle_optimizer(opt_generator)

        self.toggle_optimizer(opt_discriminator)

        valid = torch.ones(X.size(0), 1, 4, 4)
        valid = valid.type_as(X)

        real_loss = self.adversarial_loss(self.discriminator(X), valid)

        fake = torch.zeros(X.size(0), 1, 4, 4)
        fake = fake.type_as(X)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_discriminator.step()
        opt_discriminator.zero_grad()
        self.untoggle_optimizer(opt_discriminator)

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_loss = round(self.g_loss.item(), 4)
        self.d_loss = round(self.d_loss.item(), 4)

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        return opt_generator, opt_discriminator
    
    def on_train_epoch_end(self):
        
        self.epoch += 1

        gen_img = self.generated_imgs[-1]
        gen_img = gen_img.to('cpu').detach().numpy().reshape(self.input_size[0], self.input_size[1], 3)
        plt.imshow(gen_img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        folder_path = 'gan_images'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(f'{folder_path}/epoch={self.epoch}-g_loss={self.g_loss}-d_loss={self.d_loss}.png')