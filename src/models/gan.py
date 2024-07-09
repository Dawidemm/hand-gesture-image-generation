import os
import torch
import lightning
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt

from src.models.generator import HandsGenerator
from src.models.discriminator import HandsDiscriminator
from src.models.weights_initializer import weights_init

from typing import Tuple


class HandsGAN(lightning.LightningModule):
    def __init__(
            self,
            input_size: Tuple[int, int],
            latent_dim: int,
            learning_rate: float=0.0002
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

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.epoch = -1

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        
        X, _ = batch

        opt_generator, opt_discriminator = self.optimizers()

        z = torch.randn(X.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(X)

        self.generated_images = self(z)

        real_image = torch.ones(X.size(0), 1, 1, 1)
        real_image = real_image.type_as(X)

        generator_loss = self.adversarial_loss(self.discriminator(self.generated_images), real_image)
        self.log("g_loss", generator_loss, prog_bar=True)

        opt_generator.zero_grad()
        self.manual_backward(generator_loss)
        opt_generator.step()

        real_loss = self.adversarial_loss(self.discriminator(X), real_image)

        fake_image = torch.zeros(X.size(0), 1, 1, 1)
        fake_image = fake_image.type_as(X)

        fake_loss = self.adversarial_loss(self.discriminator(self.generated_images.detach()), fake_image)

        discriminator_loss = real_loss + fake_loss
        self.log("d_loss", discriminator_loss, prog_bar=True)

        opt_discriminator.zero_grad()
        self.manual_backward(discriminator_loss)
        opt_discriminator.step()

        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_loss = round(self.generator_loss.item(), 4)
        self.discriminator_loss = round(self.discriminator_loss.item(), 4)

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        return opt_generator, opt_discriminator
    
    def on_train_epoch_end(self):
        
        self.epoch += 1

        gen_img = self.generated_images[0]
        gen_img = gen_img.to('cpu').detach().numpy().transpose(1, 2, 0)
        plt.imshow(gen_img)
        plt.axis('off')
        plt.tight_layout()

        folder_path = 'gan_images'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        plt.savefig(f'{folder_path}/epoch={self.epoch}-g_loss={self.generator_loss}-d_loss={self.discriminator_loss}.png')