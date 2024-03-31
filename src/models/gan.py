import torch
import torch.nn as nn
import lightning
from torch.nn.functional import binary_cross_entropy

from src.models.generator import HandsGenerator
from src.models.discriminator import HandsDiscriminator


class HandsGAN(lightning.LightningModule):
    def __init__(
            self,
            latent_dim: int,
            generator: nn.Module,
            discriminator: nn.Module,
            learning_rate: float=0.01
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # Train Discriminator
        if optimizer_idx == 0:
            z = torch.randn(batch.size(0), self.latent_dim, device=self.device)
            fake_images = self.generator(z)
            real_pred = self.discriminator(batch)
            fake_pred = self.discriminator(fake_images.detach())
            real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
            d_loss = (real_loss + fake_loss) / 2
            return d_loss

        # Train Generator
        if optimizer_idx == 1:
            z = torch.randn(batch.size(0), self.latent_dim, device=self.device)
            fake_images = self.generator(z)
            fake_pred = self.discriminator(fake_images)
            g_loss = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
            return g_loss
    
    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        return [opt_generator, opt_discriminator], []