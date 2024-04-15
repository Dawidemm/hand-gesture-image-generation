import torch
import torch.nn as nn
import lightning
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt

from src.models.generator import HandsGenerator
from src.models.discriminator import HandsDiscriminator


class HandsGAN(lightning.LightningModule):
    def __init__(
            self,
            latent_dim: int,
            generator: nn.Module,
            discriminator: nn.Module,
            learning_rate: float=0.0005
    ):
        super().__init__()

        self.automatic_optimization = False

        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate

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

    def configure_optimizers(self):
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        return opt_generator, opt_discriminator
    
    def on_train_epoch_end(self):
        
        gen_img = self.generated_imgs[:4]
        gen_img = gen_img.to('cpu').detach().numpy()
        fig, axs = plt.subplots(2, 2, figsize=(6, 5))

        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(gen_img):
                    img = gen_img[idx].reshape(128, 128, 3)
                    axs[i, j].imshow(img)
                    axs[i, j].axis('off')
                else:
                    axs[i, j].axis('off')

        self.epoch += 1

        plt.tight_layout()
        plt.savefig(f'gan_imgs/epoch={self.epoch}-g_loss={self.g_loss}-d_loss={self.d_loss}.png')