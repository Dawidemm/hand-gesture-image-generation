import lightning

from src.utils.lit_hands_datamodule import LightningHandsDatamodule
from src.models.gan import HandsGAN

from src.utils import utils

from typing import Tuple


DATASET_DIR = 'dataset'
BATCH_SIZE = 64
MAX_EPOCHS = 25

IMG_SIZE = (128, 128)
LATNET_DIM = 64


def train_gan(
        dataset_directory: str,
        batch_size: int,
        max_epochs: int,
        img_size: Tuple[int, int],
        latent_dim: int
    ):

    datamodule = LightningHandsDatamodule(
        root_directory=dataset_directory,
        batch_size=batch_size
    )
    datamodule.setup(stage='fit')

    loss_logs = utils.LossLoggerCallback()

    trainer = lightning.Trainer(
        max_epochs=max_epochs, 
        accelerator='auto',
        logger=True,
        callbacks=[loss_logs]
    )

    model = HandsGAN(
        input_size=img_size,
        latent_dim=latent_dim,
    )

    trainer.fit(model=model, datamodule=datamodule)

    utils.plot_gan_loss(
        epochs=max_epochs,
        g_loss=loss_logs.generator_losses,
        d_loss=loss_logs.discriminator_losses,
        save_dir='.',
        format='png'
    )

if __name__ == '__main__':
    train_gan(
        dataset_directory=DATASET_DIR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        img_size=IMG_SIZE,
        latent_dim=LATNET_DIM
    )