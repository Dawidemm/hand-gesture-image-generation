import lightning

from src.utils.lit_hands_datamodule import LightningHandsDatamodule
from src.models.gan import HandsGAN

from typing import Tuple


DATASET_DIR = 'dataset'
BATCH_SIZE = 128
MAX_EPOCHS = 50

IMG_SIZE = (64, 64)
LATNET_DIM = 128


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

    trainer = lightning.Trainer(
        max_epochs=max_epochs, 
        accelerator='auto',
        logger=True
    )

    model = HandsGAN(
        input_size=img_size,
        latent_dim=latent_dim,
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_gan(
        dataset_directory=DATASET_DIR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        img_size=IMG_SIZE,
        latent_dim=LATNET_DIM
    )