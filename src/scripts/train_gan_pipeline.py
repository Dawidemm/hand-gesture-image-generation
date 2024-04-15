import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping

from src.models.generator import HandsGenerator
from src.models.discriminator import HandsDiscriminator
from src.models.gan import HandsGAN
from src.utils.lit_hands_datamodule import LightningHandsDatamodule


DATASET_DIR = 'dataset'
BATCH_SIZE = 16
MAX_EPOCHS = 250

def train_gan(
        dataset_directory: str,
        batch_size: int,
        max_epochs: int
    ):

    datamodule = LightningHandsDatamodule(
        root_directory=dataset_directory,
        batch_size=batch_size
    )
    datamodule.setup(stage='train')

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator='auto',
        logger=True
    )

    generator = HandsGenerator(
        img_shape=(128, 128)
    )
    
    discriminator = HandsDiscriminator(
        img_shape=(128, 128)
    )

    model = HandsGAN(
        latent_dim=32,
        generator=generator,
        discriminator=discriminator
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_gan(
        dataset_directory=DATASET_DIR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )